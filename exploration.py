import torch
from torch import nn
from torch import distributions as torchd

import models
import networks
import tools


class Random(nn.Module):

  def __init__(self, config):
    self._config = config

  def actor(self, feat):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return tools.OneHotDist(torch.zeros(shape))
    else:
      ones = torch.ones(shape)
      return tools.ContDist(torchd.uniform.Uniform(-ones, ones))

  def train(self, start, context):
    return None, {}


class Plan2Explore(nn.Module):

  def __init__(self, config, world_model, reward=None):
    self._config = config
    self._reward = reward
    self._behavior = models.ImagBehavior(config, world_model)
    self.actor = self._behavior.actor
    stoch_size = config.dyn_stoch
    if config.dyn_discrete:
      stoch_size *= config.dyn_discrete
    size = {
        'embed': 32 * config.cnn_depth,
        'stoch': stoch_size,
        'deter': config.dyn_deter,
        'feat': config.dyn_stoch + config.dyn_deter,
    }[self._config.disag_target]
    kw = dict(
        inp_dim=config.dyn_stoch,  # pytorch version
        shape=size, layers=config.disag_layers, units=config.disag_units,
        act=config.act)
    self._networks = [
        networks.DenseHead(**kw) for _ in range(config.disag_models)]
    self._opt = tools.optimizer(config.opt, self.parameters(),
        config.model_lr, config.opt_eps, config.weight_decay)
    self._opt = tools.Optimizer(
       'ensemble', config.model_lr, config.opt_eps, config.grad_clip,
       config.weight_decay, opt=config.opt)

  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self._config.dyn_discrete:
      stoch = torch.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self._config.disag_target]
    inputs = context['feat']
    if self._config.disag_action_cond:
      inputs = torch.concat([inputs, data['action']], -1)
    metrics.update(self._train_ensemble(inputs, target))
    metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
    return None, metrics

  def _intrinsic_reward(self, feat, state, action):
    inputs = feat
    if self._config.disag_action_cond:
      inputs = torch.concat([inputs, action], -1)
    preds = [head(inputs, torch.float32).mean() for head in self._networks]
    disag = torch.reduce_mean(torch.math.reduce_std(preds, 0), -1)
    if self._config.disag_log:
      disag = torch.math.log(disag)
    reward = self._config.expl_intr_scale * disag
    if self._config.expl_extr_scale:
      reward += torch.cast(self._config.expl_extr_scale * self._reward(
          feat, state, action), torch.float32)
    return reward

  def _train_ensemble(self, inputs, targets):
    with torch.cuda.amp.autocast(self._use_amp):
      if self._config.disag_offset:
        targets = targets[:, self._config.disag_offset:]
        inputs = inputs[:, :-self._config.disag_offset]
      targets = torch.stop_gradient(targets)
      inputs = tf.stop_gradient(inputs)

      preds = [head(inputs) for head in self._networks]
      likes = torch.cat(
        [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
      )
      loss = -torch.mean(likes)
    metrics = self._expl_opt(loss, self._networks.parameters())
    return metrics


class VideoIntrBonus(nn.Module):
    def __init__(
        self,
        beta,
        k,
        intr_seq_length,
        feat_dim,
        queue_dim,
        queue_size,
        reward_norm,
        beta_type='abs',
    ) -> None:
        super().__init__()

        self.beta = beta
        self.k = k
        self.intr_seq_length = intr_seq_length
        self.tf_queue_step = 0
        self.tf_queue_size = queue_size
        shape = (feat_dim, queue_dim)
        self.random_projection_matrix = torch.nn.Parameter(
            torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) / queue_dim),
            requires_grad=False,
        )
        self.register_buffer('queue', torch.zeros(queue_size, queue_dim))
        self.intr_rewnorm = tools.StreamNorm(**reward_norm)

        self.beta_type = beta_type
        if self.beta_type == 'rel':
            self.plain_rewnorm = tools.StreamNorm()

    def construct_queue(self, seq_feat):
        with torch.no_grad():
            seq_size = seq_feat.shape[0]
            self.queue.data[seq_size:] = self.queue.data[:-seq_size].clone()
            self.queue.data[:seq_size] = seq_feat.data

            self.tf_queue_step = self.tf_queue_step + seq_size
            self.tf_queue_step = min(self.tf_queue_step, self.tf_queue_size)
        return self.queue[: self.tf_queue_step]

    def compute_bonus(self, data, feat):
        data['reward'] = data['reward'].squeeze()
        with torch.no_grad():
            seq_feat = feat
            # NOTE: seq_feat [B, T, D], after unfold [B, T-S+1, D, S]
            seq_feat = seq_feat.unfold(dimension=1, size=self.intr_seq_length, step=1).mean(dim=-1)
            seq_feat = torch.matmul(seq_feat, self.random_projection_matrix)
            b, t, d = (seq_feat.shape[0], seq_feat.shape[1], seq_feat.shape[2])
            seq_feat = torch.reshape(seq_feat, (b * t, d))
            queue = self.construct_queue(seq_feat)
            dist = torch.norm(seq_feat[:, None, :] - queue[None, :, :], dim=-1)
            int_rew = -1.0 * torch.topk(
                -dist, k=min(self.k, queue.shape[0])
            ).values.mean(1)
            int_rew = int_rew.detach()
            int_rew, int_rew_mets = self.intr_rewnorm(int_rew)
            int_rew_mets = {f"intr_{k}": v for k, v in int_rew_mets.items()}
            int_rew = torch.reshape(int_rew, (b, t))

            plain_reward = data["reward"]
            if self.beta_type == 'abs':
                data["reward"] = data["reward"][:, :t] + self.beta * int_rew.detach()
                data["reward"] = data["reward"].unsqueeze(-1)
            elif self.beta_type == 'rel':
                self.plain_rewnorm.update(data["reward"])
                beta = self.beta * self.plain_rewnorm.mag.item()
                data["reward"] = data["reward"][:, :t] + beta * int_rew.detach()
                int_rew_mets["abs_beta"] = beta
                int_rew_mets["plain_reward_mean"] = self.plain_rewnorm.mag.item()
            else:
                raise NotImplementedError

            if int_rew_mets['intr_mean'] < 1e-5:
                print("intr_rew too small:", int_rew_mets['intr_mean'])

            int_rew_mets["plain_reward_mean"] = plain_reward.mean().item()
            int_rew_mets["intr_mag"] = self.intr_rewnorm.mag.item()

        return data, t, int_rew_mets
