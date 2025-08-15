from time import process_time_ns
import torch
from numpy.ma.core import shape
from numpy.testing.print_coercion_tables import print_new_cast_table
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torch.fx.experimental.migrate_gradual_types.operation import op_neq
import networks
import tools
to_np = lambda x: x.detach().cpu().numpy()
import beta_vae
import torch.nn.functional as F
import wandb
import exploration as expl


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config

    self.pretrain = config.beta_vae_pretrain
    self.distillation = config.distillation

    self.fine_tune = config.fine_tune
    self.naive_fine_tune = config.naive_fine_tune

    self.beta_vae = config.beta_vae
    if self.beta_vae:
      embed_size = config.z_dim

      self.z_dim = config.z_dim
      self.encoder = beta_vae.BetaVAE_H_Encoder(z_dim=embed_size, nc=3)
      if self.distillation:
        self.encoder_distillation = beta_vae.BetaVAE_H_Encoder(z_dim=embed_size, nc=3)
    else:
      self.encoder = networks.ConvEncoder(config.grayscale,
                                          config.cnn_depth, config.act, config.encoder_kernels)
      if config.size[0] == 64 and config.size[1] == 64:
        embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
        embed_size *= 2 * 2
      else:
        raise NotImplemented(f"{config.size} is not applicable now")

      # DisWM pretrain beta-VAE encoder network
      if self.distillation:
        self.encoder_distillation = networks.ConvEncoder(config.grayscale,
                                          config.cnn_depth, config.act, config.encoder_kernels)


    if self.fine_tune:
      num_actions = config.source_action_num
      import pretrain_networks
      feat_size = config.dyn_stoch + config.dyn_deter
      self.dynamics_af = pretrain_networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        num_actions, embed_size, config.device)
      self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, feat_size, config.device)
    elif self.naive_fine_tune:
      num_actions = config.source_action_num
      import pretrain_networks
      self.dynamics_af = pretrain_networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        num_actions, embed_size, config.device)
    else:
      self.dynamics = networks.RSSM(
          config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
          config.dyn_input_layers, config.dyn_output_layers,
          config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
          config.act, config.dyn_mean_act, config.dyn_std_act,
          config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
          config.num_actions, embed_size, config.device)
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    if self.beta_vae:
      self.heads['image'] = beta_vae.BetaVAE_H_Decoder(shape=shape, feat_size=feat_size, z_dim=embed_size, nc=3)
    else:
      self.heads['image'] = networks.ConvDecoder(
          feat_size,  # pytorch version
          config.cnn_depth, config.act, shape, config.decoder_kernels,
          config.decoder_thin)
    self.heads['reward'] = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    if config.distillation:
      self.encoder_distillation.requires_grad_(False)
    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)

    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)

    # the beta value of beta-VAE
    self.beta = config.beta_vae_beta

    if config.intr_beta != 0 and config.fine_tune:
      self.intr_bonus = expl.VideoIntrBonus(
        config.intr_beta, config.k, config.intr_seq_length,
        feat_size,
        config.queue_dim,
        config.queue_size,
        config.intr_reward_norm,
        config.beta_type,
      )

  def _train(self, data):
    data = self.preprocess(data)
    if self.naive_fine_tune:
      data['action'] = torch.nn.functional.pad(data['action'], (0, self._config.action_num_gap))
    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):

        if self.beta_vae:
          mu, logvar, embed = self.encoder(data, pred=False)
          total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
          total_b_vae_kld = total_kld[0]
          if self.distillation:
            mu_dis, logvar_dis, embed_dis = self.encoder_distillation(data, pred=False)
        else:
          embed = self.encoder(data)
          if self.distillation:
            embed_dis = self.encoder_distillation(data)


        if self.fine_tune:
          post_af, prior_af = self.dynamics_af.observe_action_free(embed)
          embed_af = self.dynamics_af.get_feat(post_af)

          post, prior = self.dynamics.observe(embed_af, data['action'])
        elif self.naive_fine_tune:
          post, prior = self.dynamics_af.observe(embed, data['action'])
        else:
          post, prior = self.dynamics.observe(embed, data['action'])
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        if self.naive_fine_tune:
          kl_loss, kl_value = self.dynamics_af.kl_loss(
              post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        else:
          kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        if self.fine_tune:
          kl_loss_af, kl_value_af = self.dynamics.kl_loss(
            post_af, prior_af, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        

        if self.distillation:
          kl_loss_dis = self.kl_loss_distillation(embed, embed_dis)

        if self.naive_fine_tune:
          feat = self.dynamics_af.get_feat(post)
        else:
          feat = self.dynamics.get_feat(post)
        ## apv-finetune intrinsic reward
        if self.fine_tune:
          plain_reward = data["reward"]
          if self._config.intr_beta != 0:
            data, intr_rew_len, int_rew_mets = self.intr_bonus.compute_bonus(data, embed_af)

        losses = {}
        likes = {}
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)

          inp = feat if grad_head else feat.detach()

          if name == 'reward' and self.fine_tune:
            inp = inp[:, :intr_rew_len]
          pred = head(inp)
          like = pred.log_prob(data[name])
          likes[name] = like

          # compute KL loss
          if name == 'image' and self.beta_vae:
            # add beta-VAE KL loss for DisWM
            losses[name] = -torch.mean(like) * self._scales.get(name, 1.0) + self.beta * total_b_vae_kld
          else:
            losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

        if self.fine_tune:
          model_loss = sum(losses.values()) + kl_loss + kl_loss_af
        elif self.distillation:
          if self._config.cross_domain:   # We use this trick for cross-domain tasks (eg. dmc_walker_walk -> mujoco_pusher-v5), excessive distillation loss leads to gradient explosion.  
            if self._step >= 50000:
              eta = max(2 - self._step / 50000, 0.1) * 0.1  # eta: eta value for distillation loss.
              model_loss = sum(losses.values()) + kl_loss + eta * kl_loss_dis 
            else:
              model_loss = sum(losses.values()) + kl_loss
          else:
            eta = max(1 - self._step / 50000, 0.1) * 0.1
            model_loss = sum(losses.values()) + kl_loss + eta * kl_loss_dis
        else:
          model_loss = sum(losses.values()) + kl_loss
      metrics = self._model_opt(model_loss, self.parameters())



    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    if self.beta_vae:
      metrics['beta_vae_kl_loss'] = to_np(total_b_vae_kld)
    if self.fine_tune:
      metrics['action_free_kl_loss'] = to_np(kl_loss_af)
    if self.distillation:
      metrics['distillation_kl_loss'] = to_np(kl_loss_dis)
    metrics['kl'] = to_np(torch.mean(kl_value))
    if self.naive_fine_tune:
      with torch.cuda.amp.autocast(self._use_amp):
        metrics['prior_ent'] = to_np(torch.mean(self.dynamics_af.get_dist(prior).entropy()))
        metrics['post_ent'] = to_np(torch.mean(self.dynamics_af.get_dist(post).entropy()))
        context = dict(
          embed=embed, feat=self.dynamics_af.get_feat(post),
          kl=kl_value, postent=self.dynamics_af.get_dist(post).entropy())
    else:
      with torch.cuda.amp.autocast(self._use_amp):
        metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
        metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
        context = dict(
            embed=embed, feat=self.dynamics.get_feat(post),
            kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs



  def video_pred(self, data):
    data = self.preprocess(data)
    if self.naive_fine_tune:
      data['action'] = torch.nn.functional.pad(data['action'], (0, self._config.action_num_gap))
    truth = data['image'][:6] + 0.5
    if self.beta_vae:
      mu, logvar, embed = self.encoder(data, pred=True)
    else:
      embed = self.encoder(data)
    if self.fine_tune:
      embed, _ = self.dynamics_af.observe_action_free(embed[:6, :5])
      embed = self.dynamics_af.get_feat(embed)
    if self.naive_fine_tune:
      states, _ = self.dynamics_af.observe(embed[:6, :5], data['action'][:6, :5])
      recon = self.heads['image'](
        self.dynamics_af.get_feat(states)).mode()[:6]
      reward_post = self.heads['reward'](
        self.dynamics_af.get_feat(states)).mode()[:6]
      init = {k: v[:, -1] for k, v in states.items()}
      prior = self.dynamics_af.imagine(data['action'][:6, 5:], init)
      openl = self.heads['image'](self.dynamics_af.get_feat(prior)).mode()
      reward_prior = self.heads['reward'](self.dynamics_af.get_feat(prior)).mode()
    else:
      states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
      recon = self.heads['image'](
          self.dynamics.get_feat(states)).mode()[:6]
      reward_post = self.heads['reward'](
          self.dynamics.get_feat(states)).mode()[:6]
      init = {k: v[:, -1] for k, v in states.items()}
      prior = self.dynamics.imagine(data['action'][:6, 5:], init)
      openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
      reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    return torch.cat([truth, model, error], 2)

  # Reparameterize function. 
  def reparameterize(self, mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + eps * std

  # Distillation kl loss function.
  def kl_loss_distillation(self, embed, embed_distillation):
    embed = torch.nn.functional.log_softmax(embed, dim=-1)
    embed_distillation = torch.nn.functional.softmax(embed_distillation, dim=-1)
    loss = F.kl_div(embed, embed_distillation, reduction='batchmean')
    return loss

  # Beta-VAE KL loss function.
  def kl_divergence(self, mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
      mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
      logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - (logvar.exp() + 1e-6))
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

  def reconstruction_loss(self, x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0
    if distribution == 'bernoulli':
      recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
      x_recon = F.sigmoid(x_recon)
      recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
      recon_loss = None
    return recon_loss
  

  # Beta-VAE traversal function.
  def traverse(self, data, limit=9, inter=3, loc=-1):
    data = self.preprocess(data)
    mu, logvar, embed = self.encoder(data, pred=True)
    interpolation = torch.arange(-limit, limit + 0.1, inter)
    rows = []
    for row in range(self.z_dim):
      samples = []
      if loc != -1 and row != loc:
        continue
      z = embed.clone()
      for val in interpolation:
        z[:, :, row] = val + embed[:, :, row]
        if self.fine_tune:
          post, prior = self.dynamics_af.observe_action_free(z[:1, 49:])
          embed_af = self.dynamics_af.get_feat(post)
          states, _ = self.dynamics.observe(embed_af, data['action'][:1, 49:])
        else:
          states, _ = self.dynamics.observe(z[:1, 49:], data['action'][:1, 49:])
        recon = self.heads['image'](self.dynamics.get_feat(states)).mode()[:1]
        model = recon + 0.5
        samples.append(model)
      samples = torch.cat(samples, 0)
      rows.append(samples)
    # save transversal image for wandb
    image = torch.cat(rows, 2)
    image = image.reshape(7, self.z_dim * 64, 64, 3) * 255
    image = image.permute(1, 0, 2, 3).reshape(self.z_dim*64, 7*64, 3)
    image = np.clip(image.cpu().numpy(), 0, 255).astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    image = wandb.Image(image, caption="z_dim:{}, val:gt+({})".format(row + 1, val))
    wandb.log({"traversal image": image})
    return torch.cat(rows, 2)


class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.actor = networks.ActionHead(
        feat_size,  # pytorch version
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    self.value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)
      self._updates = 0
    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)

  def _train(
      self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}

    with tools.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(self._use_amp):

        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats)

        reward = objective(imag_feat, imag_state, imag_action)

        actor_ent = self.actor(imag_feat).entropy()

        if self._config.naive_fine_tune:
          state_ent = self._world_model.dynamics_af.get_dist(
            imag_state).entropy()
        else:
          state_ent = self._world_model.dynamics.get_dist(
              imag_state).entropy()

        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target)

        actor_loss, mets = self._compute_actor_loss(
            imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
            weights)

        metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_value_target)
        value_input = imag_feat

    with tools.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())
        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))
    return imag_feat, imag_state, imag_action, weights, metrics


  def _imagine(self, start, policy, horizon, repeats=None):
    if self._config.naive_fine_tune:
      dynamics = self._world_model.dynamics_af
    else:
      dynamics = self._world_model.dynamics
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = feat.detach() if self._stop_grad_actor else feat
      action = policy(inp).sample()
      if self._config.naive_fine_tune:
        action = torch.nn.functional.pad(action, (0, self._config.action_num_gap))

      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action
    feat = 0 * dynamics.get_feat(start)
    action = policy(feat).mode()
    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon)], (start, feat, action))
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
      weights):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1


