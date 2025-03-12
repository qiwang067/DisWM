from time import process_time_ns

import torch
# from caffe2.python.fakelowp.test_utils import print_net
from numpy.ma.core import shape
from numpy.testing.print_coercion_tables import print_new_cast_table
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torch.fx.experimental.migrate_gradual_types.operation import op_neq

import pretrain_networks as networks
import tools
to_np = lambda x: x.detach().cpu().numpy()

import beta_vae
import torch.nn.functional as F
import wandb


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config

    self.beta_vae = config.beta_vae
    if self.beta_vae:
      embed_size = config.z_dim
      self.z_dim = config.z_dim
      self.encoder = beta_vae.BetaVAE_H_Encoder(z_dim=embed_size, nc=3)
    else:
      self.encoder = networks.ConvEncoder(config.grayscale,
                                          config.cnn_depth, config.act, config.encoder_kernels)
      if config.size[0] == 64 and config.size[1] == 64:
        embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
        embed_size *= 2 * 2
      else:
        raise NotImplemented(f"{config.size} is not applicable now")


    num_actions = config.pretrain_action_num
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        num_actions, embed_size, config.device)
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size       # (3, 64, 64)
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


    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)

    self.beta = config.beta_vae_beta


  def _train(self, data):
    data = self.preprocess(data)

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):

        if self.beta_vae:
          mu, logvar, embed = self.encoder(data, pred=False)
          total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
          total_b_vae_kld = total_kld[0]
        else:
          embed = self.encoder(data)
        post, prior = self.dynamics.observe_action_free(embed)
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        losses = {}
        likes = {}
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          feat = self.dynamics.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat)
          like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like)


        if self.beta_vae:
          model_loss = sum(losses.values()) + kl_loss + self.beta * total_b_vae_kld
        else:
          model_loss = sum(losses.values()) + kl_loss
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    wandb.log({'kl_balance': kl_balance,
               'kl_free': kl_free,
               'kl_scale': kl_scale,
    })

    if self.beta_vae:
      wandb.log({'beta_vae_kl_loss': to_np(total_b_vae_kld)})
      metrics['beta_vae_kl_loss'] = to_np(total_b_vae_kld)

    metrics['kl'] = to_np(torch.mean(kl_value))
    wandb.log({'kl': to_np(torch.mean(kl_value))})
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      wandb.log({'prior_ent': to_np(torch.mean(self.dynamics.get_dist(prior).entropy())),
                 'post_ent': to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
                 })
      context = dict(
          embed=embed, feat=self.dynamics.get_feat(post),
          kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    if self.beta_vae:
      mu, logvar, embed = self.encoder(data, pred=True)
    else:
      embed = self.encoder(data)
    states, _ = self.dynamics.observe_action_free(embed[:6, :5])
    recon = self.heads['image'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    print(torch.cat([truth, model, error], 2).shape)
    return torch.cat([truth, model, error], 2)

  def reparameterize(self, mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + eps * std

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

  def traverse(self, data, path, limit=9, inter=3, loc=-1):
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
        states, _ = self.dynamics.observe_action_free(z[:1, 49:])
        recon = self.heads['image'](self.dynamics.get_feat(states)).mode()[:1]
        model = recon + 0.5
        samples.append(model)
      samples = torch.cat(samples, 0)
      rows.append(samples)

    image = torch.cat(rows, 2)
    image = image.reshape(7, self.z_dim * 64, 64, 3) * 255
    image = image.permute(1, 0, 2, 3).reshape(self.z_dim*64, 7*64, 3)
    image = np.clip(image.cpu().numpy(), 0, 255).astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    image.save(path)
    image = wandb.Image(image, caption="z_dim:{}, val:gt+({})".format(row + 1, val))
    # save transversal image
    wandb.log({"traversal image": image})

    return torch.cat(rows, 2)