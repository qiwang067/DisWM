import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()

import tools
import pathlib
import pretrain_model

import wandb
import wrappers
import functools
import numpy as np
from datetime import datetime

class Dreamer(nn.Module):

  def __init__(self, config, logger, dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = dataset
    self._wm = pretrain_model.WorldModel(self._step, config)

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return state


  def _train(self, data):
    metrics = {}
    post, context, mets = self._wm._train(data)
    metrics.update(mets)
    start = post

    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  random = np.random.RandomState(2)
  print(len(random.choice(list(episodes.values()))))
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, mode):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == 'gymnasium':
    if mode == "train":
      test = False
    else:
      test = True
    env = wrappers.gymnasium_wrapper(task, config.action_repeat, config.size, test=test, max_delta=config.max_delta,
                                     step_std=config.step_std)
    env = wrappers.NormalizeActions(env)
  elif suite == "dmc2gym":
    if mode == "train":
      test = False
    else:
      test = True
    env = wrappers.dmc2gym_wrapper(config, task, test)
    env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  return env


class Beta_VAE_Pretrain:
    def __init__(self):
        super(Beta_VAE_Pretrain, self).__init__()

    def load_pretrain_episode(self, directory):

        episodes = tools.load_episodes(directory)

        return episodes

    def train(self, config):

        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = pathlib.Path(config.logdir).expanduser()
        logdir = logdir / config.task
        print('Logdir', logdir)
        logdir = logdir / 'seed_{}'.format(config.seed)
        logdir = logdir / timestamp
        config.traindir = config.traindir or logdir / 'train_eps'
        config.steps //= config.action_repeat
        config.act = getattr(torch.nn, config.act)

        print('Logdir', logdir)
        logdir.mkdir(parents=True, exist_ok=True)



        print('Load Pre-train Datasets.')
        directory = config.pretrain_datasets_path
        pretrain_directory = pathlib.Path(directory).expanduser()
        train_eps = self.load_pretrain_episode(directory)
        pretrain_total_steps = count_steps(pretrain_directory)
        logger = tools.Logger(logdir, config.action_repeat * pretrain_total_steps)
        train_dataset = make_dataset(train_eps, config)


        print('Create envs.')


        make = lambda mode: make_env(config, mode)
        train_envs = [make('train') for _ in range(config.envs)]
        acts = train_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]


        print('Create agent.')

        agent = Dreamer(config, logger, train_dataset).to(config.device)
        agent.requires_grad_(requires_grad=False)

        if (logdir / 'latest_model.pt').exists():
            agent.load_state_dict(torch.load(logdir / 'latest_model.pt'))
            agent._should_pretrain._once = False

        state = None
        print('Start Pre-training.')
        count_pre_train_step = 0
        while count_pre_train_step + 1 < config.pretrain_steps:
            for pretrain_step in range(0, 100000, 50):
                agent._train(next(train_dataset))

            count_pre_train_step += 100000
            print('Saving checkpoints.')
            check_point_path = pathlib.Path(logdir / str(count_pre_train_step)).expanduser()
            check_point_path.mkdir(parents=True, exist_ok=True)
            print('step:', count_pre_train_step)
            torch.save(agent._wm.encoder.state_dict(), check_point_path / 'encoder.pt')
            torch.save(agent._wm.heads['image'].state_dict(), check_point_path / 'decoder.pt')
            torch.save(agent._wm.dynamics.state_dict(), check_point_path / 'rssm.pt')
            if config.beta_vae:
                agent._wm.traverse(next(train_dataset), path=check_point_path / 'traverse.png')
        torch.save(agent._wm.encoder.state_dict(), logdir / 'encoder.pt')
        torch.save(agent._wm.heads['image'].state_dict(), logdir / 'decoder.pt')
        torch.save(agent._wm.dynamics.state_dict(), logdir / 'rssm.pt')

        for env in train_envs:
            try:
                env.close()
                wandb.finish()
            except Exception:
                pass

