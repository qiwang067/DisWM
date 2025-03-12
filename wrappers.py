import threading
import traceback
import sys
import cloudpickle
import atexit

import gym
import numpy as np
from PIL import Image
from numpy.distutils.command.config import config
from panda_gym import reward_type
from functools import partial

import os

import dmc2gym


class DeepMindLabyrinth(object):

  ACTION_SET_DEFAULT = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
      (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
      (0, 0, 0, 0, 1, 0, 0),    # Fire
  )

  ACTION_SET_MEDIUM = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (0, 0, 0, 0, 0, 0, 0),    # Idle.
  )

  ACTION_SET_SMALL = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
  )

  def __init__(
      self, level, mode, action_repeat=4, render_size=(64, 64),
      action_set=ACTION_SET_DEFAULT, level_cache=None, seed=None,
      runfiles_path=None):
    assert mode in ('train', 'test')
    import deepmind_lab
    if runfiles_path:
      print('Setting DMLab runfiles path:', runfiles_path)
      deepmind_lab.set_runfiles_path(runfiles_path)
    self._config = {}
    self._config['width'] = render_size[0]
    self._config['height'] = render_size[1]
    self._config['logLevel'] = 'WARN'
    if mode == 'test':
      self._config['allowHoldOutLevels'] = 'true'
      self._config['mixerSeed'] = 0x600D5EED
    self._action_repeat = action_repeat
    self._random = np.random.RandomState(seed)
    self._env = deepmind_lab.Lab(
        level='contributed/dmlab30/'+level,
        observations=['RGB_INTERLEAVED'],
        config={k: str(v) for k, v in self._config.items()},
        level_cache=level_cache)
    self._action_set = action_set
    self._last_image = None
    self._done = True

  @property
  def observation_space(self):
    shape = (self._config['height'], self._config['width'], 3)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(self._action_set))

  def reset(self):
    self._done = False
    self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
    obs = self._get_obs()
    return obs

  def step(self, action):
    raw_action = np.array(self._action_set[action], np.intc)
    reward = self._env.step(raw_action, num_steps=self._action_repeat)
    self._done = not self._env.is_running()
    obs = self._get_obs()
    return obs, reward, self._done, {}

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._last_image

  def close(self):
    self._env.close()

  def _get_obs(self):
    if self._done:
      image = 0 * self._last_image
    else:
      image = self._env.observations()['RGB_INTERLEAVED']
    self._last_image = image
    return {'image': image}



class DeepMindControl:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class gymnasium_wrapper:
  # wrapper for gymnasium_mujoco_pusher_v5 environment
  def __init__(
      self, name, action_repeat=4, size=(128, 128), test=None, max_delta=None, step_std=None, color_setting=False):
    assert size[0] == size[1]

    # camera configs
    DEFAULT_CAMERA_CONFIG = {
      "trackbodyid": -1,
      "distance": 2.0,
      "elevation": -90,
      "lookat": np.array([0, -0.5, 0.2]),
    }

    import gymnasium.wrappers
    import gymnasium as gym
    from gym.spaces import Box

    env = gym.make(id=name, render_mode='rgb_array', default_camera_config=DEFAULT_CAMERA_CONFIG)
    env._get_obs = lambda: None
    self._env = env
    self._action_repeat = action_repeat
    self._size = size[0]
    id = self._env.unwrapped.model.body("object").geomadr
    self._env.unwrapped.model.geom_rgba[id] = np.array([0, 0, 0, 1])


    if color_setting is not None:
      self.test = test
      self.max_delta = max_delta
      self.step_std = step_std
      self._current_rgb = None

  @property
  def observation_space(self):
    obs_space = {'observation': self._env.observation_space, 'image': gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)}
    return obs_space

  @property
  def action_space(self):
    spec = self._env.action_space   # Box(-2.0, 2.0, (7,), float32)
    return spec

  def _reset_color(self, test):
    if not test:
      self.origin_color = np.array([0.7, 0.3, 0.5])

      self._max_rgb = np.clip(self.origin_color + self.max_delta, 0.0, 1.0)
      self._min_rgb = np.clip(self.origin_color - self.max_delta, 0.0, 1.0)

    else:
      self.origin_color = np.array([0.5, 0.7, 0.3])

      self._max_rgb = np.clip(self.origin_color + self.max_delta, 0.0, 1.0)
      self._min_rgb = np.clip(self.origin_color - self.max_delta, 0.0, 1.0)

    r = np.random.RandomState().uniform(size=self._min_rgb.shape)
    self._current_rgb = self._min_rgb + r * (self._max_rgb - self._min_rgb)
    self._env.unwrapped.model.geom("table").rgba[:3] = self._min_rgb + np.random.RandomState().uniform(size=self._min_rgb.shape) * (self._max_rgb - self._min_rgb)
    self._env.unwrapped.model.geom("uar").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("ua").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("ef").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("sp").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("fa").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("sl").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("e1").rgba[:3] = self._current_rgb
    self._env.unwrapped.model.geom("e2").rgba[:3] = self._current_rgb

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      observation, reward, done, _, info = self._env.step(action)
      reward += reward or 0
      if done:
        break
    obs = {'observation': observation}
    image = self.render()
    image = Image.fromarray(image)
    image = image.resize((self._size, self._size))
    obs['image'] = np.array(image)

    return obs, reward, done, info

  def reset(self):
    observation = self._env.reset()
    self._reset_color(test=self.test)
    obs = {'observation': observation[0]}
    image = self.render()
    image = Image.fromarray(image)
    image = image.resize((self._size, self._size))
    obs['image'] = np.array(image)
    return obs

  def render(self):
    return self._env.render()


class dmc2gym_wrapper:

  def __init__(self, config, name, test):
    domain, task = name.split("_", 1)
    camera_id = 2 if domain == 'quadruped' else 0
    if not test:
      difficulty = config.difficulty
      dynamic = config.dynamic
      camera_kwargs = config.camera_kwargs
      background_kwargs = config.background_kwargs
      background_dataset_path = config.background_dataset_path
      background_dataset_videos = config.background_dataset_videos
      color_kwargs = config.colour_kwargs
    else:
      difficulty = config.test_difficulty
      dynamic = config.test_dynamic
      camera_kwargs = config.test_camera_kwargs
      background_kwargs = config.test_background_kwargs
      background_dataset_path = config.test_background_dataset_path
      background_dataset_videos = config.test_background_dataset_videos
      color_kwargs = config.test_colour_kwargs
    self._env = dmc2gym.make(domain_name=domain,
                             task_name=task,
                             test=test,
                             seed=config.seed,
                             difficulty=None if difficulty == "None" else difficulty,
                             dynamic=dynamic,
                             background_dataset_path=None if background_dataset_path == "None" else os.path.join(
                               os.getcwd(), background_dataset_path),
                             background_dataset_videos=None if background_dataset_videos == "None" else background_dataset_videos,
                             background_kwargs=None if background_kwargs == "None" else background_kwargs,
                             camera_kwargs=None if camera_kwargs == "None" else camera_kwargs,
                             color_kwargs=None if color_kwargs == "None" else color_kwargs,
                             visualize_reward=False,
                             from_pixels=True,
                             height=config.image_size,
                             width=config.image_size,
                             frame_skip=config.action_repeat,
                             camera_id=camera_id,
                             channels_first=False)
    self._action_repeat = config.action_repeat
    self._size = config.size

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      if len(value.shape) == 0:
        shape = (1,)
      else:
        shape = value.shape
      spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
    spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum[0], spec.maximum[0], spec.shape, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for repeat in range(self._action_repeat):
      time_step, reward, done, info = self._env.step(action)
      reward += reward or 0
      if done:
        break

    obs = dict({"image": time_step})
    # # There is no terminal state in DMC
    if info['discount'] == 0:
      obs["is_terminal"] = True
    else:
      obs["is_terminal"] = False
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict({"image": time_step})
    obs["is_terminal"] = False
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get("mode", "rgb_array") != "rgb_array":
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class CollectDataset:

  def __init__(self, env, callbacks=None, config=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    if isinstance(action, dict):
      transition.update(action)
    else:
      transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      for key, value in self._episode[1].items():
        if key not in self._episode[0]:
          self._episode[0][key] = 0 * value
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}

      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    # Missing keys will be filled with a zeroed out version of the first
    # transition, because we do not know what action information the agent will
    # pass yet.
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.discrete = True
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class SelectAction:

  def __init__(self, env, key):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    return self._env.step(action[self._key])

class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy="thread"):
    print(1)
    self._pickled_ctor = cloudpickle.dumps(constructor)
    print(1)
    if strategy == "process":
      import multiprocessing as mp

      context = mp.get_context("spawn")
    elif strategy == "thread":
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access("obs_space")()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access("act_space")()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call("step", action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call("reset")
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError("Lost connection to environment worker.")
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError("Received message of unexpected type {}".format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError("Received message of unknown type {}".format(message))
    except Exception:
      stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
      print("Error in environment process: {}".format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.

