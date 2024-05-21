import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional, List
from collections import defaultdict

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange
 
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@dataclass
class Config:
    # wandb params
    project: str = "ClORL"
    group: str = "iql"
    name: str = "iql"
    
    # model params
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_learning_rate: float = 3e-4
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    tanh_squash_distribution: bool = False
    decay_schedule: str = "cosine"
    temperature: float = 0.1

    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_activations: str = "relu"
    critic_learning_rate: float = 3e-4
    tau: float = 0.005

    value_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    value_learning_rate: float = 3e-4
    expectile: float = 0.8
    
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize_states: bool = False
    gamma: float = 0.99
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    
    # classification
    # n_classes: int = 101
    # sigma_frac: float = 0.75
    # v_min: float = float('inf')
    # v_max: float = float('inf')

    _wandb: Dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def calc_return_to_go(is_sparse_reward, rewards, terminals, gamma):
    """
    A config dict for getting the default high/low rewrd values for each envs
    This is used in calc_return_to_go func in sampler.py and replay_buffer.py
    """
    if len(rewards) == 0:
        return []
    reward_neg = 0
    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        # assuming failure reward is negative
        # use r / (1-gamma) for negative trajctory
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (1 - terminals[-i - 1])
            prev_return = return_to_go[-i - 1]

    return return_to_go


def convert_to_classification(labels, num_bins=10):
    labels = np.array(labels)
    # Determine bin edges to evenly divide the range of labels
    bin_edges = np.linspace(labels.min(), labels.max(), num_bins + 1)

    # Use numpy.digitize to assign each label to a bin
    bins = np.digitize(labels, bin_edges)

    return bins


def convert_to_classification_equal_samples(labels, num_bins=10):
    labels = np.array(labels)
    # Determine the percentiles to evenly divide the data into bins
    percentiles = np.linspace(0, 100, num_bins + 1)

    # Compute the bin edges based on the percentiles
    bin_edges = np.percentile(labels, percentiles)

    # Use numpy.digitize to assign each label to a bin
    bins = np.digitize(labels, bin_edges)

    return bins


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["dones"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, min_ret, max_ret, max_episode_steps=1000):
    # if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
    #     dataset["rewards"] = dataset["rewards"] / (max_ret - min_ret) * max_episode_steps
    if "antmaze" in env_name:
        dataset["rewards"] = dataset["rewards"] * 100.0


def qlearning_dataset(env, dataset_name, normalize_reward=False, dataset=None, terminate_on_end=False, discount=0.99,
                       **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, next_actins, rewards,
     and a terminal flag.
    Args:
       env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    if normalize_reward:
        dataset['rewards'] = ReplayBuffer.normalize_reward(dataset_name, dataset['rewards'])
    N = dataset['rewards'].shape[0]
    is_sparse = "antmaze" in dataset_name
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []
    print("SIZE", N)
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    episode_rewards = []
    episode_terminals = []
    for i in range(N - 1):
        if episode_step == 0:
            episode_rewards = []
            episode_terminals = []

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
            # print(len(mc_returns_), len(episode_rewards), end=";")
            continue
        if done_bool or final_timestep:
            episode_step = 0
            # mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
            # print(i, len(mc_returns_), len(episode_rewards))

        episode_rewards.append(reward)
        episode_terminals.append(done_bool)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1
    if episode_step != 0:
        mc_returns_ += calc_return_to_go(is_sparse, episode_rewards, episode_terminals, discount)
    print("SHAPE", np.array(mc_returns_).shape, np.array(reward_).shape, np.array(done_).shape)
    assert np.array(mc_returns_).shape == np.array(reward_).shape
    print(np.array(action_).max(), np.array(action_).min())

    cls_rewards = np.array(mc_returns_)
    # to_probs, from_probs = hl_gauss_transform(jnp.min(cls_rewards), jnp.max(cls_rewards), num_bins=n_classes,
    #                                           sigma=sigma)
    # to_probs = jax.vmap(to_probs)
    # from_probs = jax.vmap(from_probs)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'mc_returns': np.array(mc_returns_),
    }, jnp.min(cls_rewards), jnp.max(cls_rewards)


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std

@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1
    min: float = 0
    max: float = 1

    def create_from_d4rl(
            self,
            dataset_name: str,
            normalize_reward: bool = False,
            is_normalize: bool = False,
    ):
        d4rl_data, self.min, self.max = qlearning_dataset(gym.make(dataset_name), dataset_name)
        print("Min/Max", self.min, self.max)
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_observations"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32),
            "mc_returns": jnp.asarray(d4rl_data["mc_returns"], dtype=jnp.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            modify_reward(buffer, dataset_name, self.min, self.max)

        self.data = buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(
            self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree.map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


# @jax.jit
def transform_to_probs(target: jax.Array, support: jax.Array, sigma: float) -> jax.Array:
    cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    bin_probs = cdf_evals[1:] - cdf_evals[:-1]
    return bin_probs / z


transform_to_probs = jax.vmap(transform_to_probs, in_axes=(0, None, None))


# @jax.jit
def transform_from_probs(probs: jax.Array, support: jax.Array) -> jax.Array:
    centers = (support[:-1] + support[1:]) / 2
    return jnp.sum(probs * centers)


transform_from_probs = jax.vmap(transform_from_probs, in_axes=(0, None))
transform_from_probs = jax.vmap(transform_from_probs, in_axes=(0, None))


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward: float) -> float:
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# https://github.com/Howuhh/sac-n-jax/blob/a0d4b8ab8b457658e416cd554faa47506bc2367c/sac_n_jax_flax.py#L91C1-L98C63
class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.MultivariateNormalDiag(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())



# https://github.com/ikostrikov/implicit_q_learning/blob/master/policy.py
class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = False

    @nn.compact
    def __call__(self, states: jnp.ndarray, temperature: float = 1.0, training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True,
                      dropout_rate=self.dropout_rate)(states,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)


        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist

        #if self.tanh_squash_distribution:
        #    return TanhNormal(loc=means, scale=jnp.exp(log_stds) * temperature)
        #else:
        #    return distrax.MultivariateNormalDiag(nn.tanh(means), jnp.exp(log_stds) * temperature)

# Here was partial(jax.jit)
def _sample_actions(key: jax.random.PRNGKey,
                    apply_fn: Callable,
                    actor_params: TrainState,
                    states: np.ndarray,
                    temperature: float = 1.0) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    dist = apply_fn(actor_params, states, temperature)
    key, random_dist_key = jax.random.split(key)
    return key, dist.sample(seed=random_dist_key)


def sample_actions(key: jax.random.PRNGKey,
                   apply_fn: Callable,
                   actor_params: TrainState,
                   states: np.ndarray,
                   temperature: float = 1.0) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    return _sample_actions(key, apply_fn, actor_params, states, temperature)

class MLP(nn.Module):
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/common.py#L28-L43
    
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class ValueCritic(nn.Module):
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/value_net.py#L9-L15
    
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(states)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/value_net.py#L18-L28
    
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, states: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([states, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/value_net.py#L31-L42
    
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    # n_classes: int

    @nn.compact
    def __call__(self, states: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(states, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(states, actions)
        return critic1, critic2


class CriticTrainState(TrainState):
    target_params: FrozenDict
    # support: jax.Array
    # sigma: float


def update_q(
    key: jax.random.PRNGKey,
    critic: CriticTrainState,
    target_value: TrainState,
    batch: Dict[str, Any],
    gamma: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/critic.py#L32-L50
    
    next_v = target_value.apply_fn(target_value.params, batch["next_states"])

    target_q = batch["rewards"] + gamma * (1 - batch["dones"]) * next_v

    def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict]:
        q1, q2 = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    (loss, loss_metrics), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        critic.params
    )
    new_critic = critic.apply_gradients(grads=grads)
    new_metrics = metrics.update(loss_metrics)

    return key, new_critic, new_metrics


def update_v(
    key: jax.random.PRNGKey,
    critic: CriticTrainState,
    value: TrainState,
    batch: Dict[str, Any],
    expectile: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/critic.py#L13-L29

    q1, q2 = critic.apply_fn(critic.target_params, batch["states"], batch["actions"])
    q = jnp.minimum(q1, q2)

    
    def expectile_loss(diff, expectile=0.8):
        # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/critic.py#L8-L10
        weight = jnp.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)
        

    def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict]:
        v = value.apply_fn(value_params, batch["states"])
        value_loss = expectile_loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    (loss, loss_metrics), grads = jax.value_and_grad(value_loss_fn, has_aux=True)(
        value.params
    )

    new_value = value.apply_gradients(grads=grads)
    new_metrics = metrics.update(loss_metrics)

    return key, new_value, new_metrics

def update_actor(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    value: TrainState,
    batch: Dict[str, Any],
    temperature: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/actor.py#L9-L30
    
    key, random_dropout_key = jax.random.split(key, 2)
    v = value.apply_fn(value.params, batch["states"])

    q1, q2 = critic.apply_fn(critic.target_params, batch["states"], batch["actions"])
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict]:
        dist = actor.apply_fn(actor_params, batch["states"], training=True, rngs={'dropout': random_dropout_key})
        log_probs = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': (q - v).mean()}

    (loss, loss_metrics), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        actor.params
    )

    grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)

    new_actor = actor.apply_gradients(grads=grads)
    new_metrics = metrics.update(loss_metrics)

    return key, new_actor, new_metrics


def update_target(
    key: jax.random.PRNGKey,
    critic: CriticTrainState,
    tau: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/learner.py#L17-L22
    
    # new_target_params = jax.tree_map(
    #     lambda p, tp: p * tau + tp * (1 - tau), critic.params, critic.target_params)
    
    # return key, critic.replace(target_params=new_target_params), metrics

    new_target_params = optax.incremental_update(critic.params, critic.target_params, tau)
    return key, critic.replace(target_params=new_target_params), metrics


@jax.jit
def update_iql(
    key: jax.random.PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    value: TrainState,
    batch: Dict[str, Any],
    metrics: Metrics,
    gamma: float,
    expectile: float,
    tau: float,
    temperature: float,
) -> Tuple[jax.random.PRNGKey, TrainState, CriticTrainState, TrainState, Metrics]:
    # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/learner.py#L26-L45
    
    key, new_value, new_metrics = update_v(key, critic, value, batch, expectile, metrics)
    
    key, new_actor, new_metrics = update_actor(key, actor, critic, new_value, batch, temperature, new_metrics)

    key, new_critic, new_metrics = update_q(key, critic, new_value, batch, gamma, new_metrics)

    key, new_critic, new_metrics = update_target(key, new_critic, tau, new_metrics)
    
    
    return key, new_actor, new_critic, new_value, new_metrics


def evaluate(key: jax.random.PRNGKey, env: gym.Env, params: jax.Array, action_fn: Callable, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            key, action = action_fn(key, params, obs)
            action = np.asarray(jax.device_get(action))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return np.array(returns)


@pyrallis.wrap()
def train(config: Config):
    dict_config = asdict(config)

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, value_key, critic_key = jax.random.split(key, 4)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    init_state = buffer.data["states"][0][None, ...]
    init_action = buffer.data["actions"][0][None, ...]

    actor_module = NormalTanhPolicy(
        action_dim=init_action.shape[-1],
        hidden_dims=config.actor_hidden_dims,
        state_dependent_std=config.state_dependent_std,
        dropout_rate=config.dropout_rate,
        log_std_scale=config.log_std_scale,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        tanh_squash_distribution=config.tanh_squash_distribution
    )

    if config.decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-config.actor_learning_rate, config.num_epochs * config.num_updates_on_epoch)
        optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    else:
        optimizer = optax.adam(learning_rate=config.actor_learning_rate)

    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optimizer,
    )

    activations = None
    if config.critic_activations.lower() == 'relu':
        activations = nn.relu
    else:
        raise ValueError(f'Unsupported activations: {activations}')
        

    critic_module = DoubleCritic(
        hidden_dims=config.critic_hidden_dims,
        activations = activations,
        # n_classes=config.n_classes
    )

    # v_min, v_max = config.v_min, config.v_max
    # if v_min == float('inf'):
    #     v_min = buffer.min
    # if v_max == float('inf'):
    #     v_max = buffer.max
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        # support=jnp.linspace(v_min, v_max, config.n_classes + 1, dtype=jnp.float32),
        # sigma=config.sigma_frac * (v_max - v_min) / config.n_classes,
        tx=optax.adam(learning_rate=config.critic_learning_rate)
    )

    value_module = ValueCritic(hidden_dims = config.value_hidden_dims)

    value = TrainState.create(
        apply_fn=value_module.apply,
        params=value_module.init(value_key, init_state),
        tx=optax.adam(learning_rate=config.value_learning_rate))

    update_iql_partial = partial(
        update_iql,
        gamma=config.gamma, tau=config.tau,
        expectile=config.expectile, temperature=config.temperature
    )

    def iql_loop_update_step(i: int, carry: TrainState):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)
        
        key, new_actor, new_critic, new_value, new_metrics = update_iql_partial(
            key=key, actor=carry["actor"],
            critic=carry["critic"], value=carry["value"],
            metrics=carry["metrics"], batch=batch,
        )

        carry.update(key=key, actor=new_actor, critic=new_critic, metrics=new_metrics, value=new_value)
        return carry

    # metrics
    bc_metrics_to_log = [
        "critic_loss",
        "q_min",
        "q1", "q2", "v", "adv",
        "value_loss",
        "actor_loss",
        "batch_entropy",
        "bc_mse_policy",
        "bc_mse_random",
        "action_mse",
        "actions_q_entropy",
        "actor_entropy_diff_abs",
        "actor_entropy_diff",
        "q_entropy",
    ]
    # shared carry for update loops
    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "value": value,
        "buffer": buffer,
    }

    @jax.jit
    def actor_action_fn(key: jax.random.PRNGKey, params: jax.Array, obs: jax.Array):
        key, actions = sample_actions(key, actor.apply_fn, params, obs, temperature=0.0)
        return key, jnp.clip(actions, -1, 1)

    for epoch in trange(config.num_epochs, desc="IQL Epochs"):
        # metrics for accumulation during epoch and logging to wandb
        # we need to reset them every epoch
        
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)
    
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=iql_loop_update_step,
            init_val=update_carry,
        )
        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"IQL/{k}": v for k, v in mean_metrics.items()}}
        )

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            update_carry["key"], eval_key = jax.random.split(update_carry["key"])
            eval_returns = evaluate(
                eval_key,
                eval_env,
                update_carry["actor"].params,
                actor_action_fn,
                config.eval_episodes,
                seed=config.eval_seed,
            )
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            wandb.log({
                "epoch": epoch,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score),
            })


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("An exception occured")
        print(e)

