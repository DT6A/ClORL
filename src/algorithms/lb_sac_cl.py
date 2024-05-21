import copy
import math

import gym
import d4rl
import pyrallis
import numpy as np
import wandb
import uuid

import jax
import chex
import optax
import distrax
import jax.numpy as jnp

import flax
import flax.linen as nn

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState


@dataclass
class Config:
    # wandb params
    project: str = "ClORL"
    group: str = "LB-SAC"
    name: str = "lb-sac-ce"
    # model params
    hidden_dim: int = 256
    critic_n_hiddens: int = 3
    critic_ln: bool = True
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 17e-4
    critic_learning_rate: float = 17e-4
    alpha_learning_rate: float = 17e-4
    normalize_reward: bool = False
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 8192
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    # general params
    train_seed: int = 10
    eval_seed: int = 42
    # classification
    n_classes: int = 101
    sigma_frac: float = 0.75
    v_min: float = float('inf')
    v_max: float = float('inf')
    v_expand: float = 0.0
    v_expand_mode: str = "both"

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"


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
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
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

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )



def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
    support: jax.Array
    sigma: float

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(self.params, self.target_params, tau)
        return self.replace(target_params=new_target_params)


# SAC-N networks
class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def identity(x: Any) -> Any:
    return x


# WARN: only for [-1, 1] action bounds, scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = False
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state):
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        net = nn.Sequential(layers)

        log_sigma_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = net(state)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            # nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
            nn.Dense(self.n_classes, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action)#.squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3
    n_classes: int = 21

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens, self.n_classes)(state, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.array([jnp.log(self.init_value)]))
        return jnp.exp(log_alpha)


# SAC-N losses
def update_actor(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: TrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array]
) -> Tuple[TrainState, Dict[str, Any]]:
    def actor_loss_fn(actor_params):
        actions_dist = actor.apply_fn(actor_params, batch["states"])
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=key)

        # q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        logits = critic.apply_fn(critic.params, batch["states"], actions)
        probs = nn.softmax(logits, axis=-1)
        q_values = transform_from_probs(probs, critic.support).min(0)

        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) - q_values).mean()

        batch_entropy = -actions_logp.sum(-1).mean()
        return loss, batch_entropy

    (loss, batch_entropy), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    info = {
        "batch_entropy": batch_entropy,
        "actor_loss": loss
    }
    return new_actor, info


def update_alpha(
        alpha: TrainState,
        entropy: float,
        target_entropy: float
) -> Tuple[TrainState, Dict[str, Any]]:
    def alpha_loss_fn(alpha_params):
        alpha_value = alpha.apply_fn(alpha_params)
        loss = (alpha_value * (entropy - target_entropy)).mean()
        return loss

    loss, grads = jax.value_and_grad(alpha_loss_fn)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)
    info = {
        "alpha": alpha.apply_fn(alpha.params),
        "alpha_loss": loss
    }
    return new_alpha, info


def update_critic(
        key: jax.random.PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: Dict[str, jax.Array],
        gamma: float,
        tau: float,
) -> Tuple[TrainState, Dict[str, Any]]:
    next_actions_dist = actor.apply_fn(actor.params, batch["next_states"])
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=key)

    logits = critic.apply_fn(critic.target_params, batch["next_states"], next_actions)
    probs = nn.softmax(logits, axis=-1)
    next_q = transform_from_probs(probs, critic.support).min(0)
    # next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1)
    target_q = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        target_probs = transform_to_probs(target_q, critic.support, critic.sigma)

        # loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        loss = optax.softmax_cross_entropy(logits=q, labels=target_probs[None, ...]).mean(1).sum(0)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic.params)
    new_critic = critic.apply_gradients(grads=grads).soft_update(tau=tau)
    info = {
        "critic_loss": loss
    }
    return new_critic, info


# evaluation
@jax.jit
def eval_actions_jit(actor: TrainState, obs: jax.Array) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs)
    action = dist.mean()
    return action


def transform_to_probs(target: jax.Array, support: jax.Array, sigma: float) -> jax.Array:
    cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    bin_probs = cdf_evals[1:] - cdf_evals[:-1]
    return bin_probs / (z + 1e-6)


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


def evaluate(env: gym.Env, actor: TrainState, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)

    returns = []
    for _ in trange(num_episodes, leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = eval_actions_jit(actor, obs)
            obs, reward, done, _ = env.step(np.asarray(jax.device_get(action)))
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


@pyrallis.wrap()
def main(config: Config):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True
    )

    wandb.mark_preempting()

    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, False
    )

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    target_entropy = -np.prod(eval_env.action_space.shape)

    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)
    init_state = jnp.asarray(eval_env.observation_space.sample())
    init_action = jnp.asarray(eval_env.action_space.sample())

    actor_module = Actor(action_dim=np.prod(eval_env.action_space.shape), hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=config.num_critics,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
        n_classes=config.n_classes,
    )

    v_min, v_max = config.v_min, config.v_max
    if v_min == float('inf'):
        v_min = buffer.min
    if v_max == float('inf'):
        v_max = buffer.max

    expand = (v_max - v_min) * config.v_expand
    if config.v_expand_mode == "both":
        v_min -= expand / 2
        v_max += expand / 2
    elif config.v_expand_mode == "min":
        v_min -= expand
    elif config.v_expand_mode == "max":
        v_max += expand
    else:
        raise ValueError("Invalid expansion")

    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        support=jnp.linspace(v_min, v_max, config.n_classes + 1, dtype=jnp.float32),
        sigma=config.sigma_frac * (v_max - v_min) / config.n_classes,
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )

    def update_networks(key, actor, critic, alpha, batch):
        actor_key, critic_key = jax.random.split(key)

        new_actor, actor_info = update_actor(actor_key, actor, critic, alpha, batch)
        new_alpha, alpha_info = update_alpha(alpha, actor_info["batch_entropy"], target_entropy)
        new_critic, critic_info = update_critic(critic_key, new_actor, critic, new_alpha, batch, config.gamma, config.tau)

        return new_actor, new_critic, new_alpha, {**actor_info, **critic_info, **alpha_info}

    @jax.jit
    def update_step(_, carry):
        key, update_key, batch_key = jax.random.split(carry["key"], 3)
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)

        actor, critic, alpha, update_info = update_networks(
            key=update_key,
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
        )
        update_info = jax.tree.map(lambda c, u: c + u, carry["update_info"], update_info)
        carry.update(key=key, actor=actor, critic=critic, alpha=alpha, update_info=update_info)

        return carry

    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }
    for epoch in trange(config.num_epochs):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["update_info"] = {
            "critic_loss": jnp.array([0.0]),
            "actor_loss": jnp.array([0.0]),
            "alpha_loss": jnp.array([0.0]),
            "alpha": jnp.array([0.0]),
            "batch_entropy": jnp.array([0.0])
        }
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        update_info = jax.tree.map(lambda v: v.item() / config.num_updates_on_epoch, update_carry["update_info"])
        wandb.log({"epoch": epoch, **update_info})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = evaluate(eval_env, update_carry["actor"], config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0

            wandb.log({
                "epoch": epoch,
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })


if __name__ == "__main__":
    main()
