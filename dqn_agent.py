import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
import rlax
import random

from collections import namedtuple, deque
from flax.training import train_state
from jax import jit, vmap
from tqdm import tqdm

from dqn_model import MLP

Params = namedtuple("params", "policy target")
Experience = namedtuple("experience", "state action reward next_state done")
ExpCfg = namedtuple("exp_cfg", "env_name architecture n_episodes n_steps batch_size replay_size learning_rate target_update_frequency gamma eps_schedule")


class ReplayBuffer:
	"""Replay buffer to store and sample experience tuples."""

	def __init__(self, buffer_size, seed):
		random.seed(seed)
		self.memory = deque(maxlen=buffer_size)

	def add(self, state, action, reward, next_state, done):
		e = Experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self, batch_size):
		"""Randomly sample a batch of experiences from memory."""
		s, a, r, n, d = zip(*random.sample(self.memory, k=batch_size))
		return (
			jnp.vstack(s, dtype=float), jnp.vstack(a, dtype=int), jnp.vstack(r, dtype=float),
			jnp.vstack(n, dtype=float), jnp.vstack(d, dtype=float)
		)

	def __len__(self):
		return len(self.memory)


@jit
def policy(key, state, trainstate, epsilon, greedy=False):
	"""Epsilon-greedy policy. Maps state to action."""
	state = jnp.expand_dims(state, axis=0)
	q = jnp.squeeze(trainstate.apply_fn(trainstate.params, state))
	a_eps = rlax.epsilon_greedy(epsilon).sample(key, q)
	a_grd = rlax.greedy().sample(key, q)
	action = jax.lax.select(greedy, a_grd, a_eps)
	return action

@vmap
def q_learning_loss(q, target_q, action, reward, done, gamma):
	"""Compute q-learning loss through TD-learning."""
	td_target = reward + gamma*target_q.max()*(1.-done)
	td_error = jax.lax.stop_gradient(td_target) - q[action]
	return td_error**2

@vmap
def double_q_learning_loss(q, target_q, action, action_select, reward, done, gamma):
	"""Compute double q-learning loss through TD-learning (action selected by policy network)."""
	td_target = reward + gamma*target_q[action_select]*(1.-done)
	td_error = jax.lax.stop_gradient(td_target) - q[action]
	return td_error**2

@jit
def train_step(trainstate, target_params, batch, gamma=0.9):
	"""Perform a single training step, i.e. compute loss and update model parameters."""
	def loss_fn(policy_params):
		"""Compute avg loss for a batch of experiences."""
		state, action, reward, next_state, done = batch
		q = trainstate.apply_fn(policy_params, state)
		target_q = trainstate.apply_fn(target_params, next_state)
		action_select = trainstate.apply_fn(policy_params, next_state).argmax(-1)
		g = jnp.array([gamma] * state.shape[0])
		return jnp.mean(double_q_learning_loss(q, target_q, action, action_select, reward, done, g))

	# compute loss and gradients, then apply gradients
	loss, grad = jax.value_and_grad(loss_fn)(trainstate.params)
	trainstate = trainstate.apply_gradients(grads=grad)
	return trainstate, loss


class DQNAgent:
	def __init__(self, cfg: ExpCfg, seed=0) -> None:
		# create environment and replay buffer
		self.env = gym.make(cfg.env_name)
		self.replay_buffer = ReplayBuffer(cfg.replay_size, seed=0)
		# stored properties
		self.cfg = cfg
		self.key = jax.random.PRNGKey(seed)
		self.n_actions = self.env.action_space.n
		self.n_state = self.env.observation_space.shape[0]
		# network and parameters
		self.network = MLP([self.n_state] + cfg.architecture + [self.n_actions])
		state, _ = self.env.reset()
		self.target_params = self.network.init(self.key, jnp.expand_dims(state, axis=0))
		# training state
		self.eps_fn = optax.polynomial_schedule(*cfg.eps_schedule, transition_steps=cfg.n_episodes)
		self.trainstate = train_state.TrainState.create(
			apply_fn=self.network.apply,
			params=self.network.init(self.key, jnp.expand_dims(state, axis=0)),
			tx=optax.adam(cfg.learning_rate)
		)
	
	def train(self):
		mean_loss_per_episode = []
		reward_per_episode = []

		for episode in tqdm(range(self.cfg.n_episodes)):
			state, _ = self.env.reset()
			ep_reward = 0
			ep_loss = []

			for t in range(self.cfg.n_steps):
				epsilon = self.eps_fn(t)
				action = policy(self.key, state, self.trainstate, epsilon)
				next_state, reward, done, _, _ = self.env.step(int(action))
				self.replay_buffer.add(state, action, reward, next_state, done)
				ep_reward += reward

				if len(self.replay_buffer) > self.cfg.batch_size:
					batch = self.replay_buffer.sample(self.cfg.batch_size)
					self.trainstate, loss = train_step(self.trainstate, self.target_params, 
													   batch, self.cfg.gamma)
					ep_loss.append(float(loss))

				if t % self.cfg.target_update_frequency == 0:
					self.target_params = self.trainstate.params
				
				if done: break

				state = next_state
			
			mean_loss_per_episode.append(jnp.array(ep_loss).mean())
			reward_per_episode.append(ep_reward)
		
		return jnp.array(mean_loss_per_episode), jnp.array(reward_per_episode)

	def eval(self, n_episodes=100, epsilon=0.05):
		reward_per_episode = []

		for episode in tqdm(range(n_episodes)):
			ep_reward = 0
			state, _ = self.env.reset()

			for t in range(self.cfg.n_steps):
				action = policy(self.key, state, self.trainstate, epsilon)
				state, reward, done, _, _ = self.env.step(int(action))
				ep_reward += reward
				if done: break
			reward_per_episode.append(ep_reward)

		return jnp.array(reward_per_episode)

def run_experiment(cfg: ExpCfg):
	# create environment
	env = gym.make(cfg.env_name)
	replay_buffer = ReplayBuffer(cfg.replay_size, seed=0)

	# logging
	mean_loss_per_episode = []
	reward_per_episode = []
	key = jax.random.PRNGKey(0)

	# Build and initialize the action selection and target network.
	num_actions = env.action_space.n
	state, _ = env.reset()

	# initialize the model and optimizer
	network = MLP([state.shape[0]] + cfg.architecture + [num_actions])
	target_params = network.init(key, jnp.expand_dims(state, axis=0))
	optimizer = optax.adam(cfg.learning_rate)
	epsilon_by_frame = optax.polynomial_schedule(*cfg.eps_schedule, transition_steps=cfg.n_episodes)
	trainstate = train_state.TrainState.create(
		apply_fn=network.apply, params= network.init(key, jnp.expand_dims(state, axis=0)), 
		tx=optimizer
	)

	for episode in tqdm(range(cfg.n_episodes)):

		# Initialize episode
		state, _ = env.reset()
		ep_reward = 0
		mean_ep_loss = []

		for t in range(cfg.n_steps):
			# generate an action from the agent's policy
			epsilon = epsilon_by_frame(t)
			action = policy(key, state, trainstate, epsilon, num_actions)

			# step the environment
			next_state, reward, done, _, _ = env.step(int(action))

			# store to replay buffer, update episode reward
			replay_buffer.add(state, action, reward, next_state, done)
			ep_reward += reward

			# update value model when there's enough data
			if len(replay_buffer) > cfg.batch_size:
				batch = replay_buffer.sample(cfg.batch_size)
				trainstate, loss = train_step(trainstate, target_params, batch, cfg.gamma)
				mean_ep_loss.append(float(loss))

			# update target model parameters
			if t % cfg.target_update_frequency == 0:
				target_params = trainstate.params

			# Terminate episode when absorbing state reached.
			if done: break

			# Cycle the state
			state = next_state

		# Update episodic statistics
		mean_loss_per_episode.append(jnp.array(mean_ep_loss).mean())
		reward_per_episode.append(ep_reward)

	return trainstate.params, jnp.array(reward_per_episode), jnp.array(mean_loss_per_episode)

def eval_agent(cfg: ExpCfg, params, epsilon=0.05):
	env = gym.make(cfg.env_name)
	key = jax.random.PRNGKey(0)

	state, _ = env.reset()
	num_actions = env.action_space.n
	network = MLP([state.shape[0]] + cfg.architecture + [num_actions])
	netstate = train_state.TrainState.create(
		apply_fn=network.apply, params=params, tx=optax.adam(.0)
	)

	reward_per_episode = []

	for _ in tqdm(range(cfg.n_episodes)):
		state, _ = env.reset()
		ep_reward = 0
		done = False
		for _ in range(cfg.n_steps):
			action = policy(key, state, netstate, epsilon, num_actions)
			state, reward, done, _, _ = env.step(int(action))
			ep_reward += reward
			if done: break
		reward_per_episode.append(ep_reward)

	return jnp.array(reward_per_episode)
