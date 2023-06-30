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

from model import MLP

Params = namedtuple("params", "policy target")
Experience = namedtuple("experience", "state action reward next_state done")

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

def run_experiment(env_name = 'CartPole-v1', 
		   		   architecture = [512],
				   num_episodes=1000, 
				   num_steps=50,
				   batch_size=32, 
				   replay_size=1000, 
				   learning_rate=1e-3,
				   target_update_frequency=10,
				   gamma=0.9):

	# create environment
	env = gym.make(env_name)
	replay_buffer = ReplayBuffer(replay_size, seed=0)

	# logging
	mean_loss_per_episode = []
	reward_per_episode = []
	key = jax.random.PRNGKey(0)

	# Build and initialize the action selection and target network.
	num_actions = env.action_space.n
	state, _ = env.reset()

	# initialize the model and optimizer
	network = MLP([state.shape[0]] + architecture + [num_actions])
	target_params = network.init(key, jnp.expand_dims(state, axis=0))
	policy_params = network.init(key, jnp.expand_dims(state, axis=0))
	optimizer = optax.adam(learning_rate)
	opt_state = optimizer.init(policy_params)
	epsilon_by_frame = optax.polynomial_schedule(init_value=1.0, end_value=0.01, 
							                     transition_steps=num_episodes, power=1.)
	trainstate = train_state.TrainState.create(
		apply_fn=network.apply, params=policy_params, tx=optimizer
	)

	for episode in tqdm(range(num_episodes)):

		# Initialize episode
		state, _ = env.reset()
		ep_reward = 0
		mean_ep_loss = []

		for t in range(num_steps):
			# generate an action from the agent's policy
			epsilon = epsilon_by_frame(t)
			action = policy(key, state, trainstate, epsilon, num_actions)

			# step the environment
			next_state, reward, done, _, _ = env.step(int(action))

			# store to replay buffer, update episode reward
			replay_buffer.add(state, action, reward, next_state, done)
			ep_reward += reward

			# update value model when there's enough data
			if len(replay_buffer) > batch_size:
				batch = replay_buffer.sample(batch_size)
				trainstate, loss = train_step(trainstate, target_params, batch, gamma)
				mean_ep_loss.append(float(loss))

			# update target model parameters
			if t % target_update_frequency == 0:
				target_params = trainstate.params

			# Terminate episode when absorbing state reached.
			if done: break

			# Cycle the state
			state = next_state

		# Update episodic statistics
		mean_loss_per_episode.append(jnp.array(mean_ep_loss).mean())
		reward_per_episode.append(ep_reward)

		if episode % 100 == 0:
			if len(replay_buffer) > batch_size:
				print("Episode #{}, Reward {}, Avg loss {}".format(episode, ep_reward, jnp.array(mean_ep_loss).mean()))
			else:
				print("Episode #{}, Reward {}".format(episode, ep_reward))

	return trainstate.params, reward_per_episode, mean_loss_per_episode
