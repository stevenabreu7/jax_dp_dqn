import flax
import gym
import jax
import jax.numpy as jnp
import optax
import random

from collections import namedtuple, deque
import flax.linen as nn
from tqdm import tqdm
from jax import jit, vmap

from model import MLP

Params = namedtuple("params", "policy target")
Experience = namedtuple("experience", "state action reward next_state done")

class ReplayBuffer:
	"""Replay buffer to store and sample experience tuples."""

	def __init__(self, buffer_size, seed):
		""" Initialize a ReplayBuffer object."""
		self.memory = deque(maxlen=buffer_size)
		random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = Experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self, batch_size):
		"""Randomly sample a batch of experiences from memory."""
		s, a, r, n, d = zip(*random.sample(self.memory, k=batch_size))
		return (
			jnp.vstack(s, dtype=float), jnp.vstack(a, dtype=jnp.int64), jnp.vstack(r, dtype=float),
			jnp.vstack(n, dtype=float), jnp.vstack(d, dtype=float)
		)

	def __len__(self):
		return len(self.memory)


# @jit
def policy(key, x, apply_model, policy_params, epsilon, num_actions):
	"""Epsilon-greedy policy. Maps state to action."""
	prob_rnd = jax.random.uniform(key)
	# model-predicted q-values for all actions
	q = jnp.squeeze(apply_model(policy_params, jnp.expand_dims(x, axis=0)))
	# TODO: use jax.lax.cond instead of if-else
	# a = jax.lax.cond(prob < epsilon, key, partial(rand, num_actions=num_actions), q, jnp.argmax)
	# if prob_rnd < epsilon, choose random action, else greedy action
	if prob_rnd < epsilon:
		return jax.random.randint(key, (1,), 0, num_actions)[0]
	else:
		return jnp.argmax(q)

# @vmap
def q_learning_loss_single(q, target_q, action, action_select, reward, done, gamma):
	"""Compute q-learning loss through TD-learning."""
	# TODO: is this correct or action_select <> action?
	td_target = reward + gamma*(1.-done)*target_q[action_select]
	td_error = jax.lax.stop_gradient(td_target) - q[action]
	return td_error**2

q_learning_loss = vmap(q_learning_loss_single, in_axes=((0,0,0,0,0,0,None)))

# @jit
def train_step(optimizer, opt_state, apply_model, params, batch, gamma=0.9):
	"""Perform a single training step, i.e. compute loss and update model parameters."""
	def loss_fn(policy_params):
		"""Compute avg loss for a batch of experiences."""
		state, action, reward, next_state, done = batch
		q = apply_model(policy_params, state)
		target_q = apply_model(params.target, next_state)
		action_select = apply_model(policy_params, next_state).argmax(-1)
		# print('qshape', q.shape)
		return jnp.mean(q_learning_loss(q, target_q, action, action_select, reward, done, gamma))

	# compute loss and gradients, then apply gradients
	loss, grad = jax.value_and_grad(loss_fn)(params.policy)
	# optimizer = optimizer.apply_gradient(grad)
	updates, opt_state = optimizer.update(grad, opt_state)
	params = Params(
		optax.apply_updates(params.policy, updates),
		params.target
	)
	# params.policy = optax.apply_updates(params.policy, updates)
	return opt_state, loss


def run_experiment(
		env_name = 'CartPole-v1', 
		architecture = [512],
		num_episodes=1000, 
		num_steps=50,
		batch_size=32, 
		replay_size=1000, 
		learning_rate=1e-3,
		target_update_frequency=10,
		gamma=0.9):

	# Create environment
	env = gym.make(env_name)
	replay_buffer = ReplayBuffer(replay_size, seed=0)

	# logging
	ep_losses = []
	ep_rewards = []
	key = jax.random.PRNGKey(0)

	# Build and initialize the action selection and target network.
	num_actions = env.action_space.n
	state, _ = env.reset()

#   module = DQN.partial(num_actions=num_actions)
#   _, initial_params = module.init(key, jnp.expand_dims(state, axis=0))

	# initialize the model
	network = MLP([state.shape[0]] + architecture + [num_actions])
	params = Params(
		network.init(key, jnp.expand_dims(state, axis=0)),
		network.init(key, jnp.expand_dims(state, axis=0))
	)
	# model = MLP([state.shape[0]] + architecture + [num_actions])
	# initial_params = model.init(key, jnp.expand_dims(state, axis=0))
	# # model = nn.Module(model, initial_params)
	# # initialize the target network
	# target_params = model.init(key, jnp.expand_dims(state, axis=0))
	# # target_model = nn.Module(model, initial_params)

	# build and initialize optimizer
	optimizer = optax.adam(learning_rate)
	epsilon_by_frame = optax.polynomial_schedule(init_value=1.0, end_value=0.01, 
							                     transition_steps=num_episodes, power=1.)
	opt_state = optimizer.init(params.policy)
	# optimizer = flax.optim.Adam(1e-3).create(network)

	for n in tqdm(range(num_episodes)):
		state, _ = env.reset()

		# Initialize statistics
		ep_reward = 0.

		for t in range(num_steps):
			# generate an action from the agent's policy
			epsilon = epsilon_by_frame(t)
			action = policy(key, state, network.apply, params.policy, epsilon, num_actions)

			# step the environment
			next_state, reward, done, _, _ = env.step(int(action))

			# store to replay buffer, update episode reward
			replay_buffer.add(state, action, reward, next_state, done)
			ep_reward += reward

			# update value model when there's enough data
			if len(replay_buffer) > batch_size:
				batch = replay_buffer.sample(batch_size)
				opt_state, loss = train_step(optimizer, opt_state, network.apply, params, batch, gamma)
				ep_losses.append(float(loss))

			#Update Target model parameters
			if t % target_update_frequency == 0:
				params = Params(params.policy, params.policy)
				# target_model = target_model.replace(params=optimizer.target.params)

			# Terminate episode when absorbing state reached.
			if done: break

			# Cycle the state
			state = next_state

		# Update episodic statistics
		ep_rewards.append(ep_reward)

		if n % 100 == 0:
			if len(replay_buffer) > batch_size:
				print("Episode #{}, Return {}, Loss {}".format(n, ep_reward, loss))
			else:
				print("Episode #{}, Return {}".format(n, ep_reward))

	return params.policy, ep_rewards, ep_losses
