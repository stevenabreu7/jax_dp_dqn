import random
import numpy as np
import os
from dqn_agent_jax import run_experiment

n_trials = 10
n_episodes = 1000
n_steps = 50

output = 'trial-v0'
if not os.path.exists(output):
    os.makedirs(output)

if __name__ == "__main__":
    # Run experimental trials.
    # for seed in range(n_trials):
    # for now, only run a single trial
    seed = 0
    print("Trial #{} with {} episodes, each with max {} steps".format(seed, n_episodes, n_steps))
    random.seed(seed)
    qfunc, returns, losses = run_experiment(num_episodes=n_episodes, num_steps=n_steps)
    np.savetxt('{}/returns-{}.csv'.format(output, seed), returns, delimiter=',')
    np.savetxt('{}/losses-{}.csv'.format(output, seed), losses.mean(), delimiter=',')
