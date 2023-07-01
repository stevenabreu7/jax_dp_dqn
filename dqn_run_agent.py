import random
import numpy as np
import os
from dqn_agent import run_experiment, ExpCfg, eval_agent


cfg = ExpCfg(
    env_name = 'CartPole-v1',
    n_episodes = 500,
    n_steps = 100,
    architecture = [128, 128],
    replay_size = 1000,
    batch_size = 32,
    learning_rate = 1e-3,
    target_update_frequency = 10,
    gamma = 0.9,
    eps_schedule = (1.0, 0.05, 0.999)
)

trial_name = 'trial-{}-{}e-{}s-{}lr-{}g-{}fr-{}bs-{}rs'.format(
    'x'.join(map(str, cfg.architecture)), cfg.n_episodes, cfg.n_steps, cfg.learning_rate, 
    cfg.gamma, cfg.target_update_frequency, cfg.batch_size, cfg.replay_size
)
os.makedirs(f'dqn_results/{trial_name}', exist_ok=True)

if __name__ == "__main__":
    # run experimental trial
    seed = 0
    random.seed(seed)
    print("Trial #{} with {} episodes (max {} steps)".format(seed, cfg.n_episodes, cfg.n_steps))
    # returns reward and mean loss per episode
    params, reward, loss = run_experiment(cfg) 
    loss_nonan = loss[np.isnan(loss) == False]
    print('reward min/max/mean:', reward.min(), reward.max(), reward.mean())
    print('loss min/max/mean:', loss_nonan.min(), loss_nonan.max(), loss_nonan.mean())
    np.savetxt(f'dqn_results/{trial_name}/reward.csv', reward, delimiter=',')
    np.savetxt(f'dqn_results/{trial_name}/loss.csv', loss, delimiter=',')
    np.save(f'dqn_results/{trial_name}/params.npy', params)

    # eval RL agent
    reward = eval_agent(cfg, params, epsilon=0.05)
    print('eval reward min/max/mean:', reward.min(), reward.max(), reward.mean())
    np.savetxt(f'dqn_results/{trial_name}/reward-eval.csv', reward, delimiter=',')
