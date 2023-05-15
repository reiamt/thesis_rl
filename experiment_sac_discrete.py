import gym
import torch
import gym_trading
from collections import deque
import wandb
import numpy as np
from agent.algorithms.SAC_discrete.agent import SAC
from agent.algorithms.SAC_discrete.buffer import ReplayBuffer
from agent.algorithms.SAC_discrete.utils import collect_random, save

env_args = {
    "symbol": 'XBTUSD',
    "fitting_file": 'XBTUSD_20200101_20200102_merge.csv.xz',
    "testing_file": 'XBTUSD_2020-01-03.csv.xz',
    "max_position": 10,
    "window_size": 100,
    "seed": 1,
    "action_repeats": 5,
    "training": True,
    "format_3d": False,
    "reward_type": 'trade_completion',
    "transaction_fee": True,
    "id": 'market-maker-v0'    
}

agent_args = {
    "env": "market-maker-v0",
    "numer_of_training_steps": 1e3
}

if __name__ == '__main__':
    
    env = gym.make(**env_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    config = {*env_args, *agent_args} #doesnt work, only shows keys of dicts, values are lost
    
    print(env.observation_space.shape)
    print(env.observation_space.shape[0])
    print(env.action_space.n)
    

    with wandb.init(project='SAC_disc', name='sac'):
        feature_shape = (1,*env.observation_space.shape)
        print(feature_shape)
        agent = SAC(state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n,
                    device=device)
        
        wandb.watch(agent, log='gradients', log_freq=10)

        buffer = ReplayBuffer(buffer_size=100_000, batch_size=256, device=device)

        collect_random(env=env, dataset=buffer, num_samples=10000)

        for i in range(1, 100):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

                average10.append(rewards)
                total_steps += episode_steps
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            
                wandb.log({"Reward": rewards,
                        "Average10": np.mean(average10),
                        "Steps": total_steps,
                        "Policy Loss": policy_loss,
                        "Alpha Loss": alpha_loss,
                        "Bellmann error 1": bellmann_error1,
                        "Bellmann error 2": bellmann_error2,
                        "Alpha": current_alpha,
                        "Steps": steps,
                        "Episode": i,
                        "Buffer size": buffer.__len__()})
                
                if i % config.save_every == 0:
                    save(config, save_name="SAC_discrete", model=agent.actor_local, wandb=wandb, ep=0)