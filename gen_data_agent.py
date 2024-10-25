# Written Sept 2024, by Will Solow
# Code to support data generation given a single policy and single farm
# Allows the user to specify a set of years and locations (thus historical weather data)
# And a specific farm from which to gather data from 

import gymnasium as gym
import numpy as np
import wofost_gym
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys

from utils import Args
import tyro
import utils
import wofost_gym.policies as policies
from inspect import getmembers, isfunction
from rl_algs.ppo import Agent as ppo
from rl_algs.sac import Actor as sac
from rl_algs.dqn import QNetwork as dqn

def make_env(kwargs):
    env_id, env_kwargs = utils.get_gym_args(kwargs)
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = utils.wrap_env_reward(env, kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env)
        return env
    return thunk


def gen_data(env, args, pol):
    # Set the location of the weather data 
    years = np.arange(start=args.year_range[0],stop=args.year_range[1]+1,step=1)
    latitudes = np.arange(start=args.lat_range[0],stop=args.lat_range[1]+.5,step=.5)
    longitudes = np.arange(start=args.long_range[0],stop=args.long_range[1]+.5,step=.5)

    lat_long = [(i,j) for i in latitudes for j in longitudes]
    
    # Create all the location-year pairs 
    loc_yr = [[loc, yr] for yr in years for loc in lat_long]

    # Data list, to convert to array later

    # Go through every year possibility
    obs_arr = []

    df = pd.DataFrame(columns=["Year", "Latitude", "Longitude", "Date"]+ \
                      args.npk_args.output_vars+args.npk_args.weather_vars+["Days Elapsed"])

    for pair in loc_yr:
        print(pair)
        # Reset Gym environment to desired location and year 
        obs, info = env.reset(**{'year':pair[1], 'location':pair[0]})

        done = False
        while not done:
            action = pol(obs)
            next_obs, reward, done, trunc, info = env.step(action)

            # Append data/location, observation and reward
            obs_arr.append(np.concatenate(([pair[1], pair[0][0], pair[0][1], env.date.strftime('%m/%d/%Y')],obs, [reward])))

            obs = next_obs
    
            if done:
                obs, info = env.reset()
                break

    # Save all data as dataframe
    df = pd.DataFrame(data=obs_arr, columns=["Year", "Latitude", "Longitude", "Date"]+args.output_vars+args.weather_vars+["Days Elapsed", "Rewards"])
    df.to_csv(f'{args.save_folder}')
    
    return df
        
if __name__ == "__main__":

    args = tyro.cli(Args)

    env_id, env_kwargs = utils.get_gym_args(args)

    # Make the gym environment - should be as a SyncVectorEnv to support
    # Easy loading from PPO/SAC/DQN agentzs

    # Load the desired policy
    if args.policy_name == None:
        if args.agent_path == None:
            env = gym.make(env_id, **env_kwargs)
            env = utils.wrap_env_reward(env, args)

            policy = policies.default_policy
        else:
            assert args.agent_type is not None, "Specify Agent Type (SAC/DQN/PPO)"

            envs= gym.vector.SyncVectorEnv([make_env(args) for i in range(1)],)

            env = gym.make(args.env_id, **env_kwargs)
            env = utils.wrap_env_reward(env, args)
    
            if args.agent_type == 'PPO':
                policy = ppo(envs)
            elif args.agent_type == 'SAC':
                policy = sac(envs)
            elif args.agent_type == 'DQN':
                policy = dqn(envs)
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
            policy.load_state_dict(torch.load(args.agent_path, map_location=device, weights_only=True))

    else:
        try:
            env = gym.make(args.env_id, **env_kwargs)
            env = utils.wrap_env_reward(env, args)

            policy = dict(getmembers(policies, isfunction))[args.policy_name]
        except:
            print(f'No policy {args.policy_name} found in policies.py')

    df = gen_data(env, args, policy)


    sys.exit(0)
    df = pd.read_csv(args.save_folder, index_col=0)
    np_arr = df.to_numpy()

    sim_starts = np.argwhere(np_arr[:,-2]==1).flatten().astype('int32')

    arr = []
    for i in range(len(sim_starts)):
        if i+1 >= len(sim_starts):
            arr.append(np_arr[sim_starts[i]:])
        else:
            arr.append(np_arr[sim_starts[i]:sim_starts[i+1]])

    plt.figure(0)
    plt.title('Cumulative Rewards')
    for j in range(len(arr)):
        curr_arr = [arr[j][k][-1] for k in range(len(arr[j]))]
        plt.plot(np.clip(np.cumsum(curr_arr),a_min=0,a_max=None))
        plt.xlabel('Days')
    plt.show()

    
    all_vars = args.output_vars + args.weather_vars
    for i in range(len(all_vars)):
        plt.figure(i+1)
        plt.title(all_vars[i])
        for j in range(len(arr)):
            plt.plot(np.array(arr[j])[:,i+4])  
            plt.xlabel('Days')
        plt.show()
