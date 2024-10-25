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

    for pair in loc_yr:
        # Reset Gym environment to desired location and year 
        obs, info = env.reset(**{'year':pair[1], 'location':pair[0]})

        done = False
        while not done:
            action = pol(obs)
            next_obs, reward, done, trunc, info = env.step(action)

            # Append data/location, observation and reward
            obs_arr.append(np.concatenate(([pair[1], pair[0][0], pair[0][1], env.date.strftime('%m/%d/%Y')], list(obs.values()), [reward])))

            obs = next_obs
    
            if done:
                obs, info = env.reset()
                break

    # Save all data as dataframe
    df = pd.DataFrame(data=obs_arr, columns=["Year", "Latitude", "Longitude", "Date"]\
                      +args.npk_args.output_vars+args.npk_args.weather_vars+["Days Elapsed", "Rewards"])
    df.to_csv(f'{args.save_folder+"_"+str(p)+".csv"}')
    
    return df
        
if __name__ == "__main__":

    args = tyro.cli(Args)

    env_id, env_kwargs = utils.get_gym_args(args)

    env = gym.make(env_id, **env_kwargs)
    env = wofost_gym.wrappers.NPKDictObservationWrapper(env)
    env = wofost_gym.wrappers.NPKDictActionWrapper(env)
    env = utils.wrap_env_reward(env, args)

    pols = [policies.Below_N(env, threshold=10, amount=1), policies.Below_N(env,threshold=5, amount=3), \
            policies.Below_I(env,threshold=.3, amount=2), policies.Below_I(env,threshold=.4, amount=1), \
            policies.Interval_N(env,amount=1, interval=7), policies.Interval_N(env,amount=3, interval=28), \
            policies.Interval_W(env,amount=1, interval=7), policies.Interval_W(env,amount=3, interval=28)]

    

    for p in pols:
        df = gen_data(env, args, p)
