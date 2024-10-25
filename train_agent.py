"""File for training a Deep RL Agent
Agents Supported:
 - PPO
 - SAC
 - DQN

Written by: Will Solow, 2024
"""
import sys
import tyro
from dataclasses import dataclass
from rl_algs.ppo import Args as PPO_Args
from rl_algs.dqn import Args as DQN_Args
from rl_algs.sac import Args as SAC_Args

import rl_algs.ppo as ppo
import rl_algs.dqn as dqn
import rl_algs.sac as sac
from wofost_gym.args import NPK_Args

@dataclass
class Args:
    # Agent configuration
    """Algorithm Parameters for PPO"""
    ppo: PPO_Args
    """Algorithm Parameters for DQN"""
    dqn: DQN_Args
    """Algorithm Parameters for SAC"""
    sac: SAC_Args

    # Environment configuration
    """Parameters for the NPK Gym Environment"""
    npk_args: NPK_Args

    """RL Agent Type: PPO, SAC, DQN"""
    agent_type: str = 'PPO'

    """Location of data folder which contains multiple runs"""
    save_folder: str = "data/"

    # Environment configuration
    """Environment ID"""
    env_id: str = "lnpkw-v0"
    """Env Reward Function"""
    env_reward: str = "RewardFertilizationThresholdWrapper"
    """Maximum N to apply"""
    max_n: float = 10
    """Maximum P to apply"""
    max_p: float = 5
    """Maximum K to apply"""
    max_k: float = 5
    """Maximum W to apply"""
    max_w: float = 5

    """Path"""
    base_fpath: str = "/Users/wsolow/Projects/agaid_crop_simulator/"
    """Relative path to agromanagement configuration file"""
    agro_fpath: str = "env_config/agro_config/annual_agro_npk.yaml"
    """Relative path to crop configuration file"""
    crop_fpath: str = "env_config/crop_config/"
    """Relative path to site configuration file"""
    site_fpath: str = "env_config/site_config/"

if __name__ == "__main__":
    
    args = tyro.cli(Args)
    if args.agent_type == 'PPO':
        ppo.main(args)

    if args.agent_type == 'DQN':
        dqn.main(args.dqn)

    if args.agent_type == 'SAC':
        sac.main(args.sac)