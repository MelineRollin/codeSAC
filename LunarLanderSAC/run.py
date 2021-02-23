import gym
import lib.sac
import lib.random_action
import torch

env = gym.make('LunarLanderContinuous-v2')
save_period = 999999
store_probability = 1
run_name = "base"

lib.sac.sac(env, run_name=run_name, save_period=save_period,
            store_probability=store_probability)
