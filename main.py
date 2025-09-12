import numpy as np
from config_SimPy import *
from promp_env import MetaEnv
from AIIS_META.Policies.Gaussian.Meta_Gaussian import MetaGaussianPolicy
from AIIS_META.Baselines.linear_baseline import LinearBaseline
from AIIS_META.Algos.MAML.promp import ProMP

def main():
    env = MetaEnv()
    policy = MetaGaussianPolicy(5, np.prod(env.observation_space.shape), np.prod(env.action_space.shape), hidden = [64, 64, 64])
    meta_algo = ProMP(env = env, max_path_length = SIM_TIME, policy = policy, outer_iters=5)
    meta_algo.learn(1000)

main()

