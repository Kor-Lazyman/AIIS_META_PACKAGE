import numpy as np
from config_SimPy import *
from promp_env import MetaEnv
from AIIS_META.Policies.Gaussian.Meta_Gaussian import MetaGaussianPolicy
from AIIS_META.Baselines.linear_baseline import LinearBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Policies.Simple_Mlp import SimpleMLP
def main():
    env = MetaEnv()
    policy = SimpleMLP(np.prod(env.observation_space.shape), np.prod(env.action_space.shape), hidden_layers = [64, 64, 64])
    agent = MetaGaussianPolicy(policy = policy, meta_batch_size=5)
    meta_algo = ProMP(env = env, max_path_length = SIM_TIME, agent = agent, policy = policy,outer_iters=5)
    meta_algo.learn(1000)

main()

