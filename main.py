import numpy as np
from config_SimPy import *
from promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
import torch
def main():

    env = MetaEnv()
    policy = SimpleMLP(np.prod(env.observation_space.shape), np.prod(env.action_space.shape), hidden_layers = [64, 64, 64],alpha=1e-3)
    agent = MetaGaussianAgent(policy = policy, meta_batch_size=4)
    meta_algo = ProMP(env = env, max_path_length = SIM_TIME, agent = agent, policy = policy,optimizer=torch.optim.Adam(agent.parameters(), lr = 0.1e-3), baseline=LinearFeatureBaseline(min=0, max=INVEN_LEVEL_MAX*2+1),outer_iters=5)
    meta_algo.learn(1000)

main()

