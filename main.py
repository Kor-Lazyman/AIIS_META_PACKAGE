import numpy as np
from envs.config_SimPy import *
from envs.promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
import torch
from envs.config_folders import *
def main():

    env = MetaEnv()
    policy = SimpleMLP(np.prod(env.observation_space.shape), 
                       np.prod(env.action_space.shape),
                    hidden_layers = [64, 64, 64])
    
    agent = MetaGaussianAgent(policy = policy, num_tasks=5)

    meta_algo = ProMP(env = env, max_path_length = SIM_TIME,
                      agent = agent, optimizer=torch.optim.Adam(agent.parameters(), lr = 0.001),
                      baseline=LinearFeatureBaseline(min=0, max=INVEN_LEVEL_MAX*2+1),
                      tensor_log=TENSORFLOW_LOGS,
                        num_tasks=5, outer_iters=5, parallel=True, rollout_per_task=5, clip_eps=0.3, device=torch.device("cpu"))
    meta_algo.learn(1000)

if __name__ == "__main__":
    main()

