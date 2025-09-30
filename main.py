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
    
    agent = MetaGaussianAgent(policy = policy, num_tasks=4)

    meta_algo = ProMP(env = env, max_path_length = SIM_TIME,
                      agent = agent, optimizer=torch.optim.Adam(agent.parameters(), lr = 0.0005),
                      baseline=LinearFeatureBaseline(),
                      tensor_log=TENSORFLOW_LOGS,
                    num_tasks=4, outer_iters=5, parallel=True, rollout_per_task=5, clip_eps=0.3, device=torch.device("cpu"))
    meta_algo.learn(1000)

if __name__ == "__main__":
    main()
    params = {
        "Layers":[64, 64, 64], # layers of Network
        "rollout_per_task": 5,
        "num_task": 4, # Number of tasks
        "max_path_length": SIM_TIME,
        "tensor_log": TENSORFLOW_LOGS,
        "alpha": 0.002,
        "beta": 0.0005,
        "outer_iters": 5, # number of ProMp steps without re-sampling
        "clip_eps": 0.3, # clip range for ProMP(outer) update
        "num_inner_grad": 1,
        "epochs":1000,
        "discount": 0.99,
        "gae_lambda": 1,
        "parallel": True
    }

