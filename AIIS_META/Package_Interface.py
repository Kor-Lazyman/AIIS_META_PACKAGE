import numpy as np
class model:
    def __init__(self, env, 
                policy, 
                layers, 
                algo, 
                baseline = None, 
                alpha = 1e-3, 
                beta = 1e-3, 
                num_tasks = 5, 
                inner_grad_steps = 1, 
                meta_batch_size = 5,
                clip_eps = 0.2 ,
                device = 'cuda'):
        if policy == "Gaussian":
            from Policies.Gaussian.Meta_Gaussian import MetaGaussianPolicy as selected_policy

        self.policy = selected_policy(
                 obs_dim = np.prod(env.observation_space.shape),
                 out_dim = np.prod(env.action_space.shape),
                 hidden = layers)
        
        if algo == "ProMP":
            from Algos.MAML.promp import ProMP as selected_algo
        
        self.algo = selected_algo(env = env,
                 policy = self.policy,                      # PolicyAPI 구현체
                 alpha = alpha,
                 beta = beta,
                 inner_grad_steps = inner_grad_steps,
                 num_tasks = num_tasks,
                 clip_eps = clip_eps,
                 device=device)
        if baseline != None:
            if baseline == "linear":
                from Baselines.linear_baseline import LinearBaseline as selected_baseline
            elif baseline == "zero":
                from Baselines.zero_baseline import ZeroBaseline as selected_baseline

            self.baseline = selected_baseline()
            
        else:
            self.baseline == None
        
    
    def learn(self):
        self.algo.learn()