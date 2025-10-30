# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import torch.optim as optim
from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class MAML_BASE(nn.Module):
    """
    Base class for MAML-style meta-learning algorithms.
    
    Args:
        env: Wrapped gym-compatible environment
        max_path_length: Maximum trajectory length per rollout
        agent: Policy network (e.g., MLP) that outputs actions and log-probabilities
        alpha: Inner-loop learning rate
        beta: Outer-loop (meta) learning rate
        tensor_log: Directory path for TensorBoard logging
        baseline: Baseline estimator (e.g., value function or linear baseline)
        inner_grad_steps: Number of inner adaptation gradient steps
        num_tasks: Number of meta-tasks sampled per meta-iteration
        rollout_per_task: Number of rollouts per task
        outer_iters: Number of meta-updates per set of tasks
        parallel: Whether to use multiprocessing for sampling
        clip_eps: PPO/ProMP clipping epsilon
        init_inner_kl_penalty: Initial KL penalty coefficient
        discount: Discount factor (γ) for GAE
        gae_lambda: λ parameter for Generalized Advantage Estimation
        normalize_adv: Whether to normalize advantages per batch
        trainable_learning_rate: If True, inner learning rates α_i are learnable
        device: Computation device (e.g., torch.device('cuda'))
    """

    def __init__(self,
                 env,
                 max_path_length,
                 agent,
                 alpha,
                 beta,
                 tensor_log,
                 baseline=None,
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 rollout_per_task: int = 5,
                 outer_iters: int = 5,
                 parallel: bool = False,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2,
                 discount: float = 0.99,
                 gae_lambda: float = 1.0,
                 normalize_adv=True,
                 trainable_learning_rate=True,
                 device=torch.device('cuda')):
        super().__init__()

        # ----- Core components -----
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=beta)
        self.env = env

        # ----- Meta parameters -----
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.device = device
        self.alpha = alpha
        self.beta = beta

        # KL divergence penalty coefficients (inner adaptation regularization)
        self.inner_kl_coeff = torch.full(
            (inner_grad_steps,), init_inner_kl_penalty, dtype=torch.float32, device=device
        )

        # ----- Sample processing & logging -----
        self.sample_processor = MetaSampleProcessor(
            baseline=baseline,
            discount=discount,
            gae_lambda=gae_lambda,
            normalize_adv=normalize_adv
        )
        self.writer = SummaryWriter(log_dir=tensor_log)

        # Meta-sampler for collecting trajectories from multiple tasks
        self.sampler = sampler(
            self.env,
            self.agent,
            self.rollout_per_task,
            self.num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=parallel
        )

        # Create per-parameter inner learning rates (α_i)
        self._create_step_size_tensors(trainable_learning_rate)

    # =============================================================
    # Hooks to be implemented by subclasses (ProMP, MAML, etc.)
    # =============================================================
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inner objective (task-specific loss). 
        Example: -(ratio * advantage).mean() for PPO-style methods."""
        raise NotImplementedError

    def outer_obj(self, params: Dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        """Outer objective (meta-level loss). 
        Example: PPO-clip loss + optional KL penalty."""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Optional: KL(old || new) computation for adaptive inner penalty."""
        raise NotImplementedError

    # =============================================================
    # Utility functions
    # =============================================================
    @staticmethod
    def _safe_key(name: str) -> str:
        """Replace '.' with '__' since ParameterDict keys cannot contain '.'."""
        return name.replace('.', '__')

    def _create_step_size_tensors(self, trainable_learning_rate) -> None:
        """
        Create α_i step-size tensors matching each parameter's shape.
        - If trainable=True, α_i are registered as learnable nn.Parameters.
        - If trainable=False, α_i are fixed tensors (non-trainable).
        """
        pdict = nn.ParameterDict()
        for name, p in self.agent.named_parameters():
            key = self._safe_key(name)
            init = torch.full_like(p, fill_value=self.alpha, device=self.device)
            
            if trainable_learning_rate:
                pdict[key] = nn.Parameter(init, requires_grad=True)   # Learnable α_i
            else:
                pdict[key] = nn.Parameter(init, requires_grad=False)  # Fixed α_i
        self.inner_step_sizes = pdict

    # =============================================================
    # Meta-learning loop (abstract skeleton)
    # =============================================================
    def inner_loop(self, epoch) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        Inner adaptation loop.
        Should:
          1. Collect trajectories per task with current meta-parameters
          2. Compute task-specific gradients
          3. Return list of adapted parameter dicts and processed paths
        """
        raise NotImplementedError

    def outer_loop(self,
                   paths: List[Dict],                 # Post-update rollouts per task
                   adapted_params_list: List[Dict[str, torch.Tensor]]):
        """
        Outer meta-optimization loop.
        Should:
          1. Compute meta-objective across tasks
          2. Backpropagate through inner updates
          3. Update meta-parameters (θ) using self.optimizer
        """
        raise NotImplementedError

    def learn(self, epochs: int):
        """
        Full meta-training orchestration.
        For each epoch:
          1. Sample new task distributions
          2. Run inner-loop adaptation for each task
          3. Run outer-loop optimization to update meta-parameters
        """
        for epoch in range(epochs):
            print
            # Refresh sampled tasks
            self.sampler.update_tasks()

            # Inner loop: returns adapted parameter dictionaries & processed trajectories
            adapted_params_list, paths = self.inner_loop(epoch)

            # Logging and reporting
            logging_infos = self.env.report_scalar(paths, self.rollout_per_task)

            for key in logging_infos.keys():
                if isinstance(logging_infos[key], (int, float, np.generic, torch.Tensor)):
                    self.writer.add_scalar(f"{key}", logging_infos[key], global_step=epoch)
                    

                elif isinstance(logging_infos[key], dict):
                    self.writer.add_scalars(f"{key}", logging_infos[key], global_step=epoch)

                else:
                    print("This API Support only int, float, numpy, torch, dictionary types for logging.")


            print("Outer Learning Start")
            

            # Outer loop: optimize meta-parameters
            self.outer_loop(paths, adapted_params_list)
