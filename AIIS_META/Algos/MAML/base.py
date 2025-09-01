# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MAML_BASE(nn.Module):
    """
    틀 전용: 내부/외부 목적, 업데이트, 샘플러 호출만 자리만들기
    - 내부 구현(로그확률, KL, PPO-clip 등)은 전부 TODO
    """
    def __init__(self,
                 policy: nn.Module,
                 alpha: float = 1e-3,     # inner lr
                 beta: float = 1e-3,      # outer lr
                 inner_grad_steps: int = 1,              # num_inner_grad_steps
                 num_tasks: int = 4,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2):
        super().__init__()
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=beta)

    # -------- 필수 훅(빈 틀) --------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """TODO: -E[ratio * A] 등 inner 목적 반환"""
        raise NotImplementedError

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """TODO: PPO-clip + (선택) penalty 포함한 outer 목적 반환"""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """TODO: KL(old||new) 계산(로그만/패널티용)"""
        raise NotImplementedError

    # -------- 유틸(필요 최소) --------
    def clone_params(self) -> Dict[str, torch.Tensor]:
        return {n: p.clone().detach().requires_grad_(True)
                for n, p in self.policy.named_parameters()}

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor], scale: float = 1.0):
        self.optimizer.zero_grad()
        for n, p in self.policy.named_parameters():
            p.grad = base_grads[n] * scale
        self.optimizer.step()

    # -------- 메인 루프(최소화) --------
    def learn(self, sampler, total_iters: int):
        """
        sampler 인터페이스(권장):
          - sampler.collect_inner(policy, num_tasks) -> List[dict]
          - sampler.collect_outer(policy, adapted_params_list) -> List[dict]
        각 batch dict 예시 키: {'obs','actions','advantages','dist_info_old', ...}
        """
        for _ in range(total_iters):
            # 1) inner 데이터 수집
            task_ids = sampler.select_tasks(self.num_tasks)

            adapted_params_list = []

            for t in task_ids:
                params = self.clone_params()
                for k in range(self.inner_grad_steps):
                    traj_in = sampler.rollout(task_id=t, params=params, phase="inner")
                    # (선택) KL 로깅/패널티용
                    _ = self.step_kl(traj_in, params)  # 필요 시 사용
                    loss_in = self.inner_obj(traj_in, params)
                    grads = torch.autograd.grad(loss_in, params.values(), create_graph=False)
                    params = {name: p - self.alpha * g for (name, p), g in zip(params.items(), grads)}
                adapted_params_list.append((t, params))

            # 2) 태스크별 outer 목적 계산 (적응 완료 후에 rollout 수집)
            base_grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
            for t, params in adapted_params_list:
                traj_out = sampler.rollout(task_id=t, params=params, phase="outer")
                loss_out = self.outer_obj(traj_out, params)  # PPO-clip 등 포함(최소화 기준)
                grads_t = torch.autograd.grad(loss_out, params.values(), create_graph=False) # gradient 직접 반영 하기 위해 task 별 gradient 계산
                for (name, _p), g in zip(params.items(), grads_t):
                    base_grads[name] += g

            # 3) 메타 파라미터 업데이트(태스크 평균)
            self.apply_base_grads(base_grads, scale=1.0 / self.num_tasks)
