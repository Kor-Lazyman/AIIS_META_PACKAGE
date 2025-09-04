# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from Utils import utils  # rollout 등 외부 유틸 사용 가정

class MAML_BASE(nn.Module):
    """
    MAML 틀:
      - inner_loop: meta-parameter로부터 태스크별 적응 파라미터 생성(FO-MAML 방식)
      - outer_loop: 적응 파라미터들로 outer objective의 grad를 모아 meta-parameter를 업데이트
      - learn: 위 두 함수를 호출만 (orchestration)
    """
    def __init__(self,
                 env,
                 policy: nn.Module,
                 alpha: float = 1e-3,     # inner lr
                 beta: float = 1e-3,      # outer lr
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2):
        super().__init__()
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=beta)

    # --------- 훅(오버라이드 지점) ---------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: -(ratio * adv).mean()  (최소화 기준)"""
        raise NotImplementedError

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: PPO-clip + (필요 시) KL penalty 포함 (최소화 기준)"""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(옵션) KL(old||new) 측정/패널티용"""
        raise NotImplementedError

    # --------- 유틸 ---------
    def clone_params(self, from_params: Optional[Dict[str, torch.Tensor]] = None, num_tasks = 1
                     ) -> Dict[str, torch.Tensor]:
        parm_list = []
        """파라미터 dict를 복제해 gradient 대상 텐서로 준비"""
        for x in range(num_tasks):
            if from_params is None:
                src = dict(self.policy.named_parameters())
            else:
                src = from_params
            parm_list.append({n: p.clone().detach().requires_grad_(True) for n, p in src.items()})
        return parm_list

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor], scale: float = 1.0):
        """누적된 meta-gradient를 실제 파라미터에 적용"""
        self.optimizer.zero_grad()
        for n, p in self.policy.named_parameters():
            # None 가드 (일부 파라미터에 grad가 없을 수 있음)
            if base_grads[n] is not None:
                p.grad = base_grads[n] * scale
        self.optimizer.step()

    # --------- inner / outer 분리 ---------
    def inner_loop(self, sampler,
                   task_ids: Optional[List[int]],
                   base_params: Optional[Dict[str, torch.Tensor]]
                   ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        meta-parameter(=base_params)로부터 태스크별 적응 파라미터 생성
        반환: [(task_id, adapted_params), ...]
        """

        adapted_params_list: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        for k in range(self.inner_grad_steps):
            # 각 task 별로 sample 수집
            traj_in = sampler.obtain_samples()
            params = self.clone_params(base_params, num_tasks = self.num_tasks)
            for t in range(self.num_tasks):
                # (옵션) KL 측정/패널티용(반환값은 버려도 됨)
                try:
                    _ = self.step_kl(traj_in, params[t])
                except NotImplementedError:
                    pass

                # inner 목적 계산 및 1-step GD (FO-MAML, no create_graph)
                loss_in = self.inner_obj(traj_in, params[t])
                grads = torch.autograd.grad(loss_in, params[t].values(), create_graph=False, allow_unused=True)
                params[t] = {name: p - self.alpha * g if g is not None else p
                          for (name, p), g in zip(params[t].items(), grads)}
            adapted_params_list.append((task_ids[t], params[t]))

        return adapted_params_list

    def outer_loop(self,
                   adapted_params_list: List[Tuple[int, Dict[str, torch.Tensor]]]
                   ) -> float:
        """
        적응 파라미터로 outer objective의 grad를 모아 meta-parameter를 업데이트
        반환: 평균 outer loss (모니터링용)
        """
        # meta-parameter grad 누적 버퍼
        base_grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        losses = []

        for t, params in adapted_params_list:
            # outer phase rollout
            traj_out = utils.rollout(task_id=t, params=params, phase="outer")

            # outer 목적(최소화 기준). 예: PPO-clip + inner KL penalty 등
            loss_out = self.outer_obj(traj_out, params)
            losses.append(loss_out.detach())

            grads_t = torch.autograd.grad(loss_out, params.values(), create_graph=False, allow_unused=True)
            for (name, _p), g in zip(params.items(), grads_t):
                if g is not None:
                    base_grads[name] += g

        # 태스크 평균으로 meta 파라미터 업데이트
        scale = 1.0 / max(len(adapted_params_list), 1)
        self.apply_base_grads(base_grads, scale=scale)
        return torch.stack(losses).mean().item() if len(losses) > 0 else 0.0

    # --------- 학습 루프(오케스트레이션) ---------
    def learn(self, sampler, total_iters: int):
        """
        learn은 inner/outer 호출만 담당:
          1) inner_loop(...) -> adapted_params_list
          2) outer_loop(...) -> meta-parameter update
        """
        for _ in range(total_iters):
            # 1) inner: meta-parameter -> adapted params
            base_params = dict(self.policy.named_parameters())
            adapted = self.inner_loop(sampler, task_ids=None, base_params=base_params)

            # 2) outer: adapted params로 meta-parameter 업데이트
            _ = self.outer_loop(sampler, adapted_params_list=adapted)
