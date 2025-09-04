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
                 rollout_per_task = 5,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2,
                 device = torch.device('cuda')):
        super().__init__()
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty)
        self.device = device
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
    
    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        import numpy as np
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return torch.as_tensor(x, device=self.device)
    # --------- inner / outer 분리 ---------
    def inner_loop(self, sampler,
                   task_ids: Optional[List[int]],
                   base_params: Optional[Dict[str, torch.Tensor]]
                   ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        meta-parameter(=base_params)로부터 태스크별 적응 파라미터 생성
        반환: [(task_id, adapted_params), ...]
        """
        for grad in range(self.inner_grad_steps):
            # 1) pre-update trajectories 수집
        #    기대 반환: Dict[int, List[traj_dict]] 또는 List[List[traj_dict]]
            paths = sampler.obtain_samples()

            # 2) 태스크별 traj 리스트 얻기 (키가 dict일 수도, list index일 수도 있으니 정규화)
            if isinstance(paths, dict):
                # 키를 정렬해 0..num_tasks-1 순서로 뽑음
                task_keys = sorted(list(paths.keys()))[:self.num_tasks]
                paths_by_task = [paths[k] for k in task_keys]
            else:
                # list-like로 가정
                paths_by_task = [paths[i] for i in range(self.num_tasks)]

            # 3) 메타 파라미터 복제(태스크별 1세트)
            params_list = self.clone_params(base_params, num_tasks=self.num_tasks)
            adapted_params_per_task: List[Tuple[int, Dict[str, torch.Tensor]]] = []
            for i in range(self.num_tasks):
                params_i = params_list[i]
                traj_list = paths_by_task[i]
                assert len(traj_list) >= 1, f"Task index {i} has no trajectories."

                # (옵션) KL 측정이 필요하면 이 지점에서 수행 가능
                try:
                    _ = self.step_kl({"task_index": i}, params_i)
                except NotImplementedError:
                    pass

                loss_sum = 0.0
                count = 0
                for traj in traj_list:
                    batch = {
                        "observations": self.to_tensor(traj["observations"]),
                        "actions":      self.to_tensor(traj["actions"]),
                        "rewards":      self.to_tensor(traj["rewards"]),
                        "env_infos":    traj.get("env_infos", {}),
                        "agent_infos":  traj.get("agent_infos", {}),
                        "task_index":   i,
                    }
                    loss_sum = loss_sum + self.inner_obj(batch, params_i)
                    count += 1

                loss_in = loss_sum / float(max(count, 1))

                grads = torch.autograd.grad(
                    loss_in, list(params_i.values()),
                    create_graph=False, allow_unused=True
                )
                new_params_i = {}
                for (name, p), g in zip(params_i.items(), grads):
                    new_params_i[name] = p - self.alpha * g if g is not None else p
                adapted_params_per_task.append(new_params_i)

        return adapted_params_per_task
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
