# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
class MAML_BASE(nn.Module):
    """
    MAML 틀:
      - inner_loop: meta-parameter로부터 태스크별 적응 파라미터 생성(FO-MAML 방식)
      - outer_loop: 적응 파라미터들로 outer objective의 grad를 모아 meta-parameter를 업데이트
      - learn: 위 두 함수를 호출만 (orchestration)
    """
    def __init__(self,
                env,
                max_path_length,
                policy: nn.Module,
                alpha: float = 1e-3,     # inner lr
                beta: float = 1e-3,      # outer lr
                inner_grad_steps: int = 1,
                num_tasks: int = 4,
                outer_iters: int = 5,
                parallel = False,
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
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty)
        self.device = device

        self.sampler = sampler(self.env,
            self.policy,
            self.rollout_per_task,
            self.num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=parallel)
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

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor], scale: float = 1.0):
        """누적된 meta-gradient를 실제 파라미터에 적용"""
        self.optimizer.zero_grad()
        for n, p in self.policy.named_parameters():
            # None 가드 (일부 파라미터에 grad가 없을 수 있음)
            if base_grads[n] is not None:
                p.grad = base_grads[n] * scale
        self.optimizer.step()
    

    # --------- inner / outer 분리 ---------
    def inner_loop(self,
                   base_params: Optional[Dict[str, torch.Tensor]]
                   ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        meta-parameter(=base_params)로부터 태스크별 적응 파라미터 생성
        반환: [(task_id, adapted_params), ...]
        """
        # 3) 메타 파라미터 복제(태스크별 1세트)
        params_list = self.clone_params(base_params, num_tasks=self.num_tasks)
        for grad in range(self.inner_grad_steps+1):
            # 1) pre-update trajectories 수집
            # 기대 반환: Dict[int, List[traj_dict]] 또는 List[List[traj_dict]]
            paths = self.sampler.obtain_samples(params_list)

            # 2) 태스크별 traj 리스트 얻기 (키가 dict일 수도, list index일 수도 있으니 정규화)
            if isinstance(paths, dict):
                # 키를 정렬해 0..num_tasks-1 순서로 뽑음
                task_keys = sorted(list(paths.keys()))[:self.num_tasks]
                paths_by_task = [paths[k] for k in task_keys]
            else:
                # list-like로 가정
                paths_by_task = [paths[i] for i in range(self.num_tasks)]
            # 업데이트
            if self.inner_grad_steps != grad:
                adapted_params_per_task: List[Tuple[int, Dict[str, torch.Tensor]]] = []
                # task별 준비
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
                    # tensor로 traj 전환, 수집한 모든 task별 traj로 loss를 계산
                    for traj in traj_list:
                        batch = {
                            "observations": self.to_tensor(traj["observations"]),
                            "actions":      self.to_tensor(traj["actions"]),
                            "rewards":      self.to_tensor(traj["rewards"]),
                            "env_infos":    traj.get("env_infos", {}),
                            "agent_infos":  traj.get("agent_infos", {}),
                            "task_index":   i,
                        }
                        # inner_obj를 통해 loss 계산(inner_obj는 custom 필요)
                        loss_sum = loss_sum + self.inner_obj(batch, params_i)
                        count += 1
                    # loss의 평균
                    loss_in = loss_sum / float(max(count, 1))
                    # gradient update
                    grads = torch.autograd.grad(
                        loss_in, list(params_i.values()),
                        create_graph=False, allow_unused=True
                    )
                    for (name, p), g in zip(params_i.items(), grads):
                        params_list[i][name] = p - self.alpha * g if g is not None else p
                    # inner parameter 저장
                    adapted_params_per_task.append(params_list[i])

        return adapted_params_per_task, paths
    
    def outer_loop(self, paths, adapted_params_list):
        base_grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        losses = []

        for t, params in adapted_params_list:
            # paths[t] : 해당 태스크의 trajectory 리스트 -> 텐서 배치로 변환
            traj_list = paths[t]
            assert len(traj_list) >= 1
            batch = traj_list

            loss_out = self.outer_obj(batch, params)         # 태스크별 배치로 메타손실
            losses.append(loss_out.detach())

            grads_t = torch.autograd.grad(loss_out, tuple(params.values()),
                                        create_graph=False, allow_unused=True)
            for (name, _), g in zip(params.items(), grads_t):
                if g is not None:
                    base_grads[name] += g

        self.apply_base_grads(base_grads, scale=1.0/len(adapted_params_list))
        return torch.stack(losses).mean().item() if losses else 0.0

    # --------- 학습 루프(오케스트레이션) ---------
    def learn(self, epochs):
        """
        learn은 inner/outer 호출만 담당:
          1) inner_loop(...) -> adapted_params_list
          2) outer_loop(...) -> meta-parameter update
        """
        for _ in range(epochs):
            # 1) inner: meta-parameter -> adapted params
            base_params = dict(self.policy.named_parameters())
            adapted_params_per_task, paths = self.inner_loop(task_ids=None, base_params=base_params)
            for x in range(self.outer_iters):
                # 2) outer: adapted params로 meta-parameter 업데이트
                _ = self.outer_loop(paths, adapted_params_per_task)
