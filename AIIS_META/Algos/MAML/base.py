# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
from contextlib import contextmanager

from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
from AIIS_META.Sampler.base import SampleProcessor
from AIIS_META.Baselines import linear_baseline


class MAML_BASE(nn.Module):
    """
    MAML 틀 (FO-MAML):
      - inner_loop: meta-parameter로부터 태스크별 적응 파라미터 생성(1차 근사)
      - outer_loop: 적응 파라미터들에서의 outer objective grad를 모아 meta-parameter 업데이트
      - learn: inner/outer 오케스트레이션
    """
    def __init__(self,
                 env,
                 max_path_length,
                 agent,
                 policy,
                 alpha: float = 1e-3,     # inner lr
                 beta: float = 1e-3,      # outer lr
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 outer_iters: int = 5,
                 parallel: bool = False,
                 rollout_per_task: int = 5,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2,
                 baseline = linear_baseline.LinearFeatureBaseline(),
                 device = torch.device('cuda')):
        super().__init__()
        self.env = env
        self.agent = agent
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty, dtype=torch.float32, device=device)
        self.device = device

        self.sample_processor = SampleProcessor(baseline=baseline)
        self.sampler = sampler(self.env,
                               self.agent,
                               self.rollout_per_task,
                               self.num_tasks,
                               max_path_length,
                               envs_per_task=None,
                               parallel=parallel)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=beta)

    # -------------------- 훅(오버라이드 지점) --------------------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: -(ratio * adv).mean()  (최소화 기준)"""
        raise NotImplementedError

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: PPO-clip + (필요 시) KL penalty 포함 (최소화 기준)"""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(옵션) KL(old||new) 측정/패널티용"""
        raise NotImplementedError

    # -------------------- 내부 유틸 --------------------
    @contextmanager
    def use_params(self, state_dict: Dict[str, torch.Tensor]):
        """agent의 파라미터를 임시로 state_dict로 바꿨다가, 블록 종료시 원복"""
        orig = copy.deepcopy(self.agent.state_dict())
        self.agent.policy.load_state_dict(state_dict, strict=True)
        try:
            yield
        finally:
            self.agent.policy.load_state_dict(orig, strict=True)

    def _zero_like_paramdict(self) -> Dict[str, torch.Tensor]:
        return {n: torch.zeros_like(p, device=p.device) for n, p in self.agent.policy.named_parameters()}

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor]):
        """FO-MAML: 태스크별/스텝별로 모은 grad를 meta-parameter에 적용"""
        self.optimizer.zero_grad(set_to_none=True)
        for n, p in self.policy.named_parameters():
            g = base_grads.get(n, None)
            if g is not None:
                # grad를 그대로 할당 (alpha로 스케일 X) → 스케일은 self.beta가 담당
                p.grad = g.clone()
        self.optimizer.step()

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)
    # -------------------- 학습 루프(오케스트레이션) --------------------
    def learn(self, epochs: int):
        """
        1) inner_loop -> adapted_state_dicts, last_paths
        2) (반복) outer_loop -> meta update
        """
        for _ in range(epochs):
            base_sd = copy.deepcopy(self.policy.state_dict())
            adapted_state_dicts, paths = self.inner_loop(base_state_dict=base_sd)
            for _ in range(self.outer_iters):
                _ = self.outer_loop(paths, adapted_state_dicts)

    # -------------------- inner / outer --------------------
    @torch.no_grad()
    def _clone_state_dicts(self, base_state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        return [copy.deepcopy(base_state_dict) for _ in range(self.num_tasks)]

    def inner_loop(self,
                   base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        meta-parameter로부터 태스크별 적응 파라미터 생성
        반환: (adapted_state_dicts, last_paths)
        - FO 근사: inner step에서의 업데이트는 2차항 무시(create_graph=False)
        """
        if base_state_dict is None:
            base_state_dict = copy.deepcopy(self.policy.state_dict())

        adapted_state_dicts = self._clone_state_dicts(base_state_dict)
        last_paths = None

        for k in range(self.inner_grad_steps):
            # 1) 현 시점의 적응 파라미터들로 수집
            last_paths = self.sampler.obtain_samples(adapted_state_dicts)  # -> list[task] of paths
            # 2) 태스크별 처리 -> 배치
            processed_batches: List[dict] = []
            for task_id in range(self.num_tasks):
                batch_i = self.sample_processor.process_samples(last_paths[task_id])
                processed_batches.append(batch_i)

            # 3) 태스크별로 inner loss grad 계산 후, adapted_state_dicts 갱신(SGD, lr=self.alpha)
            new_adapted = []
            for task_id in range(self.num_tasks):
                with self.use_params(adapted_state_dicts[task_id]):
                    # grad 필요 → no @torch.no_grad
                    loss = self.inner_obj(processed_batches[task_id], adapted_state_dicts[task_id])

                    # 1차 근사: create_graph=False (2차 미분 무시)
                    grads = torch.autograd.grad(
                        loss,
                        [p for _, p in self.agent.policy.named_parameters()],
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )

                    # state_dict 업데이트: θ' = θ - α * g
                    updated_sd = copy.deepcopy(adapted_state_dicts[task_id])
                    for (name, param), g in zip(self.policy.named_parameters(), grads):
                        if g is None:  # allow_unused=True 방어
                            continue
                        updated_sd[name] = (updated_sd[name] - self.alpha * g).detach()
                    new_adapted.append(updated_sd)

            adapted_state_dicts = new_adapted  # 다음 inner step 을 위해 교체

        return adapted_state_dicts, last_paths

    def outer_loop(self,
                   paths,  # inner_loop에서 마지막으로 수집된 경로(또는 새로 수집 가능)
                   adapted_state_dicts: List[Dict[str, torch.Tensor]]):
        """
        FO-MAML: 각 태스크의 적응 파라미터에서 outer loss의 grad를 계산하고,
        그 grad를 meta-parameter에 그대로 적용(평균)한다.
        """
        # 필요 시: 새 데이터로 outer 수집하고 싶다면 아래 한 줄로 대체 가능
        # paths = self.sampler.obtain_samples(adapted_state_dicts)
        
        # 태스크별 배치 생성
        processed_batches: List[dict] = []

        for task_id in range(self.num_tasks):
            batch_i = self.sample_processor.process_samples(paths[task_id])
            processed_batches.append(batch_i)

        # 그라디언트 누적 버퍼
        accum_grads: Dict[str, torch.Tensor] = self._zero_like_paramdict()
        num_accum = 0

        for task_id in range(self.num_tasks):
            with self.use_params(adapted_state_dicts[task_id]):
                # outer loss (예: PPO-clip + KL penalty 등) — 최소화 기준
                loss_outer = self.outer_obj(processed_batches[task_id], adapted_state_dicts[task_id])

                # grad 계산(2차 미분 없음)
                grads = torch.autograd.grad(
                    loss_outer,
                    [p for _, p in self.agent.policy.named_parameters()],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )
                for (name, _), g in zip(self.agent.policy.named_parameters(), grads):
                    if g is None:
                        continue
                    accum_grads[name] += g
                num_accum += 1

        # 태스크 평균
        if num_accum > 0:
            for n in accum_grads:
                accum_grads[n] /= float(num_accum)

        # 메타 파라미터에 적용
        self.apply_base_grads(accum_grads)

        # (옵션) KL 모니터링
        kl_vals = []
        for task_id in range(self.num_tasks):
            with self.use_params(adapted_state_dicts[task_id]):
                try:
                    kl = self.step_kl(processed_batches[task_id], adapted_state_dicts[task_id])
                    kl_vals.append(float(kl.detach().cpu()))
                except NotImplementedError:
                    break

        return {
            "outer_loss": float(loss_outer.detach().cpu()) if 'loss_outer' in locals() else None,
            "mean_kl": (sum(kl_vals)/len(kl_vals)) if kl_vals else None
        }
