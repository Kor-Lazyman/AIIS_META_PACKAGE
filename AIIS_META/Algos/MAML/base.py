# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
from contextlib import contextmanager

from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
from torch.utils.tensorboard import SummaryWriter
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
                 optimizer,
                 tensor_log,
                 baseline = None,
                 alpha: float = 0.02,     # inner lr
                 beta: float = 1e-3,      # outer lr
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 rollout_per_task: int = 5,
                 outer_iters: int = 5,
                 parallel: bool = False,
                 discount = 0.99,
                 gae_lambda = 1,
                 normalize_adv = True,
                 device = torch.device('cuda')):
        super().__init__()
        self.optimizer = optimizer
        self.env = env
        self.agent = agent
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.device = device
        self.sample_processor = MetaSampleProcessor(baseline=baseline, discount = discount, gae_lambda = gae_lambda, normalize_adv=normalize_adv, device=device)
        self.writer = SummaryWriter(log_dir=tensor_log)
        self.sampler = sampler(self.env,
                               self.agent,
                               self.rollout_per_task,
                               self.num_tasks,
                               max_path_length,
                               envs_per_task=None,
                               parallel=parallel)
        

    # -------------------- 훅(오버라이드 지점) --------------------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: -(ratio * adv).mean()  (최소화 기준)"""
        raise NotImplementedError

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: PPO-clip + (필요 시) KL penalty 포함 (최소화 기준)"""
        raise NotImplementedError


    # -------------------- 내부 유틸 --------------------

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)
    # -------------------- 학습 루프(오케스트레이션) --------------------
    def learn(self, epochs: int):
        """
        epochs: 총 학습 수
        """
        # epochs 만큼 학습을 진행
        for epoch in range(epochs):
            # 샘플 추출
            self.env.sample_tasks(self.num_tasks)
            base_sd = copy.deepcopy(self.agent.state_dict())
            adapted_state_dicts, paths = self.inner_loop(base_state_dict=base_sd)
            # meta_parameter update
            for iter in range(self.outer_iters):
                self.outer_loop(paths)
            # Reward 추출 및 기록
            reward, cost_dict, _ = self.env.log_diagnostics(paths)
            self.writer.add_scalar("Reward", reward, global_step=epoch)
            self.writer.add_scalars("Costs",cost_dict, global_step=epoch)

    # -------------------- inner / outer --------------------
    def inner_loop(self,
                base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        부모(MAML_BASE)의 state_dict 기반 inner_loop를 확장:
        반환: (adapted_state_dicts, last_paths)
        """
        last_paths = None

        # 1) 현재 메타 파라미터를 Theta_O로 설정
        adapted_state_dicts = [copy.deepcopy(base_state_dict) for _ in range(self.num_tasks)]
        # 2) Inner Adaption 시작
        for step in range(self.inner_grad_steps + 1):
            # 3) Theta로 inner_loop 학습
            if step != self.inner_grad_steps:
                # inner_Trajectory 수집
                last_paths = self.sampler.obtain_samples(adapted_state_dicts, post_update = False)  # list[task] of paths
                # Advantage를 계산
                last_paths = self.sample_processor.process_samples(last_paths)

                # 4) 태스크별 inner 업데이트
                for task_id in range(self.num_tasks):
                    batch = last_paths[task_id]
                    # (a) 현재 파라미터 로드 후 inner_loss를 계산
                    loss_in = self.inner_obj(self.dummy_agents[task_id], batch)
                    # (b) Gradient를 계산
                    grad = torch.autograd.grad(
                    loss_in,
                    [p for _, p in self.dummy_agents[task_id].named_parameters()],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                    )
                    # (c) θ' = θ - α * g  (state_dict 갱신)
                    with torch.no_grad():
                        for p, g in zip(self.dummy_agents[task_id].parameters(), grad):
                            if g is not None:
                                p.add_(-self.alpha, g)
            # 5) 마지막(Adapted Samples로 Trajectory 수집)
            elif step == self.inner_grad_steps:
                last_paths = self.sampler.obtain_samples(adapted_state_dicts, post_update = True)  # list[task] of paths
                last_paths = self.sample_processor.process_samples(last_paths)

                return adapted_state_dicts, last_paths

    def outer_loop(self, paths):  # inner_loop에서 마지막으로 수집된 경로(또는 새로 수집 가능)):
        """
        FO-MAML: 각 태스크의 적응 파라미터에서 outer loss의 grad를 계산하고,
        그 grad를 meta-parameter에 그대로 적용(평균)한다.
        """
        # 필요 시: 새 데이터로 outer 수집하고 싶다면 아래 한 줄로 대체 가능
        # paths = self.sampler.obtain_samples(adapted_state_dicts)
    
        # 3) 태스크별 inner 업데이트 + 이번 step KL 측정
        loss_outs = []
        for task_id in range(self.num_tasks):
            # (a) 현재 파라미터 로드 후 inner loss/grad
            batch = paths[task_id]
            loss_outs.append(self.outer_obj(batch))
        
        # (a) 현재 파라미터 로드 후 inner loss/grad
        loss_out = sum(loss_outs)/len(loss_outs)
        self.optimizer.zero_grad(set_to_none=True)
        loss_out.backward()                 # retain_graph=False (기본)
        self.optimizer.step()
    
        # (선택) 평균 내기: 필요 시
        # for name in grad_accumulator:
        #     grad_accumulator[name] /= len(grads)
