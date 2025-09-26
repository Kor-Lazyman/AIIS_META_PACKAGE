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
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2,
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
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty, dtype=torch.float32, device=device)
        self.device = device
        self.sample_processor = MetaSampleProcessor(baseline=baseline, normalize_adv=True,device=device)
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

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(옵션) KL(old||new) 측정/패널티용"""
        raise NotImplementedError

    # -------------------- 내부 유틸 --------------------

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
        for epoch in range(epochs):
            self.env.sample_tasks(self.num_tasks)
            base_sd = copy.deepcopy(self.agent.state_dict())
            adapted_state_dicts, paths = self.inner_loop(base_state_dict=base_sd)
            for iter in range(self.outer_iters):
                self.outer_loop(paths, adapted_state_dicts)
            reward, cost_dict, _ = self.env.log_diagnostics(paths)
            self.writer.add_scalar("Reward", reward, global_step=epoch)
            self.writer.add_scalars("Costs",cost_dict, global_step=epoch)

    # -------------------- inner / outer --------------------
    def inner_loop(self,
                   base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        부모(MAML_BASE)의 state_dict 기반 inner_loop를 확장:
          - 각 inner step마다 KL(old||new)를 추정하여 저장
          - step별 KL penalty 계수(inner_kl_coeff) 적응
          - clip eps anneal 계수 갱신
        반환: (adapted_state_dicts, last_paths)
        """
        self.sampler.agent = self.agent
        base_state_dict = copy.deepcopy(self.agent.state_dict())

        # 태스크별 state_dict 클론
        adapted_state_dicts = [copy.deepcopy(base_state_dict) for _ in range(self.num_tasks)]
        last_paths = None

        # step별 KL 누계
        inner_kls_per_step = torch.zeros(self.inner_grad_steps,
                                         dtype=torch.float32,
                                         device=self.device)

        for step in range(self.inner_grad_steps + 1):
            # 1) 현재 적응 파라미터들로 수집
            last_paths = self.sampler.obtain_samples()  # list[task] of paths

            if step == self.inner_grad_steps:
                # outer에서 사용할 마지막 경로/적응 파라미터 반환
                self._last_inner_kls = inner_kls_per_step.detach()
                if self.adaptive_inner_kl_penalty and self.inner_grad_steps > 0:
                    self._adapt_inner_kl_coeff(self._last_inner_kls, self.target_kl_diff)
                # clip-anneal 업데이트
                self.anneal_coeff *= self.anneal_factor
                return adapted_state_dicts, last_paths

            else:
                # 2) 태스크별 배치 생성
                processed_batches = self.sample_processor.process_samples(last_paths)

                # 3) 태스크별 inner 업데이트 + 이번 step KL 측정
                for task_id in range(self.num_tasks):
                    batch = processed_batches[task_id]
                    # (a) 현재 파라미터 로드 후 inner loss/grad
                    loss_in = self.inner_obj(batch, adapted_state_dicts[task_id])
                    grads = torch.autograd.grad(
                    loss_in,
                    [p for _, p in self.sampler.agents[task_id].policy.named_parameters()],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                    )

                    # (b) θ' = θ - α * g  (state_dict 갱신)
                    for (name, _), g in zip(self.sampler.agents[task_id].policy.named_parameters(), grads):
                        if g is None:
                            continue
                        adapted_state_dicts[task_id][name] = (adapted_state_dicts[task_id][name] + self.alpha * g).detach()
                    '''
                    # (c) KL(old||new) 추정 (old=현재 batch의 logp, new=updated_sd 기준 logp)
                    with self.use_params(updated_sd), torch.no_grad():
                        logp_new = self.agent.log_prob(batch["observations"], batch["actions"])
                        kl_est = self._kl_from_logps(batch["agent_infos"]["logp"], logp_new)
                        inner_kls_per_step[step] += kl_est / float(self.num_tasks)
                    '''

    def outer_loop(self,
                   paths,  # inner_loop에서 마지막으로 수집된 경로(또는 새로 수집 가능)
                   adapted_state_dicts: List[Dict[str, torch.Tensor]]):
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
