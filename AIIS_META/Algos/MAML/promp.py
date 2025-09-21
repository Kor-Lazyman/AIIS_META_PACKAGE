# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import torch
from typing import Dict, List, Tuple, Optional, Any
from .base import MAML_BASE
from AIIS_META.Utils import utils
from torch.autograd import detect_anomaly
class ProMP(MAML_BASE):
    """
    Proximal Meta-Policy Search
      - Inner:  -(ratio * A).mean()
      - Outer:  PPO-clip + (최근 inner 단계 KL 평균에 대한 penalty)
      - KL(old||new) 추정: E_old[ logp_old - logp_new ]
    주의: 이 클래스는 MAML_BASE의 state_dict 스왑(use_params) 방식을 그대로 사용한다.
    """

    def __init__(self,
                 env: Any,
                 max_path_length: int,
                 agent,                         # 베이스와 동일: policy 역할의 nn.Module
                 policy,
                 optimizer,
                 baseline,
                 alpha: float = 1e-3,           # inner lr
                 beta: float  = 1e-3,           # outer lr
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 outer_iters: int = 5,
                 clip_eps: float = 0.2,
                 target_kl_diff: float = 0.01,
                 init_inner_kl_penalty: float = 1e-2,
                 adaptive_inner_kl_penalty: bool = True,
                 anneal_factor: float = 1.0,    # 1.0이면 고정, <1.0이면 점감
                 device: Optional[torch.device] = None):
        super().__init__(env, max_path_length, agent, policy, optimizer, baseline,
                         alpha, beta, inner_grad_steps, num_tasks,
                         outer_iters=outer_iters,
                         clip_eps=clip_eps,
                         init_inner_kl_penalty=init_inner_kl_penalty,
                         device=(device if device is not None else torch.device("cpu")))
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.target_kl_diff = target_kl_diff
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.anneal_factor = anneal_factor
        self.anneal_coeff = 1.0
        # step별 KL penalty 계수/최근 KL
        self.inner_kl_coeff = torch.full((inner_grad_steps,),
                                         init_inner_kl_penalty,
                                         dtype=torch.float32,
                                         device=self.device)
        self._last_inner_kls = torch.zeros(inner_grad_steps,
                                           dtype=torch.float32,
                                           device=self.device)

    # ---------- 유틸 ----------
    def _ratio(self, logp_new, logp_old, advs, clip = False):
        total_surr = 0

        eps = self.clip_eps * self.anneal_coeff
        for x in range(len(logp_new)):
            surr = torch.exp(logp_new[x] - logp_old[x])
            if clip:
                total_surr+=torch.min(torch.clamp(surr, 1.0-eps , 1.0+eps))*advs[x]
            
            else:
                total_surr+=surr

        return total_surr

    @staticmethod
    def _kl_from_logps(logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        # KL(old||new) = E_old[ logp_old - logp_new ]
        return (logp_old - logp_new).mean()

    # ---------- 훅 구현(목적함수) ----------
    def inner_obj(self, batchs: dict, base, adapted_state_dicts) -> torch.Tensor:

        """
        -(ratio * A).mean()
        주의: MAML_BASE.inner_loop가 use_params로 이미 현재 파라미터를 로드한 상태.
        params 인자는 서명 일치용이며 사용하지 않아도 됨.
        """
        batchs["logp"] = []
        surrs = []
        for idx in range(len(batchs["actions"])):
            dev = next(self.agent.parameters()).device
            obs = self._to_tensor(batchs["observations"][idx], dev, torch.float32)
            adv = self._to_tensor(batchs["advantages"][idx], dev, torch.float32)

            _, logp_outer =self.agent.get_outer_actions(obs)
            logp_inner = batchs["agent_infos"][idx]["logp"]
            surrs.append(self._ratio(logp_new=logp_inner, logp_old=logp_outer, advs=adv, clip = False))

        return sum(surrs)/len(surrs)

    def outer_obj(self, batchs) -> torch.Tensor:
        """
        -(ratio * A).mean()
        주의: MAML_BASE.inner_loop가 use_params로 이미 현재 파라미터를 로드한 상태.
        params 인자는 서명 일치용이며 사용하지 않아도 됨.
        """
        batchs["logp"] = []
        surrs = []
        for idx in range(len(batchs["actions"])):
            dev = next(self.agent.parameters()).device
            obs = self._to_tensor(batchs["observations"][idx], dev, torch.float32)
            adv = self._to_tensor(batchs["advantages"][idx], dev, torch.float32)

            _, logp_outer =self.agent.get_outer_actions(obs)
            logp_inner = batchs["agent_infos"][idx]["logp"]
            surrs.append(self._ratio(logp_new=logp_inner, logp_old=logp_outer, advs=adv, clip = True))
        
        return sum(surrs)/len(surrs)

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        KL(old||new) = E_old[ logp_old - logp_new ]
        (모니터링/로깅용; outer_loop 끝에서 호출됨)
        """
        obs = batch["observations"]
        acts = batch["actions"]
        logp_old = batch["agent_infos"]["logp"]
        logp_new = self.agent.log_prob(obs, acts)
        return self._kl_from_logps(logp_old, logp_new)

    # ---------- inner_loop override: step별 KL 추정/패널티 적응/클립 앤닐 ----------
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

        base_state_dict = copy.deepcopy(self.policy.state_dict())

        # 태스크별 state_dict 클론
        adapted_state_dicts = [copy.deepcopy(base_state_dict) for _ in range(self.num_tasks)]
        last_paths = None

        # step별 KL 누계
        inner_kls_per_step = torch.zeros(self.inner_grad_steps,
                                         dtype=torch.float32,
                                         device=self.device)

        for step in range(self.inner_grad_steps + 1):
            # 1) 현재 적응 파라미터들로 수집
            last_paths = self.sampler.obtain_samples(adapted_state_dicts)  # list[task] of paths

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
                new_adapted: List[Dict[str, torch.Tensor]] = []
                for task_id in range(self.num_tasks):
                    print(task_id)
                    batch = processed_batches[task_id]
                    # (a) 현재 파라미터 로드 후 inner loss/grad
                    loss_in = self.inner_obj(batch, base_state_dict, adapted_state_dicts[task_id])
                    grads = torch.autograd.grad(
                    loss_in,
                    [p for _, p in self.sampler.agents[task_id].policy.named_parameters()],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                    )

                    # (b) θ' = θ - α * g  (state_dict 갱신)
                    updated_sd = copy.deepcopy(adapted_state_dicts[task_id])
                    for (name, _), g in zip(self.sampler.agents[task_id].policy.named_parameters(), grads):
                        if g is None:
                            continue
                        updated_sd[name] = (updated_sd[name] - self.alpha * g).detach()
                    '''
                    # (c) KL(old||new) 추정 (old=현재 batch의 logp, new=updated_sd 기준 logp)
                    with self.use_params(updated_sd), torch.no_grad():
                        logp_new = self.agent.log_prob(batch["observations"], batch["actions"])
                        kl_est = self._kl_from_logps(batch["agent_infos"]["logp"], logp_new)
                        inner_kls_per_step[step] += kl_est / float(self.num_tasks)
                    '''

                adapted_state_dicts = new_adapted  # 다음 step을 위해 교체

    # ---------- KL penalty 계수 적응 ----------
    def _adapt_inner_kl_coeff(self, inner_kls: torch.Tensor, target: float):
        """
        간단한 2배/0.5배 스케줄. 필요하면 soft-update로 교체 가능.
        """
        new_coeff = self.inner_kl_coeff.clone()
        low, high = target / 1.5, target * 1.5
        for i, kl in enumerate(inner_kls):
            v = float(kl.item())
            if v < low:
                new_coeff[i] = new_coeff[i] * 0.5
            elif v > high:
                new_coeff[i] = new_coeff[i] * 2.0
        self.inner_kl_coeff = new_coeff
