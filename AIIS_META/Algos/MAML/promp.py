# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import torch
from typing import Dict, List, Tuple, Optional, Any
from .base import MAML_BASE
from AIIS_META.Utils import utils
from torch.utils.tensorboard import SummaryWriter

class ProMP(MAML_BASE):
    """
    Proximal Meta-Policy Search
      - Inner:  -(ratio * A).mean()
      - Outer:  PPO-clip + (최근 inner 단계 KL 평균에 대한 penalty)
      - KL(old||new) 추정: E_old[ logp_old - logp_new ]
    """

    def __init__(self,
                 env: Any,
                 max_path_length: int,
                 agent,
                 optimizer,
                 baseline,
                 tensor_log,
                 alpha: float = 0.02,           # inner lr
                 beta: float  = 1e-3,           # outer lr
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 rollout_per_task: int = 5,
                 outer_iters: int = 5,
                 parallel: bool = False,
                 clip_eps: float = 0.2,
                 target_kl_diff: float = 0.01,
                 init_inner_kl_penalty: float = 1e-2,
                 adaptive_inner_kl_penalty: bool = True,
                 anneal_factor: float = 1.0,
                 discount: float = 0.99,
                 gae_lambda: float = 1,
                 normalize_adv: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__(env, max_path_length, agent, optimizer, tensor_log, baseline,
                         alpha, beta, inner_grad_steps, num_tasks, rollout_per_task,
                         outer_iters, parallel,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         normalize_adv=normalize_adv,
                         device=device)
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.writer = SummaryWriter(log_dir=tensor_log)
        self.init_inner_kl_penalty=init_inner_kl_penalty
        self.target_kl_diff = target_kl_diff
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.anneal_factor = anneal_factor
        self.anneal_coeff = 1.0

        # step별 KL penalty 계수/최근 KL 기록
        self.inner_kl_coeff = torch.full((inner_grad_steps,),
                                         init_inner_kl_penalty,
                                         dtype=torch.float32,
                                         device=self.device)
        self._last_inner_kls = torch.zeros(inner_grad_steps,
                                           dtype=torch.float32,
                                           device=self.device)

        self.dummy_agents = [copy.deepcopy(agent) for _ in range(num_tasks)]

    # ---------- PPO surrogate ----------
    def _surrogate(self, logp_new, logp_old, advs, clip: bool = False) -> torch.Tensor:
        if isinstance(logp_new, (list, tuple)):
            logp_new = torch.stack(logp_new, dim=0)
        if isinstance(logp_old, (list, tuple)):
            logp_old = torch.stack(logp_old, dim=0)
        if isinstance(advs, (list, tuple)):
            advs = torch.stack(advs, dim=0)

        logp_old = logp_old.detach()
        advs     = advs.detach()

        eps = self.clip_eps * self.anneal_coeff
        delta = (logp_new - logp_old).clamp(-20.0, 20.0)
        ratio = torch.exp(delta)

        if clip:
            r_clip = ratio.clamp(1.0 - eps, 1.0 + eps)
            term_pos = torch.minimum(ratio, r_clip) * advs
            term_neg = torch.maximum(ratio, r_clip) * advs
            term = torch.where(advs >= 0, term_pos, term_neg)
            surr_loss = -term.mean()
        else:
            surr_loss = -(ratio * advs).mean()

        return surr_loss

    # ---------- KL 계산 ----------
    def _kl_from_logps(self, logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        return (logp_old - logp_new).mean()

    # ---------- inner objective ----------
    def inner_obj(self, new_agent, batchs: dict) -> torch.Tensor:
        surrs = []
        for idx in range(len(batchs["actions"])):
            dev = next(self.agent.parameters()).device
            actions = self._to_tensor(torch.stack(batchs["actions"][idx]), dev, torch.float32)
            obs     = self._to_tensor(batchs["observations"][idx], dev, torch.float32)
            adv     = self._to_tensor(batchs["advantages"][idx], dev, torch.float32)
            logp_old = torch.stack(batchs["agent_infos"][idx]["logp"])
            logp_new = new_agent.get_outer_actions(obs, actions)

            surrs.append(self._surrogate(logp_new, logp_old, adv, clip=False))
        return sum(surrs)

    # ---------- outer objective ----------
    def outer_obj(self, batchs) -> torch.Tensor:
        batchs["logp"] = []
        surrs = []
        avg_kls = 0.0
        dev = next(self.agent.parameters()).device

        for idx in range(len(batchs["actions"])):
            actions = self._to_tensor(torch.stack(batchs["actions"][idx]), dev, torch.float32).detach()
            obs     = self._to_tensor(batchs["observations"][idx], dev, torch.float32).detach()
            adv     = self._to_tensor(batchs["advantages"][idx], dev, torch.float32).detach()
            logp_old = torch.stack(batchs["agent_infos"][idx]["logp"]).detach()
            logp_new = self.agent.get_outer_actions(obs, actions)

            surrs.append(self._surrogate(logp_new, logp_old, adv, clip=True))

            # KL 추정
            kl_est = self._kl_from_logps(logp_old, logp_new)
            avg_kls += kl_est / float(self.num_tasks)

        # 최근 inner kls 갱신
        self._last_inner_kls = avg_kls.detach().unsqueeze(0)

        outer_kl_coeff = self.inner_kl_coeff.mean() if self.inner_kl_coeff.numel() > 0 else torch.tensor(0.0, device=dev)
        return sum(surrs) / len(surrs) + outer_kl_coeff * avg_kls

    # ---------- inner loop ----------
    def inner_loop(self,
                   base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        last_paths = None
        adapted_state_dicts = [copy.deepcopy(base_state_dict) for _ in range(self.num_tasks)]

        for step in range(self.inner_grad_steps + 1):
            if step == self.inner_grad_steps:
                last_paths = self.sampler.obtain_samples(None, post_update=True)
                last_paths = self.sample_processor.process_samples(last_paths)

                if self.adaptive_inner_kl_penalty and self.inner_grad_steps > 0:
                    self._adapt_inner_kl_coeff(self._last_inner_kls)

                self.anneal_coeff *= self.anneal_factor
                return adapted_state_dicts, last_paths
            else:
                last_paths = self.sampler.obtain_samples(adapted_state_dicts, post_update=False)
                last_paths = self.sample_processor.process_samples(last_paths)

                for task_id in range(self.num_tasks):
                    batch = last_paths[task_id]
                    loss_in = self.inner_obj(self.dummy_agents[task_id], batch)

                    grad = torch.autograd.grad(
                        loss_in,
                        [p for _, p in self.dummy_agents[task_id].named_parameters()],
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )

                    with torch.no_grad():
                        for p, g in zip(self.dummy_agents[task_id].parameters(), grad):
                            if g is not None:
                                p.add_(-self.alpha, g)

    # ---------- KL penalty coeff update ----------
    def _adapt_inner_kl_coeff(self, inner_kls: torch.Tensor):
        new_coeff = self.inner_kl_coeff.clone()
        low, high = self.target_kl_diff / 1.5, self.target_kl_diff * 1.5
        for i, kl in enumerate(inner_kls):
            v = float(kl.item())
            if v < low:
                new_coeff[i] = new_coeff[i] * 0.5
            elif v > high:
                new_coeff[i] = new_coeff[i] * 2.0
        self.inner_kl_coeff = new_coeff
