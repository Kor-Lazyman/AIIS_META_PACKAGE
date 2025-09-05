# promp.py
# -*- coding: utf-8 -*-
import torch
from typing import Dict, List, Tuple, Optional, Any
from base import MAML_BASE

class ProMP(MAML_BASE):
    """
    Proximal Meta-Policy Search (분포 타입 독립, log_prob만 사용)
      - Inner: -(ratio * A).mean()
      - Outer: PPO-clip + inner-step KL penalty
      - KL(old||new) 추정: E_old[ logp_old - logp_new ] (분포 객체 불필요)
    """
    def __init__(self,
                 env: Any,
                 policy,                      # PolicyAPI 구현체
                 alpha: float = 1e-3,
                 beta: float  = 1e-3,
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 clip_eps: float = 0.2,
                 target_inner_step: float = 0.01,
                 init_inner_kl_penalty: float = 1e-2,
                 adaptive_inner_kl_penalty: bool = True,
                 anneal_factor: float = 1.0,
                 device: Optional[torch.device] = None):
        super().__init__(env, policy, alpha, beta, inner_grad_steps, num_tasks, device)
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.anneal_factor = anneal_factor
        self.anneal_coeff = 1.0

        # step별 KL penalty 계수/최근 KL
        self.inner_kl_coeff = torch.full((inner_grad_steps,),
                                         init_inner_kl_penalty,
                                         device=self.device)
        self._last_inner_kls = torch.zeros(inner_grad_steps, device=self.device)

    # ----- 내부 유틸: ratio / kl_from_logps -----
    @staticmethod
    def _ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
        return torch.exp(logp_new - logp_old)

    @staticmethod
    def _kl_from_logps(logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        """
        KL(old||new) = E_old[ logp_old - logp_new ]  (샘플이 old에서 나왔다는 가정)
        """
        return (logp_old - logp_new).mean()

    # ----- Inner / Outer 목적 -----
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs  = batch["observations"]
        acts = batch["actions"]
        adv  = batch["advantages"]
        logp_old = batch["agent_infos"]["logp"]

        logp_new = self.policy.log_prob(obs, acts, params=params)   # [N]
        ratio = self._ratio(logp_new, logp_old)
        return -(ratio * adv).mean()

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs  = batch["observations"]
        acts = batch["actions"]
        adv  = batch["advantages"]
        logp_old = batch["agent_infos"]["logp"]

        logp_new = self.policy.log_prob(obs, acts, params=params)
        ratio = self._ratio(logp_new, logp_old)

        eps = self.clip_eps * self.anneal_coeff
        unclipped = ratio * adv
        clipped   = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        surr_loss = -torch.min(unclipped, clipped).mean()

        # 최근 inner 단계별 KL 평균을 penalty로 사용(스칼라)
        inner_pen = (self.inner_kl_coeff * self._last_inner_kls).mean()

        return surr_loss + inner_pen

    # ----- inner_loop 오버라이드: step별 KL 측정/계수 적응/클립 앤닐 -----
    def inner_loop(self, sampler, base_params: Dict[str, torch.Tensor]):
        params_list = self.clone_params(base_params, num=self.num_tasks)
        inner_kls_per_step = torch.zeros(self.inner_grad_steps, device=self.device)

        for step in range(self.inner_grad_steps + 1):
            paths = sampler.obtain_samples(params_list)

            if step == self.inner_grad_steps:
                # outer에서 사용할 마지막 경로
                adapted = [(t, params_list[t]) for t in range(self.num_tasks)]
                # 기록/적응/앤닐
                self._last_inner_kls = inner_kls_per_step.detach()
                if self.adaptive_inner_kl_penalty:
                    self._adapt_inner_kl_coeff(self._last_inner_kls, self.target_inner_step)
                self.anneal_coeff *= self.anneal_factor
                return adapted, paths

            # 각 태스크 inner 업데이트 + 이번 스텝 KL 측정
            for t in range(self.num_tasks):
                trajs = paths[t]
                # (1) inner loss
                batch = self._stack_trajs(trajs)  # obs/actions/advantages/agent_infos(logp=old)
                loss_in = self.inner_obj(batch, params_list[t])

                grads = torch.autograd.grad(loss_in,
                                            list(params_list[t].values()),
                                            create_graph=False,
                                            allow_unused=True)
                new_params = {}
                for (name, p), g in zip(params_list[t].items(), grads):
                    new_params[name] = p - self.alpha * g if g is not None else p

                # (2) KL(old||new) = E_old[logp_old - logp_new] (이번 스텝 평균)
                with torch.no_grad():
                    logp_new = self.policy.log_prob(batch["observations"],
                                                    batch["actions"],
                                                    params=new_params)
                    kl_est = self._kl_from_logps(batch["agent_infos"]["logp"], logp_new)
                    inner_kls_per_step[step] += kl_est / self.num_tasks

                params_list[t] = new_params

        # 안전망(여기 오지 않음)
        adapted = [(t, params_list[t]) for t in range(self.num_tasks)]
        return adapted, paths

    def _adapt_inner_kl_coeff(self, inner_kls: torch.Tensor, target: float):
        new = self.inner_kl_coeff.clone()
        for i, kl in enumerate(inner_kls):
            v = float(kl.item())
            if v < target / 1.5:
                new[i] = new[i] / 2.0
            elif v > target * 1.5:
                new[i] = new[i] * 2.0
        self.inner_kl_coeff = new
