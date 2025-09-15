# -*- coding: utf-8 -*-
"""
Single-head diagonal Gaussian policy (out_dim 기반)
- mean: backbone -> single linear head -> [B, out_dim]
- std:
    - 기본: state-independent log_std parameter of shape [out_dim]
    - 선택: state_dependent_std=True -> head outputs 2*out_dim (mean, log_std)
Implements common methods: forward, distribution, act, log_prob, functional forward/distribution.
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.func import functional_call
from typing import Sequence, Type, List, Dict, Optional, Tuple

from AIIS_META.Policies.base import BasePolicy  # 사용 중인 base 인터페이스에 맞춰져 있어야 함


def build_mlp(in_dim: int, hidden: tuple) -> tuple:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.Tanh())
        d = h
    return nn.Sequential(*layers), d


class GaussianPolicy(BasePolicy):
    """
    out_dim 기준의 단일-head diagonal Gaussian policy.
    Args:
      obs_dim, out_dim: 차원
      hidden, activation: backbone
      learn_std: state-independent std를 학습할지 여부 (only used when state_dependent_std=False)
      init_std: 초기 std 값 (state-independent일 때)
      min_std: numerical 안정성 위한 최소 std
      state_dependent_std: True면 head가 2*out_dim 출력 (mean, log_std)
    """
    def __init__(self,
                 obs_dim: int,
                 out_dim: int,
                 gamma: float = 0.99,
                 hidden: tuple = (64, 64),
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6,
                 state_dependent_std: bool = False,
                 has_value_fn: bool = False):
        total_out_dim = int(out_dim)
        super().__init__(obs_dim=obs_dim,
                         out_dim=total_out_dim,
                         hidden=hidden,
                         build_backbone=False,
                         has_value_fn=has_value_fn)

        self.obs_dim = obs_dim
        self.out_dim = total_out_dim
        self.gamma = gamma
        self.state_dependent_std = bool(state_dependent_std)
        self.min_log_std = float(torch.log(torch.tensor(min_std)))

        # backbone
        self.backbone, feat_dim = build_mlp(obs_dim, hidden)

        # head: if state_dependent_std -> output 2*out_dim, else output out_dim (mean)
        head_out = self.out_dim
        self.head = nn.Linear(feat_dim, head_out)

        # state-independent log_std param (only used if not state_dependent_std)
        init_log_std = float(torch.log(torch.tensor(init_std)))
        if not self.state_dependent_std:
            if learn_std:
                self.log_std = nn.Parameter(torch.full((self.out_dim,), init_log_std))
            else:
                p = nn.Parameter(torch.full((self.out_dim,), init_log_std), requires_grad=False)
                # still register as parameter so it appears in state_dict
                self.register_parameter("log_std", p)
                self.log_std = p
        else:
            # placeholder for attribute existence (won't be used)
            self.register_parameter("log_std", None)

    # ---------------- forward: mean only ----------------
    def forward(self, obs: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Returns mean: [B, out_dim]
        If state_dependent_std True, head outputs mean and log_std but forward returns mean only.
        If params provided -> functional forward using that param dict.
        """
        if params is not None:
            return self._functional_forward(obs, params)

        feat = self.backbone(obs)
        out = self.head(feat)  # [B, act_or_2act]
        if self.state_dependent_std:
            mean = out[..., :self.out_dim]
        else:
            mean = out
        return mean

    # ---------------- build distribution (current params) ----------------
    def distribution(self, obs: torch.Tensor) -> Independent:
        """
        Returns a single Independent Normal distribution over out_dim.
        """
        device = self.backbone[0].weight.device
        dtype  = self.backbone[0].weight.dtype
        obs = torch.as_tensor(obs, device=device, dtype=dtype)

        feat = self.backbone(obs)
        out = self.head(feat)
        if self.state_dependent_std:
            mean = out[..., :self.out_dim]
            log_std = out[..., self.out_dim:]
            log_std = torch.clamp(log_std, min=self.min_log_std)
            std = log_std.exp()
        else:
            mean = out
            log_std = torch.clamp(self.log_std, min=self.min_log_std)
            # expand to batch
            if mean.dim() == 2:
                log_std = log_std.unsqueeze(0).expand_as(mean)
            std = log_std.exp()
        base = Normal(mean, std)
        return Independent(base, 1)

    # ---------------- functional distribution from params ----------------
    def _distribution_from_params(self, obs: torch.Tensor, params: Dict[str, torch.Tensor]) -> Independent:
        """
        params: state_dict-like mapping. Keys should match module structure:
          - 'backbone.*' for backbone
          - 'head.*' for head linear
          - 'log_std' for state-independent log_std (if used)
        """
        # backbone functional_call
        bb_keys = {k[len("backbone."):]: v for k, v in params.items() if k.startswith("backbone.")}
        feat = functional_call(self.backbone, bb_keys, (obs,))

        # head
        head_keys = {k[len("head."):]: v for k, v in params.items() if k.startswith("head.")}
        out = functional_call(self.head, head_keys, (feat,))

        if self.state_dependent_std:
            mean = out[..., :self.out_dim]
            log_std = out[..., self.out_dim:]
            log_std = torch.clamp(log_std, min=self.min_log_std)
            std = log_std.exp()
        else:
            mean = out
            # try params for 'log_std' otherwise fallback to module's param
            if "log_std" in params:
                param_log_std = params["log_std"]
            else:
                param_log_std = self.log_std
            param_log_std = torch.clamp(param_log_std, min=self.min_log_std)
            if mean.dim() == 2:
                param_log_std = param_log_std.unsqueeze(0).expand_as(mean)
            std = param_log_std.exp()

        base = Normal(mean, std)
        return Independent(base, 1)

    # ---------------- act ----------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        obs: [B, obs_dim] or [obs_dim]
        returns: actions [B, out_dim], info dict {mean, log_std, logp}
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist = self.distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterized sample (rsample) could be used; .sample() also ok
        mean = dist.mean
        # reconstruct log_std: dist.base_dist.scale is std
        log_std = torch.log(dist.base_dist.scale)
        logp = dist.log_prob(action)

        info = {"mean": mean, "log_std": log_std, "logp": logp}
        return action, info

    # ---------------- log_prob ----------------
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Returns log probability [B]
        If params provided, use those to build distribution functionally.
        """
        if params is None:
            dist = self.distribution(obs)
        else:
            dist = self._distribution_from_params(obs, params)
        return dist.log_prob(actions)

    # ---------------- functional forward (means) ----------------
    def _functional_forward(self, obs: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return mean under params (state_dict mapping). Same key scheme as _distribution_from_params.
        """
        bb_keys = {k[len("backbone."):]: v for k, v in params.items() if k.startswith("backbone.")}
        feat = functional_call(self.backbone, bb_keys, (obs,))
        head_keys = {k[len("head."):]: v for k, v in params.items() if k.startswith("head.")}
        out = functional_call(self.head, head_keys, (feat,))
        if self.state_dependent_std:
            mean = out[..., :self.out_dim]
        else:
            mean = out
        return mean
