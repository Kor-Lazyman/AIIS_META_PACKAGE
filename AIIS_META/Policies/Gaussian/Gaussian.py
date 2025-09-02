import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple, Sequence, Type
from base import BasePolicy

# ----- 공용: 간단 MLP -----
def build_mlp(in_dim: int, hidden: Sequence[int], activation: Type[nn.Module]) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), activation()]
        d = h
    return nn.Sequential(*layers), d

# ----- 상속 구현: 대각 가우시안 정책 -----
class MultiHeadDiagGaussianPolicy(nn.Module):
    """
    다중 연속 액션 헤드용 정책.
    - 공유 backbone + 헤드별 mean linear
    - 헤드별 log_std 파라미터 (상태불변), 필요 시 state-dependent로 확장 가능
    """
    def __init__(self,
                 obs_dim: int,
                 head_act_dims: Sequence[int],           # 예: [2, 3, 1] → 3개 헤드, 합계 act_dim=6
                 hidden: Sequence[int] = (64, 64),
                 activation: Type[nn.Module] = nn.Tanh,
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6):
        super().__init__()
        self.head_act_dims = list(head_act_dims)
        self.total_act_dim = sum(self.head_act_dims)

        # 공유 백본
        self.backbone, feat_dim = build_mlp(obs_dim, hidden, activation) # back bone: output 전까지의 layer들

        # 헤드별 mean 프로젝션
        self.heads = nn.ModuleList([nn.Linear(feat_dim, d) for d in self.head_act_dims])

        # 헤드별 log_std 파라미터/버퍼: [d_i]
        init_log_std = float(torch.log(torch.tensor(init_std)))
        if learn_std:
            self.log_std_params = nn.ParameterList(
                [nn.Parameter(torch.full((d,), init_log_std)) for d in self.head_act_dims]
            )
        else:
            self.log_std_params = nn.ParameterList()
            for d in self.head_act_dims:
                p = nn.Parameter(torch.full((d,), init_log_std), requires_grad=False)
                self.log_std_params.append(p)
        self.min_log_std = float(torch.log(torch.tensor(min_std)))

    # ---- helper: 헤드별 분포 만들기 ----
    def _make_head_dists(self, feat: torch.Tensor,
                         log_std_list: List[torch.Tensor]) -> List[Independent]:
        dists = []
        for i, head in enumerate(self.heads):
            mean_i = head(feat)                           # [B, d_i]
            log_std_i = torch.clamp(log_std_list[i], min=self.min_log_std)
            if log_std_i.dim() == 1:
                log_std_i = log_std_i.expand_as(mean_i)   # [B, d_i]
            dist_i = Independent(Normal(mean_i, log_std_i.exp()), 1)
            dists.append(dist_i)
        return dists

    # ---- 기본 forward: mean/log_std를 concat해서 반환(편의) ----
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs)
        means, log_stds = [], []
        for i, head in enumerate(self.heads):
            m = head(feat)                                # [B, d_i]
            ls = self.log_std_params[i]
            if ls.dim() == 1: ls = ls.expand_as(m)
            means.append(m); log_stds.append(ls)
        return torch.cat(means, dim=-1), torch.cat(log_stds, dim=-1)

    # ---- 분포 (헤드별 리스트와 합성 결과 둘 다 계산) ----
    def distribution(self, obs: torch.Tensor):
        feat = self.backbone(obs)
        log_std_list = [p for p in self.log_std_params]
        dists = self._make_head_dists(feat, log_std_list)
        return dists

    # ---- 샘플/로그확률: 헤드별 계산→concat/합산 ----
    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool=False):
        """
        obs: [B, obs_dim]
        반환: actions [B, sum(d_i)], info = dict(mean, log_std, logp, per_head=...)
        """
        dists = self.distribution(obs)
        acts, means, log_stds, logps = [], [], [], []
        for dist in dists:
            a = dist.mean if deterministic else dist.sample()
            acts.append(a)
            means.append(dist.mean)
            log_stds.append(torch.log(dist.base_dist.scale))
            logps.append(dist.log_prob(a))
        action = torch.cat(acts, dim=-1)                 # [B, total_act_dim]
        mean = torch.cat(means, dim=-1)
        log_std = torch.cat(log_stds, dim=-1)
        logp = torch.stack(logps, dim=-1).sum(-1)        # [B], 헤드 합산
        info = {
            "mean": mean, "log_std": log_std, "logp": logp,
            "per_head": [{"mean": m, "log_std": ls} for m, ls in zip(means, log_stds)]
        }
        return action, info

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, total_act_dim] (헤드 concat)
        """
        dists = self.distribution(obs)
        splits = torch.split(actions, self.head_act_dims, dim=-1)
        logps = [dist.log_prob(a) for dist, a in zip(dists, splits)]
        return torch.stack(logps, dim=-1).sum(-1)        # [B]