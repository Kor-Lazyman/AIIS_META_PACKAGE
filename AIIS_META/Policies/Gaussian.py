import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from typing import Sequence, Type, Dict, Optional, Tuple
from base import BasePolicy
# ----- 상속 구현: 대각 가우시안 정책 -----
class DiagGaussianPolicy(BasePolicy):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden: Sequence[int] = (64, 64),
                 activation: Type[nn.Module] = nn.Tanh,
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6):
        super().__init__(obs_dim, act_dim, hidden, activation)

        # 동적 MLP 구성 (mean head)
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim)]
        self.net = nn.Sequential(*layers)

        # log_std 파라미터/버퍼
        init_log_std = float(torch.log(torch.tensor(init_std)))
        if learn_std:
            self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        else:
            self.register_buffer("log_std", torch.full((act_dim,), init_log_std))

        # 최솟값(수치 안정)
        self.min_log_std = float(torch.log(torch.tensor(min_std)))

    # mean, log_std 반환
    def forward(self, obs: torch.Tensor,
                params: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if params is None:
            mean = self.net(obs)
            log_std = self.log_std
        else:
            # post-update 파라미터로 실행하고 싶으면 이 부분 구현
            mean = self._functional_forward(obs, params)  # TODO: 필요 시 구현
            log_std = params.get("log_std", self.log_std)

        # 안정화 + 브로드캐스트
        log_std = torch.clamp(log_std, min=self.min_log_std)
        if log_std.dim() == 1:
            log_std = log_std.expand_as(mean)
        return mean, log_std

    # 분포 객체 (PPO에서 사용)
    def distribution(self,
                     obs: Optional[torch.Tensor] = None,
                     params: Optional[Dict[str, torch.Tensor]] = None,
                     *,
                     mean: Optional[torch.Tensor] = None,
                     log_std: Optional[torch.Tensor] = None
                     ) -> Independent:
        if mean is None or log_std is None:
            assert obs is not None, "obs 또는 (mean, log_std) 중 하나는 필요합니다."
            mean, log_std = self.forward(obs, params)
        std = log_std.exp()
        base = Normal(mean, std)          # [B, act_dim]
        return Independent(base, 1)       # 액션 차원을 이벤트로

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor,
                   params: Optional[Dict[str, torch.Tensor]] = None,
                   deterministic: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist = self.distribution(obs, params)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action)
        info = {
            "mean": dist.mean,
            "log_std": torch.log(dist.base_dist.scale),
            "logp": logp,
        }
        return action, info

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor,
                    params: Optional[Dict[str, torch.Tensor]] = None,
                    deterministic: bool = False):
        dist = self.distribution(obs, params)
        actions = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        info = {
            "mean": dist.mean,
            "log_std": torch.log(dist.base_dist.scale),
            "logp": logp,
        }
        return actions, info

    # PPO에서 자주 쓰는 log_prob
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor,
                 params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        return self.distribution(obs, params).log_prob(actions)

    # (선택) post-update용 functional forward: params dict로 self.net과 동일 연산
    def _functional_forward(self, obs: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        TODO:
          - params에서 weight/bias를 꺼내 layer별로 직접 matmul+비선형 적용하거나
          - torch.nn.utils.stateless.functional_call 사용
        """
        raise NotImplementedError("functional forward를 구현하세요 (post-update 파라미터로 실행).")
