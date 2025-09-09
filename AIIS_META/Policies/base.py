# base.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Sequence, Type, List, Dict, Optional, Tuple


class BasePolicy(nn.Module):
    """
    알고리즘(ProMP 등)이 기대하는 최소 정책 API를 한 클래스에 통합.

    핵심 원칙
      - 분포 타입(가우시안/카테고리/혼합 등)에는 전혀 의존하지 않음.
      - 알고리즘은 오직 log_prob(obs, actions, params)만 호출.
      - 샘플러는 act(obs) 호출 시 agent_infos['logp'](샘플 시점의 log_prob)를 반드시 기록.

    서브클래스가 구현해야 할 것
      - act(self, obs, deterministic=False) -> (actions, agent_infos[필수:'logp'])
      - log_prob(self, obs, actions, params=None) -> [B] 텐서
      - (선택) _functional_forward(self, obs, params): inner-loop로 적응한 params로 forward 하고 싶을 때
      - (필요 시) forward(...) 재정의 (분포/출력 방식이 다르면)

    편의 기능
      - 기본 MLP 백본(self.net)을 옵션으로 제공(build_backbone=True). 필요 없으면 무시하거나 False로 끄면 됨.
      - get_action / get_actions_all_tasks: 기존 코드와의 이름 호환용 래퍼(선택 구현).
    """

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden: Sequence[int] = (64, 64),
                 activation: Type[nn.Module] = nn.Tanh,
                 build_backbone: bool = True,
                 has_value_fn: bool = False):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = tuple(hidden)
        self.activation = activation

        # (옵션) 단순 MLP 백본. 정책에 따라 무시/재정의 가능.
        self.net: Optional[nn.Sequential] = None
        if build_backbone:
            layers: List[nn.Module] = []
            in_dim = obs_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(activation())
                in_dim = h
            layers.append(nn.Linear(in_dim, act_dim))
            self.net = nn.Sequential(*layers)
        
        self.has_value_fn = has_value_fn  # [NEW]
    # ---------------------- 공통 forward ----------------------
    def forward(self,
                obs: torch.Tensor,
                params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        관찰 -> 정책 출력(예: 로짓/평균 등). 분포/샘플링은 여기서 하지 않는다.
        - params가 주어지면 inner-loop 적응 파라미터로 계산하도록 _functional_forward 사용 권장.
        - 기본 구현은 self.net이 있을 때만 obs를 통과시키고, 없으면 NotImplementedError.
        """
        if params is not None:
            return self._functional_forward(obs, params)

        if self.net is None:
            raise NotImplementedError(
                "BasePolicy.forward: self.net이 없고 서브클래스가 forward를 재정의하지 않았습니다."
            )
        return self.net(obs)

    # ---------------------- 최소 API (알고리즘이 의존) ----------------------
    def act(self,
            obs: torch.Tensor,
            deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
          obs: [B, obs_dim] 또는 [obs_dim]
        Returns:
          actions: [B, act_dim] 또는 [act_dim]
          agent_infos: dict로 반드시 'logp' 키 포함(샘플 시점의 log_prob, shape: [B])
        Note:
          - 분포 샘플링/결정론적 행동/로그확률 계산은 '정책별'로 구현해야 한다.
        """
        raise NotImplementedError

    def log_prob(self,
                 obs: torch.Tensor,
                 actions: torch.Tensor,
                 params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
          obs: [B, obs_dim]
          actions: [B, act_dim]
          params: state_dict 형태(선택). 주어지면 해당 파라미터로 log_prob 계산.
        Returns:
          [B] 텐서 (각 샘플의 log_prob)
        Note:
          - 분포 타입에 상관없이 로그확률만 정확히 반환하면 알고리즘 쪽은 그대로 작동한다.
        """
        raise NotImplementedError
    
    # [NEW] baseline 지원 훅
    def value_function(self,
                       obs: torch.Tensor,
                       params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        A2C/PPO 등 critic이 있는 정책에서만 구현.
        없는 경우 NotImplementedError 발생.
        """
        if not self.has_value_fn:
            raise NotImplementedError("Policy has no value_function.")
        raise NotImplementedError("Subclass with baseline must implement this.")
    # ---------------------- 이름 호환용/편의 API ----------------------
    @torch.no_grad()
    def get_action(self,
                   observation: torch.Tensor,
                   task: int = 0,
                   deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        기존 코드 호환을 위해 제공하는 래퍼. 내부적으로 act(...)를 호출한다.
        - task 인자는 시그니처 호환용일 뿐, 기본 BasePolicy에서는 사용하지 않는다.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)  # [1, obs_dim]

        action, info = self.act(observation, deterministic=deterministic)

        # 배치 1개면 [0]으로 정리
        if action.dim() > 1 and action.size(0) == 1:
            action = action[0]
            info = {
                k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == 1 else v)
                for k, v in info.items()
            }
        return action, info

    @torch.no_grad()
    def get_actions_all_tasks(self,
                              observations: List[torch.Tensor]):
        """
        메타 셋업에서 태스크별 배치로 액션을 뽑을 때 사용하려면 서브클래스에서 구현.
        예시:
          - observations: 길이 meta_batch_size, 각 [B, obs_dim]
          - return: (actions_list, agent_infos_list)
              actions_list: 길이 meta_batch_size, 각 [B, act_dim]
              agent_infos_list: 길이 meta_batch_size, 각 길이=B의 리스트(dict), 각 dict에 'logp' 포함
        """
        raise NotImplementedError
    '''
    # ---------------------- functional forward 훅 ----------------------
    def _functional_forward(self,
                            obs: torch.Tensor,
                            params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        params(dict[state_dict])로 현재 모듈과 동일 연산을 수행하고자 할 때 사용.
        - torch.func.functional_call을 쓰거나,
        - 직접 레이어별 matmul + 비선형을 구현해도 됨.
        기본 Base는 정책 구조가 다양하므로 NotImplemented로 둔다.
        """
        raise NotImplementedError
    '''