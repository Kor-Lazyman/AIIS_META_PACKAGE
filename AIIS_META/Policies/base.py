import torch, torch.nn as nn
from torch.distributions.normal import Normal
from typing import Sequence, Type

class BasePolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hidden: Sequence[int] = (64, 64),
                 activation: Type[nn.Module] = nn.Tanh):
        super().__init__()

        layers = []
        in_dim = obs_dim
        # 유동적 은닉층 설계
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
            
        # 출력층: mean
        layers.append(nn.Linear(in_dim, act_dim))

    def forward(self, obs, params=None):
        # 네트워크 설정
        '''
        obs: Observation
        params: inner loop에서 파라미터를 복제하여 사용하기 위해 사용
        '''
        raise NotImplementedError

    def get_action(self, obs, params=None, deterministic=False):
        '''
        obs: Observation
        params: inner loop에서 파라미터를 복제하여 사용하기 위해 사용
        '''
        raise NotImplementedError

    # 필요 시: placeholder 없이 파라미터 dict로 실행
    def _functional_forward(self, obs, params=None):
        '''
        obs: Observation
        params: inner loop에서 파라미터를 복제하여 사용하기 위해 사용
        '''
        # TODO: params dict(name->Tensor)로 동일 계산 수행
        raise NotImplementedError
