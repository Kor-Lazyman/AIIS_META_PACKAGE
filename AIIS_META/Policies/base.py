import torch, torch.nn as nn
from torch.distributions.normal import Normal
from typing import Sequence, Type, List, Dict, Optional, Tuple
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
    
    def get_action(self, observation: torch.Tensor, task: int=0, deterministic: bool=False):
        '''
        obs: Observation
        task: 해당 모델이 사용 할 task
        deterministic: 연속형 policy를 사용 할 때 action을 deterministic하게 출력 할 지 결정
        반환:
        action: 모델을 통해서 추출 된 action
        info: agent에 대한 정보 (자유롭게 설정)
        '''
        raise NotImplementedError
    
    def get_actions_all_tasks(self, observations: List[torch.Tensor]):
        """
        observations: 길이 meta_batch_size, 각 텐서 [B, obs_dim]
        반환:
          actions_list: 길이 meta_batch_size, 각 [B, total_act_dim]
          agent_infos_list: 길이 meta_batch_size, 각 길이=B의 리스트(dict)
        """
        raise NotImplementedError
    # 필요 시: placeholder 없이 파라미터 dict로 실행
    def _functional_forward(self, obs, params=None):
        '''
        obs: Observation
        params: inner loop에서 파라미터를 복제하여 사용하기 위해 사용
        '''
        # TODO: params dict(name->Tensor)로 동일 계산 수행
        raise NotImplementedError
