# Meta_Gaussian.py  (수정본)
import torch
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple

from Gaussian import GaussianPolicy  # 기존 구현 사용

class MetaGaussianPolicy(GaussianPolicy):
    def __init__(self, meta_batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_batch_size = meta_batch_size
        self._pre_update_mode = True
        self.policies_params_vals: Optional[List[Dict[str, torch.Tensor]]] = None

    def set_pre_update_mode(self, flag: bool = True):
        self._pre_update_mode = flag
        if flag:
            self.policies_params_vals = None

    def set_post_update_params(self, params_list: List[Dict[str, torch.Tensor]]):
        assert len(params_list) == self.meta_batch_size
        self.policies_params_vals = params_list
        self._pre_update_mode = False

    # params(dict[state_dict])로 분포 계산 (GaussianPolicy에 이미 구현되어 있음)
    def _distribution_with_params(self, obs: torch.Tensor, params: Dict[str, torch.Tensor]) -> Independent:
        return self._distribution_from_params(obs, params)

    # -----------------------------------------------------------
    # 핵심: 하나의 observation에 대해 out_dim 개수만큼의 mean/std/action/logp 정보를 반환
    # -----------------------------------------------------------
    @torch.no_grad()
    def get_action(self,
                   observation: torch.Tensor,
                   task: int = 0,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """
        Args:
          observation: [obs_dim] 또는 [1, obs_dim] (단일 관측)
          task: (post-update 모드일 때) 어느 태스크의 파라미터를 쓸지
          deterministic: True면 샘플 대신 mean 사용

        Returns:
          actions: tensor [out_dim]
          info: dict {
            "mean": tensor [out_dim],
            "std": tensor [out_dim],
            "logp_total": scalar tensor,
            "logp_per_dim": tensor [out_dim],
            "per_dim": list(len=out_dim) of dict(mean, std, action, logp)
          }
        """
        # 입력 정리: single observation 형태로 맞춤
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)   # [1, obs_dim]
        elif observation.dim() == 2 and observation.size(0) == 1:
            pass
        else:
            raise ValueError("get_action expects a single observation [obs_dim] or [1, obs_dim].")

        # 분포 선택: pre/post update
        if self._pre_update_mode:
            dist = self.distribution(observation)   # Independent(Normal(mean,[1,out_dim]), 1)
        else:
            assert self.policies_params_vals is not None, "post-update params가 없습니다."
            dist = self._distribution_with_params(observation, self.policies_params_vals[task])

        # action (batched [1, out_dim]) -> squeeze -> [out_dim]
        if deterministic:
            actions_b = dist.mean                    # [1, out_dim]
        else:
            # .rsample() 대신 .sample() 사용해도 됨. rsample는 reparam available
            try:
                actions_b = dist.rsample()
            except Exception:
                actions_b = dist.sample()
        actions = actions_b.squeeze(0)               # [out_dim]

        # per-dim statistics (still batched shape [1, out_dim], then squeeze)
        mean_b = dist.mean                           # [1, out_dim]
        std_b = dist.base_dist.scale                 # [1, out_dim]
        mean = mean_b.squeeze(0)                     # [out_dim]
        std = std_b.squeeze(0)                       # [out_dim]

        # log probs
        # - per-dim: base_dist.log_prob expects same batch shape, returns [1, out_dim] -> squeeze -> [out_dim]
        logp_per_dim_b = dist.base_dist.log_prob(actions.unsqueeze(0))  # [1, out_dim]
        logp_per_dim = logp_per_dim_b.squeeze(0)                        # [out_dim]

        # per-dim structured info (you can keep tensors or convert to python scalars .item())
        per_dim = []
        for i in range(self.out_dim):
            per_dim.append({
                "mean": mean[i],              # tensor scalar
                "std": std[i],                # tensor scalar
                "action": actions[i],         # tensor scalar
                "logp": logp_per_dim[i],      # tensor scalar
            })


        return actions, per_dim
