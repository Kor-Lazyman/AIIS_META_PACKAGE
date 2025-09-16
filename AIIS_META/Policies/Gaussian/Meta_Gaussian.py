# Meta_Gaussian.py  (수정본)
import torch
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple
from AIIS_META.Utils.utils import *
from .Gaussian import GaussianPolicy  # 기존 구현 사용

class MetaGaussianPolicy(GaussianPolicy):
    def __init__(self, meta_batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_batch_size = meta_batch_size
        self._pre_update_mode = True
        print("gaussian policy ready")

    def set_pre_update_mode(self, flag: bool = True):
        self._pre_update_mode = flag
        if flag:
            self.policies_params_vals = None

    # -----------------------------------------------------------
    # 핵심: 하나의 observation에 대해 out_dim 개수만큼의 mean/std/action/logp 정보를 반환
    # -----------------------------------------------------------
    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        obs: [B, obs_dim] or [obs_dim]
        returns: actions [B, out_dim], info dict {mean, log_std, logp}
        """
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
