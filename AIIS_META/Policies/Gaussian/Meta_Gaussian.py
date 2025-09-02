# meta_policy_torch.py
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.func import functional_call
from typing import List, Dict, Optional, Tuple

from base import BasePolicy
from Gaussian import MultiHeadDiagGaussianPolicy  # 질문에서 만든 DiagGaussianPolicy 사용

class MetaMultiHeadDiagGaussianPolicy(MultiHeadDiagGaussianPolicy):
    def __init__(self, meta_batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_batch_size = meta_batch_size
        self._pre_update_mode = True
        self.policies_params_vals: Optional[List[Dict[str, torch.Tensor]]] = None  # 태스크별 state_dict 형식

    def set_pre_update_mode(self, flag: bool=True):
        self._pre_update_mode = flag
        if flag: self.policies_params_vals = None

    def set_post_update_params(self, params_list: List[Dict[str, torch.Tensor]]):
        assert len(params_list) == self.meta_batch_size
        self.policies_params_vals = params_list # inner의 parameter를 사용
        self._pre_update_mode = False

    # ----- params(dict[state_dict])로 한 번의 분포 계산 -----
    def _distribution_with_params(self, obs: torch.Tensor, params: Dict[str, torch.Tensor]) -> List[Independent]:
        # backbone
        bb_keys = {k[len("backbone."):]: v for k, v in params.items() if k.startswith("backbone.")}
        feat = functional_call(self.backbone, bb_keys, (obs,))

        # heads
        dists = []
        for i, head in enumerate(self.heads):
            hk = f"heads.{i}."
            head_keys = {k[len(hk):]: v for k, v in params.items() if k.startswith(hk)}
            mean_i = functional_call(head, head_keys, (feat,))

            # log_std_i: ParameterList 키는 'log_std_params.{i}'
            ls_key = f"log_std_params.{i}"
            log_std_i = params.get(ls_key, self.log_std_params[i])
            log_std_i = torch.clamp(log_std_i, min=self.min_log_std)
            if log_std_i.dim() == 1: log_std_i = log_std_i.expand_as(mean_i)

            dists.append(Independent(Normal(mean_i, log_std_i.exp()), 1))
        return dists

    # ----- 단일 관측 + 태스크 선택 -----
    @torch.no_grad()
    def get_action(self, observation: torch.Tensor, task: int=0, deterministic: bool=False):
        if observation.dim() == 1: observation = observation.unsqueeze(0)  # [1, obs_dim]

        if self._pre_update_mode:
            dists = self.distribution(observation)
        else:
            assert self.policies_params_vals is not None, "post params가 없습니다."
            dists = self._distribution_with_params(observation, self.policies_params_vals[task])

        acts, means, log_stds, logps = [], [], [], []
        for dist in dists:
            a = dist.mean if deterministic else dist.sample()
            acts.append(a)
            means.append(dist.mean)
            log_stds.append(torch.log(dist.base_dist.scale))
            logps.append(dist.log_prob(a))
        action = torch.cat(acts, dim=-1)[0]                 # [total_act_dim]
        info = {
            "mean": torch.cat(means, dim=-1)[0],
            "log_std": torch.cat(log_stds, dim=-1)[0],
            "logp": torch.stack(logps, dim=-1).sum(-1)[0],
            "per_head": [{"mean": m[0], "log_std": ls[0]} for m, ls in zip(means, log_stds)]
        }
        return action, info

    # 모든 task에 대한 action들을 수집
    @torch.no_grad()
    def get_actions_all_tasks(self, observations: List[torch.Tensor]):
        """
        observations: 길이 meta_batch_size, 각 텐서 [B, obs_dim]
        반환:
          actions_list: 길이 meta_batch_size, 각 [B, total_act_dim]
          agent_infos_list: 길이 meta_batch_size, 각 길이=B의 리스트(dict)
        """
        assert isinstance(observations, list) and len(observations) == self.meta_batch_size
        actions_list, agent_infos_list = [], []

        for idx in range(self.meta_batch_size):
            obs = observations[idx]
            if self._pre_update_mode:
                dists = self.distribution(obs)
            else:
                dists = self._distribution_with_params(obs, self.policies_params_vals[idx])

            acts, means, log_stds, logps = [], [], [], []
            for dist in dists:
                a = dist.sample()
                acts.append(a)
                means.append(dist.mean)
                log_stds.append(torch.log(dist.base_dist.scale))
                logps.append(dist.log_prob(a))

            action = torch.cat(acts, dim=-1)                  # [B, total_act_dim]
            mean = torch.cat(means, dim=-1)
            log_std = torch.cat(log_stds, dim=-1)
            logp = torch.stack(logps, dim=-1).sum(-1)         # [B]

            infos = [dict(mean=mean[i], log_std=log_std[i], logp=logp[i]) for i in range(self.meta_batch_size)]
            actions_list.append(action)
            agent_infos_list.append(infos)

        return actions_list, agent_infos_list