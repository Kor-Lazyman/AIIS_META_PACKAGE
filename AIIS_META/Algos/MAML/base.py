# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from typing import Dict, List, Tuple, Optional
from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
from AIIS_META.Sampler.base import SampleProcessor
from AIIS_META.Baselines import linear_baseline
import copy
from contextlib import contextmanager

class MAML_BASE(nn.Module):
    """
    MAML 틀:
      - inner_loop: meta-parameter로부터 태스크별 적응 파라미터 생성(FO-MAML 방식)
      - outer_loop: 적응 파라미터들로 outer objective의 grad를 모아 meta-parameter를 업데이트
      - learn: 위 두 함수를 호출만 (orchestration)
    """
    def __init__(self,
                env,
                max_path_length,
                policy: nn.Module,
                alpha: float = 1e-3,     # inner lr
                beta: float = 1e-3,      # outer lr
                inner_grad_steps: int = 1,
                num_tasks: int = 4,
                outer_iters: int = 5,
                parallel = False,
                rollout_per_task = 5,
                clip_eps: float = 0.2,
                init_inner_kl_penalty: float = 1e-2,
                baseline =  linear_baseline.LinearFeatureBaseline(),
                device = torch.device('cuda')):
        super().__init__()
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty)
        self.device = device

        self.sample_processor = SampleProcessor(baseline = baseline)

        self.sampler = sampler(self.env,
            self.policy,
            self.rollout_per_task,
            self.num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=parallel)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=beta)

    # --------- 훅(오버라이드 지점) ---------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: -(ratio * adv).mean()  (최소화 기준)"""
        raise NotImplementedError

    def outer_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: PPO-clip + (필요 시) KL penalty 포함 (최소화 기준)"""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(옵션) KL(old||new) 측정/패널티용"""
        raise NotImplementedError

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor], scale: float = 1.0):
        """누적된 meta-gradient를 실제 파라미터에 적용"""
        self.optimizer.zero_grad()
        for n, p in self.policy.named_parameters():
            # None 가드 (일부 파라미터에 grad가 없을 수 있음)
            if base_grads[n] is not None:
                p.grad = base_grads[n] * scale
        self.optimizer.step()
    # --------- 학습 루프(오케스트레이션) ---------
    def learn(self, epochs):
        """
        learn은 inner/outer 호출만 담당:
          1) inner_loop(...) -> adapted_params_list
          2) outer_loop(...) -> meta-parameter update
        """
        for _ in range(epochs):
            # 1) inner: meta-parameter -> adapted params
            base_params = dict(self.policy.named_parameters())
            adapted_params_per_task, paths = self.inner_loop(task_ids=None, base_params=base_params)
            for x in range(self.outer_iters):
                # 2) outer: adapted params로 meta-parameter 업데이트
                _ = self.outer_loop(paths, adapted_params_per_task)

    # --------- inner / outer 분리 ---------
    def inner_loop(self,
                   base_params: Optional[Dict[str, torch.Tensor]]
                   ) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        meta-parameter(=base_params)로부터 태스크별 적응 파라미터 생성
        반환: [(task_id, adapted_params), ...]
        """
        # 3) 메타 파라미터 복제(태스크별 1세트)
        params_list = self.clone_params(base_params, num_tasks=self.num_tasks)
        for grad in range(self.inner_grad_steps+1):
            # 1) pre-update trajectories 수집
            # 기대 반환: Dict[int, List[traj_dict]] 또는 List[List[traj_dict]]
            paths = self.sampler.obtain_samples(params_list)

            # 2) 태스크별 traj 리스트 얻기 (키가 dict일 수도, list index일 수도 있으니 정규화)
            if isinstance(paths, dict):
                # 키를 정렬해 0..num_tasks-1 순서로 뽑음
                task_keys = sorted(list(paths.keys()))[:self.num_tasks]
                paths_by_task = [paths[k] for k in task_keys]
            else:
                # list-like로 가정
                paths_by_task = [paths[i] for i in range(self.num_tasks)]
            # 업데이트
            if self.inner_grad_steps != grad:
                adapted_params_per_task: List[Tuple[int, Dict[str, torch.Tensor]]] = []
                # task별 준비
                for i in range(self.num_tasks):
                    params_i = params_list[i]
                    self.sample_processor.process_samples(paths_by_task[i], self.policy)
                    traj_list = paths_by_task[i]
                    
                    assert len(traj_list) >= 1, f"Task index {i} has no trajectories."

                    # (옵션) KL 측정이 필요하면 이 지점에서 수행 가능
                    try:
                        _ = self.step_kl({"task_index": i}, params_i)
                    except NotImplementedError:
                        pass

                    loss_sum = 0.0
                    count = 0
                    # tensor로 traj 전환, 수집한 모든 task별 traj로 loss를 계산
                    for traj in traj_list:
                        batch = {
                            "observations": self.to_tensor(traj["observations"]),
                            "actions":      self.to_tensor(traj["actions"]),
                            "rewards":      self.to_tensor(traj["rewards"]),
                            "env_infos":    traj.get("env_infos", {}),
                            "agent_infos":  traj.get("agent_infos", {}),
                            "task_index":   i,
                        }
                        
                        # inner_obj를 통해 loss 계산(inner_obj는 custom 필요)
                        loss_sum = loss_sum + self.inner_obj(batch, params_i)
                        count += 1
                    # loss의 평균
                    loss_in = loss_sum / float(max(count, 1))
                    # gradient update
                    with self._temp_params(params_i):
                        # 현재 params_i가 policy에 로드된 상태에서 loss 기준 grad 계산
                        grads = torch.autograd.grad(
                            loss_in,
                            [p for p in self.policy.parameters() if p.requires_grad],
                            create_graph=False, allow_unused=True
                        )

                    # '가상' 업데이트된 다음 스텝 state 생성 (policy에는 적용 X)
                    params_list[i] = self._apply_grads_to_state(
                        base_state=params_i, grads=grads, lr=self.alpha
                    )
                    for (name, p), g in zip(params_i.items(), grads):
                        params_list[i][name] = p - self.alpha * g if g is not None else p
                    # inner parameter 저장
                    adapted_params_per_task.append(params_list[i])

        return adapted_params_per_task, paths 
    
    def outer_loop(self, paths, adapted_params_list):
        base_grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        losses = []

        for t, params in adapted_params_list:
            # paths[t] : 해당 태스크의 trajectory 리스트 -> 텐서 배치로 변환
            traj_list = paths[t]
            assert len(traj_list) >= 1
            batch = traj_list

            with self._temp_params(params):
                loss_out = self.outer_obj(batch, params)     # 현재 모듈 파라미터 기준
                losses.append(loss_out.detach())
                grads_t = torch.autograd.grad(
                    loss_out,
                    [p for p in self.policy.parameters() if p.requires_grad],
                    create_graph=False, allow_unused=True
                )
            for (name, p), g in zip(self.policy.named_parameters(), grads_t):
                if p.requires_grad and g is not None:
                    base_grads[name] += g

        # 메타 파라미터 업데이트(평균)
        self.apply_base_grads(base_grads, scale=1.0/len(adapted_params_list))
        return torch.stack(losses).mean().item() if losses else 0.0


    # ===============Utils===============

    def _stack_trajs(self, trajs, *, adv_key: str = "advantages"):
        """
        trajs: List[dict], 각 dict 예:
        {
            "observations": (T, obs_dim) np.ndarray/torch.Tensor
            "actions":      (T, act_dim) np.ndarray/torch.Tensor
            "advantages":   (T,)         np.ndarray/torch.Tensor  # 없으면 adv_key로 찾음
            "agent_infos":  dict or List[dict] with keys in {"mean","log_std","logp"}  # 선택
        }

        반환:
        {
            "observations":  (N, obs_dim) torch.FloatTensor
            "actions":       (N, act_dim) torch.FloatTensor
            "advantages":    (N,)         torch.FloatTensor
            "agent_infos": {
                "mean":      (N, act_dim) torch.FloatTensor      # old mean
                "log_std":   (N, act_dim) torch.FloatTensor      # old log_std
                "logp":      (N,)         torch.FloatTensor      # old log_prob(a|old)
            },
            "dist_info_old": {
                "mean":      (N, act_dim) torch.FloatTensor,
                "log_std":   (N, act_dim) torch.FloatTensor,
            }
        }
        """
        device = getattr(self, "device", torch.device("cpu"))

        obs_list, act_list, adv_list = [], [], []
        mean_list, logstd_list, logp_list = [], [], []

        def to_t(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return torch.as_tensor(x, device=device, dtype=torch.float32)
        
        for traj in trajs:
            # 스텝 데이터
            obs = to_t(traj["observations"])      # (T, obs_dim)
            act = to_t(traj["actions"])           # (T, act_dim)
            # advantages 가져오기
            if adv_key in traj:
                adv = to_t(traj[adv_key]).view(-1)                     # (T,)
            elif "advantages" in traj:
                adv = to_t(traj["advantages"]).view(-1)                # (T,)
            else:
                raise ValueError("advantages(혹은 지정한 adv_key)이 traj에 없습니다.")

            # agent_infos 정규화: dict(T,dim) 혹은 list[dict] 모두 처리
            agent_infos = traj.get("agent_infos", None)

            if agent_infos is None:
                # 반드시 old logp / mean / log_std가 필요하므로 에러
                raise ValueError("agent_infos 없음: logp가 필요합니다.")

            if isinstance(agent_infos, list):
                # list[dict] -> dict(key -> stacked tensor)
                keys = agent_infos[0].keys()
                ai = {k: to_t([step[k] for step in agent_infos]) for k in keys}  # (T, ...)
            elif isinstance(agent_infos, dict):
                ai = {k: to_t(v) for k, v in agent_infos.items()}                # (T, ...)
            else:
                raise TypeError("agent_infos 형식이 dict 또는 list[dict]가 아닙니다.")


            # old logp 확보 (없으면 mean/log_std로 계산)
            if "logp" in ai:
                logp_old = ai["logp"].view(-1)
            
            else:
                raise ValueError("logp 없음: logp가 필요합니다.")
            # 누적
            obs_list.append(obs)
            act_list.append(act)
            adv_list.append(adv)
            logp_list.append(logp_old)

        # concat along time over all trajectories
        observations = torch.cat(obs_list, dim=0)       # (N, obs_dim)
        actions      = torch.cat(act_list, dim=0)       # (N, act_dim)
        advantages   = torch.cat(adv_list, dim=0)       # (N,)
        logp_old     = torch.cat(logp_list, dim=0)      # (N,)

        batch = {
            "observations": observations,
            "actions":      actions,
            "advantages":   advantages,
            "agent_infos": {
                "logp":    logp_old,
            }
        }
        return batch
    def clone_params(self, from_params: Optional[Dict[str, torch.Tensor]] = None, num_tasks: int = 1
                     ) -> List[Dict[str, torch.Tensor]]:
        """
        state_dict 기반 복제. 이후 모든 교체/주입은 load_state_dict를 통해 일관 처리.
        """
        base = self._snapshot_state() if from_params is None else {k: v.detach().clone() for k, v in from_params.items()}
        return [{k: v.detach().clone() for k, v in base.items()} for _ in range(num_tasks)]

    def apply_base_grads(self, base_grads: Dict[str, torch.Tensor], scale: float = 1.0):
        """누적된 meta-gradient를 실제 파라미터에 적용"""
        self.optimizer.zero_grad()
        for n, p in self.policy.named_parameters():
            # None 가드 (일부 파라미터에 grad가 없을 수 있음)
            if base_grads[n] is not None:
                p.grad = base_grads[n] * scale
        self.optimizer.step()
    
    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        import numpy as np
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return torch.as_tensor(x, device=self.device)

    # >>> CHANGED: 아래 3개 유틸 추가 (state_dict 스냅샷/임시적용/가상업데이트)
    def _snapshot_state(self) -> Dict[str, torch.Tensor]:
        """현재 policy state_dict를 디프 카피."""
        return {k: v.detach().clone() for k, v in self.policy.state_dict().items()}

    @contextmanager
    def _temp_params(self, state: Dict[str, torch.Tensor]):
        """
        컨텍스트 동안만 state를 policy에 로드했다가 종료 시 원복.
        나머지 로직(샘플링/로스계산)은 그대로.
        """
        prev = self._snapshot_state()
        self.policy.load_state_dict(state, strict=False)
        try:
            yield
        finally:
            self.policy.load_state_dict(prev, strict=False)

    def _apply_grads_to_state(self, base_state: Dict[str, torch.Tensor],
                              grads: List[Optional[torch.Tensor]],
                              lr: float) -> Dict[str, torch.Tensor]:
        """
        named_parameters() 순서의 grads를 받아 base_state에 '가상으로' 한 스텝 업데이트한
        새로운 state_dict 반환 (policy에는 적용 안 함).
        """
        new_state: Dict[str, torch.Tensor] = {}
        # 파라미터
        i = 0
        for name, p in self.policy.named_parameters():
            g = grads[i]; i += 1
            if g is None:
                new_state[name] = base_state[name].detach().clone()
            else:
                new_state[name] = (base_state[name] - lr * g.detach()).clone()
        # 버퍼(예: running stats 등) 그대로 복사
        for name, buf in self.policy.named_buffers():
            new_state[name] = base_state[name].detach().clone()
        # 혹시 빠진 키 있으면 보완
        for k, v in base_state.items():
            if k not in new_state:
                new_state[k] = v.detach().clone()
        return new_state

