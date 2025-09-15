# sample_processor_refactored.py
from AIIS_META.Utils import utils
import numpy as np
import torch

class Sampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        policy : policy object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self,
            env,
            policy,
            rollout_per_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False):
        assert hasattr(env, 'reset') and hasattr(env, 'step')
        self.env = env
        self.policy = policy
        self.batch_size = meta_batch_size
        self.max_path_length = max_path_length

    def obtain_samples(self):
        """Collect batch_size trajectories -> List[Path]"""
        raise NotImplementedError


class SampleProcessor(object):
    """
    Sample processor
      - (옵션) 주어진 external advantages 사용
      - 아니면 baseline을 fit하고 GAE로 advantage 계산
      - normalize/shift 옵션 제공

    Args:
        baseline (Baseline or None) : reward baseline object (fit/predict 필요)
        discount (float) : gamma
        gae_lambda (float) : lambda
        normalize_adv (bool) : advantage 정규화
        positive_adv (bool) : advantage를 양수로 쉬프트
    """

    def __init__(
        self,
        baseline=None,
        discount=0.99,
        gae_lambda=1.0,
        normalize_adv=False,
        positive_adv=False,
    ):
        assert 0.0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert 0.0 <= gae_lambda <= 1.0, 'gae_lambda must be in [0,1]'
        # baseline은 외부 adv를 쓰면 없어도 됨
        if baseline is not None:
            assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')
        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv

    def process_samples(self, paths, policy, use_adv_formula=False):
        for path in paths:
            rewards = path["rewards"]
            returns = utils.discount_cumsum(rewards, discount=self.discount)
            path["returns"] = returns
            
        for path in paths:
            if "advantages" in path:
                continue

            if getattr(policy, "has_value_fn", False):
                values = policy.value_function(torch.as_tensor(path["observations"], dtype=torch.float32))
                path["advantages"] = returns - values.detach().cpu().numpy()
                continue

            if use_adv_formula:
                # [NOTE] baseline 없으면 np.zeros 대신 오류 반환이 더 안전
                if self.baseline is None:
                    raise ValueError("GAE requires baseline for values.")
                values = self.baseline.predict(path)
                path["advantages"] = utils.compute_gae(rewards, values, gamma=policy.gamma, lam=self.lam)
                continue

            if self.baseline is not None:
                self.baseline.fit(paths)
                values = self.baseline.predict(path)
                path["advantages"] = returns - values
                continue

            raise ValueError("No way to compute advantages!")


    # -------- helpers --------
    def _flatten_advantages_input(self, paths, advantages):
        """external advantages를 일관된 1D로 변환"""
        if isinstance(advantages, list):
            # path 순서와 길이가 일치해야 함
            assert len(advantages) == len(paths), "advantages 리스트 길이가 paths와 다릅니다."
            return np.concatenate(advantages)
        adv = np.asarray(advantages)
        # 총 타임스텝과 길이 매칭 검증
        total_T = sum(len(p["rewards"]) for p in paths)
        assert adv.ndim == 1 and adv.shape[0] == total_T, \
            f"advantages 크기 {adv.shape}가 총 타임스텝({total_T})과 다릅니다."
        return adv

    def _compute_advantages(self, paths, all_path_baselines):
        """GAE(또는 0 baseline GAE)로 path['advantages'] 채움"""
        assert len(paths) == len(all_path_baselines)
        for idx, path in enumerate(paths):
            # 한 스텝 더 붙여 bootstrap (말단 0)
            path_baselines = np.append(all_path_baselines[idx], 0.0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
        return paths

    def _stack_path_data(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        actions      = np.concatenate([p["actions"] for p in paths])
        rewards      = np.concatenate([p["rewards"] for p in paths])
        returns      = np.concatenate([p["returns"] for p in paths])
        advantages = np.concatenate([p["advantages"] for p in paths])
        env_infos    = utils.concat_tensor_dict_list([p["env_infos"] for p in paths]) if "env_infos" in paths[0] else {}
        agent_infos  = utils.concat_tensor_dict_list([p["agent_infos"] for p in paths]) if "agent_infos" in paths[0] else {}
        return observations, actions, rewards, returns, advantages, env_infos, agent_infos
