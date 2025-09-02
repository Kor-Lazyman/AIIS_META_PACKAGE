# sample_processor_refactored.py
from Utils import utils
import numpy as np

class Sampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        policy : policy object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self, env, policy, batch_size, max_path_length):
        assert hasattr(env, 'reset') and hasattr(env, 'step')
        self.env = env
        self.policy = policy
        self.batch_size = batch_size
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

    def process_samples(self, paths, advantages=None, adv_in_paths_key=None):
        """
        Args:
            paths (list): 각 path는 dict로, 최소 {'observations','actions','rewards'} 포함
            advantages (np.ndarray | list[np.ndarray] | None):
                - 제공되면 그대로 사용 (baseline/GAE 스킵)
                - 리스트면 path 순서대로 concat
            adv_in_paths_key (str | None):
                - 제공되면 각 path[adv_in_paths_key]를 advantage로 사용

        Returns:
            dict: {
              observations, actions, rewards, returns, advantages, env_infos, agent_infos
            }
        """
        assert isinstance(paths, list), 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}

        # (1) returns 계산
        for path in paths:
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # (2) advantages 결정: external > path key > baseline+GAE/제로베이스라인
        if advantages is not None:
            adv_flat = self._flatten_advantages_input(paths, advantages)
        elif adv_in_paths_key is not None:
            # 각 path에 adv_in_paths_key가 있다고 가정
            for p in paths:
                assert adv_in_paths_key in p, f"adv_in_paths_key '{adv_in_paths_key}'가 path에 없습니다."
            adv_flat = np.concatenate([p[adv_in_paths_key] for p in paths])
        else:
            # baseline 기반 GAE (baseline 없으면 0 baseline)
            if self.baseline is not None:
                self.baseline.fit(paths, target_key="returns")
                all_path_baselines = [self.baseline.predict(p) for p in paths]
            else:
                # 0 baseline: 각 path 길이에 맞게 0 벡터
                all_path_baselines = [np.zeros_like(p["rewards"]) for p in paths]

            # advantages를 path별로 계산/저장
            paths = self._compute_advantages(paths, all_path_baselines)
            adv_flat = np.concatenate([p["advantages"] for p in paths])

        # (3) 스택 (경로를 concat한 순서에 맞춰)
        observations, actions, rewards, returns, env_infos, agent_infos = self._stack_path_data(paths)

        # (4) advantage 후처리
        if self.normalize_adv:
            adv_flat = utils.normalize_advantages(adv_flat)
        if self.positive_adv:
            adv_flat = utils.shift_advantages_to_positive(adv_flat)

        # (5) 반환 dict
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=adv_flat,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )
        return samples_data, paths

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
