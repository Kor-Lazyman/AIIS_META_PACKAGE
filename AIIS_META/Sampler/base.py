# sample_processor_refactored.py
from AIIS_META.Utils import utils
import numpy as np
import torch

class Sampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        agent : agent object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self,
            env,
            agent,
            rollout_per_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False):
        assert hasattr(env, 'reset') and hasattr(env, 'step')
        self.env = env
        self.agent = agent
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

    def process_samples(self, paths, agent, use_adv_formula=False):
        for path in paths:
            rewards = path["rewards"]
            returns = utils.discount_cumsum(rewards, discount=self.discount)
            path["returns"] = returns
            
        for path in paths:

            if "advantages" in path:
                continue

            if getattr(agent, "has_value_fn", False):
                values = agent.policy.value_function(torch.as_tensor(path["observations"], dtype=torch.float32))
                path["advantages"] = returns - values.cpu().numpy()
                continue

            if use_adv_formula:
                # [NOTE] baseline 없으면 np.zeros 대신 오류 반환이 더 안전
                if self.baseline is None:
                    raise ValueError("GAE requires baseline for values.")
                values = self.baseline.predict(path)
                path["advantages"] = utils.compute_gae(rewards, values, gamma=agent.gamma, lam=self.lam)
                continue

            if self.baseline is not None:
                self.baseline.fit(paths)
                values = self.baseline.predict(path)
                path["advantages"] = returns - values
                continue

            raise ValueError("No way to compute advantages!")
