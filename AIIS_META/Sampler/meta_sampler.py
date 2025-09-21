from AIIS_META.Sampler.base import Sampler
from AIIS_META.Utils.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from AIIS_META.Utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools
import torch
import copy
class MetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : agent object
        batch_size (int) : number of trajectories per task
        num_tasks (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            agent,
            rollout_per_task,
            num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=False
            ):
        super(MetaSampler, self).__init__(env,
            agent,
            rollout_per_task,
            num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=False)
        assert hasattr(env, 'set_task')
        self.envs_per_task = rollout_per_task if envs_per_task is None else envs_per_task
        self.num_tasks = num_tasks
        self.rollout_per_task = rollout_per_task
        self.total_samples = num_tasks * rollout_per_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0
        self.agents = [copy.deepcopy(self.agent) for _ in range(self.num_tasks)]
        # setup vectorized environment

        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.num_tasks, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.num_tasks, self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.num_tasks)
        assert len(tasks) == self.num_tasks
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, state_dict_per_task, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [num_tasks] x (batch_size) x [5] x (max_path_length)
        """
        self.update_tasks()
        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.num_tasks):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)

        # initial reset of envs
        obses = self.vec_env.reset()
        
        rollout_counter = 0

        task_id = 0
        while n_samples < self.total_samples:
            # execute agent
            t = time.time()
            obs_per_task = np.split(np.asarray(torch.tensor(obses).cpu()), self.num_tasks)

            # ---- (2) 태스크별 파라미터를 "직접 덮어쓴" 뒤, 해당 블록에 대한 액션 배치 계산 ----
            # 길이 = num_tasks * envs_per_task, 각 원소 dict
            
            # (a) 정책 파라미터를 해당 태스크의 params로 덮어쓰기
            if task_id != rollout_counter//self.rollout_per_task:
                task_id += 1
                self.agents[task_id].load_state_dict(state_dict_per_task[task_id], strict=False)
            
            actions, agent_infos = self.agents[task_id].get_actions(obs_per_task)
            
            # (b) 단일 태스크 배치에 대한 액션/정보 계산
            #     정책에 act_batch(obs_block) 메서드가 있어야 한다.
            #     반환: actions_j [envs_per_task, act_dim], infos_j (list of dict, len=envs_per_task)
            
            # step environments
            t = time.time()
           
            actions = actions.reshape(self.envs_per_task * self.num_tasks, np.prod(self.env.action_space.shape))
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
 
            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)
            
            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                
                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=running_paths[idx]["observations"],
                        actions=running_paths[idx]["actions"],
                        rewards=running_paths[idx]["rewards"],
                        env_infos=running_paths[idx]["env_infos"],
                        agent_infos=running_paths[idx]["agent_infos"],
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()
                    rollout_counter+=1

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
            
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(agent_infos) == self.num_tasks
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = sum(agent_infos, [])  # stack agent_infos

        assert len(agent_infos) == self.num_tasks * self.envs_per_task == len(env_infos)
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], env_infos=[], agent_infos=[])

