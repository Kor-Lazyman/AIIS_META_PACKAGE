import numpy as np
import scipy
import scipy.signal
import json
import torch
from typing import Dict, List, Tuple, Optional
def stack_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.asarray([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        import numpy as np
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return torch.as_tensor(x, device=self.device)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE)

    Args:
        rewards (np.ndarray): shape [T]
        values  (np.ndarray): shape [T+1] (bootstrap value 포함)
        gamma (float): discount factor
        lam (float): GAE lambda

    Returns:
        advantages (np.ndarray): shape [T]

    Note:
        - values는 trajectory 길이 T보다 1 길어야 함
          (마지막 값은 부트스트랩된 value)
        - ProMP, PPO, A2C 등의 advantage 계산에 사용 가능
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv
def discount_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

'''
def get_original_tf_name(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name given to the variable when creating it (i.e. name of the variable w/o the scope and the colons)
    """
    return name.split("/")[-1].split(":")[0]


def remove_scope_from_name(name, scope):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): full name of the variable with the scope removed
    """
    result = name.split(scope)[1]
    result = result[1:] if result[0] == '/' else result
    return result.split(":")[0]

def remove_first_scope_from_name(name):
    return name.replace(name + '/', "").split(":")[0]

def get_last_scope(name):
    """
    Args:
        name (str): full name of the tf variable with all the scopes

    Returns:
        (str): name of the last scope
    """
    return name.split("/")[-2]


def extract(x, *keys):
    """
    Args:
        x (dict or list): dict or list of dicts

    Returns:
        (tuple): tuple with the elements of the dict or the dicts of the list
    """
    if isinstance(x, dict):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def normalize_advantages(advantages):
    """
    Args:
        advantages (np.ndarray): np array with the advantages

    Returns:
        (np.ndarray): np array with the advantages normalized
    """
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8




def explained_variance_1d(ypred, y):
    """
    Args:
        ypred (np.ndarray): predicted values of the variable of interest
        y (np.ndarray): real values of the variable

    Returns:
        (float): variance explained by your estimator

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def concat_tensor_dict_list(tensor_dict_list):
    """
    Args:
        tensor_dict_list (list) : list of dicts of lists of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.concatenate([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU/CUDA).

    Args:
        seed (int): random seed
        deterministic (bool): make CUDA/cuDNN behavior deterministic (slower)

    Returns:
        None
    """
    import os, random
    import numpy as np
    import torch

    # 정규화(원하는 값 그대로 써도 되지만, 범위를 32-bit로 맞춰 저장성 보장)
    seed = int(seed) & 0xFFFFFFFF

    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU/CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Using seed {seed})")
'''