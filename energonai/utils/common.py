import torch
import functools
import signal
from typing import Optional, Dict, Union, Callable, Any
from threading import Lock
from contextlib import contextmanager

DeviceType = Union[int, str, torch.device]


def build_device_maps(world_size: int, n_proc_per_node: int, rank: Optional[int] = None) -> Dict[str, Dict[DeviceType, DeviceType]]:
    is_master = rank is None
    device_maps: Dict[str, Dict[DeviceType, DeviceType]] = {}
    if is_master:
        for i in range(world_size):
            worker_local_rank = i % n_proc_per_node
            device_maps[f'worker{i}'] = {'cpu': worker_local_rank}
    else:
        local_rank = rank % n_proc_per_node
        for i in range(world_size):
            if i != rank:
                worker_local_rank = i % n_proc_per_node
                device_maps[f'worker{i}'] = {local_rank: worker_local_rank}
        device_maps['master'] = {local_rank: 'cpu'}
    return device_maps


@contextmanager
def use_lock(lock: Lock):
    try:
        lock.acquire()
        yield
    finally:
        lock.release()


def run_once(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    called: bool = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal called
        if not called:
            func(*args, **kwargs)
            called = True
    return wrapper


class Terminator:
    lock = Lock()
    called: bool = False

    @classmethod
    def shield(cls):
        with use_lock(cls.lock):
            cls.called = True

    @classmethod
    def terminate(cls):
        with use_lock(cls.lock):
            if not cls.called:
                cls.called = True
                signal.raise_signal(signal.SIGINT)
