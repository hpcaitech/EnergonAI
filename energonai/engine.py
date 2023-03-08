import asyncio
import signal
import time
from collections import deque
from threading import Lock, Thread
from typing import Any, Callable, Deque, Dict, Hashable, List, Optional, Tuple

import torch.distributed.rpc as trpc
import torch.nn as nn
from colossalai.logging import get_dist_logger

from .batch_mgr import BatchManager, SubmitEntry
from .pipe import Pipe
from .task import TaskEntry
from .utils import Terminator, build_device_maps, use_lock
from .worker import launch_workers


class QueueFullError(Exception):
    pass


class AsyncEngine:
    def __init__(self, tp_world_size: int, pp_world_size: int, master_host: str, rpc_port: int, n_proc_per_node: int,
                 batch_manager: Optional[BatchManager] = None, pipe_size: int = 1, queue_size: int = 0, rpc_disable_shm: bool = True) -> None:
        self.lock = Lock()
        self.logger = get_dist_logger('energonai')
        if batch_manager is None:
            self.batch_manager = BatchManager()
        else:
            assert isinstance(batch_manager, BatchManager)
            self.batch_manager = batch_manager
        self.world_size = tp_world_size * pp_world_size

        rpc_options = {}
        if rpc_disable_shm:
            # SHM may lead to timeout error. Disabling SHM and only enabling uv transport can solve this problem.
            # See https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
            # This is a workaround and may be solved in the future.
            rpc_options['_transports'] = ['uv']
        trpc.init_rpc('master', rank=0, world_size=self.world_size + 1,
                      rpc_backend_options=trpc.TensorPipeRpcBackendOptions(
                          init_method=f'tcp://{master_host}:{rpc_port}',
                          device_maps=build_device_maps(self.world_size, n_proc_per_node),
                          **rpc_options
                      ))
        self.from_worker_pipes: List[Pipe] = []
        for i in range(self.world_size):
            pipe = Pipe(f'{i}_to_m', f'worker{i}', 'master')
            self.from_worker_pipes.append(pipe)
        self.submit_pipes: List[Pipe] = []
        self.completion_pipes: List[Pipe] = []
        for i, pipe in enumerate(self.from_worker_pipes):
            worker_pp_rank = pipe.recv()
            if worker_pp_rank == 0:
                self.submit_pipes.append(Pipe(f'm_to_{i}', 'master', f'worker{i}', max_size=pipe_size))
            if worker_pp_rank == pp_world_size - 1:
                self.completion_pipes.append(pipe)

        self.running: bool = False
        self.submit_thread = None
        self.completion_thread = None
        self.queue_size = queue_size
        self.submit_queue: Deque[SubmitEntry] = deque()
        self.batch_info: Dict[Hashable, Any] = {}
        self.timer_info: Dict[Hashable, Tuple[int, float]] = {}
        self.completion_map: Dict[Hashable, Any] = {}

        self.logger.info('Engine start')
        self._start()
        self.register_sigint()

    def _submit_loop(self) -> None:
        while self.running:
            if len(self.submit_queue) > 0:
                task_entry, batch_info = self.batch_manager.make_batch(self.submit_queue)
                self.batch_info[task_entry.uids] = batch_info
                self.timer_info[task_entry.uids] = (len(task_entry.uids), time.time())
                for pipe in self.submit_pipes:
                    pipe.send(task_entry)
            else:
                time.sleep(0.01)

    def _completion_loop(self) -> None:
        received_data: Dict[int, Any] = {}
        while self.running:
            for i, pipe in enumerate(self.completion_pipes):
                if i not in received_data:
                    try:
                        received_data[i] = pipe.recv_nowait()
                    except RuntimeError:
                        pass
            if len(received_data) == len(self.completion_pipes):
                # TODO: validate they are all the same
                task_entries: List[TaskEntry] = list(map(lambda k: received_data[k], sorted(received_data.keys())))
                received_data.clear()
                batch_info = self.batch_info.pop(task_entries[0].uids)
                for uid, output in self.batch_manager.split_batch(task_entries[0], **batch_info):
                    self.completion_map[uid] = output
                batch_size, start_time = self.timer_info.pop(task_entries[0].uids)
                self.logger.info(f'batch size: {batch_size}, time: {time.time() -start_time:.3f}')
            else:
                time.sleep(0.01)

    def _start(self) -> None:
        self.running = True
        self.submit_thread = Thread(target=self._submit_loop)
        self.submit_thread.start()
        self.completion_thread = Thread(target=self._completion_loop)
        self.completion_thread.start()

    def shutdown(self) -> None:
        with use_lock(self.lock):
            if not self.running:
                return
            self.running = False
        Terminator.shield()
        for i in range(self.world_size):
            trpc.rpc_sync(f'worker{i}', Terminator.terminate)
        trpc.shutdown()
        self.submit_thread.join()
        self.completion_thread.join()

    def submit(self, uid: Hashable, data: Any) -> None:
        assert self.submit_thread.is_alive()
        assert uid not in self.completion_map
        if self.queue_size > 0 and len(self.submit_queue) >= self.queue_size:
            raise QueueFullError(f'Submit queue full, size: {self.queue_size}')
        self.submit_queue.append(SubmitEntry(uid, data))

    async def wait(self, uid: Hashable) -> Any:
        assert self.completion_thread.is_alive()
        while True:
            if uid in self.completion_map:
                output = self.completion_map[uid]
                del self.completion_map[uid]
                return output
            await asyncio.sleep(0.1)

    def get(self, uid: Hashable, interval: float = 0.05) -> Any:
        assert self.completion_thread.is_alive()
        while True:
            if uid in self.completion_map:
                output = self.completion_map[uid]
                del self.completion_map[uid]
                return output
            time.sleep(interval)

    def _sigint_handler(self, *_):
        self.shutdown()
        raise KeyboardInterrupt

    def register_sigint(self):
        signal.signal(signal.SIGINT, self._sigint_handler)


def launch_engine(tp_world_size: int, pp_world_size: int, master_host: str, master_port: int, rpc_port: int,
                  model_fn: Callable[[Any], nn.Module], n_nodes: int = 1, node_rank: int = 0, batch_manager: Optional[BatchManager] = None,
                  pipe_size: int = 1, queue_size: int = 0, rpc_disable_shm: bool = True, **model_kwargs: Any) -> Optional[AsyncEngine]:
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes
    launch_workers(tp_world_size, pp_world_size, master_host, master_port, rpc_port,
                   model_fn, n_proc_per_node=n_proc_per_node, node_rank=node_rank, pipe_size=pipe_size, **model_kwargs)
    if node_rank == 0:
        engine = AsyncEngine(tp_world_size, pp_world_size, master_host, rpc_port,
                             n_proc_per_node, batch_manager=batch_manager, pipe_size=pipe_size, queue_size=queue_size, rpc_disable_shm=rpc_disable_shm)
        return engine
