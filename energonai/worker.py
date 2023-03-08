import time
from contextlib import contextmanager
from typing import Any, Callable

import colossalai
import torch
import torch.distributed.rpc as trpc
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger

from .pipe import Pipe
from .task import TaskEntry
from .utils import Terminator, build_device_maps


class Worker:
    def __init__(self, rank: int, tp_world_size: int, pp_world_size: int, master_host: str, master_port: int, rpc_port: int, n_proc_per_node: int,
                 model_fn: Callable[[Any], nn.Module], pipe_size: int = 1, rpc_disable_shm: bool = True, **model_kwargs: Any) -> None:
        self.global_rank = rank
        self.world_size = tp_world_size * pp_world_size
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        disable_existing_loggers(exclude=['energonai', 'colossalai'])
        colossalai.launch({'parallel': {'tensor': {'mode': '1d', 'size': tp_world_size},
                          'pipeline': pp_world_size}}, rank, self.world_size, master_host, master_port)
        self.tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        self.pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if gpc.is_initialized(ParallelMode.PIPELINE) else 0

        self.model: nn.Module = model_fn(**model_kwargs).cuda()

        self.rpc_name = f'worker{self.global_rank}'
        rpc_options = {}
        if rpc_disable_shm:
            # SHM may lead to timeout error. Disabling SHM and only enabling uv transport can solve this problem.
            # See https://discuss.pytorch.org/t/rpc-behavior-difference-between-pytorch-1-7-0-vs-1-9-0/124772/5
            # This is a workaround and may be solved in the future.
            rpc_options['_transports'] = ['uv']
        trpc.init_rpc(self.rpc_name, rank=self.global_rank + 1, world_size=self.world_size + 1,
                      rpc_backend_options=trpc.TensorPipeRpcBackendOptions(
                          init_method=f'tcp://{master_host}:{rpc_port}',
                          device_maps=build_device_maps(self.world_size, n_proc_per_node, rank=self.global_rank),
                          **rpc_options
                      ))
        self.to_master_pipe = Pipe(f'{self.global_rank}_to_m', self.rpc_name, 'master')
        self.to_master_pipe.send(self.pp_rank)

        if self.pp_rank == 0:
            self.input_pipe = Pipe(f'm_to_{self.global_rank}', 'master', self.rpc_name, max_size=pipe_size)
        else:
            pp_prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
            self.input_pipe = Pipe(f'{pp_prev_rank}_to_{self.global_rank}',
                                   f'worker{pp_prev_rank}', self.rpc_name, max_size=pipe_size)
        if self.pp_rank == self.pp_world_size - 1:
            self.output_pipe = self.to_master_pipe
        else:
            pp_next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
            self.output_pipe = Pipe(f'{self.global_rank}_to_{pp_next_rank}', self.rpc_name,
                                    f'worker{pp_next_rank}', max_size=pipe_size)

        self.logger = get_dist_logger('energonai')
        self.logger.info(f'{self.rpc_name} start')
        self._start()

    @contextmanager
    def _lifespan(self):
        try:
            yield
        finally:
            self._shutdown()

    def _start(self) -> None:
        with self._lifespan():
            while True:
                try:
                    task_entry: TaskEntry = self.input_pipe.recv_nowait()
                    with torch.inference_mode():
                        outputs = self._forward(task_entry.batch)
                    self.output_pipe.send(TaskEntry(task_entry.uids, outputs))
                except RuntimeError:
                    time.sleep(0.01)

    def _shutdown(self) -> None:
        Terminator.shield()
        trpc.rpc_sync('master', Terminator.terminate)
        trpc.shutdown()

    def _forward(self, inputs: Any) -> Any:
        if isinstance(inputs, (tuple, list)):
            outputs = self.model(*inputs)
        elif isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)
        return outputs


def launch_workers(tp_world_size: int, pp_world_size: int, master_host: str, master_port: int, rpc_port: int,
                   model_fn: Callable[[Any], nn.Module], n_proc_per_node: int = 1, node_rank: int = 0, pipe_size: int = 1, rpc_disable_shm: bool = True,
                   **model_kwargs: Any) -> None:
    ctx = mp.get_context('spawn')
    procs = []
    for i in range(n_proc_per_node):
        rank = n_proc_per_node * node_rank + i
        p = ctx.Process(target=Worker, args=(rank, tp_world_size, pp_world_size,
                                             master_host, master_port, rpc_port, n_proc_per_node, model_fn, pipe_size, rpc_disable_shm), kwargs=model_kwargs)
        procs.append(p)
        p.start()
