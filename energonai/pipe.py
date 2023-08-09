import torch.distributed.rpc as trpc
import time
from queue import Queue, Empty
from typing import Dict
from threading import Lock
from typing import Any
from .utils import use_lock
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pipe')
logger.setLevel(level=logging.WARNING)


def rpc_queue_can_put(q: trpc.RRef) -> bool:
    q = q.local_value()
    return not q.full()


def rpc_queue_put(q: trpc.RRef, data: Any) -> None:
    q = q.local_value()
    q.put(data)


class Pipe:
    _queues: Dict[str, Queue] = {}
    _lock = Lock()

    def __init__(self, name: str, src: str, dest: str, max_size: int = 0) -> None:
        self.rpc_info = trpc.get_worker_info()
        self.name = name
        self.src = src
        self.dest = dest
        self.remote_queue: trpc.RRef = None
        self.local_queue: Queue[Any] = None
        logger.debug('Pipe*'*30)
        with use_lock(self._lock):
            logger.debug(f'1_self._lock:{self.rpc_info.name}')
            if src == self.rpc_info.name:
                assert name not in self._queues, f'pipe {name} already exists on {self.rpc_info.name}'
                logger.debug(f'2_name not in self._queues:{name not in self._queues}')
                self.remote_queue = self.get_remote_queue(max_size)
                logger.debug(f'3_self.remote_queue:{self.remote_queue}')
                self._queues[name] = self.remote_queue
                logger.debug(f'4_self._queues[name]:{self._queues}')
                time.sleep(3)

    @classmethod
    def rpc_create_local_queue(cls, name: str, max_size: int) -> Queue:
        with use_lock(cls._lock):
            logger.debug('rpc_create_local_queue')
            assert name not in cls._queues, f'pipe {name} already exists'
            cls._queues[name] = Queue(max_size)
            logger.debug(f'cls._queues[name]:{cls._queues[name]}')
            return cls._queues[name]

    def get_remote_queue(self, max_size: int) -> trpc.RRef:
        # logger.debug(f'get_remote_queue,{self.dest,self.name,max_size}')
        return trpc.remote(self.dest, self.rpc_create_local_queue, args=(self.name, max_size))

    def prepare_local_queue(self) -> None:
        logger.debug('prepare_local_queue')
        logger.debug(f'self.local_queue_1:{self.local_queue}')
        logger.debug(f'self.local_queue is None: {self.local_queue is None}')
        if self.local_queue is None:
            with use_lock(self._lock):
                if self.name in self._queues:
                    logger.debug(f'self.name{self.name}')
                    self.local_queue = self._queues[self.name]

    def recv(self) -> Any:
        assert self.dest == self.rpc_info.name
        while True:
            self.prepare_local_queue()
            if self.local_queue is not None:
                return self.local_queue.get()
            time.sleep(0.01)

    def recv_nowait(self) -> Any:
        assert self.dest == self.rpc_info.name
        logger.debug(f'assert self.dest == self.rpc_info.name:{self.dest == self.rpc_info.name}')
        self.prepare_local_queue()
        if self.local_queue is not None:
            try:
                logger.debug(f'self.local_queue_2:{self.local_queue}')
                return self.local_queue.get_nowait()
            except Empty:
                logger.debug('pipe is empty')
                raise RuntimeError('pipe is empty')
        logger.debug('local queue is not created')
        raise RuntimeError('local queue is not created')

    def send(self, data: Any) -> None:
        assert self.src == self.rpc_info.name
        while not trpc.rpc_sync(self.dest, rpc_queue_can_put, args=(self.remote_queue, )):
            time.sleep(0.1)
        trpc.rpc_sync(self.dest, rpc_queue_put, args=(self.remote_queue, data))
