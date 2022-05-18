import random
from energon.core import global_context as gpc
from energon.context.parallel_mode import ParallelMode
import socket


def free_port():
    while True:
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = random.randint(20000, 65000)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except Exception:
            continue


def is_using_pp():
    return gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1
