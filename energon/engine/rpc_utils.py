import random
import string
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote

# rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=- 1.0)
# rpc.rpc_async(to, func, args=None, kwargs=None, timeout=- 1.0)
# rpc.remote(to, func, args=None, kwargs=None, timeout=- 1.0)

def call_method(method, rref, *args, **kwargs):
        return method(rref.local_value(), *args, **kwargs)

def remote_cls_method(method, rref, *args, **kwargs):
        args = [method, rref] + list(args)
        return rpc.remote(rref.owner(), call_method, args=args, kwargs=kwargs)

def sync_cls_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

def async_cls_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), call_method, args=args, kwargs=kwargs)

def get_random_string(length=16):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
#     print("Random string of length", length, "is:", result_str)