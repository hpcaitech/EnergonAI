from torch.nn import Module

from colossalai.logging import get_dist_logger

# pytorch rpc
import torch.distributed.rpc as rpc

from .rpc_utils import remote_cls_method
from .rpc_worker import RPCWorker
from .pipeline_msg_dict import CircleInt

from energonai.initialize import launch_from_multiprocess



logger = get_dist_logger('energonai')


class InferenceEngine(Module):

    def __init__(self,
                 model_class,
                 model_config,
                 model_type,
                 max_batch_size: int = 1,
                 tp_init_size: int = -1,
                 pp_init_size: int = -1,
                 auto_pp: bool = False,
                 host: str = 'localhost',
                 port: int = 29500,
                 dtype=None,
                 ):
        """
        Args:
            model: torch.nn.Module
            dtype: data-type by which inference is executed
        """
        super().__init__()

        self.model_class = model_class
        self.model_config = model_config
        self.model_type = model_type
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        # for gpc
        self.rank = 0
        self.global_world_size = pp_init_size * tp_init_size
        self.host = host
        self.port = port
        self.tp_size = tp_init_size
        self.pp_size = pp_init_size

        # for TP, PP
        self.rrefs = []
        self.auto_pp = auto_pp
        
        # for rpc
        self.WORKER_NAME = "wok{}"
        self._init_dist_rpc()
        self._init_model()
        self.key = CircleInt()

    def _init_dist_rpc(self):
        r"""
        Based on global_context, init the rpc connection.
        """
        launch_from_multiprocess(tp_size=self.tp_size,
                                 pp_size=self.pp_size,
                                 rank=self.rank,
                                 local_rank=self.rank,
                                 world_size=self.global_world_size,
                                 host=self.host,
                                 port=self.port)
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16,
        # _transports=["uv"] TODO: potentially a bug
                                                             )
        rpc.init_rpc(self.WORKER_NAME.format(0),
                     rank=0,
                     world_size=self.global_world_size,
                     rpc_backend_options=rpc_backend_options)
        logger.info(f'RPC STATUS: RPC of Rank: 0 is initialized.')

    def _init_model(self):
        for i in range(self.global_world_size):
            logger.info(f'RPC STATUS: rank: {self.rank} calls rank: {i} to init model.')
            ob_info = rpc.get_worker_info(self.WORKER_NAME.format(i))
            self.rrefs.append(
                rpc.remote(ob_info,
                           RPCWorker,
                           args=(self.model_class, self.model_config, self.model_type, self.dtype,
                                 self.max_batch_size, self.auto_pp)))

    def run(self, inputs):
        output = None

        # self.prioirty = time.time()
        for rref in self.rrefs:
            output = remote_cls_method(RPCWorker.run, rref, self.key.val, inputs)

        self.key.addOne()
        return output

    def clear(self):
        rpc.shutdown()
