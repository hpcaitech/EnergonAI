import functools
from torch.fx.node import Node
from energonai.context import MEATCONFIG


partition_counter_0 = 0

# partition_nums: nums of each submodule
def _naive_equal_partition(node: Node, partition_nums):
    global partition_counter_0
    partition = partition_counter_0 // partition_nums
    partition_counter_0 = partition_counter_0 + 1
    return partition

def naive_equal_partition(partition_nums):
    mod_partition = functools.partial(_naive_equal_partition, partition_nums = partition_nums)
    return mod_partition

partition_counter_1 = 0

# partition_nums: nums of each submodule
def _module_equal_partition(node: Node, partition_nums):
    global partition_counter_1
    partition = partition_counter_1 // partition_nums
    if node.op == 'call_module':
        partition_counter_1 = partition_counter_1 + 1
    return partition

def module_equal_partition(partition_nums):
    mod_partition = functools.partial(_module_equal_partition, partition_nums = partition_nums)
    return mod_partition



from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
partition_counter_2 = -1 # for embedding layer
# partition_nums: nums of each submodule
def _transformer_partition(node: Node, depth):
    global partition_counter_2
    assert gpc.is_initialized(ParallelMode.PIPELINE), "Pipeline communication group should be initialized!"
    
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    partition_nums = depth // pipeline_size    
    partition = abs(partition_counter_2) // partition_nums
    if node.op == 'call_module':
        partition_counter_2 = partition_counter_2 + 1
    return partition

def transformer_partition(depth):
    mod_partition = functools.partial(_transformer_partition, depth = depth)
    return mod_partition