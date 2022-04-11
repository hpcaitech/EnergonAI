from energon.logging import get_dist_logger


def partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    return parts
