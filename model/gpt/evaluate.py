import torch
import argparse

from model.pipeline_gpt1d import GPT2_small_pipeline_1D, GPT2_exlarge_pipeline_1D, GPT3_pipeline_1D
from energon.engine import InferenceEngine
from energon.logging import get_dist_logger
from energon.core import global_context as gpc
from energon.context import ParallelMode
from energon.utils import get_timers

MODEL_CLASSES = {
    "gpt2_small": GPT2_small_pipeline_1D,
    "gpt2_exlarge": GPT2_exlarge_pipeline_1D,
    "gpt3": GPT3_pipeline_1D,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_para_size", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument("--pipe_para_size", type=int, default=1, help="Pipeline Parallel Size")
    parser.add_argument("--iteration", type=int, default=10, help="Pipeline Parallel Size")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit precision instead of 32-bit")
    parser.add_argument("--model_name", default=None, type=str, required=True, help="Shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    args = parser.parse_args()
    
    dtype=torch.float
    if args.fp16:
        # print("FP16")
        dtype=torch.half

    config = {'num_chunks':1, 'checkpoint':False, 'dtype':dtype, 'embed_split_hidden':False}


    input_ids = torch.randint(1, 10, (1, 512), dtype=torch.int64)
    attention_mask = torch.randint(0, 1, (1, 1, 512), dtype=torch.int64)
    hidden_states = None
    sample = dict(hidden_states=hidden_states, input_ids=input_ids, attention_mask=attention_mask)

    # print(MODEL_CLASSES[args.model_name])

    engine = InferenceEngine(MODEL_CLASSES[args.model_name], config, sample, pp_init_size = args.pipe_para_size, tp_init_size = args.tensor_para_size, dtype = torch.half)



    # prof = torch.profiler.profile(
    #             schedule=torch.profiler.schedule(wait=1,
    #                                              warmup=1,
    #                                              active=2,
    #                                              repeat=1),
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gpt3_pp{}tp{}'.format(pp, tp)),
    #             profile_memory=True,
    #             record_shapes=True,
    #             with_stack=True)

    # prof.start()

    output = engine.run()

    timer = get_timers()

    torch.distributed.barrier()
    timer('evaluate-time').start()

    for i in range(args.iteration):

        # torch.distributed.barrier()
        timer('latency-time').start()
        output = engine.run()
        # torch.distributed.barrier()
        timer('latency-time').stop()
        
        # prof.step()

    # prof.stop()

    torch.distributed.barrier()
    timer('evaluate-time').stop()


    logger = get_dist_logger()
    evaluate_elapsed = timer('evaluate-time').elapsed()
    latency_elapsed = timer('latency-time').elapsed()

    logger.info(f'Throughput, '
                f'Pipeline Rank/ Tensor Rank: {args.pipe_para_size}/{gpc.get_world_size(ParallelMode.PARALLEL_1D)},'
                f'Time: {args.iteration/evaluate_elapsed}')
    logger.info(f'Latency, '
                f'Pipeline Rank/ Tensor Rank: {args.pipe_para_size}/{gpc.get_world_size(ParallelMode.PARALLEL_1D)},'
                f'Time: {latency_elapsed/args.iteration}')

    logger.info(f'max memory allocated, '
                f'Pipeline Rank/ Tensor Rank: {args.pipe_para_size}/{gpc.get_world_size(ParallelMode.PARALLEL_1D)},'
                f'memory: {torch.cuda.max_memory_allocated()/1e9} GB')




    
# if output is not None:
# print(output.shape)

# print(engine._model.model)
# engine.switch(2,2)

# for i in range(10):
#     output = engine.run()
#     if output is not None:
#         print(output.shape)

if __name__ == "__main__":
    main()