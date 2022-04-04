from .pipeline_meta import PipelineMeta
import queue

class PipelineMsg:
    def __init__(self, sample, pipe_meta):
        self.sample = sample
        self.pipe_meta = pipe_meta


# Here must use the Queue to ensure the order
class PipelineMsgQueue:
    def __init__(self):

        self.pipeline_msg_queue = queue.Queue()
        

    def enqueue(self, sample, pipe_meta):
        pipe_msg = PipelineMsg(sample, pipe_meta)
        self.pipeline_msg_queue.put(pipe_msg)

    def top(self):
        pipe_msg = self.pipeline_msg_queue.get()
        return pipe_msg.sample, pipe_msg.pipe_meta

'''
    dict(key : pipe_msg[sample, pipe_meta])
'''
class PipelineMsgDict:
    def __init__(self):

        self.pipeline_msg_dict = dict()
        

    def enqueue(self, key, sample, pipe_meta):
        pipe_msg = PipelineMsg(sample, pipe_meta)
        self.pipeline_msg_dict[key] = pipe_msg

    def deque(self, key):
        self.pipeline_msg_dict.pop(key)

    def get_sample(self, key):
        return self.pipeline_msg_dict[key].sample
    
    def get_meta(self, key):
        return self.pipeline_msg_dict[key].pipe_meta
        
