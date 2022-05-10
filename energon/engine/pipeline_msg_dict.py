from .pipeline_meta import PipelineMeta
import time


class CircleInt:
    def __init__(self, minValue=0, maxValue=10000):
        self.minValue = minValue
        self.maxValue = maxValue
        self.num = minValue

    def addOne(self):
        if (self.num < self.maxValue):
            self.num = self.num + 1
        else:
            self.num = self.minValue

    @property
    def val(self):
        return self.num


class PipelineMsg:
    def __init__(self, sample, pipe_meta):
        self.sample = sample
        self.pipe_meta = pipe_meta


class PipelineMsgDict:
    def __init__(self):
        self.pipeline_msg_dict = dict()

    def enqueue(self, key, sample, pipe_meta):
        pipe_msg = PipelineMsg(sample, pipe_meta)
        self.pipeline_msg_dict[key] = pipe_msg

    def top(self, key):
        while key not in self.pipeline_msg_dict:
            time.sleep(0.002)

        pipe_msg = self.pipeline_msg_dict.pop(key)

        return pipe_msg.sample, pipe_msg.pipe_meta


'''
    dict(key : pipe_msg[sample, pipe_meta])
'''
