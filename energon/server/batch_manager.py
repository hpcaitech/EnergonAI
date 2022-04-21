import queue
from threading import Event
import threading
from engine_server import InferenceEngine


class Batch_Manager():
    def __init__(self, engine: InferenceEngine, batch_acc: int=20, max_wait_time: int=5):
        self.engine = engine
        self.batch_acc = batch_acc
        self.max_wait_time = max_wait_time
        self.req_queue = queue.Queue()
        self.res_dict = dict()

    def fetch_inference_res(self, time_stamp: int):
        if time_stamp not in self.res_dict.keys():
            return "Error: Inference may have failed"
        res = self.res_dict.pop(time_stamp)
        return res

    def append_req(self, time_stamp: int, input_str: str):
        self.req_queue.put({})