from energonai.context import MEATCONFIG
import time
import queue
import redis
import threading
from concurrent.futures import ThreadPoolExecutor

class single_request:

    def __init__(self, input_token, time_stamp):
        """
        class to store related information for a single request.
        """
        self.input_token = input_token
        self.time_stamp = time_stamp

class QueueManagerGen:
    """
    This batch manager is used for maintaining a queue of request and control the number of requests to be processed.
    """
    def __init__(self,
                 engine,
                 tokenizer,
                 max_batch_size = 1,
                 max_concurrent_user = 2,
                 **kwargs):
        super().__init__()

        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_concurrent_user = max_concurrent_user
        self.working_worker = 0
        self.req_queue = queue.Queue(maxsize=self.max_concurrent_user)
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        # self.pool = ThreadPoolExecutor(max_workers=self.max_concurrent_user)
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.running_flag = True
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.main_thread.start()
    
    def length(self):
        # print(len(self.req_list))
        return len(self.req_list)

    def insert_req(self, time_stamp, req):
        """
        Build a single_request class with the input string and then insert it into the queue.
        """
        if(self.req_queue.qsize() < self.max_concurrent_user):
            input_token =self.tokenizer(req.prompt, return_tensors="pt")
            input_token['tgt_len'] = req.max_tokens    
            input_token['top_k'] = req.top_k
            input_token['top_p'] = req.top_p
            input_token['temperature'] = req.temperature
            tmp_req = single_request(input_token, time_stamp)
            self.req_queue.put(tmp_req)
            return True
        else:
            return False
    
    def subscribe_result(self, time_stamp):
        """
        waiting for the result and send back.
        """
        sub = self.publisher.pubsub()
        # print(f'Subscribe {time_stamp}')
        sub.subscribe(time_stamp)
        predictions = ''
        for message in sub.listen():
            if message is not None and isinstance(message, dict):
                predictions = message.get('data')
                if not isinstance(predictions, int):
                    break
        sub.unsubscribe()
        return predictions

    def processing_batch(self):
        """
        The main task that continuously wrap batch, puts the batch into the inference engine,
        and starts new processes that wait for and publish the inference result.
        """
        while self.running_flag:
            if not self.req_queue.empty():
                req = self.req_queue.get()
                output = self.engine.run(req.input_token).to_here()
                output = output[0, :].tolist()
                try:
                    output = self.tokenizer.decode(output)
                except:
                    self.publisher.publish(req.time_stamp, output)
                else:
                    self.publisher.publish(req.time_stamp, "Decoder Failed")
            time.sleep(0.005)