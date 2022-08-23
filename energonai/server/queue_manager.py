from energonai.context import MEATCONFIG
import time
import queue
import redis
import threading
from concurrent.futures import ThreadPoolExecutor

class single_request:

    def __init__(self, prompt, input, time_stamp, cur_len, tgt_len):
        """
        class to store related information for a single request.
        """
        self.prompt = prompt
        self.input = input
        self.time_stamp = time_stamp
        self.cur_len = cur_len
        self.tgt_len = tgt_len

class QueueManager:
    """
    This batch manager is used for maintaining a queue of request and control the number of requests to be processed.
    """
    def __init__(self,
                 engine,
                 tokenizer,
                 max_batch_size = 4,
                 max_concurrent_user = 8,
                 **kwargs):
        super().__init__()

        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_concurrent_user = max_concurrent_user
        self.working_worker = 0
        self.req_queue = queue.Queue(maxsize=self.max_concurrent_user)
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        self.pool = ThreadPoolExecutor(max_workers=self.max_concurrent_user)
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.running_flag = True
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.main_thread.start()
    
    def length(self):
        # print(len(self.req_list))
        return len(self.req_list)

    def insert_req(self, time_stamp, prompt, tgt_len):
        """
        Build a single_request class with the input string and then insert it into the queue.
        """
        if(self.req_queue.qsize() < self.max_concurrent_user):
            input = self.tokenizer(prompt, return_tensors="pt")
            cur_len = input['input_ids'].shape[1]
            tmp_req = single_request(prompt, input, time_stamp, cur_len, tgt_len)
            self.req_queue.put(tmp_req)
            return True
        else:
            return False
    
    def subscribe_result(self, time_stamp):
        """
        waiting for the result and send back.
        """
        sub = self.publisher.pubsub()
        sub.subscribe(str(time_stamp))
        predictions = ''
        for message in sub.listen():
            if message is not None and isinstance(message, dict):
                predictions = message.get('data')
                if not isinstance(predictions, int):
                    break
        return predictions

    def processing_batch(self):
        """
        The main task that continuously wrap batch, puts the batch into the inference engine,
        and starts new processes that wait for and publish the inference result.
        """
        while self.running_flag:
            if(self.working_worker < self.max_concurrent_user) and (not self.req_queue.empty()):
                self.working_worker = self.working_worker + 1
                req = self.req_queue.get()
                self.pool.submit(self.generation_task, req)
            time.sleep(0.1)
    
    def generation_task(self, req: single_request):
        while(req.cur_len < req.tgt_len):
            output = self.engine.run(req.input)
            predictions = output.to_here()
            req.prompt += self.tokenizer.decode(predictions)
            # if '<|endoftext|>' in req.prompt:
            #     break
            req.input = self.tokenizer(req.prompt, return_tensors="pt")
            req.cur_len = req.cur_len + 1

        self.publisher.publish(req.time_stamp, req.prompt)
        self.working_worker = self.working_worker - 1