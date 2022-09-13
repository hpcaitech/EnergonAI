"""
------------------------------------------
Class Batch Manager.
a naive version that is used for cases in which padding is not needed.
------------------------------------------
"""
import time
import redis
from energonai.context import MEATCONFIG
import threading
from readerwriterlock import rwlock
import logging
from concurrent.futures import ThreadPoolExecutor


class single_request:

    def __init__(self, input_, time_stamp: float, input_str: str):
        """
        class to store related information for a single request.
        :param input_: The output of GPT2Tokenizer.tokenizer, a dict including input_ids and attention_mask
        :param time_stamp: The time stamp when we receive the request. We use the time stamp as a index to
                            identify the request.
        :param input_str: The input string of the request.
        """
        self.input_ = input_
        self.text = input_str
        self.time_ = time_stamp
        self.seq_len = input_['input_ids'].shape[1]


class Manager:
    """
    Base class of batch manager.
    """

    def __init__(self):
        pass

    def insert_req(self, time_stamp: float, input_ids, input_str: str):
        pass


class Naive_Batch_Manager(Manager):
    """
    This batch manager is mainly used for maintaining a queue of request to be processed. The requests in the
    queue is wrapped into batches and then sent into the inference engine.
    """

    def __init__(self, forward_func,
                 result_process):
        """
        :param forward_func a function of calling a forward propagation, returning a RPC ref.
        :param result_process a function to process the output of the model before returning the result.
        """
        super().__init__()
        self.req_list = []
        self.max_batch_size = MEATCONFIG['max_batch_size']
        self.max_sequence_length = MEATCONFIG['max_sequence_length']
        self.req_list_lock = rwlock.RWLockFair()
        self.write_lock = self.req_list_lock.gen_wlock()
        self.running_flag = True
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        self.max_workers = MEATCONFIG['pp_init_size'] + 2
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.working_workers = 0
        self.forward_func = forward_func
        self.result_process = result_process
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.main_thread.start()

    def insert_req(self, time_stamp: float, input_ids, input_str: str):
        """
        Build a single_request class with the input string and then insert it into the queue.
        """
        tmp_req = single_request(input_ids, time_stamp, input_str)
        self.write_lock.acquire()
        self.req_list.append(tmp_req)
        self.write_lock.release()

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

    def wrap_batch(self):
        """
        Simply wrap batches by the order of insertion.
        """
        self.write_lock.acquire()
        result_batch = self.req_list[0:min(self.max_batch_size, len(self.req_list))]
        del self.req_list[0:min(self.max_batch_size, len(self.req_list))]
        self.write_lock.release()
        return result_batch

    def processing_batch(self):
        """
        The background process that continuously calls wrap_batch, puts the batch into the inference engine,
        and starts new processes that wait for and publish the inference result.
        """
        while self.running_flag:
            if (self.working_workers < self.max_workers) and (len(self.req_list) > 0):
                target_batch = self.wrap_batch()
                pad_len = max([p.seq_len for p in target_batch])
                logging.info("A batch with {} requests and length of {} packed, in-batch length: {}".format(
                    len(target_batch), pad_len, [p.seq_len for p in target_batch]))
                input_text = [i.text for i in target_batch]
                self.working_workers = self.working_workers + 1
                output_ = self.forward_func(input_list=input_text)
                self.pool.submit(self.publish_result, output_, target_batch)
            time.sleep(0.001)

    def publish_result(self, output, target_batch):
        """
        Background process that waits for the inference result and uses the publisher of Redis to publish it to
        the waiting requests.
        :param output: the rpc reference of the inference result.
        :param target_batch: the input batch
        """
        predictions = output.to_here()
        for i in range(len(target_batch)):
            temp_st = target_batch[i].time_
            chosen_pred = predictions[i]
            result = self.result_process(chosen_pred)
            self.publisher.publish(str(temp_st), result)

        self.working_workers = self.working_workers - 1

