"""
------------------------------------------
Class Batch Manager and the function for generating cached cost.
This code modifies the batch wrapping algorithm of Turbo Transformer.
------------------------------------------
"""
import time

import torch.cuda
from scipy import stats
import numpy as np
from energon.engine import InferenceEngine
import random
import redis
import os
from tqdm import trange
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
    queue is wrapped into batches according to the sequence length and the priority calculated with the equation
    in function cal_priority and then sent into the inference engine.
    """

    def __init__(self, engine: InferenceEngine, max_batch_size: int = 32, tokenizer=None, pad_token=None):
        """
        :param engine: The InferenceEngine from energon.engine
        :param max_batch_size: the max number of requests that can be wrapped into one batch.
        """
        super().__init__()
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.req_list = []
        self.req_list_lock = rwlock.RWLockFair()
        self.write_lock = self.req_list_lock.gen_wlock()
        self.tokenizer = tokenizer
        if self.tokenizer and pad_token:
            self.tokenizer.pad_token = pad_token    # GPT2Tokenizer.eos_token
        self.running_flag = True
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        self.pool = ThreadPoolExecutor(max_workers=16)
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
        # red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        # sub = red.pubsub()
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
        Given a sorted sequence list, calculate the best way to wrap the batch with DP according to the
        cached cost.
        The algorithm in this function comes from the paper of Turbo Transformer.
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

            if len(self.req_list) > 0:
                st_ = time.time()
                target_batch = self.wrap_batch()
                pad_len = max([p.seq_len for p in target_batch])
                logging.info("A batch with {} requests and length of {} packed, in-batch length: {}".format(
                    len(target_batch), pad_len, [p.seq_len for p in target_batch]))
                input_text = [i.text for i in target_batch]
                if self.tokenizer:
                    input_ids = self.tokenizer(input_text, padding="longest", return_tensors="pt")
                else:
                    input_ids = input_text
                print(time.time() - st_)
                output = self.engine.run(input_ids)
                self.pool.submit(self.publish_result, output, target_batch)
                # self.publish_result(output, target_batch, start_time)
                # pub_thread = threading.Thread(target=self.publish_result, args=(output, target_batch, start_time))
                # pub_thread.start()
            time.sleep(0.08)

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
            if self.tokenizer:
                text_ = self.tokenizer.decode(int(chosen_pred))
            else:
                text_ = chosen_pred
            self.publisher.publish(str(temp_st), text_)
