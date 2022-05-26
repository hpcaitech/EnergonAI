"""
------------------------------------------
Class Batch Manager and the function for generating cached cost.
This code modifies the batch wrapping algorithm of Turbo Transformer.
------------------------------------------
"""
import time
from collections import deque
import torch.cuda
import numpy as np
import scipy.stats as stats
from energonai.engine import InferenceEngine
import random
import redis
import math
import os
from tqdm import trange
import threading
from readerwriterlock import rwlock
import logging
from concurrent.futures import ThreadPoolExecutor

class gamma_dist:
    def __init__(self, alpha_, loc_, beta_, max_list_len, max_seq_len):
        self.alpha = alpha_
        self.loc = loc_
        self.beta = beta_
        self.max_list_len = max_list_len
        self.max_seq_len = max_seq_len

    def complete_req_list(self, req_list):
        new_size = self.max_list_len - len(req_list)
        res = stats.gamma.rvs(self.alpha, loc=self.loc, scale=self.beta, size=new_size)
        res = [math.floor(i) + 1 for i in res]
        res = [i if i < self.max_seq_len else self.max_seq_len for i in res]
        new_req = [single_request(None, None, None, i) for i in res]
        new_req.extend(req_list)
        new_req.sort(key=lambda x: x.seq_len)
        return new_req


class single_request:

    def __init__(self, input_, time_stamp, input_str, seq_len : int=0):
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
        if input_:
            self.seq_len = input_['input_ids'].shape[1]
        else:
            self.seq_len = seq_len


class Manager:
    """
    Base class of batch manager.
    """

    def __init__(self):
        pass

    def insert_req(self, time_stamp: float, input_ids, input_str: str):
        pass


class new_Batch_Manager(Manager):
    """
    This batch manager is mainly used for maintaining a queue of request to be processed. The requests in the
    queue is wrapped into batches according to the sequence length and the priority calculated with the equation
    in function cal_priority and then sent into the inference engine.
    """

    def __init__(self,
                 engine: InferenceEngine,
                 model_name: str,
                 pp: int,
                 tp: int,
                 max_sequence_length: int,
                 max_batch_size: int = 32,
                 tokenizer=None,
                 pad_token=None,
                 rm_padding=False,
                 load_history=False,
                 step: int = 1,
                 repeat_round: int = 3,
                 max_wait_time=1,
                 his_len: int = 300):
        """
        :param engine: The InferenceEngine from energon.engine
        :param cached_cost: The output of function generate_cached_cost
        :param init_mu: initial mean value we suppose for incoming sequence length.
        :param init_theta: initial variance value we suppose for incoming sequence length.
        :param max_batch_size: the max number of requests that can be wrapped into one batch.
        :param lr: the learning rate we use to update the mean and variance that we suppose for the normal
                    distribution of sequence length.
        """
        super().__init__()
        self.engine = engine
        self.max_batch_size = max_batch_size
        if load_history:
            self.req_history = self.load_history(his_len)
        else:
            self.req_history = deque(maxlen=his_len)
        self.max_wait_time = max_wait_time
        self.req_list = []
        self.req_list_lock = rwlock.RWLockFair()
        self.max_his_length = his_len
        self.gamma_dist_ = self.init_gamma_dist(max_sequence_length)
        self.write_lock = self.req_list_lock.gen_wlock()
        self.cached_cost = self.generate_cached_cost(engine,
                                                     model_name,
                                                     pp=pp,
                                                     tp=tp,
                                                     max_seq_len=max_sequence_length,
                                                     max_batch_size=max_batch_size,
                                                     step=step,
                                                     repeat_round=repeat_round,
                                                     tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.rm_padding = rm_padding
        if self.tokenizer and pad_token:
            self.tokenizer.pad_token = pad_token  # GPT2Tokenizer.eos_token
        self.running_flag = True
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        self.pool = ThreadPoolExecutor(max_workers=pp + 1)
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.main_thread.start()

    def init_gamma_dist(self, max_seq_len):
        if len(self.req_history) == 0:
            return gamma_dist(alpha_=0.022, loc_=11, beta_=3.34,
                              max_list_len=5 * self.max_batch_size,
                              max_seq_len=max_seq_len)
        else:
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(self.req_history)
            return gamma_dist(alpha_=fit_alpha, loc_=fit_loc, beta_=fit_beta,
                              max_list_len=5 * self.max_batch_size,
                              max_seq_len=max_seq_len)

    def generate_cached_cost(self,
                             engine,
                             model_name: str,
                             pp: int,
                             tp: int,
                             max_seq_len: int = 1024,
                             max_batch_size: int = 16,
                             step: int = 1,
                             repeat_round: int = 3,
                             tokenizer=None):
        """
        Test the running time for different sequence length and batch size on the current machine.
        :param engine: InferenceEngine from energon.engine
        :type engine: InferenceEngine
        :param max_seq_len: The max sequence length that is measured.
        :param max_batch_size: The max batch size that is measured.
        :param step: Run time is measured every other 'step' of sequence length
        :param repeat_round: We inference current batch 'repeat_round' times and take average.
        """
        logging.log(0, "fetching cached cost")
        cached_name = "cached_cost_{}_pp{}_tp{}_{}_{}_{}_{}.npy".format(model_name, pp, tp, max_seq_len, max_batch_size,
                                                                        step, repeat_round)
        if os.path.exists(cached_name):
            logging.log(0, "loading cached cost from file")
            cached_cost = np.load(cached_name).tolist()
        else:
            logging.log(0, "generating new cached cost")
            cached_cost = [[0 for i in range(max_batch_size + 1)] for j in range(max_seq_len + 1)]
            warm_up_str = "test test test"
            if tokenizer:
                warm_up_input = tokenizer(warm_up_str, return_tensors="pt")
            else:
                warm_up_input = warm_up_str
            for tt in range(5):
                output = engine.run(warm_up_input)
                predictions = output.to_here()
            input_text = ""
            for tmp_len in trange(1, max_seq_len + 1, step):
                input_text += "test "
                for tmp_batch in range(1, max_batch_size + 1):
                    batched_text = [input_text for _ in range(tmp_batch)]
                    if tokenizer:
                        input_token = tokenizer(batched_text, return_tensors="pt")
                    else:
                        input_token = batched_text
                    output = engine.run(input_token)
                    predictions = output.to_here()
                    if tokenizer:
                        tokenizer.decode(predictions)
                    start_time = time.time()
                    for k in range(repeat_round):
                        if tokenizer:
                            input_token = tokenizer(batched_text, return_tensors="pt")
                        else:
                            input_token = batched_text
                        output = engine.run(input_token)
                        predictions = output.to_here()
                        if tokenizer:
                            tokenizer.decode(predictions)
                    time_cost = (time.time() - start_time) / repeat_round
                    cached_cost[tmp_len][tmp_batch] = time_cost
                    for k in range(1, step):
                        cached_cost[tmp_len + k][tmp_batch] = time_cost
            np.save(cached_name, np.array(cached_cost))
        logging.log(0, "cached cost loaded")
        return cached_cost

    def load_history(self, his_len):
        try:
            f = open("req_history.txt", 'r')
        except Exception as e:
            print("history file does not exist", e)
            return
        his = f.readlines()
        his = [int(i.replace('\n', '')) for i in his]
        self.req_history = his[:his_len]
        return

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

    def insert_req(self, time_stamp: float, input_ids, input_str: str):
        """
        Build a single_request class with the input string and then insert it into the queue.
        """
        tmp_req = single_request(input_ids, time_stamp, input_str)
        self.write_lock.acquire()
        self.req_list.append(tmp_req)
        self.write_lock.release()

    def wrap_batch(self):
        """
        Given a sorted sequence list, calculate the best way to wrap the batch with DP according to the
        cached cost.
        The algorithm in this function comes from the paper of Turbo Transformer.
        """
        self.write_lock.acquire()
        new_req_list = self.gamma_dist_.complete_req_list(self.req_list)
        print("befor req num: {}".format(len(self.req_list)))
        self.req_list = []
        self.write_lock.release()
        states = [0]
        start_idx_list = [0]
        for i in range(1, len(new_req_list) + 1):
            j = i - 1
            start_idx = i - 1
            cur_length = new_req_list[i - 1].seq_len
            min_cost = self.cached_cost[cur_length][1] + states[j]
            while j > max(0, i - self.max_batch_size):
                tmp_cost = states[j - 1] + \
                           self.cached_cost[cur_length][i - j + 1]
                if tmp_cost < min_cost:
                    min_cost = tmp_cost
                    start_idx = j - 1
                j -= 1
            states.append(min_cost)
            start_idx_list.append(start_idx)
        i = len(new_req_list)
        res_start = -1
        res_end = -1
        max_priority = 0
        cur_timestamp = time.time()
        while i > 0:
            end_idx = i
            start_idx = start_idx_list[i]
            current_batch = new_req_list[start_idx:end_idx]
            current_priority = self.cal_priority(current_batch, cur_timestamp)
            if current_priority > max_priority:
                max_priority = current_priority
                res_start = start_idx
                res_end = end_idx
            i = start_idx - 1
        temp_batch = new_req_list[res_start:res_end]
        del new_req_list[res_start:res_end]

        result_batch = []
        for req in temp_batch:
            if req.input_:
                result_batch.append(req)
        print("a fake batch of {} is wrapped".format(len(temp_batch)))
        print("a batch of {} req is wrapped".format(len(result_batch)))
        self.write_lock.acquire()
        print("meanwhile, {} new req came".format(len(self.req_list)))
        for req_ in new_req_list:
            if req_.input_:
                self.req_list.append(req_)
        print("after, {} req is left".format(len(self.req_list)))
        self.write_lock.release()
        return result_batch

    def cal_priority(self, batch, cur_time):
        completeness = np.sum([1 if i.input_ else 0 for i in batch]) / len(batch)
        earliest_time_stamp = min([j.time_ if j.time_ else cur_time for j in batch])
        if cur_time - earliest_time_stamp > self.max_wait_time:
            completeness = 1.1
        return completeness

    def update_distribution(self):
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(self.req_history)
        self.gamma_dist_.alpha = fit_alpha
        self.gamma_dist_.loc = fit_loc
        self.gamma_dist_.beta = fit_beta
        return

    def processing_batch(self):
        """
        The background process that continuously calls wrap_batch, puts the batch into the inference engine,
        and starts new processes that wait for and publish the inference result.
        """
        round_cnt = 0
        while self.running_flag:
            if len(self.req_list) > 0:
                round_cnt += 1
                target_batch = self.wrap_batch()
                pad_len = target_batch[-1].seq_len
                logging.info("A batch with {} requests and length of {} packed, in-batch length: {}".format(
                    len(target_batch), pad_len, [p.seq_len for p in target_batch]))
                input_text = [i.text for i in target_batch]
                if self.tokenizer:
                    input_ids = self.tokenizer(input_text, padding="longest", return_tensors="pt")
                else:
                    input_ids = input_text
                if self.rm_padding:
                    input_ids['seq_lens'] = torch.Tensor([p.seq_len for p in target_batch])
                output = self.engine.run(input_ids)
                self.pool.submit(self.publish_result, output, target_batch)
                # self.update_norm(target_batch)
                # self.publish_result(output, target_batch, start_time)
                # pub_thread = threading.Thread(target=self.publish_result, args=(output, target_batch, start_time))
                # pub_thread.start()
                if round_cnt == 10 and len(self.req_history) >= self.max_his_length - 1:
                    round_cnt = 0
                    self.update_distribution()
            time.sleep(0.08)

    def publish_result(self, output, target_batch):
        """
        Background process that waits for the inference result and uses the publisher of Redis to publish it to
        the waiting requests.
        :param output: the rpc reference of the inference result.
        :param target_batch: the input batch
        """
        predictions = output.to_here()
        print("sending back the results of {} req".format(len(target_batch)))
        for i in range(len(target_batch)):
            temp_st = target_batch[i].time_
            chosen_pred = predictions[i]
            if self.tokenizer:
                text_ = self.tokenizer.decode(int(chosen_pred))
            else:
                text_ = chosen_pred
            self.publisher.publish(str(temp_st), text_)
