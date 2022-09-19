"""
------------------------------------------
Class Batch Manager and the function for generating cached cost.
Part of the algorithm comes from Turbo Transformer.
------------------------------------------
"""
import time
from collections import deque
import numpy as np
import scipy.stats as stats
import redis
import math
import os
from tqdm import trange
import threading
from readerwriterlock import rwlock
import logging
from concurrent.futures import ThreadPoolExecutor
from energonai.context import MEATCONFIG

class gamma_dist:
    """
    Data structure for recording the distribution of the requests
    """
    def __init__(self, alpha_, loc_, beta_):
        self.alpha = alpha_
        self.loc = loc_
        self.beta = beta_
        self.max_list_len = 5 * MEATCONFIG['max_batch_size']
        self.max_seq_len = MEATCONFIG['max_sequence_length']

    def complete_req_list(self, req_list):
        """
        Use fake requests to fill the req list to a certain number so that the scheduler
        can perform better. The length of the fake requests is generated with the current
        recorded distribution.
        """
        new_size = self.max_list_len - len(req_list)
        if new_size <= 0:
            req_list.sort(key=lambda x: x.seq_len)
            return req_list
        res = stats.gamma.rvs(self.alpha, loc=self.loc, scale=self.beta, size=new_size)
        res = [math.floor(i) + 1 for i in res]
        res = [i if i < self.max_seq_len else self.max_seq_len for i in res]
        new_req = [single_request(input_=None, time_stamp=None, seq_len=i, input_str=None) for i in res]
        new_req.extend(req_list)
        new_req.sort(key=lambda x: x.seq_len)
        return new_req


class single_request:

    def __init__(self, input_, time_stamp, input_str, seq_len=None):
        """
        class to store related information for a single request.
        :param input_: The output of GPT2Tokenizer.tokenizer, a dict including input_ids and attention_mask
        :param time_stamp: The time stamp when we receive the request. We use the time stamp as a index to
                            identify the request.
        """
        self.input_ = input_
        self.time_ = time_stamp
        if seq_len is None:
            if MEATCONFIG['model_type'] in ['gpt', 'bert']:
                self.seq_len = input_['input_ids'].shape[1]
            elif MEATCONFIG['model_type'] == 'vit':
                self.seq_len = input_.shape[-1]
        else:
            self.seq_len = seq_len
        self.text = input_str


class Manager:
    """
    Base class of batch manager.
    """

    def __init__(self):
        pass

    def insert_req(self, time_stamp: float, input_ids, input_str):
        pass


class Dynamic_Batch_Manager(Manager):
    """
    This batch manager is mainly used for maintaining a queue of request to be processed. The requests in the
    queue is wrapped into batches and then sent into the inference engine.
    """

    def __init__(self,
                 forward_func,
                 result_process,
                 load_history=False,
                 his_len: int = 300, **kwargs):
        super().__init__()
        print(MEATCONFIG.config)
        self.max_batch_size = MEATCONFIG['max_batch_size']
        self.max_sequence_length = MEATCONFIG['max_sequence_length']
        self.forward_func = forward_func
        self.publisher = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
        self.result_process = result_process
        if load_history:
            self.load_history(his_len)
        else:
            self.req_history = deque(maxlen=his_len)
        self.req_list = []
        self.req_list_lock = rwlock.RWLockFair()
        self.write_lock = self.req_list_lock.gen_wlock()
        self.max_his_length = his_len
        self.gamma_dist_ = self.init_gamma_dist(self.max_sequence_length)
        self.cached_cost = self.generate_cached_cost()
        self.running_flag = True
        self.max_workers = MEATCONFIG['pp_init_size'] + 2
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.working_workers = 0
        self.main_thread = threading.Thread(target=self.processing_batch)
        self.main_thread.start()

    def init_gamma_dist(self, max_seq_len):
        if len(self.req_history) == 0:
            return gamma_dist(alpha_=0.022, loc_=11, beta_=3.34,)
        else:
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(self.req_history)
            return gamma_dist(alpha_=fit_alpha, loc_=fit_loc, beta_=fit_beta)

    def generate_cached_cost(self):
        """
        Test the running time for different sequence length and batch size on the current machine.
        """
        logging.log(0, "fetching cached cost")
        cached_name = "cached_cost_{}_pp{}_tp{}_{}_{}_{}_{}.npy"\
            .format(MEATCONFIG['model_class'].__name__, MEATCONFIG['pp_init_size'], MEATCONFIG['tp_init_size'],
                    MEATCONFIG['max_sequence_length'], MEATCONFIG['max_batch_size'],
                    MEATCONFIG['step'], MEATCONFIG['repeat_round'])
        if os.path.exists(cached_name):
            logging.log(0, "loading cached cost from file")
            cached_cost = np.load(cached_name).tolist()
        else:
            logging.log(0, "generating new cached cost")
            cached_cost = [[0 for i in range(MEATCONFIG['max_batch_size'] + 1)] for j in range(MEATCONFIG['max_sequence_length'] + 1)]
            for tt in range(5):
                output_ = self.forward_func(seq_len=MEATCONFIG['max_sequence_length'] - 1, batch_size=MEATCONFIG['max_batch_size'] - 1)
                pred = output_.to_here()
                for bs in range(MEATCONFIG['max_batch_size'] - 1):
                    res = self.result_process(int(pred[bs]))
            input_text = ""
            for tmp_len in trange(1, MEATCONFIG['max_sequence_length'] + 1, MEATCONFIG['step']):
                input_text += "test "
                for tmp_batch in range(1, MEATCONFIG['max_batch_size'] + 1):
                    self.forward_func(seq_len=tmp_len, batch_size=tmp_batch)
                    start_time = time.time()
                    for k in range(MEATCONFIG['repeat_round']):
                        output_ = self.forward_func(seq_len=tmp_len, batch_size=tmp_batch)
                        pred = output_.to_here()
                        for bs in range(tmp_batch):
                            res = self.result_process(int(pred[bs]))
                    time_cost = (time.time() - start_time) / MEATCONFIG['repeat_round']
                    cached_cost[tmp_len][tmp_batch] = time_cost
                    for k in range(1, MEATCONFIG['step']):
                        cached_cost[tmp_len + k][tmp_batch] = time_cost
            np.save(cached_name, np.array(cached_cost))
        logging.log(0, "cached cost loaded")
        return cached_cost

    def load_history(self, his_len):
        """
        load the distribution history
        """
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

    def insert_req(self, time_stamp: float, input_ids, input_str: str):
        """
        Build a single_request class with the input string and then insert it into the queue.
        """
        tmp_req = single_request(input_ids, time_stamp, input_str)
        self.write_lock.acquire()
        self.req_list.append(tmp_req)
        self.req_history.append(tmp_req.seq_len)
        self.write_lock.release()

    def wrap_batch(self):
        """
        Given a sorted sequence list, calculate the best way to wrap the batch with DP according to the
        cached cost.
        Part of this function comes from the paper of Turbo Transformer.
        """
        self.write_lock.acquire()
        new_req_list = self.gamma_dist_.complete_req_list(self.req_list)
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
            i = start_idx
        temp_batch = new_req_list[res_start:res_end]
        del new_req_list[res_start:res_end]

        result_batch = []
        for req in temp_batch:
            if req.input_:
                result_batch.append(req)
        self.write_lock.acquire()
        for req_ in new_req_list:
            if req_.input_:
                self.req_list.append(req_)
        self.write_lock.release()
        return result_batch

    def cal_priority(self, batch, cur_time):
        completeness = np.sum([1 if i.input_ else 0 for i in batch]) / len(batch)
        earliest_time_stamp = min([j.time_ if j.time_ else cur_time for j in batch])
        if cur_time - earliest_time_stamp > MEATCONFIG['max_wait_time']:
            completeness = 1.1
        return completeness

    def update_distribution(self):
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(self.req_history)
        self.gamma_dist_.alpha = fit_alpha
        self.gamma_dist_.loc = fit_loc
        self.gamma_dist_.beta = fit_beta
        f = open("req_history.txt", 'w')
        for i in self.req_history:
            f.write(i)
        return

    def processing_batch(self):
        """
        The background process that continuously calls wrap_batch, puts the batch into the inference engine,
        and starts new processes that wait for and publish the inference result.
        """
        round_cnt = 0
        while self.running_flag:
            if (self.working_workers < self.max_workers) and (len(self.req_list) > 0):
                round_cnt += 1
                target_batch = self.wrap_batch()
                pad_len = target_batch[-1].seq_len
                logging.info("A batch with {} requests and length of {} packed, in-batch length: {}".format(
                    len(target_batch), pad_len, [p.seq_len for p in target_batch]))
                input_text = [i.text for i in target_batch]
                self.working_workers = self.working_workers + 1
                output_ = self.forward_func(input_list=input_text)
                self.pool.submit(self.publish_result, output_, target_batch)
                if round_cnt == 10 and len(self.req_history) >= self.max_his_length - 1:
                    round_cnt = 0
                    self.update_distribution()
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
