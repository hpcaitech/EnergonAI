import torch
import time
from scipy import stats
import numpy as np
from engine_server import InferenceEngine


class single_request():
    def __init__(self, input_, time_stamp: float):
        self.input_ = input_
        self.time_ = time_stamp
        self.seq_len = len(input_['input_ids'])


class Batch_Manager():
    def __init__(self, engine: InferenceEngine, cached_cost: list,
                 init_mu: int, init_theta: int,
                 max_batch_size: int = 32, lr: float = 0.01,
                 max_seq_len=1024):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.lr = lr
        self.mu = init_mu
        self.theta = init_theta
        self.max_seq_len = max_seq_len
        # self.normal_weight = self._init_normal_dist_weight()
        self.req_list = []
        self.cached_cost = cached_cost

    def insert_req(self, time_stamp: float, input_ids):
        tmp_req = single_request(input_ids, time_stamp)
        self.req_list.append(tmp_req)
        self.req_list.sort(key=lambda x: x.seq_len)

    def cal_priority(self, batch_list: list, cur_stamp: float):
        cur_len = batch_list[-1].seq_len
        earliest_timestamp = min([i.time_ for i in batch_list])

        wait_time = cur_stamp - earliest_timestamp
        batch_size = len(batch_list)
        appear_possibility_weight = 1.0 / self.normal_weight[cur_len]

        #TODO adjust the euqation
        priority = appear_possibility_weight * batch_size * np.exp(wait_time)
        return priority

    # def _init_normal_dist_weight(self):
    #     temp_weight_list = [0]
    #     for i in range(1, self.max_seq_len):
    #         temp_weight_list.append(stats.norm(self.mu, self.theta).cdf(i) -
    #                                 stats.norm(self.mu, self.theta).cdf(i - 1))
    #     return temp_weight_list

    def cal_norm_weight(self, seq_len):
        return stats.norm(self.mu, self.theta).cdf(seq_len) - \
               stats.norm(self.mu, self.theta).cdf(seq_len - 1)

    def update_norm(self, batch_: list):
        new_mu = np.mean([i.seq_len for i in batch_])
        new_theta = np.std([i.seq_len for i in batch_])
        delta_mu = new_mu - self.mu
        delta_theta = new_theta - self.theta
        self.mu += self.lr * delta_mu
        self.theta += self.lr * delta_theta
        return

    def wrap_batch(self):
        states = [0]
        start_idx_list = []
        for i in range(1, len(self.req_list) + 1):
            j = i - 1
            start_idx = i - 1
            cur_length = self.req_list[i - 1].seq_len
            min_cost = self.cached_cost[cur_length][1] + states[j]
            while j > 0:
                tmp_cost = states[j - 1] + \
                           self.cached_cost[cur_length][i - j + 1] * (i - j + 1)
                if tmp_cost < min_cost:
                    min_cost = tmp_cost
                    start_idx = j - 1
                j -= 1
            states[i] = min_cost
            start_idx_list[i] = start_idx
        i = len(self.req_list)
        res_start = -1
        res_end = -1
        max_priority = -1
        cur_timestamp = time.time()
        while i > 0:
            end_idx = i
            start_idx = start_idx_list[i]
            current_batch = self.req_list[start_idx: end_idx]
            current_priority = self.cal_priority(current_batch, cur_timestamp)
            if current_priority > max_priority:
                max_priority = current_priority
                res_start = start_idx
                res_end = end_idx
            i = start_idx - 1
        result_batch = self.req_list[res_start: res_end]
        del self.req_list[res_start: res_end]
        self.update_norm(result_batch)
        return result_batch
