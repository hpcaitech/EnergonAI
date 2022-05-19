from concurrent.futures import ThreadPoolExecutor

import requests
import threading
import math
import torch
import random
import os
import numpy as np
import time

latency = []
finish_time = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_test_dataset(text_num: int = 100, max_len: int = 1024):
    file_name = "test_set_{}_{}.txt".format(text_num, max_len)
    if os.path.exists(file_name):
        f = open(file_name)
        res_text_list = f.readlines()
    else:
        tmp_str = "test "
        len_list = torch.randint(low=1, high=max_len, size=(1, text_num))
        # len_list = [math.floor(random.uniform(1, max_len)) for _ in range(text_num)]
        res_text_list = [(tmp_str * len_list[0][i]) + "\n" for i in range(text_num)]
        f = open(file_name, "w")
        f.writelines(res_text_list)
    res_text_list = [i.replace(" \n", "").replace('\n', '') for i in res_text_list]
    return res_text_list


def generate_raising_dataset(text_num: int = 100, max_len: int = 1024):
    file_name = "raising_set_{}_{}.txt".format(text_num, max_len)
    if os.path.exists(file_name):
        f = open(file_name)
        res_text_list = f.readlines()
    else:
        tmp_str = "test "
        # len_list = torch.randint(low=1, high=max_len, size=(1, text_num))
        len_list = [1024 - i for i in range(text_num)]
        res_text_list = [(tmp_str * len_list[i]) + "\n" for i in range(text_num)]
        f = open(file_name, "w")
        f.writelines(res_text_list)
    res_text_list = [i.replace(" \n", "").replace('\n', '') for i in res_text_list]
    return res_text_list


def send_request(input_: str, url_: str, port: str, num: int, record=False):
    global latency
    url_ = url_ + ":" + port + "/model_with_padding_naive"
    # url_ = url_ + ":" + port + "/model_with_padding"
    # url_ = url_ + ":" + port + "/server_random"
    params = {"input_str": input_}
    start_ = time.time()
    response = requests.post(url=url_, json=params).text
    if record:
        lat = time.time() - start_
        latency.append(lat)
        # print("latency: {}, {}".format(num, lat))
        finish_time.append(time.time())
    print(response)


def test_batch():
    global latency
    setup_seed(42)
    pool = ThreadPoolExecutor(max_workers=64)
    ip_ = "http://127.0.0.1"
    port_ = "8020"
    req_num = 512
    seq_len = 256
    # req_list = ["test " * 10 for _ in range(req_num)]
    req_list = generate_test_dataset(req_num, seq_len)
    print(min([len(k) for k in req_list]))
    # req_list = generate_raising_dataset(req_num, seq_len)
    # send_request(req_list[0], ip_, port_, -1)
    send_request('1', ip_, port_, -1)
    time.sleep(1)
    st__ = time.time()
    for i in range(req_num):
        pool.submit(send_request, req_list[i], ip_, port_, i, True)
    time.sleep(40)
    print(np.mean(latency))
    print(req_num / (max(finish_time) - st__))


if __name__ == "__main__":
    test_batch()
