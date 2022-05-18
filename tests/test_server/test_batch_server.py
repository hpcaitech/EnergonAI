import requests
import threading
import torch
import random
import os
import numpy as np
import time

latency = []


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
        res_text_list = [(tmp_str * len_list[0][i]) + "\n" for i in range(text_num)]
        f = open(file_name, "w")
        f.writelines(res_text_list)
    res_text_list = [i.replace(" \n", "").replace('\n', '') for i in res_text_list]
    return res_text_list


def send_request(input_: str, url_: str, port: str):
    global latency
    url_ = url_ + ":" + port + "/model_with_padding"
    params = {"input_str": input_}
    start_ = time.time()
    response = requests.post(url=url_, json=params).text
    latency.append(time.time() - start_)
    print(response)


def test_batch():
    global latency
    ip_ = "http://127.0.0.1"
    port_ = "8020"
    req_num = 50
    seq_len = 64
    # req_list = ["test " * 10 for _ in range(req_num)]
    req_list = generate_test_dataset(req_num, seq_len)
    for i in range(req_num):
        time.sleep(0.005)
        temp_thread = threading.Thread(target=send_request, args=(req_list[i], ip_, port_))
        temp_thread.start()
    time.sleep(20)
    print(np.mean(latency))


if __name__ == "__main__":
    test_batch()

