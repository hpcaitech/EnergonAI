import torch
from typing import List, Deque, Tuple, Hashable, Any
from energonai import BatchManager, SubmitEntry, TaskEntry
import numpy as np
import logging
from colossalai.logging import get_dist_logger
# logging.basicConfig(level=logging.info,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=get_dist_logger('batch')
# TODO 设置log等级
# logger.setLevel(level=logging.WARNING)

class BatchManagerForGeneration(BatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs['input_ids']) for inputs in batch_inputs)
        outputs = {'input_ids': [], 'attention_mask': [],'position_ids':[]}
        for inputs in batch_inputs:
            input_ids, attention_mask, position_ids = inputs['input_ids'], inputs['attention_mask'],inputs['position_ids']
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            
            # attention_mask = [0] * padding_len + attention_mask
            attention_mask = np.pad(attention_mask,
                                    pad_width=[(0, 0), (padding_len, 0), (padding_len, 0)],
                                    mode='constant', constant_values=True)
            position_ids = np.pad(position_ids,pad_width=[(0, 0), (padding_len, 0)])

            outputs['input_ids'].append(input_ids)
            outputs['attention_mask'].append(attention_mask)
            outputs['position_ids'].append(position_ids)
        for k in outputs:
            if isinstance(outputs[k],list):
                outputs[k]=np.stack(outputs[k])
            outputs[k] = torch.tensor(outputs[k])
            # if isinstance(outputs[k], list):
            #     outputs[k] = np.array(outputs[k])
            # outputs[k] = torch.from_numpy(outputs[k])
        logger.info(f"type(outputs:{type(outputs)}")
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data['top_k'], data['top_p'], data['temperature'])
    

    def pro_batch(self,batch,out_batch):
        for i in range(len(batch[0]['input_ids'])):
            tmp={}
            tmp['input_ids']=batch[0]['input_ids']
            tmp['attention_mask']=batch[0]['attention_mask']
            tmp['position_ids']=batch[0]['position_ids']
            tmp['max_tokens']=batch[0]['max_tokens']
            tmp['do_sample']=batch[0]['do_sample']
            tmp['top_k']=batch[0]['top_k']
            tmp['top_p']=batch[0]['top_p']
            tmp['temperature']=batch[0]['temperature']
            out_batch.append(tmp)
        return out_batch

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        out_batch=[]
        out_batch= self.pro_batch(batch,out_batch)
        while len(out_batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data['max_tokens'] > entry.data['max_tokens']:
                break
            e = q.popleft()
            out_batch=self.pro_batch(e,out_batch)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(out_batch)
        trunc_lens = []
        for data in out_batch:
            trunc_lens.append(max_len + data['max_tokens'])
        inputs['top_k'] = entry.data['top_k']
        inputs['top_p'] = entry.data['top_p']
        inputs['temperature'] = entry.data['temperature']
        inputs['max_tokens'] = max_len + entry.data['max_tokens']
        # TODO inputs,在后续是1*4的，必须把这个解决掉，让其成为2*4的张量
        return TaskEntry(tuple(uids), inputs), {'trunc_lens': trunc_lens}

    def split_batch(self, task_entry: TaskEntry, trunc_lens: List[int] = []) -> List[Tuple[Hashable, Any]]:
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, task_entry.batch, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval