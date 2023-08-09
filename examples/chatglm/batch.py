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
        logger.info(f'type(batch_inputs){type(batch_inputs)}')
        logger.info(f'len(batch_inputs){len(batch_inputs)}')
        logger.info(f'batch_inputs{batch_inputs}')
        logger.info('这段话会出现几次，表明是否有多线程')
        max_len = max(len(inputs['input_ids']) for inputs in batch_inputs)
        outputs = {'input_ids': [], 'attention_mask': [],'position_ids':[]}
        for inputs in batch_inputs:
            input_ids, attention_mask, position_ids = inputs['input_ids'], inputs['attention_mask'],inputs['position_ids']
            logger.info('&'*10)
            logger.info(f'attention_mask{attention_mask}')
            logger.info(type(attention_mask))
            logger.info(f'attention_mask.shape{attention_mask.shape}')
            logger.info('$'*10)

            padding_len = max_len - len(input_ids)
            logger.info(f'max_len:{max_len}')
            logger.info(f'padding_len:{padding_len}')
            logger.info(f'input_ids:{input_ids}')

            input_ids = [self.pad_token_id] * padding_len + input_ids
            
            # attention_mask = [0] * padding_len + attention_mask
            attention_mask = np.pad(attention_mask,
                                    pad_width=[(0, 0), (padding_len, 0), (padding_len, 0)],
                                    mode='constant', constant_values=True)
            position_ids = np.pad(position_ids,pad_width=[(0, 0), (padding_len, 0)])

            outputs['input_ids'].append(input_ids)
            outputs['attention_mask'].append(attention_mask)
            outputs['position_ids'].append(position_ids)
            # outputs['position_ids'].append(position_ids)
        for k in outputs:
            logger.info(f"k:{k}")
            logger.info(f"type(outputs[k]): {type(outputs[k])}")
            logger.info(f"outputs[k]: {outputs[k]}")
            if isinstance(outputs[k],list):
                outputs[k]=np.stack(outputs[k])
            outputs[k] = torch.tensor(outputs[k])
            # if isinstance(outputs[k], list):
            #     outputs[k] = np.array(outputs[k])
            # outputs[k] = torch.from_numpy(outputs[k])

            logger.info(f"outputs[k].shape:{outputs[k].shape}")
        logger.info(f'output:{outputs}')
        logger.info(f'output["input_ids"]:{outputs["input_ids"]}')
        logger.info(f'output["input_ids"].shape:{outputs["input_ids"].shape}')
        logger.info(f'output["attention_mask"]:{outputs["attention_mask"]}')
        logger.info(f'output["attention_mask"].shape:{outputs["attention_mask"].shape}')
        logger.info(f'output["position_ids"]:{outputs["position_ids"]}')
        logger.info(f'output["position_ids"].shape:{outputs["position_ids"].shape}')
        # import pdb;pdb.set_trace()
        logger.info(f"type(outputs:{type(outputs)}")
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data['top_k'], data['top_p'], data['temperature'])

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data['max_tokens'] > entry.data['max_tokens']:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(batch)
        trunc_lens = []
        for data in batch:
            trunc_lens.append(max_len + data['max_tokens'])
        inputs['top_k'] = entry.data['top_k']
        inputs['top_p'] = entry.data['top_p']
        inputs['temperature'] = entry.data['temperature']
        inputs['max_tokens'] = max_len + entry.data['max_tokens']
        return TaskEntry(tuple(uids), inputs), {'trunc_lens': trunc_lens}

    def split_batch(self, task_entry: TaskEntry, trunc_lens: List[int] = []) -> List[Tuple[Hashable, Any]]:
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, task_entry.batch, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval