import asyncio
import time
import torch
from threading import Thread
from typing import Any, Dict, Deque
from collections import namedtuple, deque
from energonai.logging import get_dist_logger

GenerationArgs = namedtuple('GenerationArgs', ['top_k', 'top_p', 'temperature'])
SubmitEntry = namedtuple('SubmitEntry', ['inputs', 'args', 'decode_steps'])


class Executor:
    def __init__(self, engine, pad_token_id: int = 0, max_batch_size: int = 1) -> None:
        self.engine = engine
        self.pad_token_id = pad_token_id
        self.max_batch_size = max_batch_size
        self.running: bool = False
        self.thread = None
        self.ready_map: Dict[int, Any] = {}
        self.submit_queue: Deque[SubmitEntry] = deque()
        self.logger = get_dist_logger()

    def _start(self) -> None:
        self.running = True
        while self.running:
            if len(self.submit_queue) > 0:
                inputs, entry_ids, trunc_lens, decode_steps = self._make_batch()
                start = time.time()
                outputs = self.engine.run(inputs).to_here()
                for entry_id, output, trunc_len in zip(entry_ids, outputs, trunc_lens):
                    self.ready_map[entry_id] = output[:trunc_len]
                self.logger.info(
                    f'batch size: {len(entry_ids)}, decode steps: {decode_steps}, time: {time.time()-start:.3f} s')

    def _make_batch(self):
        entry = self.submit_queue.popleft()
        batch = [entry]
        while len(batch) < self.max_batch_size:
            if len(self.submit_queue) == 0:
                break
            if self.submit_queue[0].args != entry.args:
                break
            if self.submit_queue[0].decode_steps > entry.decode_steps:
                break
            batch.append(self.submit_queue.popleft())
        inputs, max_len = self._left_padding([e.inputs for e in batch])
        entry_ids = []
        trunc_lens = []
        for e in batch:
            entry_ids.append(id(e))
            trunc_lens.append(max_len + e.decode_steps)
        inputs['top_k'] = entry.args.top_k
        inputs['top_p'] = entry.args.top_p
        inputs['temperature'] = entry.args.temperature
        inputs['max_tokens'] = max_len + entry.decode_steps
        return inputs, entry_ids, trunc_lens, entry.decode_steps

    def start(self):
        self.thread = Thread(target=self._start)
        self.thread.start()

    def submit(self, inputs, max_tokens, top_k, top_p, temperature):
        if not self.running:
            raise RuntimeError('executor is shutdown')
        args = GenerationArgs(top_k, top_p, temperature)
        entry = SubmitEntry(inputs, args, max_tokens)
        self.submit_queue.append(entry)
        return id(entry)

    async def wait(self, entry_id):
        while True:
            if entry_id in self.ready_map:
                output = self.ready_map[entry_id]
                del self.ready_map[entry_id]
                return output
            await asyncio.sleep(0.1)

    def teardown(self):
        self.running = False
        self.thread.join()

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs['input_ids']) for inputs in batch_inputs)
        outputs = {'input_ids': [], 'attention_mask': []}
        for inputs in batch_inputs:
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            outputs['input_ids'].append(input_ids)
            outputs['attention_mask'].append(attention_mask)
        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])
        return outputs, max_len
