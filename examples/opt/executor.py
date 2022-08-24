import asyncio
import time
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from typing import Any, Dict, Deque, List
from collections import namedtuple, deque

GenerationArgs = namedtuple('GenerationArgs', ['top_k', 'top_p', 'temperature', 'max_tokens'])
SubmitEntry = namedtuple('SubmitEntry', ['text', 'args'])


class Executor:
    def __init__(self, engine, tokenizer, max_batch_size: int = 1) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.running: bool = False
        self.thread = None
        self.ready_map: Dict[int, Any] = {}
        self.submit_queue: Deque[SubmitEntry] = deque()

    def _start(self) -> None:
        self.running = True
        while self.running:
            if len(self.submit_queue) > 0:
                inputs, entry_ids = self._make_batch()
                print(f'Run batch size = {len(entry_ids)}')
                outputs = self.engine.run(inputs)
                outputs = self.tokenizer.decoce(outputs)
                for entry_id, output in zip(entry_ids, outputs):
                    self.ready_map[entry_id] = output

    def _make_batch(self):
        entry = self.submit_queue.popleft()
        batch = [entry]
        while len(batch) < self.max_batch_size:
            if len(self.submit_queue) == 0:
                break
            if self.submit_queue[0].args != entry.args:
                break
            batch.append(self.submit_queue.popleft())
        batch_text = [e.text for e in batch]
        inputs = self.tokenizer(batch_text, padding=True)
        inputs['top_k'] = entry.args.top_k
        inputs['top_p'] = entry.args.top_p
        inputs['temperature'] = entry.args.temperature
        inputs['max_tokens'] = entry.args.max_tokens
        return inputs, [id(e) for e in batch]

    def start(self):
        self.thread = Thread(target=self._start)
        self.thread.start()

    def submit(self, promt, max_tokens, top_k, top_p, temperature):
        if not self.running:
            raise RuntimeError('executor is shutdown')
        args = GenerationArgs(top_k, top_p, temperature, max_tokens)
        entry = SubmitEntry(promt, args)
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
