from typing import Any, Hashable, Tuple, Deque, Iterable
from dataclasses import dataclass
from .task import TaskEntry


@dataclass
class SubmitEntry:
    uid: Hashable
    data: Any


class BatchManager:
    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        return TaskEntry((entry.uid, ), entry.data), {}

    def split_batch(self, task_entry: TaskEntry, **kwargs: Any) -> Iterable[Tuple[Hashable, Any]]:
        return [(task_entry.uids[0], task_entry.batch)]
