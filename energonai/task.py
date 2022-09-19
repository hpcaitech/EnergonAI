from dataclasses import dataclass
from typing import Hashable, Tuple, Any


@dataclass
class TaskEntry:
    uids: Tuple[Hashable, ...]
    batch: Any
