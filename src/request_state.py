from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RequestLifecycle(str, Enum):
    NOT_ARRIVED = "NOT_ARRIVED"
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    DONE = "DONE"


@dataclass(frozen=True)
class TraceRequest:
    """Immutable trace fields parsed from one JSONL row."""

    req_id: int
    chat_id: int
    parent_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    request_type: str
    turn: int
    hash_ids: List[int]


@dataclass
class RequestState:
    """Mutable runtime fields layered on top of immutable trace content."""

    trace: TraceRequest
    state: RequestLifecycle = RequestLifecycle.NOT_ARRIVED
    prefill_progress_tokens: int = 0
    generated_tokens: int = 0
    reused_blocks: int = 0
    computed_blocks: int = 0
    dma_blocks: int = 0
    enqueued_time: Optional[float] = None
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    tags: dict = field(default_factory=dict)

    @property
    def req_id(self) -> int:
        return self.trace.req_id

    @property
    def timestamp(self) -> float:
        return self.trace.timestamp

    @property
    def input_length(self) -> int:
        return self.trace.input_length

    @property
    def output_length(self) -> int:
        return self.trace.output_length

    def set_state(self, state: RequestLifecycle) -> None:
        self.state = state
