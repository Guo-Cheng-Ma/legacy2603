from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from .request_state import RequestLifecycle, RequestState


@dataclass
class SchedulerSnapshot:
    sim_time: float
    num_not_arrived: int
    num_waiting: int
    num_active: int
    num_done: int


class ContinuousScheduler:
    """FIFO continuous-batching scheduler skeleton for trace-driven execution."""

    def __init__(self, requests: List[RequestState], max_batch_size: int):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        self.requests = list(requests)
        self.max_batch_size = max_batch_size

        self.sim_time = 0.0
        self.arrival_idx = 0
        self.wait_queue: Deque[RequestState] = deque()
        self.active: Dict[int, RequestState] = {}
        self.done: Dict[int, RequestState] = {}

    def _admit_arrivals(self) -> int:
        admitted = 0
        while self.arrival_idx < len(self.requests):
            req = self.requests[self.arrival_idx]
            if req.timestamp > self.sim_time:
                break
            req.set_state(RequestLifecycle.WAITING)
            req.enqueued_time = self.sim_time
            self.wait_queue.append(req)
            self.arrival_idx += 1
            admitted += 1
        return admitted

    def _backfill_active_slots(self) -> int:
        activated = 0
        while len(self.active) < self.max_batch_size and self.wait_queue:
            req = self.wait_queue.popleft()
            req.set_state(RequestLifecycle.PREFILLING)
            if req.start_time is None:
                req.start_time = self.sim_time
            self.active[req.req_id] = req
            activated += 1
        return activated

    def step_dry_run(self, dt: float = 0.0) -> SchedulerSnapshot:
        if dt < 0:
            raise ValueError("dt must be >= 0")

        self._admit_arrivals()
        self._backfill_active_slots()
        self.sim_time += dt

        num_done = len(self.done)
        num_active = len(self.active)
        num_waiting = len(self.wait_queue)
        num_not_arrived = len(self.requests) - (self.arrival_idx + num_waiting + num_active + num_done)

        return SchedulerSnapshot(
            sim_time=self.sim_time,
            num_not_arrived=max(num_not_arrived, 0),
            num_waiting=num_waiting,
            num_active=num_active,
            num_done=num_done,
        )
