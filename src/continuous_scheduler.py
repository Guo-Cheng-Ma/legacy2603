import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .kv_cache import KVCacheManager
from .request_state import RequestLifecycle, RequestState


@dataclass
class SchedulerSnapshot:
    sim_time: float
    num_not_arrived: int
    num_waiting: int
    num_active: int
    num_done: int


class ContinuousScheduler:
    """FIFO continuous-batching scheduler with chunked prefill and decode steps."""

    def __init__(
        self,
        requests: List[RequestState],
        max_batch_size: int,
        system=None,
        kv_cache: Optional[KVCacheManager] = None,
        prefill_chunk_tokens: int = 128,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if prefill_chunk_tokens <= 0:
            raise ValueError("prefill_chunk_tokens must be > 0")

        self.requests = list(requests)
        self.max_batch_size = max_batch_size
        self.system = system
        self.kv_cache = kv_cache
        self.prefill_chunk_tokens = prefill_chunk_tokens

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

    def _blocks_for_chunk(self, req: RequestState):
        if req.prefill_progress_tokens >= req.input_length:
            return []

        chunk_end = min(req.input_length, req.prefill_progress_tokens + self.prefill_chunk_tokens)
        block_start = req.prefill_progress_tokens // 16
        block_end = int(math.ceil(chunk_end / 16.0))
        return list(range(block_start, block_end))

    def _step_prefill(self) -> float:
        prefill_latency = 0.0

        for req in list(self.active.values()):
            if req.state != RequestLifecycle.PREFILLING:
                continue

            block_indices = self._blocks_for_chunk(req)
            if not block_indices:
                req.set_state(RequestLifecycle.DECODING)
                continue

            missed_blocks = 0
            for block_idx in block_indices:
                hash_id = req.trace.hash_ids[block_idx]
                hit = self.kv_cache.probe(hash_id) if self.kv_cache is not None else False
                if hit:
                    req.reused_blocks += 1
                else:
                    missed_blocks += 1
                    req.computed_blocks += 1
                    if self.kv_cache is not None:
                        self.kv_cache.insert(hash_id)

            chunk_end = min(req.input_length, req.prefill_progress_tokens + self.prefill_chunk_tokens)
            req.prefill_progress_tokens = chunk_end

            if missed_blocks > 0 and self.system is not None:
                effective_tokens = min(missed_blocks * 16, req.input_length)
                estimate = self.system.estimate_prefill_microbatch(1, max(1, effective_tokens))
                prefill_latency += estimate['latency']

            if req.prefill_progress_tokens >= req.input_length:
                req.set_state(RequestLifecycle.DECODING)

        return prefill_latency

    def _step_decode(self) -> float:
        decoding = [r for r in self.active.values() if r.state == RequestLifecycle.DECODING]
        if not decoding:
            return 0.0

        decode_latency = 0.0
        max_context_len = max(r.input_length + r.generated_tokens + 1 for r in decoding)
        if self.system is not None:
            estimate = self.system.estimate_decode_step(len(decoding), max_context_len)
            decode_latency = estimate['latency']

        done_ids = []
        finished_time = self.sim_time + decode_latency
        for req in decoding:
            req.generated_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = finished_time
            if req.generated_tokens >= req.output_length:
                req.set_state(RequestLifecycle.DONE)
                req.finish_time = finished_time
                done_ids.append(req.req_id)

        for req_id in done_ids:
            self.done[req_id] = self.active.pop(req_id)

        return decode_latency

    def step(self) -> SchedulerSnapshot:
        self._admit_arrivals()
        self._backfill_active_slots()

        prefill_latency = self._step_prefill()
        decode_latency = self._step_decode()
        step_latency = max(prefill_latency, decode_latency)

        self.sim_time += step_latency

        # Immediate backfill on completion in same scheduling iteration.
        self._admit_arrivals()
        self._backfill_active_slots()

        return self.snapshot()

    def step_dry_run(self, dt: float = 0.0) -> SchedulerSnapshot:
        if dt < 0:
            raise ValueError("dt must be >= 0")

        self._admit_arrivals()
        self._backfill_active_slots()
        self.sim_time += dt

        return self.snapshot()

    def run(self, max_steps: int = 10_000_000) -> SchedulerSnapshot:
        steps = 0
        while len(self.done) < len(self.requests):
            prev_time = self.sim_time
            prev_done = len(self.done)
            self.step()
            steps += 1
            if steps >= max_steps:
                raise RuntimeError("scheduler exceeded max_steps")

            # If no progress is possible, jump to next arrival time.
            if self.sim_time == prev_time and len(self.done) == prev_done and not self.active and self.arrival_idx < len(self.requests):
                self.sim_time = self.requests[self.arrival_idx].timestamp

        return self.snapshot()

    def snapshot(self) -> SchedulerSnapshot:
        num_done = len(self.done)
        num_active = len(self.active)
        num_waiting = len(self.wait_queue)
        num_not_arrived = len(self.requests) - (
            self.arrival_idx + num_waiting + num_active + num_done
        )

        return SchedulerSnapshot(
            sim_time=self.sim_time,
            num_not_arrived=max(num_not_arrived, 0),
            num_waiting=num_waiting,
            num_active=num_active,
            num_done=num_done,
        )
