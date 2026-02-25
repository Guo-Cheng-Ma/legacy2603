import math
from collections import deque
from typing import Deque, Dict, List, Optional

from .continuous_scheduler import SchedulerSnapshot
from .kv_cache import KVCacheManager
from .request_state import RequestLifecycle, RequestState


class StaticBatchScheduler:
    """FIFO static batching scheduler: no backfill within a running batch."""

    def __init__(
        self,
        requests: List[RequestState],
        max_batch_size: int,
        system=None,
        kv_cache: Optional[KVCacheManager] = None,
        pipe_level: bool = False,
        parallel_ff: bool = False,
        debug: bool = False,
        debug_interval: int = 100,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        self.requests = list(requests)
        self.max_batch_size = max_batch_size
        self.system = system
        self.kv_cache = kv_cache
        self.pipe_level = bool(pipe_level)
        self.parallel_ff = bool(parallel_ff)
        self.debug = debug
        self.debug_interval = max(1, int(debug_interval))

        self.step_count = 0
        self.sim_time = 0.0
        self.arrival_idx = 0
        self.wait_queue: Deque[RequestState] = deque()
        self.done: Dict[int, RequestState] = {}

        self.prefill_energy_pj = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.decode_energy_pj = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.prefill_work_time_s = 0.0
        self.decode_work_time_s = 0.0
        self.prefill_steps = 0
        self.decode_steps = 0

        self.num_batches = 0
        self.total_batch_size = 0
        self.total_batch_max_output_tokens = 0
        self.total_decode_padded_tokens = 0
        self._next_batch_id = 0

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[TRACE][static][step={self.step_count}][t={self.sim_time:.6f}] {message}")

    def _admit_arrivals(self) -> int:
        admitted = 0
        admitted_ids = []
        while self.arrival_idx < len(self.requests):
            req = self.requests[self.arrival_idx]
            if req.timestamp > self.sim_time:
                break
            req.set_state(RequestLifecycle.WAITING)
            req.enqueued_time = self.sim_time
            self.wait_queue.append(req)
            admitted_ids.append(req.req_id)
            self.arrival_idx += 1
            admitted += 1
        if admitted > 0:
            self._log(f"admit arrivals: +{admitted} reqs {admitted_ids[:8]}")
        return admitted

    def _form_batch(self) -> List[RequestState]:
        batch = []
        while self.wait_queue and len(batch) < self.max_batch_size:
            batch.append(self.wait_queue.popleft())
        return batch

    def _prefill_batch(self, batch: List[RequestState], batch_id: int) -> float:
        if not batch:
            return 0.0

        batch_max_miss_tokens = 0
        for req in batch:
            req.set_state(RequestLifecycle.PREFILLING)
            req.start_time = self.sim_time
            req.tags["batch_id"] = batch_id
            req.tags["batch_start_s"] = self.sim_time

            missed_blocks = 0
            hit_blocks = 0
            for block_idx in range(int(math.ceil(req.input_length / 16.0))):
                hash_id = req.trace.hash_ids[block_idx]
                hit = self.kv_cache.probe(hash_id) if self.kv_cache is not None else False
                if hit:
                    req.reused_blocks += 1
                    hit_blocks += 1
                else:
                    req.computed_blocks += 1
                    missed_blocks += 1
                    if self.kv_cache is not None:
                        self.kv_cache.insert(hash_id)

            req.prefill_progress_tokens = req.input_length
            req.set_state(RequestLifecycle.DECODING)
            miss_tokens = min(req.input_length, missed_blocks * 16)
            batch_max_miss_tokens = max(batch_max_miss_tokens, miss_tokens)
            self._log(
                "prefill req={} hit={} miss={} progress={}/{}".format(
                    req.req_id,
                    hit_blocks,
                    missed_blocks,
                    req.prefill_progress_tokens,
                    req.input_length,
                )
            )

        prefill_latency = 0.0
        if batch_max_miss_tokens > 0 and self.system is not None:
            estimate = self.system.estimate_prefill_microbatch(
                len(batch),
                max(1, batch_max_miss_tokens),
            )
            prefill_latency = estimate["latency"]
            energy = estimate.get("energy", [0, 0, 0, 0, 0, 0])
            self.prefill_energy_pj = [
                self.prefill_energy_pj[i] + float(energy[i]) for i in range(len(self.prefill_energy_pj))
            ]
            self.prefill_steps += 1
            self.prefill_work_time_s += prefill_latency

        return prefill_latency

    def _decode_batch(self, batch: List[RequestState], prefill_latency: float) -> float:
        if not batch:
            return 0.0

        max_output_tokens = max(req.output_length for req in batch)
        max_input_tokens = max(req.input_length for req in batch)
        decode_latency_total = 0.0

        for step_idx in range(max_output_tokens):
            step_latency = 0.0
            context_len = max_input_tokens + step_idx + 1
            if self.system is not None:
                estimate = self.system.estimate_decode_step(
                    len(batch),
                    context_len,
                    pipe=self.pipe_level,
                    parallel_ff=self.parallel_ff,
                )
                step_latency = estimate["latency"]
                energy = estimate.get("energy", [0, 0, 0, 0, 0, 0])
                self.decode_energy_pj = [
                    self.decode_energy_pj[i] + float(energy[i]) for i in range(len(self.decode_energy_pj))
                ]
            decode_latency_total += step_latency
            self.decode_steps += 1

            finished_time = self.sim_time + prefill_latency + decode_latency_total
            for req in batch:
                if step_idx < req.output_length:
                    req.generated_tokens += 1
                    if req.first_token_time is None:
                        req.first_token_time = finished_time

            if self.debug and (step_idx < 5 or (step_idx + 1) % self.debug_interval == 0):
                self._log(
                    "decode batch={} step={}/{} max_context_len={} est_latency={:.6f}".format(
                        len(batch),
                        step_idx + 1,
                        max_output_tokens,
                        context_len,
                        step_latency,
                    )
                )

        self.decode_work_time_s += decode_latency_total
        return decode_latency_total

    def _finish_batch(self, batch: List[RequestState], batch_finish_time: float) -> None:
        max_output_tokens = max(req.output_length for req in batch) if batch else 0
        for req in batch:
            req.finish_time = batch_finish_time
            req.set_state(RequestLifecycle.DONE)
            req.tags["batch_finish_s"] = batch_finish_time
            self.done[req.req_id] = req

        self.total_decode_padded_tokens += sum(max_output_tokens - req.output_length for req in batch)

    def step(self) -> SchedulerSnapshot:
        self.step_count += 1
        self._admit_arrivals()

        if not self.wait_queue:
            if self.arrival_idx < len(self.requests):
                self.sim_time = self.requests[self.arrival_idx].timestamp
                self._log(f"time jump to next arrival: {self.sim_time:.6f}")
                self._admit_arrivals()
            return self.snapshot()

        batch = self._form_batch()
        batch_size = len(batch)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self.num_batches += 1
        self.total_batch_size += batch_size
        max_output_tokens = max(req.output_length for req in batch) if batch else 0
        self.total_batch_max_output_tokens += max_output_tokens
        self._log(f"start batch={batch_id} size={batch_size} max_output={max_output_tokens}")

        prefill_latency = self._prefill_batch(batch, batch_id=batch_id)
        decode_latency = self._decode_batch(batch, prefill_latency)
        batch_latency = prefill_latency + decode_latency
        batch_finish_time = self.sim_time + batch_latency
        self._finish_batch(batch, batch_finish_time)
        self.sim_time = batch_finish_time
        self._log(f"finish batch={batch_id} latency={batch_latency:.6f} total_done={len(self.done)}")

        return self.snapshot()

    def run(self, max_steps: int = 10_000_000) -> SchedulerSnapshot:
        self._log(
            "start run requests={} max_batch_size={}".format(
                len(self.requests),
                self.max_batch_size,
            )
        )
        steps = 0
        while len(self.done) < len(self.requests):
            self.step()
            steps += 1
            if steps >= max_steps:
                raise RuntimeError("static scheduler exceeded max_steps")
        self._log(f"run finished total_done={len(self.done)} sim_time={self.sim_time:.6f}")
        return self.snapshot()

    def completed_requests(self):
        return [self.done[key] for key in sorted(self.done.keys())]

    def batch_stats(self):
        avg_batch_size = (self.total_batch_size / self.num_batches) if self.num_batches > 0 else 0.0
        avg_batch_max_output = (
            self.total_batch_max_output_tokens / self.num_batches
        ) if self.num_batches > 0 else 0.0
        return {
            "num_batches": self.num_batches,
            "avg_batch_size": avg_batch_size,
            "avg_batch_max_output_tokens": avg_batch_max_output,
            "decode_padded_tokens": self.total_decode_padded_tokens,
        }

    def energy_snapshot(self):
        prefill_total_pj = sum(self.prefill_energy_pj)
        decode_total_pj = sum(self.decode_energy_pj)
        total_pj = prefill_total_pj + decode_total_pj
        snapshot = {
            "prefill_energy_pj": prefill_total_pj,
            "decode_energy_pj": decode_total_pj,
            "total_energy_pj": total_pj,
            "prefill_energy_nj": prefill_total_pj / 1000.0,
            "decode_energy_nj": decode_total_pj / 1000.0,
            "total_energy_nj": total_pj / 1000.0,
            "prefill_work_time_s": self.prefill_work_time_s,
            "decode_work_time_s": self.decode_work_time_s,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
        }
        components = ["dram", "l2", "l1", "reg", "alu", "comm"]
        for i, name in enumerate(components):
            prefill_comp_pj = float(self.prefill_energy_pj[i])
            decode_comp_pj = float(self.decode_energy_pj[i])
            total_comp_pj = prefill_comp_pj + decode_comp_pj
            snapshot[f"prefill_{name}_energy_pj"] = prefill_comp_pj
            snapshot[f"decode_{name}_energy_pj"] = decode_comp_pj
            snapshot[f"total_{name}_energy_pj"] = total_comp_pj
            snapshot[f"prefill_{name}_energy_nj"] = prefill_comp_pj / 1000.0
            snapshot[f"decode_{name}_energy_nj"] = decode_comp_pj / 1000.0
            snapshot[f"total_{name}_energy_nj"] = total_comp_pj / 1000.0
        return snapshot

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            sim_time=self.sim_time,
            num_not_arrived=max(len(self.requests) - self.arrival_idx, 0),
            num_waiting=len(self.wait_queue),
            num_active=0,
            num_done=len(self.done),
        )
