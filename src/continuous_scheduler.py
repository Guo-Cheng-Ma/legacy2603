import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .kv_cache import KVCacheManager
from .config import get_hetero_transfer_bandwidths
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
        pipe_level: bool = False,
        parallel_ff: bool = False,
        debug: bool = False,
        debug_interval: int = 100,
        kv_arch_cfg: Optional[dict] = None,
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
        self.pipe_level = bool(pipe_level)
        self.parallel_ff = bool(parallel_ff)
        self.debug = debug
        self.debug_interval = max(1, int(debug_interval))
        self.step_count = 0
        self.transfer_bw = get_hetero_transfer_bandwidths(kv_arch_cfg)

        self.sim_time = 0.0
        self.arrival_idx = 0
        self.wait_queue: Deque[RequestState] = deque()
        self.active: Dict[int, RequestState] = {}
        self.done: Dict[int, RequestState] = {}
        # Energy is accumulated in pJ from estimator outputs.
        self.prefill_energy_pj = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.decode_energy_pj = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Work counters accumulate estimated stage times before overlap.
        self.prefill_work_time_s = 0.0
        self.decode_work_time_s = 0.0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.migration_time_s = 0.0
        self.dma_time_s = 0.0
        self.pcie_time_s = 0.0
        self.migration_energy_nj = 0.0
        self.dma_energy_nj = 0.0
        self.pcie_energy_nj = 0.0
        self.dma_transfer_blocks = 0
        self.pcie_transfer_blocks = 0
        self.migration_bytes = 0

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[TRACE][scheduler][step={self.step_count}][t={self.sim_time:.6f}] {message}")

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

    def _backfill_active_slots(self) -> int:
        activated = 0
        activated_ids = []
        while len(self.active) < self.max_batch_size and self.wait_queue:
            req = self.wait_queue.popleft()
            req.set_state(RequestLifecycle.PREFILLING)
            if req.start_time is None:
                req.start_time = self.sim_time
            self.active[req.req_id] = req
            activated_ids.append(req.req_id)
            activated += 1
        if activated > 0:
            self._log(f"backfill active: +{activated} reqs {activated_ids[:8]}")
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
            hit_blocks = 0
            l1_hit_blocks = 0
            l2_hit_blocks = 0
            l3_hit_blocks = 0
            l1_to_l2_moves = 0
            l2_to_l3_moves = 0
            l3_drop_blocks = 0
            dma_blocks = 0
            pcie_blocks = 0
            for block_idx in block_indices:
                hash_id = req.trace.hash_ids[block_idx]
                if self.kv_cache is None:
                    missed_blocks += 1
                    req.computed_blocks += 1
                    continue

                access = self.kv_cache.access(hash_id)
                if self.debug and (access.hit_tier is not None and access.hit_tier != "L1"):
                    self._log(
                        "kv hit req={} hash_id={} tier={} promote_to_l1=yes".format(
                            req.req_id,
                            hash_id,
                            access.hit_tier,
                        )
                    )
                if self.debug and (access.l1_to_l2 > 0 or access.l2_to_l3 > 0 or access.l3_drops > 0):
                    self._log(
                        "kv evict req={} hash_id={} l1_to_l2={} l2_to_l3={} l3_drop={}".format(
                            req.req_id,
                            hash_id,
                            access.l1_to_l2,
                            access.l2_to_l3,
                            access.l3_drops,
                        )
                    )
                if self.debug and (access.dma_blocks > 0 or access.pcie_blocks > 0):
                    self._log(
                        "kv xfer req={} hash_id={} dma_blocks={} pcie_blocks={}".format(
                            req.req_id,
                            hash_id,
                            access.dma_blocks,
                            access.pcie_blocks,
                        )
                    )
                req.l1_hit_blocks += access.l1_hit
                req.l2_hit_blocks += access.l2_hit
                req.l3_hit_blocks += access.l3_hit
                req.l1_to_l2_blocks += access.l1_to_l2
                req.l2_to_l3_blocks += access.l2_to_l3
                req.l3_drop_blocks += access.l3_drops
                req.dma_blocks += access.dma_blocks
                req.migration_bytes += access.migration_blocks * self.kv_cache.kv_bytes_per_block

                l1_hit_blocks += access.l1_hit
                l2_hit_blocks += access.l2_hit
                l3_hit_blocks += access.l3_hit
                l1_to_l2_moves += access.l1_to_l2
                l2_to_l3_moves += access.l2_to_l3
                l3_drop_blocks += access.l3_drops
                dma_blocks += access.dma_blocks
                pcie_blocks += access.pcie_blocks

                if access.is_hit:
                    req.reused_blocks += 1
                    hit_blocks += 1
                else:
                    missed_blocks += 1
                    req.computed_blocks += 1

            chunk_end = min(req.input_length, req.prefill_progress_tokens + self.prefill_chunk_tokens)
            req.prefill_progress_tokens = chunk_end

            if self.system is not None and self.kv_cache is not None:
                if dma_blocks > 0:
                    dma_bytes = dma_blocks * self.kv_cache.kv_bytes_per_block
                    dma_est = self.system.estimate_kv_dma(
                        dma_bytes,
                        bw_bps=self.transfer_bw["dma_bw_bps"],
                    )
                    prefill_latency += dma_est['latency']
                    self.migration_time_s += dma_est['latency']
                    self.dma_time_s += dma_est['latency']
                    self.migration_energy_nj += dma_est['energy_nj']
                    self.dma_energy_nj += dma_est['energy_nj']
                    self.dma_transfer_blocks += dma_blocks
                    self.migration_bytes += dma_bytes
                if pcie_blocks > 0:
                    pcie_bytes = pcie_blocks * self.kv_cache.kv_bytes_per_block
                    pcie_est = self.system.estimate_kv_pcie(
                        pcie_bytes,
                        bw_bps=self.transfer_bw["pcie_bw_bps"],
                    )
                    prefill_latency += pcie_est['latency']
                    self.migration_time_s += pcie_est['latency']
                    self.pcie_time_s += pcie_est['latency']
                    self.migration_energy_nj += pcie_est['energy_nj']
                    self.pcie_energy_nj += pcie_est['energy_nj']
                    self.pcie_transfer_blocks += pcie_blocks
                    self.migration_bytes += pcie_bytes

            if missed_blocks > 0 and self.system is not None:
                effective_tokens = min(missed_blocks * 16, req.input_length)
                estimate = self.system.estimate_prefill_microbatch(1, max(1, effective_tokens))
                prefill_latency += estimate['latency']
                energy = estimate.get("energy", [0, 0, 0, 0, 0, 0])
                self.prefill_energy_pj = [
                    self.prefill_energy_pj[i] + float(energy[i]) for i in range(len(self.prefill_energy_pj))
                ]

            if req.prefill_progress_tokens >= req.input_length:
                req.set_state(RequestLifecycle.DECODING)
            self._log(
                "prefill req={} chunk_blocks={} hit={} miss={} hit_tiers=[{},{},{}] moves=[l1_to_l2={},l2_to_l3={},l3_drop={}] xfer=[dma_blocks={},pcie_blocks={}] progress={}/{}".format(
                    req.req_id,
                    len(block_indices),
                    hit_blocks,
                    missed_blocks,
                    l1_hit_blocks,
                    l2_hit_blocks,
                    l3_hit_blocks,
                    l1_to_l2_moves,
                    l2_to_l3_moves,
                    l3_drop_blocks,
                    dma_blocks,
                    pcie_blocks,
                    req.prefill_progress_tokens,
                    req.input_length,
                )
            )

        return prefill_latency

    def _step_decode(self) -> float:
        decoding = [r for r in self.active.values() if r.state == RequestLifecycle.DECODING]
        if not decoding:
            return 0.0

        decode_latency = 0.0
        max_context_len = max(r.input_length + r.generated_tokens + 1 for r in decoding)
        if self.system is not None:
            estimate = self.system.estimate_decode_step(
                len(decoding),
                max_context_len,
                pipe=self.pipe_level,
                parallel_ff=self.parallel_ff,
            )
            decode_latency = estimate['latency']
            energy = estimate.get("energy", [0, 0, 0, 0, 0, 0])
            self.decode_energy_pj = [
                self.decode_energy_pj[i] + float(energy[i]) for i in range(len(self.decode_energy_pj))
            ]
        self._log(
            "decode batch={} max_context_len={} est_latency={:.6f}".format(
                len(decoding),
                max_context_len,
                decode_latency,
            )
        )

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
        if done_ids:
            self._log(f"complete reqs: {done_ids[:8]} (total_done={len(self.done)})")

        return decode_latency

    def step(self) -> SchedulerSnapshot:
        self.step_count += 1
        self._admit_arrivals()
        self._backfill_active_slots()

        prefill_latency = self._step_prefill()
        decode_latency = self._step_decode()
        if prefill_latency > 0:
            self.prefill_steps += 1
        if decode_latency > 0:
            self.decode_steps += 1
        self.prefill_work_time_s += prefill_latency
        self.decode_work_time_s += decode_latency
        step_latency = max(prefill_latency, decode_latency)

        self.sim_time += step_latency

        # Immediate backfill on completion in same scheduling iteration.
        self._admit_arrivals()
        self._backfill_active_slots()

        if self.debug and (self.step_count <= 20 or self.step_count % self.debug_interval == 0):
            snapshot = self.snapshot()
            self._log(
                "summary active={} waiting={} not_arrived={} done={}".format(
                    snapshot.num_active,
                    snapshot.num_waiting,
                    snapshot.num_not_arrived,
                    snapshot.num_done,
                )
            )

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
        self._log(
            "start run requests={} max_batch_size={} prefill_chunk_tokens={}".format(
                len(self.requests),
                self.max_batch_size,
                self.prefill_chunk_tokens,
            )
        )
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
                self._log(f"time jump to next arrival: {self.sim_time:.6f}")

        self._log(f"run finished total_done={len(self.done)} sim_time={self.sim_time:.6f}")
        return self.snapshot()

    def completed_requests(self):
        return [self.done[key] for key in sorted(self.done.keys())]

    def energy_snapshot(self):
        prefill_total_pj = sum(self.prefill_energy_pj)
        decode_total_pj = sum(self.decode_energy_pj)
        total_pj = prefill_total_pj + decode_total_pj
        model_energy_nj = total_pj / 1000.0
        total_energy_nj = model_energy_nj + self.migration_energy_nj
        snapshot = {
            "prefill_energy_pj": prefill_total_pj,
            "decode_energy_pj": decode_total_pj,
            "total_energy_pj": total_pj,
            "prefill_energy_nj": prefill_total_pj / 1000.0,
            "decode_energy_nj": decode_total_pj / 1000.0,
            "model_energy_nj": model_energy_nj,
            "migration_energy_nj": self.migration_energy_nj,
            "dma_energy_nj": self.dma_energy_nj,
            "pcie_energy_nj": self.pcie_energy_nj,
            "total_energy_nj": total_energy_nj,
            "prefill_work_time_s": self.prefill_work_time_s,
            "decode_work_time_s": self.decode_work_time_s,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "migration_time_s": self.migration_time_s,
            "dma_time_s": self.dma_time_s,
            "pcie_time_s": self.pcie_time_s,
            "migration_bytes": self.migration_bytes,
            "dma_transfer_blocks": self.dma_transfer_blocks,
            "pcie_transfer_blocks": self.pcie_transfer_blocks,
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
        num_done = len(self.done)
        num_active = len(self.active)
        num_waiting = len(self.wait_queue)
        # arrival_idx already counts all admitted requests (waiting + active + done),
        # so not_arrived should not subtract queue/state counts again.
        num_not_arrived = len(self.requests) - self.arrival_idx

        return SchedulerSnapshot(
            sim_time=self.sim_time,
            num_not_arrived=max(num_not_arrived, 0),
            num_waiting=num_waiting,
            num_active=num_active,
            num_done=num_done,
        )
