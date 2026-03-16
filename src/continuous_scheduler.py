import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from .kv_cache import KVCacheManager, KVTransferDomainDemands, reduce_transfer_demands
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
        self.die_type = str((kv_arch_cfg or {}).get("DIE_TYPE", "attacc")).lower()
        self.charge_eviction_latency = self.die_type in {"attacc", "uniform"}
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
        self.callback_dma_transfer_blocks = 0
        self.callback_pcie_transfer_blocks = 0
        self.eviction_dma_transfer_blocks = 0
        self.eviction_pcie_transfer_blocks = 0
        self.migration_bytes = 0
        self.callback_migration_time_s = 0.0
        self.total_eviction_time_s = 0.0
        self.charged_eviction_time_s = 0.0
        self.overlapped_eviction_time_s = 0.0
        self.callback_dma_overlapped_time_s = 0.0
        self.callback_hbm_read_overlapped_time_s = 0.0
        self.callback_hbm_write_overlapped_time_s = 0.0
        self.callback_hbm_overlapped_time_s = 0.0
        self.callback_nvlink_overlapped_time_s = 0.0
        self.callback_pcie_overlapped_time_s = 0.0
        self.eviction_dma_overlapped_time_s = 0.0
        self.eviction_hbm_read_overlapped_time_s = 0.0
        self.eviction_hbm_write_overlapped_time_s = 0.0
        self.eviction_hbm_overlapped_time_s = 0.0
        self.eviction_nvlink_overlapped_time_s = 0.0
        self.eviction_pcie_overlapped_time_s = 0.0
        self.max_callback_dma_active_domains = 0
        self.max_callback_hbm_read_active_domains = 0
        self.max_callback_hbm_write_active_domains = 0
        self.max_callback_nvlink_active_domains = 0
        self.max_callback_pcie_active_domains = 0
        self.max_eviction_dma_active_domains = 0
        self.max_eviction_hbm_read_active_domains = 0
        self.max_eviction_hbm_write_active_domains = 0
        self.max_eviction_nvlink_active_domains = 0
        self.max_eviction_pcie_active_domains = 0
        self.cross_die_time_s = 0.0
        self.cross_card_time_s = 0.0
        self.cross_die_transfer_blocks = 0
        self.cross_card_transfer_blocks = 0
        self.replica_fanout_time_s = 0.0
        self.replica_fanout_bytes = 0
        self.replica_fanout_blocks = 0
        self.avoided_cross_card_callbacks = 0

        if self.kv_cache is not None:
            for req in self.requests:
                if req.home_card is None or req.home_die is None:
                    req.home_card, req.home_die = self.kv_cache.assign_request_home(req.req_id)

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[TRACE][scheduler][step={self.step_count}][t={self.sim_time:.6f}] {message}")

    def _accumulate_overlap_metrics(self, prefix: str, overlap) -> None:
        for field_name in [
            "dma_overlapped_time_s",
            "hbm_read_overlapped_time_s",
            "hbm_write_overlapped_time_s",
            "hbm_overlapped_time_s",
            "nvlink_overlapped_time_s",
            "pcie_overlapped_time_s",
        ]:
            metric_name = f"{prefix}_{field_name}"
            setattr(self, metric_name, getattr(self, metric_name) + getattr(overlap, field_name))
        for field_name in [
            "active_dma_domains",
            "active_hbm_read_domains",
            "active_hbm_write_domains",
            "active_nvlink_domains",
            "active_pcie_domains",
        ]:
            metric_name = {
                "active_dma_domains": f"max_{prefix}_dma_active_domains",
                "active_hbm_read_domains": f"max_{prefix}_hbm_read_active_domains",
                "active_hbm_write_domains": f"max_{prefix}_hbm_write_active_domains",
                "active_nvlink_domains": f"max_{prefix}_nvlink_active_domains",
                "active_pcie_domains": f"max_{prefix}_pcie_active_domains",
            }[field_name]
            setattr(self, metric_name, max(getattr(self, metric_name), getattr(overlap, field_name)))

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
            same_die_l1_to_l1_moves = 0
            cross_die_l1_to_l1_moves = 0
            l1_to_l2_moves = 0
            l2_to_l3_moves = 0
            l3_drop_blocks = 0
            same_die_l1_to_l2_moves = 0
            cross_die_l1_to_l2_moves = 0
            cross_card_l1_to_l2_moves = 0
            cross_die_l2_to_l2_moves = 0
            cross_card_l2_to_l2_moves = 0
            callback_same_die_blocks = 0
            callback_cross_die_blocks = 0
            callback_cross_card_blocks = 0
            callback_from_l3_blocks = 0
            callback_demands = KVTransferDomainDemands()
            eviction_demands = KVTransferDomainDemands()
            callback_time_s = 0.0
            eviction_time_s = 0.0
            same_die_time_s = 0.0
            cross_die_time_s = 0.0
            cross_card_time_s = 0.0
            pcie_time_s = 0.0
            migration_bytes = 0
            replica_fanout_time_s = 0.0
            replica_fanout_bytes = 0
            replica_fanout_blocks = 0
            for block_idx in block_indices:
                hash_id = req.trace.hash_ids[block_idx]
                if self.kv_cache is None:
                    missed_blocks += 1
                    req.computed_blocks += 1
                    continue

                access = self.kv_cache.access(
                    hash_id,
                    target_card=req.home_card,
                    target_die=req.home_die,
                    sim_time=self.sim_time,
                    chat_id=req.trace.chat_id,
                    request_type=req.trace.request_type,
                    turn_class=req.turn_class,
                )
                if self.debug and (access.hit_tier is not None and access.hit_tier != "L1"):
                    self._log(
                        "kv hit req={} hash_id={} tier={} promote_to_l1=yes".format(
                            req.req_id,
                            hash_id,
                            access.hit_tier,
                        )
                    )
                if self.debug and (
                    access.same_die_l1_to_l1 > 0
                    or access.cross_die_l1_to_l1 > 0
                    or access.l1_to_l2 > 0
                    or access.cross_die_l2_to_l2 > 0
                    or access.cross_card_l2_to_l2 > 0
                    or access.l2_to_l3 > 0
                    or access.l3_drops > 0
                ):
                    self._log(
                        "kv evict req={} hash_id={} l1_to_l1=[same_die={},cross_die={}] l1_to_l2=[same_die={},cross_die={},cross_card={}] l2_to_l2=[cross_die={},cross_card={}] l2_to_l3={} l3_drop={}".format(
                            req.req_id,
                            hash_id,
                            access.same_die_l1_to_l1,
                            access.cross_die_l1_to_l1,
                            access.same_die_l1_to_l2,
                            access.cross_die_l1_to_l2,
                            access.cross_card_l1_to_l2,
                            access.cross_die_l2_to_l2,
                            access.cross_card_l2_to_l2,
                            access.l2_to_l3,
                            access.l3_drops,
                        )
                    )
                if self.debug and (access.migration_bytes > 0):
                    self._log(
                        "kv xfer req={} hash_id={} same_die_t={:.6e} cross_die_t={:.6e} cross_card_t={:.6e} pcie_t={:.6e}".format(
                            req.req_id,
                            hash_id,
                            access.same_die_time_s,
                            access.cross_die_time_s,
                            access.cross_card_time_s,
                            access.pcie_time_s,
                        )
                    )
                req.same_die_l1_to_l1_blocks += access.same_die_l1_to_l1
                req.cross_die_l1_to_l1_blocks += access.cross_die_l1_to_l1
                req.l1_hit_blocks += access.l1_hit
                req.l2_hit_blocks += access.l2_hit
                req.l3_hit_blocks += access.l3_hit
                req.l1_to_l2_blocks += access.l1_to_l2
                req.same_die_l1_to_l2_blocks += access.same_die_l1_to_l2
                req.cross_die_l1_to_l2_blocks += access.cross_die_l1_to_l2
                req.cross_card_l1_to_l2_blocks += access.cross_card_l1_to_l2
                req.l2_to_l3_blocks += access.l2_to_l3
                req.cross_die_l2_to_l2_blocks += access.cross_die_l2_to_l2
                req.cross_card_l2_to_l2_blocks += access.cross_card_l2_to_l2
                req.l3_drop_blocks += access.l3_drops
                req.dma_blocks += access.dma_blocks
                req.pcie_blocks += access.pcie_blocks
                req.callback_same_die_blocks += access.callback_same_die
                req.callback_cross_die_blocks += access.callback_cross_die
                req.callback_cross_card_blocks += access.callback_cross_card
                req.migration_bytes += access.migration_bytes
                req.same_chat_hit_blocks += access.same_chat_hit
                req.cross_chat_hit_blocks += access.cross_chat_hit
                req.single_turn_hit_blocks += access.single_turn_hit
                req.multi_turn_hit_blocks += access.multi_turn_hit
                req.replica_hit_blocks += access.replica_hit
                req.replica_l1_hit_blocks += access.replica_l1_hit
                req.replica_l2_hit_blocks += access.replica_l2_hit
                req.avoided_cross_card_blocks += access.avoided_cross_card_callback

                l1_hit_blocks += access.l1_hit
                l2_hit_blocks += access.l2_hit
                l3_hit_blocks += access.l3_hit
                same_die_l1_to_l1_moves += access.same_die_l1_to_l1
                cross_die_l1_to_l1_moves += access.cross_die_l1_to_l1
                l1_to_l2_moves += access.l1_to_l2
                l2_to_l3_moves += access.l2_to_l3
                l3_drop_blocks += access.l3_drops
                same_die_l1_to_l2_moves += access.same_die_l1_to_l2
                cross_die_l1_to_l2_moves += access.cross_die_l1_to_l2
                cross_card_l1_to_l2_moves += access.cross_card_l1_to_l2
                cross_die_l2_to_l2_moves += access.cross_die_l2_to_l2
                cross_card_l2_to_l2_moves += access.cross_card_l2_to_l2
                callback_same_die_blocks += access.callback_same_die
                callback_cross_die_blocks += access.callback_cross_die
                callback_cross_card_blocks += access.callback_cross_card
                callback_from_l3_blocks += access.callback_from_l3
                callback_demands.merge(access.callback_demands)
                eviction_demands.merge(access.eviction_demands)
                callback_time_s += access.callback_time_s
                eviction_time_s += access.eviction_time_s
                same_die_time_s += access.same_die_time_s
                cross_die_time_s += access.cross_die_time_s
                cross_card_time_s += access.cross_card_time_s
                pcie_time_s += access.pcie_time_s
                migration_bytes += access.migration_bytes
                replica_fanout_time_s += access.replica_fanout_time_s
                replica_fanout_bytes += access.replica_fanout_bytes
                replica_fanout_blocks += access.replica_fanout_blocks
                self.avoided_cross_card_callbacks += access.avoided_cross_card_callback

                if access.is_hit:
                    req.reused_blocks += 1
                    hit_blocks += 1
                else:
                    missed_blocks += 1
                    req.computed_blocks += 1

            chunk_end = min(req.input_length, req.prefill_progress_tokens + self.prefill_chunk_tokens)
            req.prefill_progress_tokens = chunk_end

            callback_overlap = reduce_transfer_demands(callback_demands)
            eviction_overlap = reduce_transfer_demands(eviction_demands)
            prefill_latency += callback_overlap.blocking_time_s
            self.migration_time_s += callback_overlap.blocking_time_s
            self.callback_migration_time_s += callback_overlap.blocking_time_s
            self._accumulate_overlap_metrics("callback", callback_overlap)
            self.total_eviction_time_s += eviction_time_s
            self._accumulate_overlap_metrics("eviction", eviction_overlap)
            if self.charge_eviction_latency:
                prefill_latency += eviction_overlap.blocking_time_s
                self.charged_eviction_time_s += eviction_overlap.blocking_time_s
            else:
                self.overlapped_eviction_time_s += eviction_overlap.blocking_time_s
            self.dma_time_s += same_die_time_s
            self.cross_die_time_s += cross_die_time_s
            self.cross_card_time_s += cross_card_time_s + replica_fanout_time_s
            self.pcie_time_s += pcie_time_s
            self.migration_time_s += replica_fanout_time_s
            self.migration_bytes += migration_bytes + replica_fanout_bytes
            self.replica_fanout_time_s += replica_fanout_time_s
            self.replica_fanout_bytes += replica_fanout_bytes
            self.replica_fanout_blocks += replica_fanout_blocks
            prefill_latency += replica_fanout_time_s
            self.dma_transfer_blocks += same_die_l1_to_l1_moves + same_die_l1_to_l2_moves + callback_same_die_blocks
            self.pcie_transfer_blocks += callback_from_l3_blocks + l2_to_l3_moves
            self.cross_die_transfer_blocks += (
                cross_die_l1_to_l1_moves + cross_die_l1_to_l2_moves + cross_die_l2_to_l2_moves + callback_cross_die_blocks
            )
            self.cross_card_transfer_blocks += (
                cross_card_l1_to_l2_moves + cross_card_l2_to_l2_moves + callback_cross_card_blocks + replica_fanout_blocks
            )
            self.callback_dma_transfer_blocks += callback_same_die_blocks
            self.callback_pcie_transfer_blocks += callback_from_l3_blocks
            self.eviction_dma_transfer_blocks += same_die_l1_to_l1_moves + same_die_l1_to_l2_moves
            self.eviction_pcie_transfer_blocks += l2_to_l3_moves

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
                "prefill req={} chunk_blocks={} hit={} miss={} hit_tiers=[{},{},{}] moves=[l1_to_l1_same_die={},l1_to_l1_cross_die={},l1_to_l2_same_die={},l1_to_l2_cross_die={},l1_to_l2_cross_card={},l2_to_l2_cross_die={},l2_to_l2_cross_card={},l2_to_l3={},l3_drop={}] xfer=[cb_same_die={},cb_cross_die={},cb_cross_card={},cb_l3={}] progress={}/{}".format(
                    req.req_id,
                    len(block_indices),
                    hit_blocks,
                    missed_blocks,
                    l1_hit_blocks,
                    l2_hit_blocks,
                    l3_hit_blocks,
                    same_die_l1_to_l1_moves,
                    cross_die_l1_to_l1_moves,
                    same_die_l1_to_l2_moves,
                    cross_die_l1_to_l2_moves,
                    cross_card_l1_to_l2_moves,
                    cross_die_l2_to_l2_moves,
                    cross_card_l2_to_l2_moves,
                    l2_to_l3_moves,
                    l3_drop_blocks,
                    callback_same_die_blocks,
                    callback_cross_die_blocks,
                    callback_cross_card_blocks,
                    callback_from_l3_blocks,
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
            "callback_migration_time_s": self.callback_migration_time_s,
            "total_eviction_time_s": self.total_eviction_time_s,
            "charged_eviction_time_s": self.charged_eviction_time_s,
            "overlapped_eviction_time_s": self.overlapped_eviction_time_s,
            "callback_dma_overlapped_time_s": self.callback_dma_overlapped_time_s,
            "callback_hbm_read_overlapped_time_s": self.callback_hbm_read_overlapped_time_s,
            "callback_hbm_write_overlapped_time_s": self.callback_hbm_write_overlapped_time_s,
            "callback_hbm_overlapped_time_s": self.callback_hbm_overlapped_time_s,
            "callback_nvlink_overlapped_time_s": self.callback_nvlink_overlapped_time_s,
            "callback_pcie_overlapped_time_s": self.callback_pcie_overlapped_time_s,
            "eviction_dma_overlapped_time_s": self.eviction_dma_overlapped_time_s,
            "eviction_hbm_read_overlapped_time_s": self.eviction_hbm_read_overlapped_time_s,
            "eviction_hbm_write_overlapped_time_s": self.eviction_hbm_write_overlapped_time_s,
            "eviction_hbm_overlapped_time_s": self.eviction_hbm_overlapped_time_s,
            "eviction_nvlink_overlapped_time_s": self.eviction_nvlink_overlapped_time_s,
            "eviction_pcie_overlapped_time_s": self.eviction_pcie_overlapped_time_s,
            "max_callback_dma_active_domains": self.max_callback_dma_active_domains,
            "max_callback_hbm_read_active_domains": self.max_callback_hbm_read_active_domains,
            "max_callback_hbm_write_active_domains": self.max_callback_hbm_write_active_domains,
            "max_callback_nvlink_active_domains": self.max_callback_nvlink_active_domains,
            "max_callback_pcie_active_domains": self.max_callback_pcie_active_domains,
            "max_eviction_dma_active_domains": self.max_eviction_dma_active_domains,
            "max_eviction_hbm_read_active_domains": self.max_eviction_hbm_read_active_domains,
            "max_eviction_hbm_write_active_domains": self.max_eviction_hbm_write_active_domains,
            "max_eviction_nvlink_active_domains": self.max_eviction_nvlink_active_domains,
            "max_eviction_pcie_active_domains": self.max_eviction_pcie_active_domains,
            "dma_time_s": self.dma_time_s,
            "pcie_time_s": self.pcie_time_s,
            "migration_bytes": self.migration_bytes,
            "dma_transfer_blocks": self.dma_transfer_blocks,
            "pcie_transfer_blocks": self.pcie_transfer_blocks,
            "cross_die_transfer_blocks": self.cross_die_transfer_blocks,
            "cross_card_transfer_blocks": self.cross_card_transfer_blocks,
            "callback_dma_transfer_blocks": self.callback_dma_transfer_blocks,
            "callback_pcie_transfer_blocks": self.callback_pcie_transfer_blocks,
            "eviction_dma_transfer_blocks": self.eviction_dma_transfer_blocks,
            "eviction_pcie_transfer_blocks": self.eviction_pcie_transfer_blocks,
            "cross_die_time_s": self.cross_die_time_s,
            "cross_card_time_s": self.cross_card_time_s,
            "replica_fanout_time_s": self.replica_fanout_time_s,
            "replica_fanout_bytes": self.replica_fanout_bytes,
            "replica_fanout_blocks": self.replica_fanout_blocks,
            "avoided_cross_card_callbacks": self.avoided_cross_card_callbacks,
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
