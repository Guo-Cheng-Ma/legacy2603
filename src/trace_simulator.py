import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from .continuous_scheduler import ContinuousScheduler
from .config import get_hetero_kv_capacities, load_hetero_kv_arch_config, get_hetero_transfer_bandwidths
from .kv_cache import KVCacheManager
from .request_state import RequestState
from .static_batch_scheduler import StaticBatchScheduler
from .trace_loader import load_request_states
from .type import DataType, DeviceType


def _activation_bytes(dtype: DataType) -> int:
    if dtype in [DataType.W16A16, DataType.W8A16]:
        return 2
    return 1


def _percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil(ratio * len(ordered))) - 1
    idx = min(max(idx, 0), len(ordered) - 1)
    return ordered[idx]


def _build_kv_cache(system, kv_arch_cfg: Dict) -> Tuple[KVCacheManager, Dict]:
    a_byte = _activation_bytes(system.model.dtype)
    kv_bytes_per_token = system.model.ndec * 2 * system.model.hdim * a_byte
    kv_bytes_per_block = kv_bytes_per_token * 16
    weight_bytes, _, _ = system.get_required_mem_capacity(batch_size=1, lin=1, lout=1)
    tier_caps = get_hetero_kv_capacities(weight_bytes_total=weight_bytes, hetero_kv_arch=kv_arch_cfg)
    bw_cfg = get_hetero_transfer_bandwidths(kv_arch_cfg)

    cache = KVCacheManager(
        kv_bytes_per_block=kv_bytes_per_block,
        topology={
            "num_cards": kv_arch_cfg["NUM_CARDS"],
            "num_die_packages_per_card": tier_caps["num_die_packages_per_card"],
            "banks_per_die": tier_caps["banks_per_die"],
            "l1_die_ids": tier_caps["l1_die_ids"],
            "l2_die_ids": tier_caps["l2_die_ids"],
            "l1_bank_capacity_bytes": max(tier_caps["l1_bank_capacity_bytes"], kv_bytes_per_block),
            "l2_kv_die_capacity_bytes": tier_caps["l2_kv_die_capacity_bytes"],
            "l3_kv_bytes": tier_caps["l3_kv_bytes"],
            "card_total_bw_bps": system.devices['GPU'].peak_memory_bandwidth,
            "nvlink_bw_bps": system.devices['GPU'].max_interface_bandwidth,
            "dma_bw_bps": bw_cfg["dma_bw_bps"],
            "pcie_bw_bps": bw_cfg["pcie_bw_bps"],
        },
    )
    return cache, tier_caps


def _collect_system_metadata(system, requests: List[RequestState], max_batch_size: int) -> Dict:
    gib = 1024.0 * 1024.0 * 1024.0
    gpu = system.devices['GPU']
    acc = system.devices['Acc']

    total_cap = gpu.aggregate_memory_capacity
    if system.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
        total_cap += acc.aggregate_memory_capacity

    bw_scale = 0.0
    if gpu.peak_memory_bandwidth > 0:
        bw_scale = acc.peak_memory_bandwidth / gpu.peak_memory_bandwidth

    opb = 0.0
    if gpu.peak_memory_bandwidth > 0:
        opb = gpu.peak_flops / gpu.peak_memory_bandwidth
        if system.model.dtype in [DataType.W8A8, DataType.W8A16]:
            opb *= 2

    hw = system.hetero_name.name
    if system.hetero_name == DeviceType.PIM:
        hw = acc.pim_type.name

    max_input = max([req.input_length for req in requests], default=0)
    max_output = max([req.output_length for req in requests], default=0)
    required_cap_est_gb = 0.0
    if requests:
        weight, kv, temp = system.get_required_mem_capacity(
            batch_size=max_batch_size,
            lin=max(1, max_input),
            lout=max(1, max_output),
        )
        required_cap_est_gb = (weight + kv + temp) / gib

    return {
        'model': system.model.name,
        'dtype': system.model.dtype.name,
        'xpu': gpu.name.name,
        'hw': hw,
        'cores': gpu.num_xpu,
        'cap': total_cap / gib,
        'bw': bw_scale,
        'sys_opb': opb,
        'power_constraint': bool(getattr(acc, "power_constraint", False)),
        'gqa_size': getattr(system.model, "gqa_size", 0),
        'ndec': system.model.ndec,
        'hdim': system.model.hdim,
        'num_heads': system.model.num_heads,
        'dhead': system.model.dhead,
        'required_cap_est_gb': required_cap_est_gb,
    }


def _summarize_requests(requests: List[RequestState], total_time: float) -> Dict:
    latencies = [req.latency for req in requests if req.latency is not None]
    ttfts = [req.ttft for req in requests if req.ttft is not None]
    queue_delays = [
        max(0.0, req.start_time - req.timestamp)
        for req in requests
        if req.start_time is not None
    ]
    input_tokens = [req.input_length for req in requests]
    output_tokens = [req.output_length for req in requests]
    prompt_blocks = [req.prompt_block_count for req in requests]
    arrival_ts = [req.timestamp for req in requests]

    num_requests = len(requests)
    total_generated_tokens = sum(req.generated_tokens for req in requests)
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    total_prompt_blocks = sum(prompt_blocks)
    total_reused_blocks = sum(req.reused_blocks for req in requests)
    total_computed_blocks = sum(req.computed_blocks for req in requests)
    num_multiturn = sum(1 for req in requests if req.trace.turn > 1)
    num_singleturn = num_requests - num_multiturn

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0
    avg_queue_delay = (sum(queue_delays) / len(queue_delays)) if queue_delays else 0.0

    throughput_req = (num_requests / total_time) if total_time > 0 else 0.0
    throughput_tok = (total_generated_tokens / total_time) if total_time > 0 else 0.0
    arrival_span_s = (max(arrival_ts) - min(arrival_ts)) if len(arrival_ts) >= 2 else 0.0
    arrival_rate_req = (num_requests / arrival_span_s) if arrival_span_s > 0 else 0.0

    return {
        'num_requests': num_requests,
        'total_time_s': total_time,
        'arrival_span_s': arrival_span_s,
        'arrival_rate_req_per_s': arrival_rate_req,
        'avg_latency_s': avg_latency,
        'p50_latency_s': _percentile(latencies, 0.50),
        'p95_latency_s': _percentile(latencies, 0.95),
        'avg_ttft_s': avg_ttft,
        'p50_ttft_s': _percentile(ttfts, 0.50),
        'p95_ttft_s': _percentile(ttfts, 0.95),
        'avg_queue_delay_s': avg_queue_delay,
        'p50_queue_delay_s': _percentile(queue_delays, 0.50),
        'p95_queue_delay_s': _percentile(queue_delays, 0.95),
        'throughput_req_per_s': throughput_req,
        'throughput_tok_per_s': throughput_tok,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_input_tokens': (total_input_tokens / num_requests) if num_requests > 0 else 0.0,
        'avg_output_tokens': (total_output_tokens / num_requests) if num_requests > 0 else 0.0,
        'p50_input_tokens': _percentile(input_tokens, 0.50),
        'p95_input_tokens': _percentile(input_tokens, 0.95),
        'p50_output_tokens': _percentile(output_tokens, 0.50),
        'p95_output_tokens': _percentile(output_tokens, 0.95),
        'max_input_tokens': max(input_tokens) if input_tokens else 0,
        'max_output_tokens': max(output_tokens) if output_tokens else 0,
        'total_prompt_blocks': total_prompt_blocks,
        'avg_prompt_blocks': (total_prompt_blocks / num_requests) if num_requests > 0 else 0.0,
        'num_multiturn_requests': num_multiturn,
        'num_singleturn_requests': num_singleturn,
        'total_generated_tokens': total_generated_tokens,
        'kv_hit_blocks': total_reused_blocks,
        'kv_miss_blocks': total_computed_blocks,
    }


def run_trace_simulation(
    system,
    trace_file: str,
    max_batch_size: int,
    prefill_chunk_tokens: int,
    kv_arch_config: str,
    timestamp_scaling: float = 1.0,
    trace_debug: bool = False,
    trace_debug_interval: int = 100,
    pipe_level: bool = False,
    is_parallel: bool = False,
    power_constraint: bool = False,
    system_name: str = "",
    gpu_name: str = "",
    pim_type: str = "",
    trace_scheduler: str = "continuous",
):
    if trace_debug:
        print(
            "[TRACE][sim] start trace_file={} max_batch_size={} prefill_chunk_tokens={} kv_arch_config={} timestamp_scaling={}".format(
                trace_file,
                max_batch_size,
                prefill_chunk_tokens,
                kv_arch_config,
                timestamp_scaling,
            )
        )
    kv_arch_cfg = load_hetero_kv_arch_config(kv_arch_config)
    bw_cfg = get_hetero_transfer_bandwidths(kv_arch_cfg)
    requests = load_request_states(trace_file, timestamp_scale=timestamp_scaling)
    if trace_debug:
        first_ts = requests[0].timestamp if requests else 0.0
        last_ts = requests[-1].timestamp if requests else 0.0
        print(
            "[TRACE][sim] parsed requests={} first_ts={:.6f} last_ts={:.6f}".format(
                len(requests),
                first_ts,
                last_ts,
            )
        )
    kv_cache, kv_caps = _build_kv_cache(system, kv_arch_cfg)
    if trace_debug:
        kv_info = kv_cache.snapshot()
        print(
            "[TRACE][sim] kv cache l1={} l2={} l3={} kv_bytes_per_block={} weight_reserved={}".format(
                kv_info["l1_capacity_bytes"],
                kv_info["l2_capacity_bytes"],
                kv_info["l3_capacity_bytes"],
                kv_cache.kv_bytes_per_block,
                kv_caps["weight_bytes_total"],
            )
        )
        print(
            "[TRACE][sim] kv arch cards={} dma_bw_bps={} pcie_bw_bps={}".format(
                kv_arch_cfg["NUM_CARDS"],
                bw_cfg["dma_bw_bps"],
                bw_cfg["pcie_bw_bps"],
            )
        )

    if trace_scheduler == "continuous":
        scheduler = ContinuousScheduler(
            requests=requests,
            max_batch_size=max_batch_size,
            system=system,
            kv_cache=kv_cache,
            prefill_chunk_tokens=prefill_chunk_tokens,
            pipe_level=pipe_level,
            parallel_ff=is_parallel,
            debug=trace_debug,
            debug_interval=trace_debug_interval,
            kv_arch_cfg=kv_arch_cfg,
        )
    elif trace_scheduler == "static":
        scheduler = StaticBatchScheduler(
            requests=requests,
            max_batch_size=max_batch_size,
            system=system,
            kv_cache=kv_cache,
            pipe_level=pipe_level,
            parallel_ff=is_parallel,
            debug=trace_debug,
            debug_interval=trace_debug_interval,
            kv_arch_cfg=kv_arch_cfg,
        )
    else:
        raise ValueError(f"unsupported trace_scheduler: {trace_scheduler}")

    if requests and requests[0].timestamp > 0:
        scheduler.sim_time = requests[0].timestamp

    snapshot = scheduler.run()

    completed = scheduler.completed_requests()
    request_rows = []
    for req in completed:
        kv_total = req.reused_blocks + req.computed_blocks
        request_rows.append({
            'req_id': req.req_id,
            'chat_id': req.trace.chat_id,
            'parent_chat_id': req.trace.parent_chat_id,
            'request_type': req.trace.request_type,
            'turn': req.trace.turn,
            'arrival_s': req.timestamp,
            'start_s': req.start_time,
            'home_card': req.home_card,
            'home_die': req.home_die,
            'batch_id': req.tags.get("batch_id"),
            'batch_start_s': req.tags.get("batch_start_s"),
            'batch_finish_s': req.tags.get("batch_finish_s"),
            'queue_delay_s': (req.start_time - req.timestamp) if req.start_time is not None else None,
            'first_token_s': req.first_token_time,
            'finish_s': req.finish_time,
            'latency_s': req.latency,
            'ttft_s': req.ttft,
            'input_tokens': req.input_length,
            'output_tokens': req.output_length,
            'prompt_blocks': req.prompt_block_count,
            'reused_blocks': req.reused_blocks,
            'computed_blocks': req.computed_blocks,
            'l1_hit_blocks': req.l1_hit_blocks,
            'l2_hit_blocks': req.l2_hit_blocks,
            'l3_hit_blocks': req.l3_hit_blocks,
            'same_die_l1_to_l1_blocks': req.same_die_l1_to_l1_blocks,
            'cross_die_l1_to_l1_blocks': req.cross_die_l1_to_l1_blocks,
            'l1_to_l2_blocks': req.l1_to_l2_blocks,
            'same_die_l1_to_l2_blocks': req.same_die_l1_to_l2_blocks,
            'cross_die_l1_to_l2_blocks': req.cross_die_l1_to_l2_blocks,
            'cross_card_l1_to_l2_blocks': req.cross_card_l1_to_l2_blocks,
            'cross_die_l2_to_l2_blocks': req.cross_die_l2_to_l2_blocks,
            'cross_card_l2_to_l2_blocks': req.cross_card_l2_to_l2_blocks,
            'l2_to_l3_blocks': req.l2_to_l3_blocks,
            'l3_drop_blocks': req.l3_drop_blocks,
            'migration_bytes': req.migration_bytes,
            'kv_hit_rate': (req.reused_blocks / kv_total) if kv_total > 0 else 0.0,
            'dma_blocks': req.dma_blocks,
            'pcie_blocks': req.pcie_blocks,
            'callback_same_die_blocks': req.callback_same_die_blocks,
            'callback_cross_die_blocks': req.callback_cross_die_blocks,
            'callback_cross_card_blocks': req.callback_cross_card_blocks,
        })

    summary = _summarize_requests(completed, snapshot.sim_time)
    summary.update(_collect_system_metadata(system, completed, max_batch_size))
    summary.update(kv_cache.snapshot())
    summary['mode'] = "trace"
    summary['trace_scheduler_mode'] = trace_scheduler
    summary['trace_file'] = trace_file
    summary['input_request_name'] = Path(trace_file).stem
    summary['system'] = system_name
    summary['gpu_name'] = gpu_name
    pim_type_map = {"bank": "BA", "bg": "BG", "buffer": "BUFFER"}
    summary['pim_type'] = pim_type_map.get(str(pim_type).lower(), str(pim_type).upper()) if pim_type else ""
    summary['pipe_level'] = int(bool(pipe_level))
    summary['is_parallel'] = int(bool(is_parallel))
    summary['power_constraint'] = bool(power_constraint)
    summary['max_batch_size'] = int(max_batch_size)
    summary['prefill_chunk_tokens'] = int(prefill_chunk_tokens)
    summary['timestamp_scaling'] = float(timestamp_scaling)
    summary['kv_capacity_mode'] = "hetero_3tier"
    summary['kv_arch_config'] = kv_arch_config
    summary['num_cards_cfg'] = kv_arch_cfg["NUM_CARDS"]
    summary['num_die_packages_per_card_cfg'] = kv_caps["num_die_packages_per_card"]
    summary['banks_per_die_cfg'] = kv_caps["banks_per_die"]
    summary['dma_bw_bps_cfg'] = bw_cfg["dma_bw_bps"]
    summary['pcie_bw_bps_cfg'] = bw_cfg["pcie_bw_bps"]
    summary['gpu_mem_per_card_gb_cfg'] = kv_caps["gpu_mem_per_card_gb"]
    summary['hispeed_kv_per_card_gb_cfg'] = kv_caps["hispeed_kv_per_card_gb"]
    summary['hicap_total_per_card_gb_cfg'] = kv_caps["hicap_total_per_card_gb"]
    summary['weight_reserved_bytes'] = kv_caps["weight_bytes_total"]
    summary['l1_kv_capacity_bytes_cfg'] = kv_caps["l1_kv_bytes"]
    summary['l2_total_bytes_cfg'] = kv_caps["l2_total_bytes"]
    summary['l2_kv_capacity_bytes_cfg'] = kv_caps["l2_kv_bytes"]
    summary['l3_kv_capacity_bytes_cfg'] = kv_caps["l3_kv_bytes"]
    summary['kv_bytes_per_block'] = kv_cache.kv_bytes_per_block
    summary['resident_blocks'] = summary.get('num_blocks', 0)
    summary['kv_cache_capacity_bytes'] = summary.get('capacity_bytes', 0)
    summary['kv_cache_capacity_gb'] = summary['kv_cache_capacity_bytes'] / (1024.0 * 1024.0 * 1024.0)
    summary['kv_cache_used_gb'] = summary.get('used_bytes', 0) / (1024.0 * 1024.0 * 1024.0)
    summary['kv_cache_free_gb'] = summary.get('free_bytes', 0) / (1024.0 * 1024.0 * 1024.0)
    summary['trace_window_s'] = snapshot.sim_time

    total_kv = summary['kv_hit_blocks'] + summary['kv_miss_blocks']
    summary['kv_hit_rate'] = (summary['kv_hit_blocks'] / total_kv) if total_kv > 0 else 0.0
    summary['kv_evictions'] = summary['evictions']
    total_tier_hits = summary.get('l1_hits', 0) + summary.get('l2_hits', 0) + summary.get('l3_hits', 0)
    summary['l1_hit_rate'] = (summary.get('l1_hits', 0) / total_tier_hits) if total_tier_hits > 0 else 0.0
    summary['l2_hit_rate'] = (summary.get('l2_hits', 0) / total_tier_hits) if total_tier_hits > 0 else 0.0
    summary['l3_hit_rate'] = (summary.get('l3_hits', 0) / total_tier_hits) if total_tier_hits > 0 else 0.0
    summary['l1_used_ratio'] = (
        summary.get('l1_used_bytes', 0) / summary.get('l1_capacity_bytes', 1)
    ) if summary.get('l1_capacity_bytes', 0) > 0 else 0.0
    summary['l2_used_ratio'] = (
        summary.get('l2_used_bytes', 0) / summary.get('l2_capacity_bytes', 1)
    ) if summary.get('l2_capacity_bytes', 0) > 0 else 0.0
    summary['l3_used_ratio'] = (
        summary.get('l3_used_bytes', 0) / summary.get('l3_capacity_bytes', 1)
    ) if summary.get('l3_capacity_bytes', 0) > 0 else 0.0
    if hasattr(scheduler, "batch_stats"):
        summary.update(scheduler.batch_stats())
    summary.update(scheduler.energy_snapshot())
    summary['dma_transfers'] = summary.get('dma_transfer_blocks', sum(req['dma_blocks'] for req in request_rows))
    summary['pcie_transfers'] = summary.get('pcie_transfer_blocks', sum(req['pcie_blocks'] for req in request_rows))
    summary['dma_time_s'] = summary.get('dma_time_s', 0.0)
    summary['pcie_time_s'] = summary.get('pcie_time_s', 0.0)
    summary['migration_time_s'] = summary.get('migration_time_s', 0.0)
    summary['migration_energy_nj'] = summary.get('migration_energy_nj', 0.0)
    summary['Lin'] = summary.get('avg_input_tokens', 0.0)
    summary['Lout'] = summary.get('avg_output_tokens', 0.0)
    summary['bs'] = summary.get('max_batch_size', 0)
    summary['required_cap'] = summary.get('required_cap_est_gb', 0.0)
    summary['s_time'] = summary.get('prefill_work_time_s', 0.0) * 1000.0
    summary['g_time (ms)'] = summary.get('decode_work_time_s', 0.0) * 1000.0
    summary['g_energy (nJ)'] = summary['total_energy_nj']
    summary['g_dram_energy'] = summary.get('total_dram_energy_nj', 0.0)
    summary['g_l2_energy'] = summary.get('total_l2_energy_nj', 0.0)
    summary['g_l1_energy'] = summary.get('total_l1_energy_nj', 0.0)
    summary['g_reg_energy'] = summary.get('total_reg_energy_nj', 0.0)
    summary['g_alu_energy'] = summary.get('total_alu_energy_nj', 0.0)
    summary['g_comm_energy'] = summary.get('total_comm_energy_nj', 0.0)
    if trace_debug:
        print(
            "[TRACE][sim] finished total_time_s={:.6f} kv_hit_blocks={} kv_miss_blocks={} kv_evictions={} total_energy_nj={:.3f}".format(
                summary['total_time_s'],
                summary['kv_hit_blocks'],
                summary['kv_miss_blocks'],
                summary['kv_evictions'],
                summary['total_energy_nj'],
            )
        )

    return {
        'summary': summary,
        'requests': request_rows,
        'snapshot': snapshot,
    }


def _ordered_record(row: Dict, columns: List[str]) -> Dict:
    return {key: row.get(key, None) for key in columns}


def _write_yaml_record(path: str, row: Dict, columns: List[str]) -> None:
    ordered_row = _ordered_record(row, columns)
    try:
        import yaml
        yaml_text = yaml.safe_dump(
            ordered_row,
            sort_keys=False,
            default_flow_style=False,
            width=4096,
        )
    except ImportError:
        yaml_lines = [
            "{}: {}".format(key, json.dumps(value, ensure_ascii=False))
            for key, value in ordered_row.items()
        ]
        yaml_text = "\n".join(yaml_lines) + "\n"

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(yaml_text)


def _write_jsonl_rows(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(_ordered_record(row, columns), ensure_ascii=False))
            handle.write("\n")


def write_trace_outputs(result, summary_path='trace_summary.yaml', requests_path='trace_requests.jsonl'):
    summary = result['summary']
    request_rows = result['requests']

    summary_cols = [
        'mode',
        'system',
        'gpu_name',
        'pim_type',
        'trace_file',
        'input_request_name',
        'model',
        'dtype',
        'xpu',
        'hw',
        'cores',
        'cap',
        'bw',
        'sys_opb',
        'power_constraint',
        'pipe_level',
        'is_parallel',
        'gqa_size',
        'ndec',
        'hdim',
        'num_heads',
        'dhead',
        'max_batch_size',
        'trace_scheduler_mode',
        'num_batches',
        'avg_batch_size',
        'avg_batch_max_output_tokens',
        'decode_padded_tokens',
        'prefill_chunk_tokens',
        'timestamp_scaling',
        'kv_capacity_mode',
        'kv_arch_config',
        'num_cards_cfg',
        'num_die_packages_per_card_cfg',
        'banks_per_die_cfg',
        'dma_bw_bps_cfg',
        'pcie_bw_bps_cfg',
        'gpu_mem_per_card_gb_cfg',
        'hispeed_kv_per_card_gb_cfg',
        'hicap_total_per_card_gb_cfg',
        'weight_reserved_bytes',
        'l1_kv_capacity_bytes_cfg',
        'l2_total_bytes_cfg',
        'l2_kv_capacity_bytes_cfg',
        'l3_kv_capacity_bytes_cfg',
        'required_cap_est_gb',
        'required_cap',
        'Lin',
        'Lout',
        'bs',
        'kv_bytes_per_block',
        'kv_cache_capacity_bytes',
        'kv_cache_capacity_gb',
        'kv_cache_used_gb',
        'kv_cache_free_gb',
        'num_requests',
        'total_time_s',
        'num_singleturn_requests',
        'num_multiturn_requests',
        'trace_window_s',
        'arrival_span_s',
        'arrival_rate_req_per_s',
        'total_input_tokens',
        'total_output_tokens',
        'avg_input_tokens',
        'avg_output_tokens',
        'p50_input_tokens',
        'p95_input_tokens',
        'p50_output_tokens',
        'p95_output_tokens',
        'max_input_tokens',
        'max_output_tokens',
        'total_prompt_blocks',
        'avg_prompt_blocks',
        'avg_latency_s',
        'p50_latency_s',
        'p95_latency_s',
        'avg_ttft_s',
        'p50_ttft_s',
        'p95_ttft_s',
        'avg_queue_delay_s',
        'p50_queue_delay_s',
        'p95_queue_delay_s',
        'throughput_req_per_s',
        'throughput_tok_per_s',
        'kv_hit_blocks',
        'kv_miss_blocks',
        'kv_hit_rate',
        'kv_evictions',
        'resident_blocks',
        'l1_num_blocks',
        'l2_num_blocks',
        'l3_num_blocks',
        'l1_hits',
        'l2_hits',
        'l3_hits',
        'l1_hit_rate',
        'l2_hit_rate',
        'l3_hit_rate',
        'l1_to_l2',
        'l2_to_l3',
        'l3_drops',
        'l1_used_bytes',
        'l2_used_bytes',
        'l3_used_bytes',
        'l1_capacity_bytes',
        'l2_capacity_bytes',
        'l3_capacity_bytes',
        'l1_used_ratio',
        'l2_used_ratio',
        'l3_used_ratio',
        'used_bytes',
        'free_bytes',
        'capacity_bytes',
        'dma_transfers',
        'pcie_transfers',
        'cross_die_transfer_blocks',
        'cross_card_transfer_blocks',
        'callback_dma_transfer_blocks',
        'callback_pcie_transfer_blocks',
        'eviction_dma_transfer_blocks',
        'eviction_pcie_transfer_blocks',
        'dma_time_s',
        'pcie_time_s',
        'cross_die_time_s',
        'cross_card_time_s',
        'migration_time_s',
        'callback_migration_time_s',
        'total_eviction_time_s',
        'charged_eviction_time_s',
        'overlapped_eviction_time_s',
        'callback_dma_overlapped_time_s',
        'callback_hbm_read_overlapped_time_s',
        'callback_hbm_write_overlapped_time_s',
        'callback_hbm_overlapped_time_s',
        'callback_nvlink_overlapped_time_s',
        'callback_pcie_overlapped_time_s',
        'eviction_dma_overlapped_time_s',
        'eviction_hbm_read_overlapped_time_s',
        'eviction_hbm_write_overlapped_time_s',
        'eviction_hbm_overlapped_time_s',
        'eviction_nvlink_overlapped_time_s',
        'eviction_pcie_overlapped_time_s',
        'max_callback_dma_active_domains',
        'max_callback_hbm_read_active_domains',
        'max_callback_hbm_write_active_domains',
        'max_callback_nvlink_active_domains',
        'max_callback_pcie_active_domains',
        'max_eviction_dma_active_domains',
        'max_eviction_hbm_read_active_domains',
        'max_eviction_hbm_write_active_domains',
        'max_eviction_nvlink_active_domains',
        'max_eviction_pcie_active_domains',
        'migration_bytes',
        'prefill_work_time_s',
        'decode_work_time_s',
        'prefill_steps',
        'decode_steps',
        's_time',
        'g_time (ms)',
        'prefill_energy_nj',
        'decode_energy_nj',
        'model_energy_nj',
        'migration_energy_nj',
        'dma_energy_nj',
        'pcie_energy_nj',
        'total_energy_nj',
        'prefill_dram_energy_nj',
        'prefill_l2_energy_nj',
        'prefill_l1_energy_nj',
        'prefill_reg_energy_nj',
        'prefill_alu_energy_nj',
        'prefill_comm_energy_nj',
        'decode_dram_energy_nj',
        'decode_l2_energy_nj',
        'decode_l1_energy_nj',
        'decode_reg_energy_nj',
        'decode_alu_energy_nj',
        'decode_comm_energy_nj',
        'total_dram_energy_nj',
        'total_l2_energy_nj',
        'total_l1_energy_nj',
        'total_reg_energy_nj',
        'total_alu_energy_nj',
        'total_comm_energy_nj',
        'g_energy (nJ)',
        'g_dram_energy',
        'g_l2_energy',
        'g_l1_energy',
        'g_reg_energy',
        'g_alu_energy',
        'g_comm_energy',
    ]
    _write_yaml_record(summary_path, summary, summary_cols)

    request_cols = [
        'req_id',
        'chat_id',
        'parent_chat_id',
        'request_type',
        'turn',
        'arrival_s',
        'start_s',
        'home_card',
        'home_die',
        'batch_id',
        'batch_start_s',
        'batch_finish_s',
        'queue_delay_s',
        'first_token_s',
        'finish_s',
        'latency_s',
        'ttft_s',
        'input_tokens',
        'output_tokens',
        'prompt_blocks',
        'reused_blocks',
        'computed_blocks',
        'l1_hit_blocks',
        'l2_hit_blocks',
        'l3_hit_blocks',
        'same_die_l1_to_l1_blocks',
        'cross_die_l1_to_l1_blocks',
        'l1_to_l2_blocks',
        'same_die_l1_to_l2_blocks',
        'cross_die_l1_to_l2_blocks',
        'cross_card_l1_to_l2_blocks',
        'cross_die_l2_to_l2_blocks',
        'cross_card_l2_to_l2_blocks',
        'l2_to_l3_blocks',
        'l3_drop_blocks',
        'migration_bytes',
        'kv_hit_rate',
        'dma_blocks',
        'pcie_blocks',
        'callback_same_die_blocks',
        'callback_cross_die_blocks',
        'callback_cross_card_blocks',
    ]
    _write_jsonl_rows(requests_path, request_rows, request_cols)


def run_trace_mode(system, args):
    return run_trace_simulation(
        system=system,
        trace_file=args.trace_file,
        max_batch_size=args.max_batch_size,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        kv_arch_config=args.kv_arch_config,
        timestamp_scaling=args.timestamp_scaling,
        trace_debug=args.trace_debug,
        trace_debug_interval=args.trace_debug_interval,
        pipe_level=args.pipeopt,
        is_parallel=args.ffopt,
        power_constraint=args.powerlimit,
        system_name=args.system,
        gpu_name=args.gpu,
        pim_type=args.pim,
        trace_scheduler=args.trace_scheduler,
    )
