import csv
import math
from typing import Dict, List

from .continuous_scheduler import ContinuousScheduler
from .kv_cache import KVCacheManager
from .request_state import RequestState
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


def _build_kv_cache(system, kv_hbm_ratio: float) -> KVCacheManager:
    kv_hbm_ratio = max(0.0, min(1.0, float(kv_hbm_ratio)))

    total_hbm = system.devices['GPU'].aggregate_memory_capacity
    if system.hetero_name == DeviceType.PIM:
        total_hbm += system.devices['Acc'].aggregate_memory_capacity

    capacity_bytes = int(total_hbm * kv_hbm_ratio)

    a_byte = _activation_bytes(system.model.dtype)
    kv_bytes_per_token = system.model.ndec * 2 * system.model.hdim * a_byte
    kv_bytes_per_block = kv_bytes_per_token * 16

    return KVCacheManager(
        capacity_bytes=max(capacity_bytes, kv_bytes_per_block),
        kv_bytes_per_block=kv_bytes_per_block,
    )


def _summarize_requests(requests: List[RequestState], total_time: float) -> Dict:
    latencies = [req.latency for req in requests if req.latency is not None]
    ttfts = [req.ttft for req in requests if req.ttft is not None]

    num_requests = len(requests)
    total_generated_tokens = sum(req.generated_tokens for req in requests)
    total_reused_blocks = sum(req.reused_blocks for req in requests)
    total_computed_blocks = sum(req.computed_blocks for req in requests)

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0

    throughput_req = (num_requests / total_time) if total_time > 0 else 0.0
    throughput_tok = (total_generated_tokens / total_time) if total_time > 0 else 0.0

    return {
        'num_requests': num_requests,
        'total_time_s': total_time,
        'avg_latency_s': avg_latency,
        'p50_latency_s': _percentile(latencies, 0.50),
        'p95_latency_s': _percentile(latencies, 0.95),
        'avg_ttft_s': avg_ttft,
        'throughput_req_per_s': throughput_req,
        'throughput_tok_per_s': throughput_tok,
        'total_generated_tokens': total_generated_tokens,
        'kv_hit_blocks': total_reused_blocks,
        'kv_miss_blocks': total_computed_blocks,
    }


def run_trace_simulation(
    system,
    trace_file: str,
    max_batch_size: int,
    prefill_chunk_tokens: int,
    kv_hbm_ratio: float,
):
    requests = load_request_states(trace_file)
    kv_cache = _build_kv_cache(system, kv_hbm_ratio)

    scheduler = ContinuousScheduler(
        requests=requests,
        max_batch_size=max_batch_size,
        system=system,
        kv_cache=kv_cache,
        prefill_chunk_tokens=prefill_chunk_tokens,
    )

    if requests and requests[0].timestamp > 0:
        scheduler.sim_time = requests[0].timestamp

    snapshot = scheduler.run()

    completed = scheduler.completed_requests()
    request_rows = []
    for req in completed:
        request_rows.append({
            'req_id': req.req_id,
            'chat_id': req.trace.chat_id,
            'parent_chat_id': req.trace.parent_chat_id,
            'arrival_s': req.timestamp,
            'start_s': req.start_time,
            'first_token_s': req.first_token_time,
            'finish_s': req.finish_time,
            'latency_s': req.latency,
            'ttft_s': req.ttft,
            'input_tokens': req.input_length,
            'output_tokens': req.output_length,
            'reused_blocks': req.reused_blocks,
            'computed_blocks': req.computed_blocks,
            'dma_blocks': req.dma_blocks,
        })

    summary = _summarize_requests(completed, snapshot.sim_time)
    summary.update(kv_cache.snapshot())
    summary['trace_file'] = trace_file

    total_kv = summary['kv_hit_blocks'] + summary['kv_miss_blocks']
    summary['kv_hit_rate'] = (summary['kv_hit_blocks'] / total_kv) if total_kv > 0 else 0.0
    summary['kv_evictions'] = summary['evictions']
    summary['dma_transfers'] = sum(req['dma_blocks'] for req in request_rows)
    summary['dma_time_s'] = 0.0

    return {
        'summary': summary,
        'requests': request_rows,
        'snapshot': snapshot,
    }


def write_trace_outputs(result, summary_path='trace_summary.csv', requests_path='trace_requests.csv'):
    summary = result['summary']
    request_rows = result['requests']

    summary_cols = [
        'trace_file',
        'num_requests',
        'avg_latency_s',
        'p50_latency_s',
        'p95_latency_s',
        'avg_ttft_s',
        'throughput_req_per_s',
        'throughput_tok_per_s',
        'kv_hit_blocks',
        'kv_miss_blocks',
        'kv_hit_rate',
        'kv_evictions',
        'dma_transfers',
        'dma_time_s',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerow({k: summary.get(k, None) for k in summary_cols})

    request_cols = [
        'req_id',
        'chat_id',
        'parent_chat_id',
        'arrival_s',
        'start_s',
        'first_token_s',
        'finish_s',
        'latency_s',
        'ttft_s',
        'input_tokens',
        'output_tokens',
        'reused_blocks',
        'computed_blocks',
        'dma_blocks',
    ]
    with open(requests_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=request_cols)
        writer.writeheader()
        for row in request_rows:
            writer.writerow({k: row.get(k, None) for k in request_cols})


def run_trace_mode(system, args):
    return run_trace_simulation(
        system=system,
        trace_file=args.trace_file,
        max_batch_size=args.max_batch_size,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        kv_hbm_ratio=args.kv_hbm_ratio,
    )
