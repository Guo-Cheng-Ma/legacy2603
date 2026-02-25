# attacc_simulator: Current Structure and Workflow

## 1) Repository structure (relevant to simulation flow)

- `main.py`
  - Single entry point for running one simulation configuration.
  - Takes fixed `--lin`, `--lout`, `--batch`.
- `src/type.py`
  - Enums for data type, layer type, device type.
- `src/config.py`
  - Hardware/model config tables and scaling factors.
- `src/model.py`
  - Builds transformer layers (`sum_decoder` + `gen_decoder`) for one fixed `(batch, lin, lout)`.
- `src/devices.py`
  - Device performance/energy model:
    - `xPU`: GPU/CPU timing + energy.
    - `PIM`: PIM timing + energy; calls Ramulator for attention score layer.
- `src/system.py`
  - Top-level simulator orchestration (`System.simulate`).
  - Aggregates per-layer time/energy, scales to all decoder layers, reports throughput/latency.
- `src/ramulator_wrapper.py`
  - Python wrapper that generates trace, runs `ramulator2`, parses cycles/command counters, caches results in CSV.
- `ramulator2/src/...`
  - Cycle-accurate DRAM/PIM simulator.
  - Uses `PIMLoadStoreTrace` frontend + `PIMDRAMSystem` + `HBM3-PIM` controller/scheduler.
- `ramulator2/trace_gen/gen_trace_attacc_{bank,bg,buffer}.py`
  - Generates PIM command trace files for a given attention problem size.
- `llm-req-inputs/qwen_thinking_blksz_16.jsonl`
  - Real request trace file with per-request timestamp, input/output length, turn, and `hash_ids` (16-token blocks).

## 2) Current execution workflow

1. `main.py` parses one fixed simulation tuple:
   - system config (`dgx`, `dgx-attacc`, etc.)
   - model/dtype
   - fixed `lin`, `lout`, `batch`.

2. `System` is initialized:
   - GPU always created.
   - Optional accelerator:
     - PIM (`System.set_accelerator(..., DeviceType.PIM, ...)`) creates `Ramulator(...)`.
     - CPU alternative also supported.

3. `System.simulate(batch, lin, lout, ...)` runs one static scenario:
   - `Transformer.build(batch, lin, lout, attn_on_hetero)` generates:
     - one summarization stage over full `lin`,
     - generation stages from `stage=1..lout-1`.
   - For each layer:
     - GPU/CPU/PIM time+energy selected by layer type.
     - If layer is attention `score` on PIM, `PIM.get_time_and_energy` calls `ramulator_wrapper.output(...)`.
   - Optional pipeline/feed-forward overlap optimizations applied for PIM mode.
   - Aggregates stage breakdown + energy and prints throughput/latency.

4. Ramulator call path for PIM attention:
   - `ramulator_wrapper.output(...)` checks cache key:
     - `(L, nhead, dhead, dbyte, pim_type, power_constraint)`.
   - On miss:
     - generates trace via `ramulator2/trace_gen/gen_trace_attacc_*.py`,
     - writes temp YAML,
     - runs `ramulator2 -f <yaml>`,
     - parses `memory_system_cycles` and PIM command counters.
   - Returns attention execution time + traffic-derived energy terms.

5. Output:
   - single CSV row per run in `output.csv`.
   - No per-request timeline or queue-level metrics.

## 3) Current behavior limits vs. your target

- No request abstraction:
  - simulator consumes scalar `(batch, lin, lout)`, not per-request records.
- No request queue / scheduler:
  - no arrival timestamps, no waiting queue, no continuous admission.
- No multi-turn/session execution:
  - `chat_id`/`parent_chat_id`/`turn` are not used.
- No explicit KV cache reuse model:
  - KV memory is only accounted as total capacity (`get_required_mem_capacity`), not block-level reuse.
- Fixed static batching only:
  - cannot replace completed requests with queued requests during runtime.

## 4) Trace format compatibility notes

- `llm-req-inputs/qwen_thinking_blksz_16.jsonl` provides:
  - `timestamp`, `input_length`, `output_length`, `turn`, `parent_chat_id`, `hash_ids`.
- In sampled file:
  - `len(hash_ids) == ceil(input_length / 16)` for all records.
  - multi-turn relationships are encoded by `parent_chat_id` chains (not repeated `chat_id` rows).

## 5) Trace mode workflow (new)

### Mode switch

- `main.py` now supports:
  - `--mode fixed` (legacy static `(batch, lin, lout)` path),
  - `--mode trace` (continuous batching + trace-driven requests).
  - `--trace-scheduler {continuous,static}` for trace-mode scheduling policy.

### Fixed mode vs Trace mode

- `fixed` mode:
  - input is scalar `--lin`, `--lout`, `--batch`,
  - executes one static shape configuration,
  - no request queue or arrival-time scheduling,
  - output is `output.csv` (single-run performance/energy breakdown).
- `trace` mode:
  - input is JSONL request stream (`--trace-file`),
  - supports per-request `timestamp`, variable input/output lengths,
  - uses either:
    - `continuous`: FIFO waiting queue + immediate backfill, or
    - `static`: FIFO static batching without backfill inside a running batch,
  - uses global KV reuse by `hash_id` with LRU eviction,
  - output is timestamped request/summary CSVs for this run.

### Static scheduler semantics (`--trace-scheduler static`)

- Batch formation:
  - FIFO over arrived requests, up to `--max-batch-size`.
  - Requests that arrive while a batch is running wait for the next batch.
- Decode semantics:
  - all requests in the batch start decode together,
  - decode runs for `max(output_length)` steps in that batch,
  - batch finishes only when the longest response finishes.
- Completion timestamps:
  - all requests in a batch are marked with the same batch finish timestamp.

### Trace mode runtime pipeline

1. `src/trace_loader.py` parses JSONL requests and validates required fields.
2. `src/kv_cache.py` manages global KV block cache (`hash_id` key, 16-token block, LRU eviction).
3. `src/continuous_scheduler.py` runs:
   - FIFO admission by `timestamp`,
   - chunked prefill with KV hit-skip,
   - one-token decode step,
   - immediate backfill when a request completes.
4. `src/trace_simulator.py` orchestrates run + metrics export.

### KV reuse semantics

- Reuse scope is global across all chats/users.
- Reuse trigger is `hash_id` hit.
- On hit: skip KV recomputation.
- On miss: compute and insert block into KV cache.

### Example trace input row

```json
{"chat_id": 0, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 502, "output_length": 1494, "type": "thinking", "turn": 1, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
```

- Validation rule: `len(hash_ids) == ceil(input_length / 16)`.
- For this sample: `ceil(502/16) = 32`, so 32 blocks is valid.

### Trace mode outputs

- `trace_summary_{dtype}_{mmdd_hhmmss}_{input_request_name}.csv`:
  - run metadata (system/gpu/pim/model/dtype, scheduler knobs),
  - aggregate latency/TTFT/queue/throughput metrics,
  - input/output token distribution stats (avg/p50/p95/max),
  - KV cache and eviction stats,
  - stage work counters (`prefill_work_time_s`, `decode_work_time_s`, steps),
  - scheduler/batch stats (`trace_scheduler_mode`, `num_batches`, `avg_batch_size`),
  - energy summary and component breakdown (dram/l2/l1/reg/alu/comm).
- `trace_requests_{dtype}_{mmdd_hhmmss}_{input_request_name}.csv`:
  - per-request type/turn, arrival/start/queue/TTFT/finish,
  - static-batch fields (`batch_id`, `batch_start_s`, `batch_finish_s`),
  - per-request input/output tokens and prompt blocks,
  - per-request KV reuse counters and hit rate.

### 3-tier KV fields in trace outputs

`trace_summary_*.csv` now includes architecture and tier-state fields:

- Capacity mode and config:
  - `kv_capacity_mode` (`hetero_3tier`)
  - `weight_reserved_bytes`
  - `l1_kv_capacity_bytes_cfg`
  - `l2_kv_capacity_bytes_cfg`
  - `l3_kv_capacity_bytes_cfg`
- Tier occupancy and hit stats:
  - `l1_num_blocks`, `l2_num_blocks`, `l3_num_blocks`
  - `l1_hits`, `l2_hits`, `l3_hits`
  - `l1_hit_rate`, `l2_hit_rate`, `l3_hit_rate`
  - `l1_used_ratio`, `l2_used_ratio`, `l3_used_ratio`
- Migration counters and penalties:
  - `dma_transfers`, `pcie_transfers`
  - `dma_time_s`, `pcie_time_s`, `migration_time_s`
  - `migration_bytes`
  - `migration_energy_nj`, `dma_energy_nj`, `pcie_energy_nj`

`trace_requests_*.csv` now includes per-request tier/migration fields:

- `l1_hit_blocks`, `l2_hit_blocks`, `l3_hit_blocks`
- `l1_to_l2_blocks`, `l2_to_l3_blocks`, `l3_drop_blocks`
- `dma_blocks`, `pcie_blocks`, `migration_bytes`

Interpretation:

- Higher `l1_hit_rate` generally improves TTFT and throughput.
- High `pcie_time_s` indicates host spill/refill is a bottleneck.
- `total_energy_nj` includes both model compute energy and migration energy.

## 6) Heterogeneous 3-tier KV architecture constants

The simulator reads 3-tier KV architecture settings from YAML:

- default file: `configs/kv_arch.yaml`
- CLI argument: `--kv-arch-config <path>`

Capacity formulas:

- `l1_kv_bytes = num_cards * hispeed_kv_per_card_gb`
- `l2_total_bytes = num_cards * hicap_total_per_card_gb`
- `l2_kv_bytes = max(0, l2_total_bytes - weight_bytes_total)`
- `l3_kv_bytes = host_kv_total_gb`

Helper API:

- `load_hetero_kv_arch_config(yaml_path)` in `src/config.py`
- `get_hetero_kv_capacities(weight_bytes_total, hetero_kv_arch=...)` in `src/config.py`
- `get_hetero_transfer_bandwidths(hetero_kv_arch=...)` in `src/config.py`

Example YAML:

```yaml
kv_arch:
  num_cards: 8
  gpu_mem_per_card_gb: 60
  hispeed_kv_per_card_gb: 20
  hicap_total_per_card_gb: 40
  host_kv_total_gb: 512
  dma_bandwidth_gbps: 1676
  pcie_bandwidth_gbps: 64
```

## 7) Validation checklist (C32)

Use this checklist after changes to trace scheduler / KV cache:

1. Build regression:
   - `cd ramulator2/build`
   - `cmake ..`
   - `make -j`
2. Tier-transition unit checks (small synthetic capacities):
   - verify L1->L2, L2->L3, and L3 drop counts increase under pressure.
   - verify L3 hit triggers direct `L3->L1` promotion.
3. Functional trace run (external request format):
   - run with `llm-req-inputs/qwen_thinking_blksz_16.jsonl` (or bounded subset for faster iteration).
   - ensure output files are generated:
     - `trace_summary_{dtype}_{mmdd_hhmmss}_{input_request_name}.csv`
     - `trace_requests_{dtype}_{mmdd_hhmmss}_{input_request_name}.csv`
4. Sanity checks on outputs:
   - `total_time_s`, `throughput_tok_per_s`, `avg_ttft_s` are populated.
   - tier metrics exist: `l1/l2/l3` hits, occupancy, and migration counters.
   - energy includes migration terms (`migration_energy_nj`) and combined `total_energy_nj`.

Recent validation snapshot:

- Synthetic KV tests: passed (`L1->L2`, `L2->L3`, `L3` drop, and direct `L3->L1` promotion).
- Trace run on bounded subset (`/tmp/qwen_thinking_blksz_16_32.jsonl`): completed, produced summary/request CSVs with 3-tier metrics.

## 8) Parallel Ramulator Cache Pre-Generation (BA)

To reduce runtime trace-generation bottlenecks, use the standalone pregen tool:

```bash
cd ramulator2/trace_gen
python3 pregen_ramulator_bank.py --config pregen_bank.yaml
```

Behavior:

- Pre-generates BA (`gen_trace_attacc_bank.py`) cache rows and writes to `ramulator.out`.
- Skips keys already existing in `ramulator.out`.
- Runs tasks in parallel (`workers` configurable, default 100).
- Uses a single atomic writer path to avoid concurrent CSV corruption.
- Keeps all temporary traces/YAMLs under `ramulator2/trace_gen/tmp/`:
  - `tmp/traces/`
  - `tmp/yamls/`

Key dimensions covered by default config:

- batch-derived `nhead` for batch sizes 1..16
- `seqlen` range 1..8192
- `dhead=128`, `dbyte=2`
- `maxlen=max(4096, seqlen)`

Config example file:

- `ramulator2/trace_gen/pregen_bank.yaml`
