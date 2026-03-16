# attacc_simulator: Current Python Frontend Workflow

This document reflects the current Python frontend after the unified-config, topology-aware trace-mode, and KV-policy changes recorded through `timeline.csv` C55. The old CLI-first description, separate `kv_arch.yaml` flow, and flat trace CSV outputs are obsolete.

## 1) Entry point and config model

- The frontend entry point is:

```bash
python3 main.py --config <yaml>
```

- `main.py` no longer exposes the old large set of runtime flags directly. It reads one unified YAML and passes the parsed namespace into the existing fixed-mode or trace-mode simulator path.
- `src/config.py` merges these sections into one runtime config:
  - `system`
  - `pim`
  - `model`
  - `workload`
  - `trace`
  - `kv_arch`
- The loader accepts normal PyYAML when available and falls back to the project's minimal parser when it is not.
- `kv_arch` is now part of the same unified config. The old standalone `configs/kv_arch.yaml` workflow is no longer the frontend contract.

Minimal config shape:

```yaml
system:
  system: dgx-attacc
  gpu: A100a
  ngpu: 8

pim:
  pim: bank
  num_pim_die: 4
  die_type: attacc
  compute_stack_l1: 4
  capacity_stack_l2: 4
  powerlimit: false
  ffopt: false
  pipeopt: false

model:
  model: GPT-175B
  word: 2

workload:
  mode: trace
  lin: 2048
  lout: 128
  batch: 1

trace:
  trace_file: llm-req-inputs/qwen_thinking_blksz_16.jsonl
  max_batch_size: 16
  prefill_chunk_tokens: 128
  trace_scheduler: continuous
  timestamp_scaling: 1.0
  trace_debug: false
  trace_debug_interval: 100

kv_arch:
  num_cards: 8
  host_kv_total_gb: 512
  dma_bandwidth_gbps: 1676
  pcie_bandwidth_gbps: 64
  eviction_policy: lru
  placement_policy: all_unique
  replica_tier: auto
  replica_reserve_ratio_l1: 0.0
  replica_reserve_ratio_l2: 0.0
  aware_policy:
    replica_min_distinct_cards: 3
    replica_min_remote_hits: 8
```

## 2) Frontend modules that define the workflow

- `main.py`
  - Loads the unified YAML.
  - Derives GPU/PIM topology from `die_type`, `num_pim_die`, `compute_stack_l1`, and `capacity_stack_l2`.
  - Dispatches into fixed mode or trace mode.
- `src/config.py`
  - Defines unified-config defaults and parsing.
  - Derives topology-aware GPU capacity, bandwidth, and KV tier capacities.
  - Resolves trace-family/model-specific KV policy presets and heuristic defaults.
  - Validates `attacc`, `vstack`, and `uniform` constraints.
- `src/system.py`
  - Runs the original layer-by-layer fixed-shape simulator.
- `src/trace_loader.py`
  - Parses request JSONL and validates the external trace format.
- `src/request_state.py`
  - Keeps immutable trace fields separate from mutable runtime state.
- `src/kv_cache.py`
  - Implements the topology-aware KV cache and migration overlap model.
- `src/continuous_scheduler.py`
  - Continuous batching with chunked prefill and immediate backfill.
- `src/static_batch_scheduler.py`
  - FIFO static batching without mid-batch backfill.
- `src/trace_simulator.py`
  - Builds the KV cache, runs the selected scheduler, and writes summary/request outputs.
- `src/ramulator_wrapper.py`
  - Still handles PIM attention timing for fixed mode and any estimator path that calls Ramulator-backed attention.

## 3) Fixed mode workflow

Fixed mode still exists. It is selected by:

```yaml
workload:
  mode: fixed
```

Runtime flow:

1. `main.py` loads the unified config and constructs the `System`.
2. `make_xpu_config()` and `make_pim_config()` derive memory capacity and off-chip bandwidth from the die topology:
   - `attacc`: dedicated PIM dies reduce available GPU HBM bandwidth and L2 capacity.
   - `vstack`: all dies are hybrid, so GPU bandwidth stays full while capacity is split by stack role.
   - `uniform`: all 5 dies are HBM-PIM, GPU memory is modeled from L1 only, and there is no L2 tier.
3. `System.simulate(batch, lin, lout, ...)` runs the original static decoder path.
4. PIM attention score layers still go through `src/ramulator_wrapper.py` and `ramulator2`.
5. Output remains a single `output.csv` file for the run.

Nothing in fixed mode uses request arrivals, schedulers, or block-level KV reuse.

## 4) Trace mode workflow

Trace mode is selected by:

```yaml
workload:
  mode: trace
```

### 4.1 Request ingest

1. `src/trace_loader.py` reads JSONL rows.
2. Each row must contain:
   - `chat_id`
   - `parent_chat_id`
   - `timestamp`
   - `input_length`
   - `output_length`
   - `type`
   - `turn`
   - `hash_ids`
3. `timestamp_scaling` is applied during load, so effective arrival rate can be changed without rewriting the trace file.
4. The block size is fixed at 16 tokens and validation requires:

```text
len(hash_ids) == ceil(input_length / 16)
```

### 4.2 KV topology and policy construction

`src/trace_simulator.py` builds a topology-aware KV cache before scheduling begins.

- KV bytes per block are derived from the current model and activation precision.
- Weight reservation is estimated from `System.get_required_mem_capacity(...)`.
- `get_hetero_kv_capacities()` derives L1/L2/L3 capacities from topology, not from manually-entered per-tier capacities.
- `get_hetero_transfer_bandwidths()` provides DMA and PCIe rates from the unified config.
- `resolve_kv_policy_config()` normalizes `eviction_policy`, `placement_policy`, replica-tier settings, and `aware_policy` knobs before the cache is created.
- In `uniform`, weight reservation comes out of L1; in `attacc`/`vstack`, weight reservation still comes out of L2.
- For known trace families (`traceA`, `traceB`, `thinking`, `coder`) and supported models, `src/config.py` can inject default replica reserve ratios when YAML leaves them unset.

Current tier model:

- L1: banked HBM KV at `(card, die, bank)`
- L2: die-local high-capacity HBM at `(card, die)`
- L3: global host spill tier
- optional replica pools:
  - `replica_tier: L1` reserves a portion of each card's L1 capacity for replicated hot blocks
  - `replica_tier: L2` reserves a portion of each card's L2 capacity for replicated hot blocks
  - `replica_tier: auto` resolves to L2 when there is usable L2 KV space, else L1

Each request is assigned a physical home `(card, die)` and each `hash_id` maps to an L1 bank by `hash_id % banks_per_die`.

### 4.3 Scheduler execution

`trace_scheduler` chooses one of two runtime policies.

Continuous mode:

- Admit requests when `timestamp <= sim_time`.
- Move arrived requests into a waiting queue.
- Backfill active slots immediately up to `max_batch_size`.
- Prefill in chunks of `prefill_chunk_tokens`.
- Run one-token decode steps across active requests.
- When a request finishes, admit another waiting request immediately.

Static mode:

- Admit arrived requests into a FIFO waiting queue.
- Form one batch up to `max_batch_size`.
- Prefill the whole prompt for that batch.
- Decode for `max(output_length)` steps across the batch.
- Requests arriving while the batch is running wait for the next batch.
- Requests in the same batch share `batch_id`, `batch_start_s`, and `batch_finish_s`.

### 4.4 KV hit, promotion, eviction, and placement behavior

The current KV model is topology-aware and YAML-selectable.

- Reuse key: `hash_id`
- Reuse scope: global across all requests
- Canonical eviction path: `L1 -> L2 -> L3 -> drop`
- Effective capacity threshold uses a 1% spare ratio, so tiers start evicting at 99% of nominal capacity

Current policy controls:

- `eviction_policy: lru`
  - container-local `OrderedDict` recency eviction
- `eviction_policy: aware`
  - keeps per-block metadata such as owner chat, request type, turn class, reuse count, last access time, distinct cards seen, and remote-hit count
  - ranks victims by short-term reuse value and expected movement cost instead of pure recency
- `placement_policy: all_unique`
  - only one canonical live copy of each block exists
- `placement_policy: hotset_broadcast`
  - canonical copy still exists
  - ultra-hot blocks may be replicated into a reserved per-card replica pool
  - replicas are checked before remote canonical callback to reduce future cross-card movement

Current hit behavior:

- local L1 canonical hit: reuse in place
- remote L1 canonical hit: callback to the request's home L1
- L2/L3 canonical hit: callback toward the request's home L1
- local replica hit:
  - `replica_tier=L1`: immediate hit
  - `replica_tier=L2`: local callback from replica L2 to home L1

Current miss insertion behavior:

- try home L1 first
- if L1 block capacity is too small for the model/block size, fall back to home L2
- if L2 is unavailable or full, fall back to L3

This fallback matters for large models such as `GPT-175B`, where one 16-token KV block can exceed a single bank-sized L1 slot.

Migration timing is modeled by transfer domain:

- same-die DMA
- cross-die HBM movement
- cross-card NVLink movement
- host spill/refill over PCIe

Independent domains can overlap. The overlap totals are exported in the trace summary.

Current latency policy:

- Callback latency (`L2/L3 -> L1`) is charged
- `attacc` mode also charges KV eviction latency
- `uniform` also charges KV eviction latency
- `vstack` keeps eviction latency overlapped
- replica fanout latency is tracked separately and exported in the trace summary

## 5) Topology rules that now matter

These rules were added after the old workflow document and must be reflected in any frontend usage.

### `attacc`

- `die_type: attacc`
- `num_pim_die` must be in `[0, 4]`
- At least one HBM die-package must remain GPU-only, otherwise the run is rejected
- L1 capacity comes from half-capacity PIM dies
- L2 capacity comes from the remaining non-PIM GPU dies
- GPU off-chip bandwidth scales with the number of remaining GPU dies

### `vstack`

- `die_type: vstack`
- `main.py` treats all 5 die-packages as hybrid
- `compute_stack_l1 + capacity_stack_l2` must equal 8 stacks per die-package
- L1 capacity is derived from compute stacks with the compute-stack capacity ratio
- L2 capacity is derived from capacity stacks
- GPU off-chip bandwidth remains full, while memory capacity reflects the stack split

### `uniform`

- `die_type: uniform`
- `main.py` treats all 5 die-packages as HBM-PIM
- Each die contributes 8 GiB of L1 space, for 40 GiB/card total
- There is no L2 tier
- All weights must fit inside aggregate L1; otherwise the run fails early
- `compute_stack_l1` and `capacity_stack_l2` are ignored
- GPU off-chip bandwidth remains nonzero and memory capacity is modeled from L1 only

## 6) Current trace outputs

Trace outputs are no longer written as flat `trace_summary_*.csv` and `trace_requests_*.csv` files.

Current output layout:

```text
results/<YYMMDD>/<trace-name>/
  S-<die_type>-<compute_stack_l1>-<model>-<ngpu>gpu-<HHMMSS>.yaml
  R-<die_type>-<compute_stack_l1>-<model>-<ngpu>gpu-<HHMMSS>.jsonl
```

`trace-name` is normalized from the input filename when it contains `traceA`, `traceB`, `coder`, or `thinking`.

Summary YAML contains:

- run metadata: system, GPU, PIM type, model, dtype, scheduler knobs
- KV policy metadata: `trace_family`, `eviction_policy_cfg`, `placement_policy_cfg`, `replica_tier_cfg`, and resolved reserve ratios
- arrival and completion metrics: total time, QPS, latency, TTFT, queue delay, throughput
- batch metrics: batch counts and padding stats for static scheduling
- topology-derived KV config: per-card memory, tier capacities, bandwidth config, bytes per block
- tier state: resident blocks, L1/L2/L3 occupancy, hit counters, hit rates, used ratios
- policy-analysis state: same-chat vs cross-chat hits, single-turn vs multi-turn hits, replica occupancy, replica fanout, broadcast promotions, and avoided cross-card callbacks
- migration stats: DMA/PCIe transfers, migration bytes, callback/eviction timing, overlap-domain maxima
- energy stats: prefill/decode/model/migration totals and component breakdown

Per-request JSONL contains:

- request identity and trace fields
- physical placement: `home_card`, `home_die`
- arrival, start, TTFT, finish, latency
- static-batch tags when static scheduling is used
- prompt block counts, computed blocks, reused blocks, per-request KV hit rate
- tier hit counts and relocation counters
- per-request DMA/PCIe and callback movement counters
- per-request policy-analysis counters: same/cross-chat hits, single/multi-turn hits, replica hits, and avoided cross-card callbacks

## 7) Practical validation checklist

Use this checklist after any frontend changes that touch configs, schedulers, cache placement, or output writers.

1. Config load:
   - run `python3 main.py --config <existing-config>.yaml`
   - confirm the YAML loads without relying on deleted legacy config files
2. Documentation sync:
   - if simulator functionality, spec, outputs, or workflow changed, update `workflow.md`, `architecture_summary.md`, and `AGENTS.md` in the same change step before committing
3. Fixed-mode smoke:
   - set `workload.mode: fixed`
   - confirm `output.csv` is produced
4. Trace-mode smoke:
   - set `workload.mode: trace`
   - confirm `results/<date>/<trace>/S-...yaml` and `R-...jsonl` are produced
5. Trace sanity checks:
   - verify `total_time_s`, `throughput_tok_per_s`, `avg_ttft_s`, and `trace_qps`
   - verify topology/KV fields such as `home_card`, `home_die`, `l1_hit_rate`, `dma_time_s`, and `migration_energy_nj`
   - when policy mode is enabled, verify `eviction_policy_cfg`, `placement_policy_cfg`, `replica_*`, and `same_chat_hit_rate`
   - verify output format assumptions: summary is YAML, requests are JSONL

## 8) Obsolete assumptions from older docs

The following are stale and should not be reintroduced into documentation or scripts:

- the old many-CLI-arguments workflow as the primary frontend contract
- separate `kv_arch.yaml` as the normal way to configure trace mode
- flat `trace_summary_*.csv` and `trace_requests_*.csv` outputs
- topology-agnostic KV accounting that only tracks global L1/L2/L3 totals
- docs that describe trace-mode cache policy as fixed pure-LRU with only unique-copy placement
- documentation that ignores `die_type`, `num_pim_die`, `compute_stack_l1`, `capacity_stack_l2`, or `timestamp_scaling`
