# Plan: Heterogeneous HBM-PIM + GPU 3-Tier KV Cache

## 1) Background and motivation

Current trace mode already supports:
- variable `input_length` / `output_length`
- global hash-block KV reuse by `hash_id`
- continuous and static scheduling policies

Current cache model is still single-pool (`--kv-hbm-ratio`) and does not model:
- heterogeneous HBM split (Hi-speed vs Hi-capacity)
- weight reservation inside Hi-capacity
- 3-tier KV migration and eviction (`Hi-speed -> Hi-capacity -> Host`)
- DMA / PCIe migration latency and energy in runtime

Goal of this extension:
- simulate an 8-card heterogeneous memory hierarchy where hot KV stays in Hi-speed, colder KV spills to Hi-capacity, and overflow spills to host memory, while preserving global prefix reuse.

## 2) Locked design decisions

- KV reuse scope stays global: any request/chat/user can reuse an existing `hash_id` block.
- Reuse unit is one 16-token hash block from trace `hash_ids`.
- `input_length` drives prefill block checks; `output_length` drives decode token count.
- Weight tensors stay resident in Hi-capacity tier and reduce available KV space there.
- `--kv-hbm-ratio` is removed from effective capacity logic (no ratio-based KV sizing).
- Migration links:
  - Hi-capacity <-> Hi-speed uses DMA bandwidth = `0.5 * HBM3 BW`
  - Host <-> GPU uses PCIe 4.0 x16 bandwidth
- Eviction policy:
  - Hi-speed full: evict LRU block to Hi-capacity
  - Hi-capacity full: evict LRU block to host

## 3) Input contract alignment

Example request (must be supported as-is):

```json
{"chat_id": 0, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 502, "output_length": 1494, "type": "thinking", "turn": 1, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
```

Required invariant:
- `len(hash_ids) == ceil(input_length / 16)`.

## 4) Target architecture and memory accounting

Per card:
- GPU memory total = `60 GB`
- Hi-speed tier (`L1`) = `20 GB` for hot KV
- Hi-capacity tier (`L2`) = `40 GB` shared by model weights + colder KV

8 cards aggregate:
- L1 total = `160 GB`
- L2 total = `320 GB`

Host tier (`L3`):
- `512 GB` KV spill space

Capacity formulas in simulator:
- `weight_bytes_total = model_weight_bytes(...)`
- `l2_kv_capacity = max(0, 320GB - weight_bytes_total)`
- `l1_kv_capacity = 160GB`
- `l3_kv_capacity = 512GB`

## 5) High-level runtime behavior

For each prompt block (`hash_id`) during prefill:
1. If block in L1: reuse immediately (no recompute, no migration).
2. Else if in L2: count reuse, add DMA latency/energy for L2->L1 promotion.
3. Else if in L3: count reuse, add PCIe latency/energy for L3->L1 refill.
4. Else: compute block, then insert into L1.

When inserting into L1 and full:
- evict LRU from L1 to L2 (add DMA latency/energy).

When inserting to L2 and full:
- evict LRU from L2 to L3 (add PCIe latency/energy).

Decode stage:
- no new prompt blocks are generated; existing decode latency estimator remains, with migration penalties accumulated from prefill-stage cache operations.

## 6) Code areas to modify

Primary files:
- `src/kv_cache.py` (replace single-pool manager with 3-tier manager)
- `src/trace_simulator.py` (build tier capacities from architecture, not ratio)
- `src/continuous_scheduler.py` (consume tier-hit/migration results and accumulate penalties)
- `src/static_batch_scheduler.py` (same KV-tier behavior as continuous mode)
- `src/system.py` and `src/devices.py` (latency/energy helpers for DMA and PCIe transfers)
- `src/config.py` (heterogeneous memory constants and bandwidth constants)
- `main.py` (deprecate or ignore `--kv-hbm-ratio`, expose hetero-tier mode flags if needed)
- `workflow.md` / `attacc-README.md` (document architecture, outputs, and new semantics)

## 7) Mandatory procedure after each implementation step

After each step below:

```bash
cd ramulator2/build
cmake ..
make -j
```

If build fails:
1. Fix the issue introduced in that step.
2. Re-run the same three commands until success.

Then commit:

```bash
cd /home/lizhuoran200/vstack/attacc_simulator
git add <step files>
git commit -m "<step message>"
```

## 8) Dependency-ordered commit plan

### C24. Add hetero-tier config schema (no behavior change)
- Depends on: current HEAD
- Files (2): `src/config.py`, `workflow.md`
- Changes:
  - add constants for L1/L2/L3 capacities and DMA/PCIe bandwidth assumptions
  - document architecture assumptions and formulas
- Commit message: `feat(config): add hetero memory tier constants and formulas`

### C25. Introduce 3-tier KV cache manager data model
- Depends on: C24
- Files (2): `src/kv_cache.py`, `src/request_state.py`
- Changes:
  - add tier enum/location tracking (`L1/L2/L3`)
  - maintain per-tier LRU structures and byte accounting
  - extend request/runtime stats for migration counters
- Commit message: `feat(kv): add three-tier kv cache data model`

### C26. Implement tiered lookup/insert/evict transitions
- Depends on: C25
- Files (2): `src/kv_cache.py`, `src/continuous_scheduler.py`
- Changes:
  - implement `access(hash_id)` returning tier-hit/miss and required migrations
  - implement L1->L2 and L2->L3 eviction paths
  - wire continuous prefill loop to use transition results
- Commit message: `feat(kv): implement tiered access and eviction transitions`

### C27. Add transfer cost model (DMA + PCIe)
- Depends on: C26
- Files (3): `src/system.py`, `src/devices.py`, `src/type.py`
- Changes:
  - add explicit transfer-type helpers (HBM_DMA, PCIE_TRANSFER)
  - return latency and energy per migration byte volume
  - keep backward compatibility with current estimator APIs
- Commit message: `feat(cost): add dma and pcie transfer latency-energy estimators`

### C28. Integrate migration penalties into continuous scheduler timing
- Depends on: C27
- Files (2): `src/continuous_scheduler.py`, `src/trace_simulator.py`
- Changes:
  - add migration latency/energy accumulation into step timing
  - report tier-hit counters, migration bytes/time, and eviction counters in summary
- Commit message: `feat(trace): integrate kv migration penalties in continuous scheduling`

### C29. Integrate 3-tier KV behavior into static scheduler
- Depends on: C28
- Files (2): `src/static_batch_scheduler.py`, `src/trace_simulator.py`
- Changes:
  - static prefill path uses same tiered cache operations
  - include static-mode tier metrics in request and summary CSV outputs
- Commit message: `feat(static): enable three-tier kv cache in static scheduler`

### C30. Replace ratio-based KV sizing with architecture-derived capacities
- Depends on: C29
- Files (2): `src/trace_simulator.py`, `main.py`
- Changes:
  - remove `--kv-hbm-ratio` from effective logic
  - compute L2 KV capacity from `L2_total - model_weight_bytes`
  - expose resulting capacities in summary CSV
- Commit message: `feat(capacity): switch to architecture-derived kv capacities`

### C31. Extend trace outputs and docs for analysis
- Depends on: C30
- Files (3): `src/trace_simulator.py`, `workflow.md`, `attacc-README.md`
- Changes:
  - add `l1/l2/l3 hits`, migration bytes/time/energy, tier occupancy, and evictions
  - document interpretation of TTFT/throughput under 3-tier cache behavior
- Commit message: `docs(trace): add three-tier kv metrics and analysis guide`

### C32. Validation and regression checklist
- Depends on: C31
- Files (2): `timeline.csv`, `workflow.md`
- Changes:
  - add validation records and expected-vs-observed checks
  - include deterministic test matrix for cache behavior and performance sanity
- Validation runs:
  1. Build regression (`cmake .. && make -j`).
  2. Functional trace run on `llm-req-inputs/qwen_thinking_blksz_16.jsonl`.
  3. Assert tier behavior:
     - repeated hashes increase hit count
     - L1 saturation causes L1->L2 moves
     - L2 saturation causes L2->L3 moves
  4. Compare TTFT/throughput/energy vs baseline to confirm migration penalties are reflected.
- Commit message: `test(trace): validate three-tier kv behavior and performance counters`

## 9) Expected feature outcomes after completing this plan

- Heterogeneous memory-aware KV simulation (L1/L2/L3) instead of single-pool ratio model.
- Automatic KV capacity computation from architecture + model weight reservation.
- Global prefix reuse preserved, now with tier-aware migration penalties.
- Continuous and static scheduler parity for tiered KV behavior.
- Trace outputs include enough data to analyze TTFT/throughput/energy impact of cache tiering.

## 10) Clarifications needed before implementation starts

1. Weight placement granularity: should `weight_bytes_total` reserve L2 once globally across 8 GPUs, or per-card then aggregated (equivalent only if tensor-parallel partitioning is strict)?
2. Host hit refill path: on L3 hit, should we model direct `L3->L1` transfer, or enforced staged `L3->L2->L1`?
3. Cross-card scope: is KV cache logically global with free migration across cards, or per-card local caches without inter-card block migration?
4. If L3 (host) is full, should we drop oldest blocks silently or treat as simulation error/backpressure?
