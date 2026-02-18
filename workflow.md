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

