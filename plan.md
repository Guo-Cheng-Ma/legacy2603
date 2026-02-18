# Plan to Add Prefix KV Reuse + Continuous Batching

## 1) Scope

Target capabilities:
- Variable per-request `input_length` / `output_length`.
- Arrival-time-driven request queue.
- Continuous batching (backfill immediately when requests finish).
- Prefix KV cache reuse based on `hash_ids` (16-token blocks), so reused blocks are not recomputed.

Non-goal for this phase:
- Do not modify low-level Ramulator DRAM scheduling policy (`ramulator2/src/dram_controller/...`) unless needed for correctness.

## 2) High-level design

### A. Add a request-level simulation pipeline

Introduce explicit request entities loaded from JSONL:
- request fields: `chat_id`, `parent_chat_id`, `timestamp`, `input_length`, `output_length`, `turn`, `hash_ids`.
- runtime fields: `arrival_time`, `start_time`, `finish_time`, `state`, `remaining_decode_tokens`, `cached_prefix_blocks`.

This becomes the input to a scheduler instead of fixed `(batch, lin, lout)`.

### B. Add KV cache manager (prefix/block aware)

Create a KV cache component that:
- tracks cached blocks by `hash_id`,
- returns how many prefix blocks in a request can be reused,
- accounts cache memory usage and evictions,
- updates recency on access.

Core behavior:
- If block is reusable from cache: do not charge recompute for that block.
- If block misses: charge prefill compute and insert block KV into cache.

### C. Refactor compute model to support partial prefill + token-step decode

Current `System.simulate` is batch-static and monolithic.  
Add new APIs for scheduler-driven execution:
- `estimate_prefill(...)` with:
  - total prompt length,
  - cached prefix tokens,
  - uncached new tokens.
- `estimate_decode_step(...)` for one decode-token step with current active batch.

This allows dynamic request mix and continuous batch updates over time.

### D. Implement continuous batching scheduler

Event-driven loop:
1. Move newly arrived requests (by timestamp) into wait queue.
2. Admit requests into active set until `max_batch_size`.
3. Process compute in steps:
   - prefill for newly admitted requests (with KV reuse),
   - decode one token-step for active requests.
4. On completion, remove request and immediately backfill from wait queue.
5. Repeat until all requests finish.

Outputs:
- per-request latency/TTFT/finish time,
- queue wait time,
- throughput timeline,
- cache hit/miss/reuse stats.

## 3) Planned file-level changes

## Existing files to modify

- `main.py`
  - Add trace-mode CLI args (trace path, max batch, cache capacity/policy, scheduler policy).
  - Keep existing fixed-mode path for backward compatibility.
- `src/system.py`
  - Split monolithic simulation into reusable primitives (`estimate_prefill`, `estimate_decode_step`).
  - Add interfaces used by scheduler for variable lengths and incremental progression.
- `src/model.py`
  - Support partial-prefill shape construction:
    - total context length vs newly computed prefix-suffix length.
  - Support decode-step construction without rebuilding full `lout` path each time.
- `src/ramulator_wrapper.py`
  - Ensure large sequence lengths from real traces are handled robustly.
  - Pass/derive `maxlen` for trace generation to avoid incorrect fixed defaults.

## New files to add

- `src/trace_loader.py`
  - JSONL parser and validation for `llm-req-inputs` format.
- `src/request_state.py`
  - Dataclasses for request static fields + runtime state.
- `src/kv_cache.py`
  - Prefix/block KV cache manager (lookup, insert, evict, stats).
- `src/continuous_scheduler.py`
  - Event loop for arrival handling, wait queue, active set, and continuous batching decisions.
- `src/trace_simulator.py`
  - High-level orchestration: loader + scheduler + system cost model + reporting.

## Ramulator-side scripts (if needed)

- `ramulator2/trace_gen/gen_trace_attacc_bank.py`
- `ramulator2/trace_gen/gen_trace_attacc_bg.py`
- `ramulator2/trace_gen/gen_trace_attacc_buffer.py`
  - Optional small patch so `max_L` is safe for very long contexts seen in trace-driven mode.

## 4) Execution plan (implementation order)

1. Build request ingestion layer
   - parse JSONL -> validated request objects.
   - keep deterministic ordering by `(timestamp, chat_id)`.
2. Build KV cache manager
   - hash-block lookup/insert/evict + memory accounting.
   - expose `get_reusable_prefix_blocks(request)` API.
3. Refactor cost model (`System`/`Transformer`)
   - add partial-prefill/decode-step APIs.
   - preserve current fixed-mode behavior.
4. Implement continuous scheduler
   - arrival queue + wait queue + active batch + completion backfill.
   - integrate KV cache decisions before prefill charging.
5. Integrate CLI and reports
   - add trace mode in `main.py`.
   - output request-level and aggregate metrics.
6. Validation
   - compare against old mode on synthetic fixed traces.
   - run real trace and inspect cache-hit and latency sanity.

## 5) Validation checklist

- Functional:
  - requests processed in timestamp order.
  - completed requests are immediately replaced when queue non-empty.
  - cached hash blocks are not recharged for prefill compute.
- Metrics:
  - TTFT, end-to-end latency, throughput, queue depth over time.
  - KV cache hit rate, evictions, effective reused tokens/blocks.
- Regression:
  - old fixed-mode path unchanged for existing command lines.

## 6) Open questions to resolve before coding

1. Reuse scope: Should same `hash_id` be reusable globally across all chats, or only within parent-linked conversation lineage (`parent_chat_id` chain)?
2. Prefix strictness: Must reuse require contiguous prefix match from block 0, or can any block hit be reused even if earlier blocks miss?
3. Position sensitivity: Should we treat same `hash_id` at different absolute positions as reusable (your statement suggests yes), or enforce position-aware reuse for model-faithful KV?
4. Cache capacity: What KV cache capacity should we model (GB), and is it shared across all GPUs/attacc units or per-device?
5. Eviction policy: Prefer `LRU`, `FIFO`, or another policy?
6. Continuous batching granularity: Token-level decode step with one-token progress per active request, correct?
7. Prefill policy: For new arrivals, do you want full prefill immediately, or chunked prefill interleaved with decode?
8. Batch admission policy: FIFO by arrival time only, or priority policy (e.g., shortest job first / smallest remaining decode)?
9. Padding model: For variable active lengths, should decode cost use max active context length (padded batch) or per-request no-padding estimate?
10. Output format: Do you want a new CSV (per-request + aggregate), and what exact columns are mandatory?
11. Ramulator depth: Is Python-side scheduler/cache extension sufficient now, or do you also want a Ramulator C++ request-queue model for memory-level contention per request stream?
12. Warm-start behavior: Should cache start empty at t=0, or preload common prefix blocks (e.g., system prompt)?

