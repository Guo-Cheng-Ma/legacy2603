# Refactored Plan: Prefix KV Reuse + Continuous Batching

## 1) Background and motivation (restored blueprint context)

Current simulator behavior:

- fixed `(batch, lin, lout)` inputs only,
- single-turn static execution style,
- no request-arrival-time queueing,
- no block-level KV cache reuse from trace `hash_id`.

Target behavior:

- trace-driven variable input/output lengths,
- per-request timestamps and continuous batching,
- prefix KV reuse by `hash_id` (reuse => skip recompute),
- multi-turn reuse support through shared cached blocks.

This document keeps the earlier architectural motivation and refactors implementation into dependency-aware, commit-sized steps.

## 2) Locked design decisions

- Reuse scope: global reuse across all requests/chats/users (confirmed).
- Reuse unit: 16-token hash block (`hash_id` from trace).
- Reuse condition: `hash_id` hit is reusable; no contiguous-prefix requirement.
- Cache placement: KV cache capacity modeled in HBM pool.
- Eviction: LRU on capacity pressure.
- Decode policy: one token per active decoding request per scheduler step.
- Prefill policy: chunked prefill interleaved with decode.
- Admission policy: strict FIFO by arrival time (`timestamp`, stable tie-break).
- Backfill policy: immediate refill of freed active slot.
- Startup state: KV cache empty at `t=0`.
- Compatibility: legacy fixed mode must remain functional.

## 2.1) Trace input contract (sample-aligned)

Sample request (provided and treated as canonical format):

```json
{"chat_id": 0, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 502, "output_length": 1494, "type": "thinking", "turn": 1, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
```

Required parsing/validation behavior:

- required keys:
  - `chat_id`, `parent_chat_id`, `timestamp`, `input_length`,
    `output_length`, `type`, `turn`, `hash_ids`
- `hash_ids` is a list of block IDs where one block = 16 input tokens
- enforce `len(hash_ids) == ceil(input_length/16)`
- using sample values: `ceil(502/16) = 32`, so 32 `hash_ids` is valid
- cache lookup/reuse for each `hash_id` is global (not scoped by `chat_id`)

## 3) Target architecture 

### Runtime flow

1. Parse JSONL trace into typed request objects.
2. Maintain arrival queue (`timestamp`) + FIFO waiting queue.
3. Keep active set up to `max_batch_size`.
4. Per scheduler step:
   - move arrived requests to waiting queue,
   - admit waiting requests to active set,
   - run prefill micro-step for `PREFILLING` requests,
   - run decode micro-step for `DECODING` requests,
   - finish completed requests and immediately backfill.
5. For each prompt block:
   - KV hit => reuse, skip recompute,
   - KV miss => compute and insert into KV cache.
6. Emit trace-level and per-request metrics.

### Components

- `TraceLoader`: trace parsing and validation.
- `RequestState`: lifecycle state + per-request runtime counters.
- `KVCacheManager`: hash table + HBM accounting + LRU eviction.
- `ContinuousScheduler`: queueing, active-set management, step loop.
- `System` micro-cost APIs: prefill/decode estimation for variable shapes.
- `TraceSimulator`: top-level orchestration and CSV export.

## 4) KV cache model details (restored)

- `kv_bytes_per_token = ndec * 2 * hdim * a_byte`
- `kv_bytes_per_block = kv_bytes_per_token * 16`
- capacity pool:
  - GPU-only: aggregate GPU HBM
  - GPU+PIM: aggregate GPU HBM + PIM HBM
  - available KV capacity = total capacity minus non-KV reserve (configurable ratio)
- insertion:
  - if free space insufficient, evict LRU blocks until enough space
  - then insert new block and mark as MRU

## 5) Continuous batching and scheduler details (restored)

Request states:

- `NOT_ARRIVED`, `WAITING`, `PREFILLING`, `DECODING`, `DONE`

Per-step policy:

1. Admit newly arrived requests into FIFO waiting queue.
2. Fill active set from wait queue until `max_batch_size`.
3. Prefill chunk processing:
   - process up to `prefill_chunk_tokens` per prefill request,
   - hits skip compute; misses compute + insert.
4. Decode processing:
   - one token per decoding request,
   - decode cost padded by max active context length.
5. Advance simulated time by step latency.
6. Complete done requests and backfill immediately.

## 6) Mandatory procedure after each commit step

After each step below (after edits), run:

```bash
cd ramulator2/build
cmake ..
make -j
```

If build fails:

1. fix bugs introduced in that step,
2. re-run the same 3 commands until build passes.

Then commit:

```bash
cd /home/lizhuoran200/vstack/attacc_simulator
git add <files changed in this step>
git commit -m "<step commit message>"
```

Constraint: each step changes 1-3 files.

## 7) Dependency-ordered commit plan (refactored granular implementation)

### C1. Add trace schema + loader

- Depends on: none
- Files (2):
  - `src/request_state.py` (new)
  - `src/trace_loader.py` (new)
- Content:
  - typed request record and default runtime fields,
  - JSONL loader + validation (`len(hash_ids) == ceil(input_length/16)`),
  - keep `type` field in request object for future policy extension,
  - deterministic sort by `(timestamp, req_id)`.
- Commit message:
  - `feat(trace): add request schema and trace loader`

### C2. Add scheduler skeleton and state machine

- Depends on: C1
- Files (2):
  - `src/request_state.py`
  - `src/continuous_scheduler.py` (new)
- Content:
  - lifecycle enum + transitions,
  - arrival/wait/active containers,
  - dry-run step loop for admission/backfill only.
- Commit message:
  - `feat(scheduler): add continuous scheduler skeleton`

### C3. Add KV cache manager with LRU

- Depends on: C1
- Files (2):
  - `src/kv_cache.py` (new)
  - `src/request_state.py`
- Content:
  - global `hash_id` lookup table,
  - HBM usage accounting,
  - LRU eviction + access APIs.
- Commit message:
  - `feat(kv): add kv cache manager and lru eviction`

### C4. Add variable-shape micro-cost APIs

- Depends on: C1
- Files (2):
  - `src/system.py`
  - `src/model.py`
- Content:
  - keep `System.simulate(...)` unchanged,
  - add APIs for prefill microbatch and decode-step estimation.
- Commit message:
  - `feat(system): add prefill/decode micro-cost apis`

### C5. Implement prefill with KV hit-skip semantics

- Depends on: C2, C3, C4
- Files (2):
  - `src/continuous_scheduler.py`
  - `src/kv_cache.py`
- Content:
  - chunked prefill execution,
  - per-block hit/miss handling,
  - counters for reused/computed blocks.
- Commit message:
  - `feat(prefill): skip kv recompute on hash hit`

### C6. Implement decode progression and immediate backfill

- Depends on: C5
- Files (2):
  - `src/continuous_scheduler.py`
  - `src/request_state.py`
- Content:
  - one-token decode per active request per step,
  - completion detection and same-step backfill,
  - timing fields (`start`, `ttft`, `finish`).
- Commit message:
  - `feat(decode): add token-step decode and immediate backfill`

### C7. Add top-level trace simulator

- Depends on: C6
- Files (2):
  - `src/trace_simulator.py` (new)
  - `src/continuous_scheduler.py`
- Content:
  - orchestration entrypoint for trace mode:
    - load trace,
    - setup scheduler/cache/system,
    - run to completion,
    - return summary data structures.
- Commit message:
  - `feat(trace): add end-to-end trace simulator orchestration`

### C8. Add CLI switch for trace mode

- Depends on: C7
- Files (2):
  - `main.py`
  - `src/trace_simulator.py`
- Content:
  - new args:
    - `--mode {fixed,trace}`
    - `--trace-file`
    - `--max-batch-size`
    - `--prefill-chunk-tokens`
    - `--kv-hbm-ratio`
  - fixed mode path unchanged.
- Commit message:
  - `feat(cli): add trace mode runtime arguments`

### C9. Add trace output CSVs

- Depends on: C8
- Files (2):
  - `src/trace_simulator.py`
  - `main.py`
- Content:
  - `trace_summary.csv`: latency/throughput/kv-hit/eviction metrics,
  - `trace_requests.csv`: per-request timeline and reuse metrics.
- Commit message:
  - `feat(output): add trace summary and per-request csv`

### C10. Add optional KV DMA penalty model

- Depends on: C9
- Files (3):
  - `src/type.py`
  - `src/devices.py`
  - `src/system.py`
- Content:
  - logical KV DMA op and estimator hooks,
  - optional reuse-locality mismatch penalty accounting.
- Commit message:
  - `feat(kv): add optional kv dma accounting`

### C11. Update project documentation

- Depends on: C10
- Files (2):
  - `workflow.md`
  - `attacc-README.md`
- Content:
  - trace mode workflow and CLI examples,
  - explicit KV reuse semantics by `hash_id`.
- Commit message:
  - `docs: add trace-mode workflow and kv reuse docs`

### C12. Validation and expectation testing step (added)

- Depends on: C11
- Files (1-3):
  - `workflow.md`
  - `attacc-README.md`
  - `plan.md` (checklist status update section)
- Content:
  - run validation scenarios and document pass/fail outcomes:
    1. fixed mode regression vs prior behavior,
    2. FIFO admission order for same/different timestamps,
    3. continuous backfill on completion,
    4. KV hit path skips recompute,
    5. LRU eviction when KV pool overflows,
    6. decode one-token-per-step behavior,
    7. output CSV consistency checks.
  - include a concise “expected vs observed” table for each scenario.
  - include one explicit sample-row check using:
    - `{"chat_id":0,"parent_chat_id":-1,"timestamp":0.0,"input_length":502,"output_length":1494,"type":"thinking","turn":1,"hash_ids":[...32 ids...]}`
    - expected parser result: valid row, 32 blocks, eligible for global KV reuse.
- Commit message:
  - `test(docs): validate expected behavior for trace mode and kv reuse`

## 8) Final validation checklist (end-of-plan gate)

1. `--mode fixed` still compiles and behaves as before.
2. Trace mode consumes variable-length requests with timestamps.
3. Scheduler enforces FIFO admission and immediate backfill.
4. KV `hash_id` hits are reused (no recomputation).
5. KV capacity obeys HBM budget and LRU eviction.
6. Decode advances one token/request/step.
7. CSV outputs are generated and metrics are internally consistent.
8. Optional DMA penalties only appear when enabled and applicable.
9. Sample trace row with `input_length=502` is parsed as 32 hash blocks.
