# Simulator Architecture Summary

This note summarizes the system architecture that the simulator is trying to model, and the software stack used to implement that model.

## 1) What system is being simulated

The simulator targets heterogeneous DGX-class inference nodes running transformer decoding across multiple GPUs, with optional attention offload to an HBM-based processing-in-memory accelerator called AttAcc.

At a high level, the modeled machine has:

- multiple GPUs participating in tensor-parallel transformer inference
- NVLink-based GPU-to-GPU communication
- optional attention acceleration on HBM-PIM instead of on the GPU cores
- optional CPU fallback for attention offload
- a trace-mode KV cache hierarchy that spans device-local HBM and host memory

The workload being modeled is autoregressive transformer inference with two phases:

- prefill: process the input prompt and build KV state
- decode: generate output tokens one step at a time

## 2) Hardware architecture the simulator describes

### Compute devices

The frontend models two GPU families:

- `A100a`
- `H100`

For each GPU type, the simulator tracks:

- device count
- peak FLOPS
- off-chip memory bandwidth
- cache capacities
- interconnect bandwidth
- energy per byte / per operation

The baseline execution model is that most transformer layers run on the GPUs, while the attention score/context path can be redirected to a heterogeneous accelerator.

### AttAcc / PIM device

AttAcc represents an HBM-PIM accelerator embedded in HBM die packages. The simulator supports three placement granularities for the PIM compute:

- `BA`: bank-level PIM
- `BG`: bank-group-level PIM
- `BUFFER`: buffer-die / pseudo-channel style PIM

Conceptually, AttAcc is used to accelerate the attention-heavy part of decoding, especially KV-heavy score/context operations, while dense FC/FFN work remains on the GPUs.

### Two die organizations

The current frontend describes two hardware organizations.

`attacc` mode:

- some HBM die packages are dedicated PIM dies
- the remaining die packages behave as conventional GPU-attached HBM
- GPU memory bandwidth and high-capacity memory shrink as more dies are assigned to PIM
- PIM dies contribute high-speed KV space

`vstack` mode:

- every die package is hybrid
- each die package is split into compute-oriented stacks and capacity-oriented stacks
- bandwidth remains GPU-like, while usable memory capacity is partitioned by stack role
- the split is controlled by `compute_stack_l1` and `capacity_stack_l2`

### HBM and KV hierarchy

The trace-mode memory model is a topology-aware 3-tier KV hierarchy:

- L1: high-speed KV in HBM, placed at `(card, die, bank)`
- L2: higher-capacity HBM KV, placed at `(card, die)`
- L3: host-memory spill tier
- optional replica pool carved out of per-card L1 or L2 when `placement_policy: hotset_broadcast`

Important details:

- the block size is 16 tokens
- each request is assigned a physical home card/die
- each KV block is identified by `hash_id`
- L1 placement is banked by `hash_id % banks_per_die`
- the eviction path is `L1 -> L2 -> L3 -> drop`
- cache management uses an effective 99% threshold via a 1% spare ratio
- canonical placement remains unique-copy even when replicas are enabled

### KV policy layer

Trace mode now exposes two policy dimensions through the unified YAML:

- `eviction_policy`
  - `lru`: recency-only eviction
  - `aware`: workload-aware eviction using per-block metadata and category-local reuse statistics
- `placement_policy`
  - `all_unique`: baseline unique-copy placement
  - `hotset_broadcast`: keep canonical unique placement, but replicate ultra-hot blocks into a reserved per-card pool

The cache metadata used by `aware` includes:

- `chat_id` locality
- `request_type`
- `turn_class` (`single` or `multi`)
- recent access time
- reuse count
- distinct cards seen
- remote-hit count

This lets the simulator model the production-style observation that reusable blocks are highly skewed, often short-lived, and not well-served by plain LFU/LRU alone.

### Data movement the simulator tries to capture

The frontend models several movement classes that matter for latency and energy:

- GPU-to-GPU collectives over NVLink
- request-home KV promotion back into L1
- same-die DMA movement
- cross-die HBM movement
- cross-card movement
- PCIe spill/refill to and from host memory
- replica fanout movement when a block is promoted into the broadcast hotset

In trace mode, the simulator also models overlap between independent transfer domains instead of charging every migration serially.

## 3) Transformer execution model

The model builder decomposes a decoder block into layer-level operations such as:

- `qkv`
- `score`
- `softmax`
- `context`
- `proj`
- `ff*`
- normalization
- communication layers (`G2G`, `X2G`)

The simulator distinguishes:

- summarization / prefill over the full input length
- generation / decode over one token at a time with growing context length

Execution is split across devices roughly as follows:

- FC, activation, normalization, and most communication are modeled on the GPU
- attention score / context / softmax can run on the accelerator path in heterogeneous mode
- PIM-backed attention timing comes from Ramulator2 rather than only from a closed-form Python estimate

## 4) Software stack

### Frontend layer: Python orchestration

The Python frontend is responsible for user-facing configuration and high-level simulation control.

- `main.py`
  - entry point
  - loads one unified YAML config
  - constructs the modeled system
  - selects fixed mode or trace mode
- `src/config.py`
  - holds hardware defaults, model tables, topology derivation, energy tables, and KV policy presets
  - converts YAML inputs into validated runtime settings
  - resolves model/trace-specific replica reserve ratios when YAML leaves them unset
- `src/type.py`
  - defines enums for devices, layer types, PIM types, interfaces, and precision

### Analytical simulation layer

These modules implement the analytical timing/energy model for the Python-side simulator.

- `src/model.py`
  - builds the transformer layer graph used for fixed mode and for trace-mode prefill/decode estimation
- `src/devices.py`
  - models GPU and CPU timing/energy analytically
  - models PIM execution and communication behavior
- `src/system.py`
  - orchestrates per-layer execution
  - aggregates performance and energy across summarization and generation stages

This layer answers questions such as:

- how long does a decoder step take on GPU vs PIM
- whether a layer is compute-bound or memory-bound
- how much DRAM/L2/L1/register/ALU/communication energy is consumed

### Trace-simulation layer

These modules turn the fixed-shape estimator into a request-driven serving simulator.

- `src/trace_loader.py`
  - reads JSONL traces and validates the request format
- `src/request_state.py`
  - stores immutable trace metadata plus mutable runtime state
- `src/kv_cache.py`
  - implements topology-aware KV placement, promotion, eviction, replica pools, and transfer overlap accounting
- `src/continuous_scheduler.py`
  - simulates FIFO continuous batching with chunked prefill and immediate backfill
- `src/static_batch_scheduler.py`
  - simulates FIFO static batching without mid-batch backfill
- `src/trace_simulator.py`
  - ties loader, cache, scheduler, and output writers together

For matrix configs stored under `configs/<family>/<trace>/...`, trace outputs are now grouped under the mirrored result hierarchy:

- `results/<date>/<family>/<trace>/S-*.yaml`
- `results/<date>/<family>/<trace>/R-*.jsonl`

The frontend also has a lightweight post-processing layer under `tools/` for analysis-ready figure generation:

- `tools/plot_generation_results.py`
  - scans trace summary YAMLs
  - deduplicates repeated runs by `(model, trace_family, mode)` while keeping the newest summary
  - exports a tidy CSV plus matplotlib figures under `figure/<yymmdd>-<hhmm>/` by default
  - renders energy breakdown normalized to `vstack-o`, normalized throughput, and TTFT/latency normalized to `vstack-o`
  - uses a broken y-axis for normalized plots when large outliers would otherwise flatten the smaller bars
  - filters out the legacy `example` trace family from grouped analysis output
  - keeps empty slots visible when a `(model, trace_family, mode)` combination is missing

This layer answers questions such as:

- how arrival rate affects TTFT and latency
- how much prefix reuse comes from shared `hash_id` blocks
- whether L1/L2/L3 pressure or PCIe spill dominates performance
- whether workload-aware eviction improves reuse over the `all_unique + lru` baseline
- whether bounded hotset replication reduces cross-card callbacks enough to justify reserved space
- how scheduling policy changes throughput and queue delay
- how the five published modes (`attacc`, `static`, `uniform`, `vstack-b`, `vstack-o`) compare after deduplication and queue-delay adjustment

### Backend layer: Ramulator2

The backend for detailed PIM-memory behavior is the modified `ramulator2/` subtree.

The Python side uses `src/ramulator_wrapper.py` to:

- generate a trace-generator input for the requested attention shape
- emit a temporary Ramulator YAML
- run the `ramulator2` binary
- parse `memory_system_cycles` and PIM command counters
- cache those results in `ramulator.out`

So the division of labor is:

- Python frontend: workload shape, system topology, scheduler, KV behavior, summary metrics
- Ramulator backend: cycle-accurate HBM/PIM behavior for attention kernels

## 5) Fixed mode vs trace mode

The simulator has two distinct operating modes.

Fixed mode:

- takes one static `(batch, lin, lout)` configuration
- builds a full decoder execution for that shape
- is useful for component-level throughput and energy studies

Trace mode:

- consumes a request trace with arrival times and per-request lengths
- simulates queueing, batching, backfill, KV reuse, migration, and completion timing
- is useful for service-level studies such as TTFT, latency, throughput, and cache pressure

Both modes share the same underlying device/model descriptions. Trace mode adds the request scheduler and KV-memory system on top of the fixed-shape estimator.

## 6) What the simulator is really trying to answer

Taken together, the frontend and backend are meant to answer a system question:

How does a multi-GPU inference server behave when attention is moved toward HBM-PIM, while KV state becomes a first-class, topology-aware resource across device HBM and host memory?

More concretely, the simulator is built to study tradeoffs among:

- GPU compute vs PIM attention offload
- die organization (`attacc` vs `vstack`)
- KV locality and reuse
- unique-copy placement vs bounded hotset replication
- LRU vs workload-aware eviction
- DMA / NVLink / PCIe movement cost
- continuous vs static batching
- throughput, TTFT, latency, and energy
