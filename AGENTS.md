# AGENTS.md - AttAcc Simulator Agent Guide

## Project Overview

Cycle-accurate simulator for **HBM-PIM + GPU hybrid inference systems**. Models transformer-based generative model (TbGM) inference on heterogeneous DGX-class hardware with optional AttAcc (Attention Accelerator) backed by HBM-PIM.

## Architecture

```
main.py                        # Entry point: --mode fixed | trace
src/
  type.py                      # Enums: DataType, LayerType, DeviceType, PIMType, GPUType, etc.
  config.py                    # Hardware configs (A100a/H100), model tables, energy tables,
                               #   3-tier KV arch YAML loader, capacity/bandwidth helpers
  model.py                     # Layer class, Transformer builder (sum_decoder + gen_decoder)
  devices.py                   # xPU (GPU/CPU) and PIM device models: timing, energy, tiling
  system.py                    # System orchestration: simulate(), estimate_prefill/decode/kv_transfer
  ramulator_wrapper.py         # Ramulator2 subprocess runner + CSV cache (ramulator.out)
  request_state.py             # TraceRequest (immutable JSONL row) + RequestState (mutable runtime)
  trace_loader.py              # JSONL parser with validation (hash_ids, block size = 16 tokens)
  kv_cache.py                  # 3-tier LRU KV cache: L1/L2/L3 with cascading eviction
  continuous_scheduler.py      # FIFO continuous batching: chunked prefill + decode + backfill
  static_batch_scheduler.py    # FIFO static batching: no mid-batch backfill
  trace_simulator.py           # Trace-mode orchestration + CSV output writers
configs/
  kv_arch.yaml                 # 3-tier KV capacity/bandwidth config
ramulator2/                    # C++ cycle-accurate DRAM/PIM simulator (DO NOT MODIFY without permission)
  ramulator2                   # Built binary
  trace_gen/                   # Trace generators + parallel pre-generation tool
llm-req-inputs/                # JSONL request traces (large files, do not cat)
ramulator.out                  # Ramulator result cache CSV (accelerates re-runs)
results/                       # Output directory for batch experiment results
timeline.csv                   # Step-by-step change log (append after each committed step)
```

## Two Simulation Modes

### Fixed Mode (`--mode fixed`)
- Static `(batch, lin, lout)` simulation.
- `System.simulate()` builds full summarization + generation decoder layers.
- PIM attention goes through `ramulator_wrapper.py` -> `ramulator2` binary.
- Output: `output.csv`.

### Trace Mode (`--mode trace`)
- JSONL-driven requests with `timestamp`, `input_length`, `output_length`, `hash_ids`.
- Scheduler choices: `--trace-scheduler continuous` (default) or `static`.
- Continuous: chunked prefill + per-token decode + immediate backfill.
- Static: batch-at-a-time, decode runs `max(output_length)` steps, no backfill.
- Global KV reuse by `hash_id` with 3-tier LRU (L1/L2/L3) and migration penalties.
- Output: `results/<date>/<trace>/S-*.yaml` + `results/<date>/<trace>/R-*.jsonl`.

## Key Constraints and Rules

1. **Do NOT modify `ramulator2/`** without explicit user permission.
2. **`rm` is banned** - never use `rm` in any command.
3. **All changes are small-step**: edit -> test/validate -> `git commit` -> `git push` -> append to `timeline.csv`.
4. **`llm-req-inputs/` files are large** - never `cat` them entirely; use `head` or load programmatically.
5. **`ramulator.out`** is a cache file that accelerates simulation - do not delete or corrupt it.
6. **Python frontend only** - modifications go in `src/` and `main.py`.
7. **Git push via SSH** - use `git push git@github.com:pku-lemonade/vStack.git main` (HTTPS proxy is unreliable).
8. **Maintain `TODO.local.md`** at the repo root as a Markdown working checklist during active tasks; keep it untracked unless the user explicitly asks to commit it.

## Python Environment

Always activate the conda environment before executing any command in this repo:
```bash
source /home/lizhuoran200/miniconda3/etc/profile.d/conda.sh && conda activate attacc_baseline
```
This environment has all required dependencies (pandas, numpy, etc.), and all shell commands in this repo should be run after this activation step.

## Default Runtime Knobs

Unless the user says otherwise, assume:
- `powerlimit: true`
- `ffopt: true`
- `pipeopt: true`

For BA Ramulator cache warmup, match the same power mode:
- use `--power-modes 1` by default

## Development Workflow

For each change step:

1. **Edit**: Make targeted changes in `src/` or `main.py`.
2. **Sync docs**: If the simulator's functionality, spec, outputs, or workflow changed, update `architecture_summary.md` and `AGENTS.md` in the same step before committing.
3. **Validate**: Run unit test or full simulation:
   ```bash
   # Fixed mode smoke test
   python3 main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bank

   # Trace mode smoke test (small subset)
   python3 main.py --mode trace --system dgx-attacc --gpu A100a --ngpu 8 --model Qwen3-32B --pim bank \
     --trace-file llm-req-inputs/example.jsonl --max-batch-size 16 --prefill-chunk-tokens 128
   ```
4. **Commit**: `git add <files> && git commit -m "<message>"`
5. **Push**: `git push`
6. **Log**: Append row to `timeline.csv` with format:
   ```
   timestamp,step,status,files,build,notes
   ```

## Warm Cache Pregeneration

Default shell prompt for `Qwen3-32B` BA cache warmup into the shared [ramulator.out](/home/lizhuoran200/vstack/attacc_simulator/ramulator.out):

```bash
source /home/lizhuoran200/miniconda3/etc/profile.d/conda.sh && conda activate attacc_baseline && cd /home/lizhuoran200/vstack/attacc_simulator/ramulator2/trace_gen && python3 pregen_ramulator_bank.py --model Qwen3-32B --ngpu 8 --num-hbm 5 --batch-min 16 --batch-max 16 --seqlen-min 1 --seqlen-max 31000 --maxlen-floor 4096 --dbyte 2 --power-modes 1 --workers 96 --flush-every 200 --ramulator-out /home/lizhuoran200/vstack/attacc_simulator/ramulator.out --tmp-dir /home/lizhuoran200/vstack/attacc_simulator/ramulator2/trace_gen/tmp
```

## Model Config Table

Models defined in `config.py:make_model_config()`:
- `[ndec, hdim, nheads, dhead, ff_scale, gqa_size]`
- Supported: GPT-175B/89B/13B, LLAMA-7B/65B, MT-76B/146B/310B/530B/1008B, OPT-66B, Qwen3-4B/32B, Mistral-Devstral2-123B, Llama-3.1-405B

## Hardware Configs

- **GPU**: A100a (HBM3, 3352 GB/s, 312 TFLOPS) or H100 (3352 GB/s, 989.4 TFLOPS)
- **PIM types**: BA (bank-level), BG (bank-group), BUFFER (pseudo-channel/buffer-die)
- **Interfaces**: NVLink3 (600 GB/s), NVLink4 (900 GB/s), PCIe4 (64 GB/s), PCIe5 (128 GB/s)

## Trace Input Format

```json
{"chat_id": 0, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 502, "output_length": 1494, "type": "thinking", "turn": 1, "hash_ids": [0, 1, ..., 31]}
```
- Validation: `len(hash_ids) == ceil(input_length / 16)`
- Block size: 16 tokens

## Energy Model

- Energy tracked per-component: `[dram, l2, l1, reg, alu, comm]` in pJ.
- Migration energy: DMA (L1<->L2) and PCIe (L2<->L3) transfers.
- Callback (promotion) latency is non-overlapped; eviction latency is overlapped.

## KV Cache Tiers

- **L1**: High-speed HBM KV (default 20 GiB/card x 8 cards = 160 GiB)
- **L2**: High-capacity HBM (40 GiB/card total, minus weights = KV remainder)
- **L3**: Host memory spill (default 512 GiB)
- Access policy: always promote to L1; cascading eviction L1->L2->L3->drop
- Spare ratio: 1% (effective 99% capacity threshold)

## Timeline CSV Format

```csv
timestamp,step,status,files,build,notes
2026-02-26T20:41:51+08:00,C39,completed,main.py|src/trace_loader.py|src/trace_simulator.py|timeline.csv,cmake+make pass,added --timestamp-scaling...
```
- Step IDs: C1, C2, ... (sequential)
- Latest step: C56
