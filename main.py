import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *
from src.trace_simulator import run_trace_simulation, write_trace_outputs

RAMULATOR = False


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def _trace_name(trace_file: str) -> str:
    trace_stem = Path(trace_file).stem
    trace_stem_lower = trace_stem.lower()
    trace_aliases = {
        'tracea': 'traceA',
        'traceb': 'traceB',
        'coder': 'coder',
        'thinking': 'thinking',
    }
    for needle, alias in trace_aliases.items():
        if needle in trace_stem_lower:
            return alias
    return trace_stem


def _result_subdirs(config_path: str, trace_file: str):
    cfg_path = Path(str(config_path or ""))
    cfg_parts_lower = [part.lower() for part in cfg_path.parts]
    family = ""
    trace_name = _trace_name(trace_file)

    if "configs" in cfg_parts_lower:
        cfg_idx = cfg_parts_lower.index("configs")
        trailing = cfg_path.parts[cfg_idx + 1:]
        if len(trailing) >= 2:
            family = trailing[0]
            trace_name = trailing[1]
    return family, trace_name


def _trace_output_paths(config_path: str, trace_file: str, die_type: str, compute_stack_l1: int,
                        model: str, ngpu: int):
    now = datetime.now()
    date_tag = now.strftime("%y%m%d")
    time_tag = now.strftime("%H%M%S")
    family, trace_name = _result_subdirs(config_path, trace_file)
    trace_dir = Path("results") / date_tag
    if family:
        trace_dir = trace_dir / family
    trace_dir = trace_dir / trace_name
    trace_dir.mkdir(parents=True, exist_ok=True)

    die_tag = str(die_type).lower()
    compute_tag = str(int(compute_stack_l1))
    model_tag = str(model)
    ngpu_tag = f"{int(ngpu)}gpu"

    summary_path = trace_dir / f"S-{die_tag}-{compute_tag}-{model_tag}-{ngpu_tag}-{time_tag}.yaml"
    requests_path = trace_dir / f"R-{die_tag}-{compute_tag}-{model_tag}-{ngpu_tag}-{time_tag}.jsonl"
    return str(summary_path), str(requests_path)


def main():
    parser = argparse.ArgumentParser(
        description="AttAcc Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default='configs/default.yaml',
        help="path to unified YAML config file")

    cli_args = parser.parse_args()
    args = load_unified_config(cli_args.config)

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0

    if args.system == 'dgx-attacc':
        die_type = args.die_type
        if die_type in ['vstack', 'uniform']:
            num_pim_die = 5  # all dies are hybrid
        else:
            num_pim_die = args.num_pim_die
            if num_pim_die >= 5:
                print("ERROR: attacc mode requires num_pim_die < 5 (GPU needs at least 1 HBM die for BW); "
                      "use die_type: vstack or uniform for all-hybrid dies")
                raise SystemExit(1)
        print("{}: ({} x {}), PIM:{}, die-type:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim, die_type,
            [args.lin, args.lout, args.batch]))
    else:
        num_pim_die = 0
        die_type = 'attacc'  # non-PIM system, ignore die_type
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024 if args.gmemcap is not None else None
    output_path = "output.csv"
    if os.path.exists(output_path):
        os.remove(output_path)

    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap,
                                  num_pim_die=num_pim_die, die_type=die_type,
                                  compute_stack_l1=args.compute_stack_l1,
                                  capacity_stack_l2=args.capacity_stack_l2)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        kv_arch_cfg = load_hetero_kv_arch_config(args.kv_arch_config)
        weight_bytes, _, _ = system.get_required_mem_capacity(batch_size=1, lin=1, lout=1)
        validate_hetero_weight_capacity(weight_bytes_total=weight_bytes, hetero_kv_arch=kv_arch_cfg)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        pim_config = make_pim_config(pim_type,
                                     InterfaceType.NVLINK3,
                                     num_pim_die=num_pim_die,
                                     power_constraint=args.powerlimit,
                                     die_type=die_type,
                                     compute_stack_l1=args.compute_stack_l1,
                                     capacity_stack_l2=args.capacity_stack_l2)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    if args.mode == 'trace':
        result = run_trace_simulation(
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
        summary = result['summary']
        summary_path, requests_path = _trace_output_paths(
            config_path=cli_args.config,
            trace_file=args.trace_file,
            die_type=args.die_type,
            compute_stack_l1=args.compute_stack_l1,
            model=args.model,
            ngpu=args.ngpu,
        )
        write_trace_outputs(result, summary_path=summary_path, requests_path=requests_path)
        print(
            "Trace mode done: requests={} total_time={:.6f}s kv_hits={} kv_misses={} outputs=[{},{}]".format(
                summary['num_requests'],
                summary['total_time_s'],
                summary['kv_hit_blocks'],
                summary['kv_miss_blocks'],
                summary_path,
                requests_path,
            )
        )
    else:
        run(system,
            args.batch,
            args.lin,
            args.lout,
            pipe=args.pipeopt,
            parallel=args.ffopt,
            output_file=output_path,
            power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
