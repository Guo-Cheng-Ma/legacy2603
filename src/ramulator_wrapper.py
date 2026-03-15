import pandas as pd
import subprocess
import math
import os
import sys
import tempfile
from src.config import *
from src.model import *
from src.type import *


class Ramulator:

    CACHE_COLUMNS = [
        'L', 'nhead', 'dhead', 'dbyte', 'pim_type', 'power_constraint',
        'cycle', 'mac', 'softmax', 'mvgb', 'mvsb', 'wrgb'
    ]

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_pim_die=5):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        self.df = self._load_cache_df(output_log)
        self.tCK = 0.769  # ns
        self.num_pim_die = num_pim_die
        self.nhead = modelinfos['num_heads']
        self.fast_mode = fast_mode

    def _load_cache_df(self, path):
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=self.CACHE_COLUMNS)

        try:
            # Tolerate stale/corrupted cache lines instead of crashing startup.
            df = pd.read_csv(path, on_bad_lines='skip')
        except TypeError:
            # Backward-compatible fallback for older pandas versions.
            df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)
        except Exception as e:
            print(f"Warning: failed to read cache file {path}: {e}")
            return pd.DataFrame(columns=self.CACHE_COLUMNS)

        if df.empty:
            return pd.DataFrame(columns=self.CACHE_COLUMNS)

        for col in self.CACHE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[self.CACHE_COLUMNS]

        numeric_cols = [
            'L', 'nhead', 'dhead', 'dbyte', 'cycle', 'mac', 'softmax', 'mvgb',
            'mvsb', 'wrgb'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Normalize bool parsing for cached files written by different runs.
        power_map = {'true': True, 'false': False}
        df['power_constraint'] = (
            df['power_constraint'].astype(str).str.strip().str.lower().map(
                power_map))

        df['pim_type'] = df['pim_type'].astype(str).str.strip()
        df = df.dropna(subset=self.CACHE_COLUMNS)
        return df

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        if self.df.empty:
            df = self._load_cache_df(self.output_log)
        else:
            df = self.df
        if df.empty:
            df = pd.DataFrame(columns=self.CACHE_COLUMNS)
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        if not self.output_log:
            return

        output_abs = os.path.abspath(self.output_log)
        output_dir = os.path.dirname(output_abs) or "."
        output_name = os.path.basename(output_abs)
        os.makedirs(output_dir, exist_ok=True)

        fd, tmp_log = tempfile.mkstemp(prefix=f"{output_name}.",
                                       suffix=".tmp",
                                       dir=output_dir)
        os.close(fd)
        try:
            self.df.to_csv(tmp_log, index=False)
            os.replace(tmp_log, output_abs)
        except FileNotFoundError:
            # Fallback for rare temp-file races from concurrent runs.
            self.df.to_csv(output_abs, index=False)
        finally:
            if os.path.exists(tmp_log):
                os.remove(tmp_log)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dhead, dbyte,
                      yaml_file, file_name):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_attacc_{}.py".format(pim_type_name))
        # Ensure maxlen is always valid for current seqlen.
        # Trace generators use maxlen for partition sizing.
        maxlen = max(int(l), 4096)

        # generate trace
        try:
            subprocess.run(
                [
                    sys.executable,
                    trace_exc,
                    "--dhead",
                    str(dhead),
                    "--nhead",
                    str(num_ops_per_hbm),
                    "--seqlen",
                    str(l),
                    "--maxlen",
                    str(maxlen),
                    "--dbyte",
                    str(dbyte),
                    "--output",
                    trace_file,
                ],
                check=True,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise

        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # remove trace
        try:
            if os.path.exists(trace_file):
                os.remove(trace_file)
        except Exception as e:
            print(f"Error: {e}")

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            # Cache keys and trace generation must follow the realized layer
            # shape, not the static model table, so warm pre-generated caches
            # match runtime lookups for models such as Qwen3.
            dhead = layer.k
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_pim_die = math.ceil(num_ops_per_attacc / self.num_pim_die)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_pim_die / minimum_heads)
                num_ops_per_pim_die = minimum_heads

            file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, num_ops_per_pim_die, dhead, layer.dbyte, int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, l, num_ops_per_pim_die,
                                        dhead,
                                        layer.dbyte, yaml_file, file_name)

            # remove yaml
            try:
                if os.path.exists(yaml_file):
                    os.remove(yaml_file)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            ## update log file

            log = [
                l, num_ops_per_pim_die, dhead, dbyte, pim_type.name,
                power_constraint
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_pim_die for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_pim_die = math.ceil(num_ops_per_attacc / self.num_pim_die)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_pim_die / minimum_heads)
            num_ops_per_pim_die = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_pim_die) & \
                      (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                      (self.df['power_constraint'] == power_constraint) &  \
                      (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cached_cols = ['cycle', 'mac', 'softmax', 'mvgb', 'mvsb', 'wrgb']
            row = row.dropna(subset=cached_cols)
            if row.empty:
                return self.run(pim_type, layer, power_constraint)

            try:
                cycle = int(row.iloc[0]['cycle'])
                mac = int(row.iloc[0]['mac'])
                softmax = int(row.iloc[0]['softmax'])
                mvgb = int(row.iloc[0]['mvgb'])
                mvsb = int(row.iloc[0]['mvsb'])
                wrgb = int(row.iloc[0]['wrgb'])
            except (TypeError, ValueError):
                # Stale or corrupted cache entries should not crash simulation.
                return self.run(pim_type, layer, power_constraint)
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 2

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_pim_die for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
