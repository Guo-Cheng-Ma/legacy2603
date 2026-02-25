from src.type import *
import copy

SCALING_FACTOR = {}
SCALING_FACTOR['MAX_COMPUTE_UTIL'] = 0.8
SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = 0.85

# Heterogeneous 3-tier KV cache architecture defaults.
# Prefer providing these via YAML in trace mode (`--kv-arch-config`).
DEFAULT_HETERO_KV_ARCH = {
    "NUM_CARDS": 8,
    "GPU_MEM_PER_CARD_GB": 60,
    "HISPEED_KV_PER_CARD_GB": 20,  # L1 (compute layers)
    "HICAP_TOTAL_PER_CARD_GB": 40,  # L2 (weights + KV)
    "HOST_KV_TOTAL_GB": 512,  # L3 host spill tier
    "DMA_BW_BPS": 1676 * 1000 * 1000 * 1000,  # default: 0.5 * A100 HBM3 BW
    "PCIE_BW_BPS": 64 * 1000 * 1000 * 1000,  # PCIe 4.0 x16
}


def gib_to_bytes(gib: float) -> int:
    return int(float(gib) * 1024 * 1024 * 1024)


def _required_numeric(cfg, key):
    if key not in cfg:
        raise ValueError(f"missing key in kv arch config: {key}")
    try:
        value = float(cfg[key])
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric value for {key}: {cfg[key]}")
    if value <= 0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return value


def _parse_simple_yaml_map(yaml_text: str) -> dict:
    """
    Minimal YAML parser for this project config format.
    Supports flat key:value pairs and one top-level nested map (e.g., kv_arch:).
    """
    root = {}
    section = None
    for raw in yaml_text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue

        if not raw.startswith(' ') and line.endswith(':'):
            section = line[:-1].strip()
            root.setdefault(section, {})
            continue

        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            parsed = value[1:-1]
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value

        target = root[section] if raw.startswith(' ') and section else root
        target[key] = parsed
    return root


def load_hetero_kv_arch_config(yaml_path: str = None) -> dict:
    """Load heterogeneous KV architecture config from YAML with defaults."""
    cfg = copy.deepcopy(DEFAULT_HETERO_KV_ARCH)
    if yaml_path is None:
        return cfg

    with open(yaml_path, 'r', encoding='utf-8') as handle:
        text = handle.read()

    try:
        import yaml
        parsed = yaml.safe_load(text) or {}
    except ImportError:
        parsed = _parse_simple_yaml_map(text)

    root = parsed.get("kv_arch", parsed)
    if not isinstance(root, dict):
        raise ValueError("kv arch YAML must be a map or contain a `kv_arch` map")

    normalized = {
        "NUM_CARDS": root.get("num_cards", cfg["NUM_CARDS"]),
        "GPU_MEM_PER_CARD_GB": root.get("gpu_mem_per_card_gb", cfg["GPU_MEM_PER_CARD_GB"]),
        "HISPEED_KV_PER_CARD_GB": root.get("hispeed_kv_per_card_gb", cfg["HISPEED_KV_PER_CARD_GB"]),
        "HICAP_TOTAL_PER_CARD_GB": root.get("hicap_total_per_card_gb", cfg["HICAP_TOTAL_PER_CARD_GB"]),
        "HOST_KV_TOTAL_GB": root.get("host_kv_total_gb", cfg["HOST_KV_TOTAL_GB"]),
        "DMA_BW_BPS": root.get("dma_bandwidth_gbps", cfg["DMA_BW_BPS"] / 1e9) * 1e9,
        "PCIE_BW_BPS": root.get("pcie_bandwidth_gbps", cfg["PCIE_BW_BPS"] / 1e9) * 1e9,
    }

    # Validate required numeric values.
    validated = {
        "NUM_CARDS": int(_required_numeric(normalized, "NUM_CARDS")),
        "GPU_MEM_PER_CARD_GB": _required_numeric(normalized, "GPU_MEM_PER_CARD_GB"),
        "HISPEED_KV_PER_CARD_GB": _required_numeric(normalized, "HISPEED_KV_PER_CARD_GB"),
        "HICAP_TOTAL_PER_CARD_GB": _required_numeric(normalized, "HICAP_TOTAL_PER_CARD_GB"),
        "HOST_KV_TOTAL_GB": _required_numeric(normalized, "HOST_KV_TOTAL_GB"),
        "DMA_BW_BPS": _required_numeric(normalized, "DMA_BW_BPS"),
        "PCIE_BW_BPS": _required_numeric(normalized, "PCIE_BW_BPS"),
    }
    return validated


def get_hetero_transfer_bandwidths(hetero_kv_arch: dict = None) -> dict:
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch
    return {
        "dma_bw_bps": float(cfg["DMA_BW_BPS"]),
        "pcie_bw_bps": float(cfg["PCIE_BW_BPS"]),
    }


def get_hetero_kv_capacities(weight_bytes_total: int = 0, hetero_kv_arch: dict = None) -> dict:
    """Return architecture-derived global KV capacities for L1/L2/L3 tiers."""
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch
    num_cards = int(cfg["NUM_CARDS"])
    l1_total = gib_to_bytes(
        cfg["HISPEED_KV_PER_CARD_GB"] * num_cards
    )
    l2_total = gib_to_bytes(
        cfg["HICAP_TOTAL_PER_CARD_GB"] * num_cards
    )
    l3_total = gib_to_bytes(cfg["HOST_KV_TOTAL_GB"])

    weight_bytes = max(0, int(weight_bytes_total))
    l2_kv = max(0, l2_total - weight_bytes)
    return {
        "l1_kv_bytes": l1_total,
        "l2_total_bytes": l2_total,
        "l2_kv_bytes": l2_kv,
        "l3_kv_bytes": l3_total,
        "weight_bytes_total": weight_bytes,
    }

# ENERGY_TABLE: pJ per byte
# Cache info: https://core.ac.uk/download/pdf/232142915.pdf
ENERGY_TABLE = {
    'GPU': {},
    'CPU': {},
    'PIM': {
        PIMType.BA: {},
        PIMType.BG: {},
        PIMType.BUFFER: {}
    }
}
ENERGY_TABLE['GPU']['reg'] = 0.0675
#4-way cache, ref: https://arxiv.org/pdf/1509.02308v1.pdf
ENERGY_TABLE['GPU'][ 'l1'] = 0.16 * 8  
ENERGY_TABLE['GPU']['l2'] = 0.3 * 8
ENERGY_TABLE['GPU']['alu'] = 0.32
ENERGY_TABLE['GPU']['mem'] = (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8
# ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['GPU'][ 'comm'] = 1.3 * 8  

## TODO: Add energy of CPU (pJ per byte)
ENERGY_TABLE['CPU']['reg'] = 0
ENERGY_TABLE['CPU']['l1'] = 0
ENERGY_TABLE['CPU']['l2'] = 0
ENERGY_TABLE['CPU']['alu'] = 0
ENERGY_TABLE['CPU']['mem'] = 0
ENERGY_TABLE['CPU']['comm'] = 0

## 2017 MICRO FGDRAM
## https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
## Cell (ACT/PRE) energy: 0.11pJ/b,
## Cell (RD/WRT) energy: 0.44pJ/b,

## RD/WR Energy (column decoder to BG MUX): 1.01 pJ/b
## RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/b
## TSV energy : 0.5 pJ/b
## Silicon interposer IO energy : 0.3 pJ/b

## energy_table = [energy between DRAM cell and PE, energy between PE and buffer die

ENERGY_TABLE['PIM'][PIMType.BA]['mem'] = (0.11 +
                                          0.44) * 8  #, (1.01 + 1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BG]['mem'] = (0.11 + 0.44 +
                                          1.01) * 8  #, (1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['mem'] = (0.11 + 0.44 + 1.01 + 1.23 +
                                              0.5) * 8  #, 0]

ENERGY_TABLE['PIM'][PIMType.BA]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BG]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BUFFER]['sram'] = 0.0034

ENERGY_TABLE['PIM'][PIMType.BA]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BG]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BUFFER]['alu'] = 0.32

ENERGY_TABLE['PIM'][PIMType.BA]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BG]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['io'] = [0.3, 0.5, 1.23, 1.01]

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['PIM'][PIMType.BA]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BG]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BUFFER]['comm'] = 10.4


def make_xpu_config(gpu_type: GPUType,
                    num_gpu=None,
                    flops=None,
                    mem_cap=None,
                    mem_bw=None,
                    power_constraint=True):
    config = {'GPU': {}, 'CPU': {}}
    config['GPU']["GPUTYPE"] = gpu_type
    config['GPU']["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu

    if gpu_type == GPUType.A100a:
        # Ref: DGX-A100 whitepaper
        config['GPU']["NUM_CORE"] = 108
        config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                    if mem_cap is None else mem_cap

        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
                                                  if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 3.8 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 192 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 40 * 1024 * 1024
        config['GPU']["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 64
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 200 * 1000 * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 96 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 256 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 64 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    elif gpu_type == GPUType.H100:
        # Ref: DGX-H100 whitepaper
        config['GPU']["NUM_CORE"] = 132
        config['GPU']["FLOPS_PER_DEVICE"] = 989.4 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                   if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
                                                 if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 256 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 50 * 1024 * 1024
        # NVLINK: 900GB/s (Read 450GB/s Write 450GB/s)
        config['GPU']["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # H100 DGX CPU configuration sapphire-rapids
        # https://www.servethehome.com/4th-gen-intel-xeon-scalable-sapphire-rapids-leaps-forward/7/
        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 56
        # 4TFLOPS per CPU (half precision)
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        # (2TB, dual processors)
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        # channels x dpc x 4400 MT/s  https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 8 * 2 * 4400 * (
            64 / 8) * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 48 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 2 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 128 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    return config


# Rank x BG x BA / 2 (tCCD)
BW_SCALE = {
    False: {
        PIMType.BA: 2 * 4 * 4 / 2,
        PIMType.BG: 2 * 4,
        PIMType.BUFFER: 1
    },
    True: {
        PIMType.BA: 9,
        PIMType.BG: 3,
        PIMType.BUFFER: 1
    }
}


def make_pim_config(pim_type: PIMType,
                    interface_type: InterfaceType,
                    opb=1,
                    num_attacc=8,
                    num_hbm=5,
                    bw_scale=None,
                    power_constraint=False):
    config = {}
    config["PIM_TYPE"] = pim_type
    config["POWER_CONSTRAINT"] = power_constraint
    config["ENERGY_TABLE"] = ENERGY_TABLE['PIM'][pim_type]

    internal_bandwidth_scale =  BW_SCALE[power_constraint][pim_type] \
                                if bw_scale is None else bw_scale
    config["NUM_ATTACC"] = num_attacc
    config["NUM_HBM"] = num_hbm
    config["MEM_CAPACITY_PER_HBM"] = 16 * 1024 * 1024 * 1024
    config[
        "MEM_BW_PER_HBM"] = 670.4 * 1000 * 1000 * 1000 * internal_bandwidth_scale
    config["FLOPS_PER_HBM"] = config["MEM_BW_PER_HBM"] * opb
    config["SOFTMAX_MEM_BW"] = 670.4 * 1000 * 1000 * 1000 * num_hbm
    config["SOFTMAX_FLOPS"] = config["SOFTMAX_MEM_BW"]

    if interface_type == InterfaceType.NVLINK3:
        config["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.NVLINK4:
        config["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE4:
        config["INTERFACE_BW"] = 64 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE5:
        config["INTERFACE_BW"] = 128 * 1000 * 1000 * 1000
    else:
        assert 0, "Invalid interface type"

    return config


def make_model_config(name, dtype):
    model_table = {}
    model_table['GPT-175B'] = [96, 12288, 96, 128, 4, 1]
    model_table['GPT-89B'] = [48, 12288, 96, 128, 4, 1]
    model_table['GPT-13B'] = [40, 5120, 40, 128, 4, 1]
    model_table['LLAMA-7B'] = [32, 4096, 32, 128, 8 / 3, 1]
    model_table['LLAMA-65B'] = [80, 8192, 64, 128, 8 / 3, 1]
    model_table['MT-76B'] = [60, 10240, 40, 128, 4, 1]
    model_table['MT-146B'] = [80, 12288, 80, 128, 4, 1]
    model_table['MT-310B'] = [96, 16384, 128, 128, 4, 1]
    model_table['MT-530B'] = [105, 20480, 128, 160, 4, 1]
    model_table['MT-1008B'] = [128, 25600, 160, 160, 4, 1]
    model_table['OPT-66B'] = [64, 9216, 72, 128, 4, 1]

    ndec, hdim, nheads, dhead, ff_scale, gqa_size = model_table[name]
    config = {
        'name': name,
        'ndec': ndec,
        'hdim': hdim,
        'num_heads': nheads,
        'dhead': dhead,
        'ff_scale': ff_scale,
        'gqa_size': gqa_size,
        'dtype': dtype
    }
    return config
