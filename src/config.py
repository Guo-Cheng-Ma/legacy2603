from src.type import *
import argparse
import copy
from pathlib import Path

SCALING_FACTOR = {}
SCALING_FACTOR['MAX_COMPUTE_UTIL'] = 0.8
SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = 0.85

HBM_DIE_PACKAGES_PER_CARD = 5
HBM_STACKS_PER_DIE_PACKAGE = 8
HBM_BANKS_PER_DIE_PACKAGE = 1024
FULL_HBM_DIE_CAPACITY_GB = 16.0
FULL_HBM_STACK_CAPACITY_GB = FULL_HBM_DIE_CAPACITY_GB / HBM_STACKS_PER_DIE_PACKAGE
PIM_DIE_CAPACITY_RATIO = 0.5
COMPUTE_STACK_CAPACITY_RATIO = 0.5

# Heterogeneous 3-tier KV cache architecture defaults.
# Prefer providing these via YAML in trace mode (`--kv-arch-config`).
DEFAULT_HETERO_KV_ARCH = {
    "NUM_CARDS": 8,
    "DIE_TYPE": "attacc",
    "NUM_PIM_DIE": 4,
    "COMPUTE_STACK_L1": 4,
    "CAPACITY_STACK_L2": 4,
    "HOST_KV_TOTAL_GB": 512,  # L3 host spill tier
    "DMA_BW_BPS": 1676 * 1000 * 1000 * 1000,  # default: 0.5 * A100 HBM3 BW
    "PCIE_BW_BPS": 64 * 1000 * 1000 * 1000,  # PCIe 4.0 x16
    "NUM_DIE_PACKAGES_PER_CARD": HBM_DIE_PACKAGES_PER_CARD,
    "STACKS_PER_DIE_PACKAGE": HBM_STACKS_PER_DIE_PACKAGE,
    "BANKS_PER_DIE_PACKAGE": HBM_BANKS_PER_DIE_PACKAGE,
    "FULL_DIE_CAPACITY_GB": FULL_HBM_DIE_CAPACITY_GB,
}

DEFAULT_AWARE_POLICY = {
    "CATEGORY_KEYS": ["request_type", "turn_class"],
    "INTRA_USER_BIAS": True,
    "SINGLE_TURN_BIAS": True,
    "TTL_MODE": "category_ema",
    "TTL_SAFETY_FACTOR": 1.0,
    "RECENCY_WEIGHT": 1.0,
    "REUSE_WEIGHT": 1.0,
    "LOCALITY_WEIGHT": 1.0,
    "COST_WEIGHT": 1.0,
    "HOTNESS_CAP": 8,
    "REPLICA_MIN_DISTINCT_CARDS": 3,
    "REPLICA_MIN_REMOTE_HITS": 8,
    "REPLICA_SCORE_THRESHOLD": 0.0,
}

DEFAULT_KV_POLICY = {
    "EVICTION_POLICY": "lru",
    "PLACEMENT_POLICY": "all_unique",
    "REPLICA_TIER": "auto",
    "REPLICA_RESERVE_RATIO_L1": None,
    "REPLICA_RESERVE_RATIO_L2": None,
    "AWARE_POLICY": copy.deepcopy(DEFAULT_AWARE_POLICY),
}

TRACE_MODEL_KV_POLICY_PRESETS = {
    "GPT-175B": {
        "traceA": {
            "REPLICA_TIER": "L1",
            "REPLICA_RESERVE_RATIO_L1": 0.20,
            "REPLICA_RESERVE_RATIO_L2": 0.0,
        },
        "traceB": {
            "REPLICA_TIER": "L1",
            "REPLICA_RESERVE_RATIO_L1": 0.15,
            "REPLICA_RESERVE_RATIO_L2": 0.0,
        },
        "thinking": {
            "REPLICA_TIER": "L1",
            "REPLICA_RESERVE_RATIO_L1": 0.05,
            "REPLICA_RESERVE_RATIO_L2": 0.0,
        },
        "coder": {
            "REPLICA_TIER": "L1",
            "REPLICA_RESERVE_RATIO_L1": 0.25,
            "REPLICA_RESERVE_RATIO_L2": 0.0,
        },
    },
    "Qwen3-32B": {
        "traceA": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.03,
        },
        "traceB": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.03,
        },
        "thinking": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.01,
        },
        "coder": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.10,
        },
    },
    "Qwen3-4B": {
        "traceA": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.02,
        },
        "traceB": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.03,
        },
        "thinking": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.10,
        },
        "coder": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.12,
        },
    },
    "Mistral-Devstral2-123B": {
        "traceA": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.25,
        },
        "traceB": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.25,
        },
        "thinking": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.07,
        },
        "coder": {
            "REPLICA_TIER": "L2",
            "REPLICA_RESERVE_RATIO_L1": 0.0,
            "REPLICA_RESERVE_RATIO_L2": 0.25,
        },
    },
}


def gib_to_bytes(gib: float) -> int:
    return int(float(gib) * 1024 * 1024 * 1024)


def bytes_to_gib(byte_count: int) -> float:
    return float(byte_count) / (1024.0 * 1024.0 * 1024.0)


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
    Supports flat key:value pairs and up to two nested map levels.
    """
    root = {}
    section = None
    subsection = None
    for raw in yaml_text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(raw) - len(raw.lstrip(' '))

        if indent == 0 and line.endswith(':'):
            section = line[:-1].strip()
            subsection = None
            root.setdefault(section, {})
            continue

        if indent == 2 and line.endswith(':') and section:
            subsection = line[:-1].strip()
            parent = root.setdefault(section, {})
            if not isinstance(parent, dict):
                parent = {}
                root[section] = parent
            parent.setdefault(subsection, {})
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
            low = value.lower()
            if low in ('true', 'yes'):
                parsed = True
            elif low in ('false', 'no'):
                parsed = False
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value

        if indent >= 4 and section and subsection:
            target = root[section][subsection]
        elif indent >= 2 and section:
            target = root[section]
            subsection = None
        else:
            target = root
            subsection = None
        target[key] = parsed
    return root


def _normalize_optional_ratio(value, key: str):
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a float in [0.0, 1.0], got {value}")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{key} must be in [0.0, 1.0], got {ratio}")
    return ratio


def _deep_merge_policy(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_policy(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_aware_policy(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {}
    key_map = {
        "category_keys": "CATEGORY_KEYS",
        "intra_user_bias": "INTRA_USER_BIAS",
        "single_turn_bias": "SINGLE_TURN_BIAS",
        "ttl_mode": "TTL_MODE",
        "ttl_safety_factor": "TTL_SAFETY_FACTOR",
        "recency_weight": "RECENCY_WEIGHT",
        "reuse_weight": "REUSE_WEIGHT",
        "locality_weight": "LOCALITY_WEIGHT",
        "cost_weight": "COST_WEIGHT",
        "hotness_cap": "HOTNESS_CAP",
        "replica_min_distinct_cards": "REPLICA_MIN_DISTINCT_CARDS",
        "replica_min_remote_hits": "REPLICA_MIN_REMOTE_HITS",
        "replica_score_threshold": "REPLICA_SCORE_THRESHOLD",
    }
    normalized = {}
    for key, value in raw.items():
        normalized[key_map.get(str(key), str(key).upper())] = value
    return normalized


def infer_trace_family(trace_file: str = "") -> str:
    name = Path(str(trace_file or "")).stem.lower()
    if "tracea" in name:
        return "traceA"
    if "traceb" in name:
        return "traceB"
    if "thinking" in name:
        return "thinking"
    if "coder" in name:
        return "coder"
    return ""


# ---------------------------------------------------------------------------
# Unified config: single YAML replaces all CLI arguments + kv_arch.yaml
# ---------------------------------------------------------------------------
_UNIFIED_DEFAULTS = {
    # system
    'system': 'dgx',
    'gpu': 'A100a',
    'ngpu': 8,
    'gmemcap': None,
    # pim
    'pim': 'bank',
    'num_pim_die': 4,
    'die_type': 'attacc',
    'compute_stack_l1': 4,
    'capacity_stack_l2': 4,
    'powerlimit': False,
    'ffopt': False,
    'pipeopt': False,
    # model
    'model': 'GPT-175B',
    'word': 2,
    # workload
    'mode': 'fixed',
    'lin': 2048,
    'lout': 128,
    'batch': 1,
    # trace
    'trace_file': 'llm-req-inputs/qwen_thinking_blksz_16.jsonl',
    'max_batch_size': 16,
    'prefill_chunk_tokens': 128,
    'trace_scheduler': 'continuous',
    'timestamp_scaling': 1.0,
    'trace_debug': False,
    'trace_debug_interval': 100,
    # kv_arch (sentinel: None means use DEFAULT_HETERO_KV_ARCH)
    'kv_arch_config': None,
}


def load_unified_config(yaml_path: str) -> argparse.Namespace:
    """Load unified YAML config and return an argparse.Namespace with all settings."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        text = f.read()

    try:
        import yaml
        parsed = yaml.safe_load(text) or {}
    except ImportError:
        parsed = _parse_simple_yaml_map(text)

    cfg = dict(_UNIFIED_DEFAULTS)

    # Flatten known sections into cfg
    for section_name in ('system', 'pim', 'model', 'workload', 'trace'):
        section = parsed.get(section_name, {})
        if isinstance(section, dict):
            for key, value in section.items():
                if key in cfg:
                    cfg[key] = value

    # kv_arch section: keep bandwidth/host overrides here and derive capacity inputs from unified config.
    kv_section = parsed.get('kv_arch', None)
    kv_cfg = dict(kv_section) if isinstance(kv_section, dict) else {}
    kv_cfg.setdefault('num_cards', cfg['ngpu'])
    kv_cfg.setdefault('die_type', cfg['die_type'])
    kv_cfg.setdefault('num_pim_die', cfg['num_pim_die'])
    kv_cfg.setdefault('compute_stack_l1', cfg['compute_stack_l1'])
    kv_cfg.setdefault('capacity_stack_l2', cfg['capacity_stack_l2'])
    cfg['kv_arch_config'] = kv_cfg

    # Type coercions
    cfg['ngpu'] = int(cfg['ngpu'])
    cfg['num_pim_die'] = int(cfg['num_pim_die'])
    cfg['compute_stack_l1'] = int(cfg['compute_stack_l1'])
    cfg['capacity_stack_l2'] = int(cfg['capacity_stack_l2'])
    cfg['word'] = int(cfg['word'])
    cfg['lin'] = int(cfg['lin'])
    cfg['lout'] = int(cfg['lout'])
    cfg['batch'] = int(cfg['batch'])
    cfg['max_batch_size'] = int(cfg['max_batch_size'])
    cfg['prefill_chunk_tokens'] = int(cfg['prefill_chunk_tokens'])
    cfg['trace_debug_interval'] = int(cfg['trace_debug_interval'])
    cfg['timestamp_scaling'] = float(cfg['timestamp_scaling'])
    cfg['powerlimit'] = bool(cfg['powerlimit'])
    cfg['ffopt'] = bool(cfg['ffopt'])
    cfg['pipeopt'] = bool(cfg['pipeopt'])
    cfg['trace_debug'] = bool(cfg['trace_debug'])
    if cfg['gmemcap'] is not None:
        cfg['gmemcap'] = int(cfg['gmemcap'])

    return argparse.Namespace(**cfg)


def load_hetero_kv_arch_config(yaml_path=None) -> dict:
    """Load heterogeneous KV architecture config from YAML path, inline dict, or defaults."""
    cfg = copy.deepcopy(DEFAULT_HETERO_KV_ARCH)
    policy_defaults = copy.deepcopy(DEFAULT_KV_POLICY)
    if yaml_path is None:
        cfg.update(policy_defaults)
        return cfg

    # Accept a pre-parsed dict (from unified config's kv_arch section)
    if isinstance(yaml_path, dict):
        root = yaml_path
    else:
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
        "DIE_TYPE": str(root.get("die_type", cfg["DIE_TYPE"])).lower(),
        "NUM_PIM_DIE": root.get("num_pim_die", cfg["NUM_PIM_DIE"]),
        "COMPUTE_STACK_L1": root.get("compute_stack_l1", cfg["COMPUTE_STACK_L1"]),
        "CAPACITY_STACK_L2": root.get("capacity_stack_l2", cfg["CAPACITY_STACK_L2"]),
        "HOST_KV_TOTAL_GB": root.get("host_kv_total_gb", cfg["HOST_KV_TOTAL_GB"]),
        "DMA_BW_BPS": root.get("dma_bandwidth_gbps", cfg["DMA_BW_BPS"] / 1e9) * 1e9,
        "PCIE_BW_BPS": root.get("pcie_bandwidth_gbps", cfg["PCIE_BW_BPS"] / 1e9) * 1e9,
        "NUM_DIE_PACKAGES_PER_CARD": root.get("num_die_packages_per_card", cfg["NUM_DIE_PACKAGES_PER_CARD"]),
        "STACKS_PER_DIE_PACKAGE": root.get("stacks_per_die_package", cfg["STACKS_PER_DIE_PACKAGE"]),
        "BANKS_PER_DIE_PACKAGE": root.get("banks_per_die_package", cfg["BANKS_PER_DIE_PACKAGE"]),
        "FULL_DIE_CAPACITY_GB": root.get("full_die_capacity_gb", cfg["FULL_DIE_CAPACITY_GB"]),
        "EVICTION_POLICY": str(root.get("eviction_policy", policy_defaults["EVICTION_POLICY"])).lower(),
        "PLACEMENT_POLICY": str(root.get("placement_policy", policy_defaults["PLACEMENT_POLICY"])).lower(),
        "REPLICA_TIER": str(root.get("replica_tier", policy_defaults["REPLICA_TIER"])).upper(),
        "REPLICA_RESERVE_RATIO_L1": root.get("replica_reserve_ratio_l1", policy_defaults["REPLICA_RESERVE_RATIO_L1"]),
        "REPLICA_RESERVE_RATIO_L2": root.get("replica_reserve_ratio_l2", policy_defaults["REPLICA_RESERVE_RATIO_L2"]),
        "AWARE_POLICY": _deep_merge_policy(
            policy_defaults["AWARE_POLICY"],
            _normalize_aware_policy(root.get("aware_policy", {})),
        ),
    }

    validated = {
        "NUM_CARDS": int(_required_numeric(normalized, "NUM_CARDS")),
        "DIE_TYPE": normalized["DIE_TYPE"],
        "NUM_PIM_DIE": int(normalized["NUM_PIM_DIE"]),
        "COMPUTE_STACK_L1": int(normalized["COMPUTE_STACK_L1"]),
        "CAPACITY_STACK_L2": int(normalized["CAPACITY_STACK_L2"]),
        "HOST_KV_TOTAL_GB": _required_numeric(normalized, "HOST_KV_TOTAL_GB"),
        "DMA_BW_BPS": _required_numeric(normalized, "DMA_BW_BPS"),
        "PCIE_BW_BPS": _required_numeric(normalized, "PCIE_BW_BPS"),
        "NUM_DIE_PACKAGES_PER_CARD": int(_required_numeric(normalized, "NUM_DIE_PACKAGES_PER_CARD")),
        "STACKS_PER_DIE_PACKAGE": int(_required_numeric(normalized, "STACKS_PER_DIE_PACKAGE")),
        "BANKS_PER_DIE_PACKAGE": int(_required_numeric(normalized, "BANKS_PER_DIE_PACKAGE")),
        "FULL_DIE_CAPACITY_GB": _required_numeric(normalized, "FULL_DIE_CAPACITY_GB"),
        "EVICTION_POLICY": normalized["EVICTION_POLICY"],
        "PLACEMENT_POLICY": normalized["PLACEMENT_POLICY"],
        "REPLICA_TIER": normalized["REPLICA_TIER"],
        "REPLICA_RESERVE_RATIO_L1": _normalize_optional_ratio(
            normalized["REPLICA_RESERVE_RATIO_L1"], "replica_reserve_ratio_l1"
        ),
        "REPLICA_RESERVE_RATIO_L2": _normalize_optional_ratio(
            normalized["REPLICA_RESERVE_RATIO_L2"], "replica_reserve_ratio_l2"
        ),
        "AWARE_POLICY": copy.deepcopy(normalized["AWARE_POLICY"]),
    }
    if validated["EVICTION_POLICY"] not in {"lru", "aware"}:
        raise ValueError(f"unsupported eviction_policy: {validated['EVICTION_POLICY']}")
    if validated["PLACEMENT_POLICY"] not in {"all_unique", "hotset_broadcast"}:
        raise ValueError(f"unsupported placement_policy: {validated['PLACEMENT_POLICY']}")
    if validated["REPLICA_TIER"] not in {"AUTO", "L1", "L2"}:
        raise ValueError(f"unsupported replica_tier: {validated['REPLICA_TIER']}")
    if validated["DIE_TYPE"] not in {"attacc", "vstack", "uniform"}:
        raise ValueError(f"unsupported die_type: {validated['DIE_TYPE']}")
    if validated["DIE_TYPE"] == "attacc":
        max_pim_die = validated["NUM_DIE_PACKAGES_PER_CARD"] - 1
        if validated["NUM_PIM_DIE"] < 0 or validated["NUM_PIM_DIE"] > max_pim_die:
            raise ValueError(f"num_pim_die must be in [0, {max_pim_die}] for attacc")
    elif validated["DIE_TYPE"] == "vstack":
        total_stacks = validated["STACKS_PER_DIE_PACKAGE"]
        if validated["COMPUTE_STACK_L1"] + validated["CAPACITY_STACK_L2"] != total_stacks:
            raise ValueError(
                "compute_stack_l1 + capacity_stack_l2 must equal "
                f"{total_stacks} for vstack"
            )
    else:
        validated["NUM_PIM_DIE"] = validated["NUM_DIE_PACKAGES_PER_CARD"]
    return validated


def resolve_kv_policy_config(model_name: str, trace_file: str, hetero_kv_arch: dict = None) -> dict:
    cfg = load_hetero_kv_arch_config(hetero_kv_arch)
    resolved = copy.deepcopy(cfg)
    trace_family = infer_trace_family(trace_file)
    preset = TRACE_MODEL_KV_POLICY_PRESETS.get(str(model_name), {}).get(trace_family, {})

    if resolved.get("REPLICA_TIER", "AUTO") == "AUTO" and "REPLICA_TIER" in preset:
        resolved["REPLICA_TIER"] = preset["REPLICA_TIER"]
    if resolved.get("REPLICA_RESERVE_RATIO_L1") is None and "REPLICA_RESERVE_RATIO_L1" in preset:
        resolved["REPLICA_RESERVE_RATIO_L1"] = preset["REPLICA_RESERVE_RATIO_L1"]
    if resolved.get("REPLICA_RESERVE_RATIO_L2") is None and "REPLICA_RESERVE_RATIO_L2" in preset:
        resolved["REPLICA_RESERVE_RATIO_L2"] = preset["REPLICA_RESERVE_RATIO_L2"]

    resolved.setdefault("AWARE_POLICY", copy.deepcopy(DEFAULT_AWARE_POLICY))
    resolved["AWARE_POLICY"] = _deep_merge_policy(DEFAULT_AWARE_POLICY, resolved["AWARE_POLICY"])

    # Keep broadcast disabled by default unless the user explicitly enables it.
    if resolved["PLACEMENT_POLICY"] == "all_unique":
        resolved["REPLICA_RESERVE_RATIO_L1"] = 0.0
        resolved["REPLICA_RESERVE_RATIO_L2"] = 0.0
    else:
        if resolved["REPLICA_RESERVE_RATIO_L1"] is None:
            resolved["REPLICA_RESERVE_RATIO_L1"] = 0.0
        if resolved["REPLICA_RESERVE_RATIO_L2"] is None:
            resolved["REPLICA_RESERVE_RATIO_L2"] = 0.0

    resolved["TRACE_FAMILY"] = trace_family
    resolved["MODEL_NAME"] = str(model_name)
    return resolved


def get_hetero_memory_topology(hetero_kv_arch: dict = None) -> dict:
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch

    num_cards = int(cfg["NUM_CARDS"])
    die_type = str(cfg["DIE_TYPE"]).lower()
    num_die_packages = int(cfg["NUM_DIE_PACKAGES_PER_CARD"])
    stacks_per_die = int(cfg["STACKS_PER_DIE_PACKAGE"])
    banks_per_die = int(cfg["BANKS_PER_DIE_PACKAGE"])
    full_die_capacity_bytes = gib_to_bytes(cfg["FULL_DIE_CAPACITY_GB"])
    full_stack_capacity_bytes = int(full_die_capacity_bytes / max(stacks_per_die, 1))

    if die_type == "vstack":
        compute_stack_l1 = int(cfg["COMPUTE_STACK_L1"])
        capacity_stack_l2 = int(cfg["CAPACITY_STACK_L2"])
        l1_die_capacity_bytes = int(compute_stack_l1 * full_stack_capacity_bytes * COMPUTE_STACK_CAPACITY_RATIO)
        l2_die_capacity_bytes = int(capacity_stack_l2 * full_stack_capacity_bytes)
        l1_die_ids = tuple(range(num_die_packages))
        l2_die_ids = tuple(range(num_die_packages))
    elif die_type == "uniform":
        l1_die_capacity_bytes = int(full_die_capacity_bytes * PIM_DIE_CAPACITY_RATIO)
        l2_die_capacity_bytes = 0
        l1_die_ids = tuple(range(num_die_packages))
        l2_die_ids = tuple()
    else:
        num_pim_die = int(cfg["NUM_PIM_DIE"])
        l1_die_capacity_bytes = int(full_die_capacity_bytes * PIM_DIE_CAPACITY_RATIO) if num_pim_die > 0 else 0
        l2_die_capacity_bytes = full_die_capacity_bytes if num_pim_die < num_die_packages else 0
        l1_die_ids = tuple(range(num_pim_die))
        l2_die_ids = tuple(range(num_pim_die, num_die_packages))

    l1_per_card_bytes = l1_die_capacity_bytes * len(l1_die_ids)
    l2_per_card_bytes = l2_die_capacity_bytes * len(l2_die_ids)
    gpu_mem_per_card_bytes = l1_per_card_bytes + l2_per_card_bytes
    l1_bank_capacity_bytes = int(l1_die_capacity_bytes / banks_per_die) if l1_die_capacity_bytes > 0 else 0

    return {
        "num_cards": num_cards,
        "die_type": die_type,
        "num_die_packages_per_card": num_die_packages,
        "stacks_per_die_package": stacks_per_die,
        "banks_per_die": banks_per_die,
        "full_die_capacity_bytes": full_die_capacity_bytes,
        "full_stack_capacity_bytes": full_stack_capacity_bytes,
        "l1_die_ids": l1_die_ids,
        "l2_die_ids": l2_die_ids,
        "l1_die_capacity_bytes": l1_die_capacity_bytes,
        "l2_die_capacity_bytes": l2_die_capacity_bytes,
        "l1_bank_capacity_bytes": l1_bank_capacity_bytes,
        "l1_per_card_bytes": l1_per_card_bytes,
        "l2_per_card_bytes": l2_per_card_bytes,
        "gpu_mem_per_card_bytes": gpu_mem_per_card_bytes,
        "l1_per_card_gb": bytes_to_gib(l1_per_card_bytes),
        "l2_per_card_gb": bytes_to_gib(l2_per_card_bytes),
        "gpu_mem_per_card_gb": bytes_to_gib(gpu_mem_per_card_bytes),
        "num_l1_dies_total": num_cards * len(l1_die_ids),
        "num_l2_dies_total": num_cards * len(l2_die_ids),
    }


def get_hetero_transfer_bandwidths(hetero_kv_arch: dict = None) -> dict:
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch
    return {
        "dma_bw_bps": float(cfg["DMA_BW_BPS"]),
        "pcie_bw_bps": float(cfg["PCIE_BW_BPS"]),
    }


def validate_hetero_weight_capacity(weight_bytes_total: int = 0, hetero_kv_arch: dict = None) -> None:
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch
    topo = get_hetero_memory_topology(cfg)
    weight_bytes = max(0, int(weight_bytes_total))
    if topo["die_type"] != "uniform":
        return

    l1_total = topo["l1_per_card_bytes"] * topo["num_cards"]
    if weight_bytes > l1_total:
        raise ValueError(
            "uniform mode requires all weights to fit in aggregate L1: "
            f"weights={bytes_to_gib(weight_bytes):.3f} GiB, "
            f"available_l1={bytes_to_gib(l1_total):.3f} GiB "
            f"({topo['num_cards']} cards x {topo['l1_per_card_gb']:.3f} GiB/card)"
        )


def get_hetero_kv_capacities(weight_bytes_total: int = 0, hetero_kv_arch: dict = None) -> dict:
    """Return architecture-derived global KV capacities for L1/L2/L3 tiers."""
    cfg = DEFAULT_HETERO_KV_ARCH if hetero_kv_arch is None else hetero_kv_arch
    topo = get_hetero_memory_topology(cfg)
    l1_total = topo["l1_per_card_bytes"] * topo["num_cards"]
    l2_total = topo["l2_per_card_bytes"] * topo["num_cards"]
    l3_total = gib_to_bytes(cfg["HOST_KV_TOTAL_GB"])

    weight_bytes = max(0, int(weight_bytes_total))
    validate_hetero_weight_capacity(weight_bytes_total=weight_bytes, hetero_kv_arch=cfg)
    if topo["die_type"] == "uniform":
        l1_weight_reserved_bytes = weight_bytes
        l2_weight_reserved_bytes = 0
        l1_kv = max(0, l1_total - weight_bytes)
        l2_kv = 0
    else:
        l1_weight_reserved_bytes = 0
        l2_weight_reserved_bytes = weight_bytes
        l1_kv = l1_total
        l2_kv = max(0, l2_total - weight_bytes)

    l1_kv_die_capacity_bytes = 0
    l1_kv_bank_capacity_bytes = 0
    if topo["num_l1_dies_total"] > 0:
        l1_kv_die_capacity_bytes = int(l1_kv / topo["num_l1_dies_total"])
        if topo["banks_per_die"] > 0:
            l1_kv_bank_capacity_bytes = int(l1_kv_die_capacity_bytes / topo["banks_per_die"])
    l2_kv_die_capacity_bytes = 0
    if topo["num_l2_dies_total"] > 0:
        l2_kv_die_capacity_bytes = int(l2_kv / topo["num_l2_dies_total"])
    return {
        "l1_kv_bytes": l1_kv,
        "l1_total_bytes": l1_total,
        "l2_total_bytes": l2_total,
        "l2_kv_bytes": l2_kv,
        "l3_kv_bytes": l3_total,
        "weight_bytes_total": weight_bytes,
        "weight_reservation_tier": "L1" if topo["die_type"] == "uniform" else "L2",
        "l1_weight_reserved_bytes": l1_weight_reserved_bytes,
        "l2_weight_reserved_bytes": l2_weight_reserved_bytes,
        "gpu_mem_per_card_bytes": topo["gpu_mem_per_card_bytes"],
        "gpu_mem_per_card_gb": topo["gpu_mem_per_card_gb"],
        "hispeed_kv_per_card_gb": bytes_to_gib(l1_kv / topo["num_cards"]) if topo["num_cards"] > 0 else 0.0,
        "hispeed_total_per_card_gb": topo["l1_per_card_gb"],
        "hicap_total_per_card_gb": topo["l2_per_card_gb"],
        "num_die_packages_per_card": topo["num_die_packages_per_card"],
        "banks_per_die": topo["banks_per_die"],
        "l1_die_capacity_bytes": topo["l1_die_capacity_bytes"],
        "l2_die_capacity_bytes": topo["l2_die_capacity_bytes"],
        "l1_kv_die_capacity_bytes": l1_kv_die_capacity_bytes,
        "l1_bank_capacity_bytes": topo["l1_bank_capacity_bytes"],
        "l1_kv_bank_capacity_bytes": l1_kv_bank_capacity_bytes,
        "l2_kv_die_capacity_bytes": l2_kv_die_capacity_bytes,
        "l1_die_ids": topo["l1_die_ids"],
        "l2_die_ids": topo["l2_die_ids"],
        "die_type": topo["die_type"],
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
                    power_constraint=True,
                    num_pim_die=0,
                    die_type='attacc',
                    compute_stack_l1=4,
                    capacity_stack_l2=4):
    topo = get_hetero_memory_topology({
        "NUM_CARDS": 1,
        "DIE_TYPE": die_type,
        "NUM_PIM_DIE": num_pim_die,
        "COMPUTE_STACK_L1": compute_stack_l1,
        "CAPACITY_STACK_L2": capacity_stack_l2,
        "NUM_DIE_PACKAGES_PER_CARD": HBM_DIE_PACKAGES_PER_CARD,
        "STACKS_PER_DIE_PACKAGE": HBM_STACKS_PER_DIE_PACKAGE,
        "BANKS_PER_DIE_PACKAGE": HBM_BANKS_PER_DIE_PACKAGE,
        "FULL_DIE_CAPACITY_GB": FULL_HBM_DIE_CAPACITY_GB,
    })
    total_hbm_dies = topo["num_die_packages_per_card"]
    if die_type in ['vstack', 'uniform']:
        raw_hbm_dies = total_hbm_dies
        raw_hbm_ratio = 1.0
        capacity_per_device_bytes = topo["gpu_mem_per_card_bytes"]
    else:  # attacc
        raw_hbm_dies = total_hbm_dies - num_pim_die
        raw_hbm_ratio = raw_hbm_dies / total_hbm_dies
        capacity_per_device_bytes = topo["l2_per_card_bytes"]

    config = {'GPU': {}, 'CPU': {}}
    config['GPU']["GPUTYPE"] = gpu_type
    config['GPU']["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu

    if gpu_type == GPUType.A100a:
        # Ref: DGX-A100 whitepaper
        config['GPU']["NUM_CORE"] = 108
        config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = capacity_per_device_bytes if mem_cap is None else mem_cap

        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = int(3352 * 1000 * 1000 * 1000 * raw_hbm_ratio) \
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
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = capacity_per_device_bytes if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = int(3352 * 1000 * 1000 * 1000 * raw_hbm_ratio) \
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
                    num_pim_die=5,
                    bw_scale=None,
                    power_constraint=False,
                    die_type='attacc',
                    compute_stack_l1=4,
                    capacity_stack_l2=4):
    config = {}
    config["PIM_TYPE"] = pim_type
    config["POWER_CONSTRAINT"] = power_constraint
    config["ENERGY_TABLE"] = ENERGY_TABLE['PIM'][pim_type]

    internal_bandwidth_scale =  BW_SCALE[power_constraint][pim_type] \
                                if bw_scale is None else bw_scale
    config["NUM_ATTACC"] = num_attacc
    config["NUM_PIM_DIE"] = num_pim_die
    topo = get_hetero_memory_topology({
        "NUM_CARDS": num_attacc,
        "DIE_TYPE": die_type,
        "NUM_PIM_DIE": num_pim_die,
        "COMPUTE_STACK_L1": compute_stack_l1,
        "CAPACITY_STACK_L2": capacity_stack_l2,
        "NUM_DIE_PACKAGES_PER_CARD": HBM_DIE_PACKAGES_PER_CARD,
        "STACKS_PER_DIE_PACKAGE": HBM_STACKS_PER_DIE_PACKAGE,
        "BANKS_PER_DIE_PACKAGE": HBM_BANKS_PER_DIE_PACKAGE,
        "FULL_DIE_CAPACITY_GB": FULL_HBM_DIE_CAPACITY_GB,
    })
    # In vstack, capacity is already accounted for on the GPU side.
    config["MEM_CAPACITY_PER_PIM_DIE"] = 0 if die_type == 'vstack' else topo["l1_die_capacity_bytes"]
    config[
        "MEM_BW_PER_PIM_DIE"] = 670.4 * 1000 * 1000 * 1000 * internal_bandwidth_scale
    config["FLOPS_PER_PIM_DIE"] = config["MEM_BW_PER_PIM_DIE"] * opb
    config["SOFTMAX_MEM_BW"] = 670.4 * 1000 * 1000 * 1000 * num_pim_die
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
    model_table["Qwen3-4B"] = [36, 2560, 32, 128, 3.8, 4]
    model_table["Qwen3-32B"] = [64, 5120, 64, 128, 5.0, 8]
    model_table["Mistral-Devstral2-123B"] = [80, 12288, 96, 128, 3.25, 16]
    model_table["Llama-3.1-405B"] = [126, 16384, 128, 128, 3.25, 16]
    

    

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
