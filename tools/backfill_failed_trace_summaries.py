#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import resolve_kv_policy_config


MODE_TO_DIE_TAG = {
    "attacc": "attacc",
    "static": "attacc",
    "uniform": "uniform",
    "vstack-b": "vstack",
    "vstack-o": "vstack",
}

METRIC_FIELDS = [
    "total_energy_nj",
    "total_dram_energy_nj",
    "total_l2_energy_nj",
    "total_l1_energy_nj",
    "total_reg_energy_nj",
    "total_alu_energy_nj",
    "total_comm_energy_nj",
    "avg_ttft_s",
    "avg_tbt_s",
    "avg_e2e_latency_s",
    "avg_latency_s",
    "avg_queue_delay_s",
    "throughput_tok_per_s",
]

TRACE_ALIASES = {
    "tracea": "traceA",
    "traceb": "traceB",
    "coder": "coder",
    "thinking": "thinking",
}

EXCEPTION_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):\s+.+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill failed tmux trace runs into synthetic S-*.yaml summaries "
            "so plotting and dedup treat them as explicit failed slots."
        )
    )
    parser.add_argument(
        "--summary-tsv",
        required=True,
        help="Path to the tmux batch summary.tsv file.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory where synthetic S-*.yaml files are written.",
    )
    parser.add_argument(
        "--status",
        default="failed",
        help="Only backfill rows with this status. Default: failed.",
    )
    return parser.parse_args()


def _parse_timestamp(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")


def _normalize_trace_name(trace_file: str) -> str:
    trace_stem = Path(str(trace_file or "")).stem
    lowered = trace_stem.lower()
    for needle, alias in TRACE_ALIASES.items():
        if needle in lowered:
            return alias
    return trace_stem or "unknown"


def _result_subdirs(config_path: str, trace_file: str, fallback_trace: str) -> Tuple[str, str]:
    cfg_path = Path(str(config_path or ""))
    cfg_parts_lower = [part.lower() for part in cfg_path.parts]
    family = ""
    trace_name = _normalize_trace_name(trace_file) or fallback_trace

    if "configs" in cfg_parts_lower:
        cfg_idx = cfg_parts_lower.index("configs")
        trailing = cfg_path.parts[cfg_idx + 1:]
        if len(trailing) >= 2:
            family = trailing[0]
            trace_name = trailing[1]
    return family, trace_name or fallback_trace or "unknown"


def _load_yaml_map(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    parsed = _parse_simple_yaml_tree(path.read_text(encoding="utf-8")) or {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_scalar(value: str):
    stripped = value.strip()
    if stripped == "":
        return ""
    if len(stripped) >= 2 and stripped[0] in {"'", '"'} and stripped[-1] == stripped[0]:
        return stripped[1:-1]

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        if any(ch in stripped for ch in ".eE"):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _parse_simple_yaml_tree(text: str) -> Dict[str, object]:
    root: Dict[str, object] = {}
    stack: List[Tuple[int, Dict[str, object]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value == "":
            child: Dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _pim_type_label(value: object) -> str:
    lowered = str(value or "").strip().lower()
    if lowered == "bank":
        return "BA"
    if lowered == "bg":
        return "BG"
    if lowered == "buffer":
        return "BUFFER"
    return str(value or "").upper()


def _read_log_text(path: Path) -> str:
    if not path.exists():
        return f"missing log file: {path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"failed to read log file {path}: {exc}"


def _extract_failure_detail(log_text: str) -> str:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if EXCEPTION_LINE_RE.match(line):
            return line
    return lines[-1] if lines else "failed without log output"


def _classify_failure(detail: str, log_text: str) -> str:
    text = f"{detail}\n{log_text}".lower()
    if "aggregate l1" in text or "available_l1" in text or "fit in aggregate l1" in text:
        return "L1 cap"
    if "out of memory" in text or re.search(r"\boom\b", text):
        return "OOM"
    if "trace.qps" in text or "timestamp_scaling" in text:
        return "Config"
    if "yaml" in text or "unsupported" in text or "invalid" in text:
        return "Config"
    if "assertionerror" in text:
        return "Assert"
    if "traceback" in text:
        return "Crash"
    return "Failed"


def _effective_kv_policy(config: Dict[str, object], model_name: str, trace_file: str) -> Dict[str, object]:
    system_cfg = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    pim_cfg = config.get("pim", {}) if isinstance(config.get("pim"), dict) else {}
    kv_arch = config.get("kv_arch", {})
    kv_dict = dict(kv_arch) if isinstance(kv_arch, dict) else {}
    kv_dict.setdefault("num_cards", _coerce_int(system_cfg.get("ngpu"), 8))
    kv_dict.setdefault("die_type", str(pim_cfg.get("die_type") or "attacc").lower())
    kv_dict.setdefault("num_pim_die", _coerce_int(pim_cfg.get("num_pim_die"), 4))
    kv_dict.setdefault("compute_stack_l1", _coerce_int(pim_cfg.get("compute_stack_l1"), 4))
    kv_dict.setdefault("capacity_stack_l2", _coerce_int(pim_cfg.get("capacity_stack_l2"), 4))
    if not model_name or not trace_file:
        return kv_dict
    try:
        return resolve_kv_policy_config(
            model_name=model_name,
            trace_file=trace_file,
            hetero_kv_arch=kv_dict,
        )
    except Exception:
        return kv_dict


def _synthetic_summary_path(
    results_root: Path,
    config_path: str,
    trace_file: str,
    fallback_trace: str,
    mode: str,
    die_tag: str,
    compute_stack_l1: int,
    model_name: str,
    ngpu: int,
    ended_at: Optional[datetime],
) -> Path:
    timestamp = ended_at or datetime.now().astimezone()
    date_tag = timestamp.strftime("%y%m%d")
    time_tag = timestamp.strftime("%H%M%S")
    family, trace_name = _result_subdirs(config_path=config_path, trace_file=trace_file, fallback_trace=fallback_trace)

    trace_dir = results_root / date_tag
    if family:
        trace_dir = trace_dir / family
    else:
        trace_dir = trace_dir / (trace_name or fallback_trace or mode)
        trace_name = trace_name or fallback_trace or mode

    if family:
        trace_dir = trace_dir / trace_name

    trace_dir.mkdir(parents=True, exist_ok=True)
    filename = f"S-{die_tag}-{int(compute_stack_l1)}-{model_name}-{int(ngpu)}gpu-{time_tag}.yaml"
    return trace_dir / filename


def _build_summary(row: Dict[str, str], results_root: Path) -> Tuple[Path, OrderedDict]:
    config_path = str(row.get("config") or "")
    config = _load_yaml_map(Path(config_path))
    system_cfg = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    pim_cfg = config.get("pim", {}) if isinstance(config.get("pim"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    trace_cfg = config.get("trace", {}) if isinstance(config.get("trace"), dict) else {}

    mode = str(row.get("mode") or "").strip()
    model_name = str(model_cfg.get("model") or row.get("model") or "").strip()
    trace_file = str(trace_cfg.get("trace_file") or "").strip()
    trace_name = str(row.get("trace") or _normalize_trace_name(trace_file) or "unknown")
    die_tag = str(pim_cfg.get("die_type") or MODE_TO_DIE_TAG.get(mode, mode or "unknown")).strip().lower()
    compute_stack_l1 = _coerce_int(pim_cfg.get("compute_stack_l1"), 4)
    ngpu = _coerce_int(system_cfg.get("ngpu"), 8)
    ended_at = _parse_timestamp(row.get("ended_at", ""))
    log_path = Path(str(row.get("log_file") or ""))
    log_text = _read_log_text(log_path)
    failure_detail = _extract_failure_detail(log_text)
    failure_short = _classify_failure(failure_detail, log_text)
    trace_family = _result_subdirs(config_path, trace_file, trace_name)[1]
    kv_policy = _effective_kv_policy(config=config, model_name=model_name, trace_file=trace_file)

    summary_path = _synthetic_summary_path(
        results_root=results_root,
        config_path=config_path,
        trace_file=trace_file,
        fallback_trace=trace_name,
        mode=mode,
        die_tag=die_tag,
        compute_stack_l1=compute_stack_l1,
        model_name=model_name,
        ngpu=ngpu,
        ended_at=ended_at,
    )

    summary = OrderedDict()
    summary["mode"] = "trace"
    summary["run_status"] = "failed"
    summary["failure_reason_short"] = failure_short
    summary["failure_reason_detail"] = failure_detail
    summary["failure_exit_code"] = _coerce_int(row.get("exit_code"), 1)
    summary["failure_log_path"] = str(log_path)
    summary["tmux_session_name"] = str(row.get("session_name") or "")
    summary["config_path"] = config_path
    summary["started_at"] = str(row.get("started_at") or "")
    summary["ended_at"] = str(row.get("ended_at") or "")
    summary["system"] = str(system_cfg.get("system") or "")
    summary["gpu_name"] = str(system_cfg.get("gpu") or "")
    summary["pim_type"] = _pim_type_label(pim_cfg.get("pim"))
    summary["trace_file"] = trace_file
    summary["input_request_name"] = Path(trace_file).stem if trace_file else trace_family
    summary["model"] = model_name
    summary["power_constraint"] = _coerce_bool(pim_cfg.get("powerlimit"))
    summary["pipe_level"] = int(_coerce_bool(pim_cfg.get("pipeopt")))
    summary["is_parallel"] = int(_coerce_bool(pim_cfg.get("ffopt")))
    summary["max_batch_size"] = _coerce_int(trace_cfg.get("max_batch_size"), 0)
    summary["trace_scheduler_mode"] = str(trace_cfg.get("trace_scheduler") or "")
    summary["prefill_chunk_tokens"] = _coerce_int(trace_cfg.get("prefill_chunk_tokens"), 0)
    summary["requested_qps"] = _coerce_float(trace_cfg.get("QPS"))
    summary["trace_family"] = trace_family
    summary["kv_arch_config"] = kv_policy if kv_policy else {}
    summary["eviction_policy_cfg"] = str(
        kv_policy.get("EVICTION_POLICY")
        or (config.get("kv_arch", {}) or {}).get("eviction_policy")
        or ""
    )
    summary["placement_policy_cfg"] = str(
        kv_policy.get("PLACEMENT_POLICY")
        or (config.get("kv_arch", {}) or {}).get("placement_policy")
        or ""
    )
    for field in METRIC_FIELDS:
        summary[field] = None

    return summary_path, summary


def _load_rows(summary_tsv: Path, wanted_status: str) -> List[Dict[str, str]]:
    with summary_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if str(row.get("status") or "") == wanted_status]
    return rows


def _write_yaml(path: Path, payload: OrderedDict) -> None:
    lines: List[str] = []
    _dump_yaml_lines(lines, payload, indent=0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False)


def _dump_yaml_lines(lines: List[str], payload: Dict[str, object], indent: int) -> None:
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            _dump_yaml_lines(lines, value, indent + 2)
        else:
            lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")


def main() -> None:
    args = parse_args()
    summary_tsv = Path(args.summary_tsv)
    results_root = Path(args.results_root)
    if not summary_tsv.exists():
        raise SystemExit(f"summary TSV does not exist: {summary_tsv}")

    rows = _load_rows(summary_tsv=summary_tsv, wanted_status=args.status)
    if not rows:
        print("matched rows: 0")
        return

    written: List[Path] = []
    for row in rows:
        summary_path, summary = _build_summary(row=row, results_root=results_root)
        _write_yaml(summary_path, summary)
        written.append(summary_path)
        print(
            "wrote\t{}\t{}\t{}\t{}".format(
                row.get("session_name", ""),
                summary.get("failure_reason_short", ""),
                summary.get("trace_family", ""),
                summary_path,
            )
        )

    print(f"matched rows: {len(rows)}")
    print(f"written summaries: {len(written)}")


if __name__ == "__main__":
    main()
