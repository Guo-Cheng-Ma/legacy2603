#!/usr/bin/env python3
import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    from matplotlib import transforms
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for tools/plot_generation_results.py. "
        "Install it in the attacc_baseline environment first."
    ) from exc


MODE_ORDER = ["attacc", "static", "uniform", "vstack-b", "vstack-o"]
MODE_TICK_ROTATION = 33
BREAK_THRESHOLD = 10.0
BREAK_MARK_SIZE = 0.007
THROUGHPUT_BASELINE_MODE = "attacc"
ENERGY_BASELINE_MODE = "vstack-o"
E2E_LATENCY_BASELINE_MODE = "vstack-o"
SKIPPED_TRACE_FAMILIES = {"example"}
MODE_COLORS = {
    "attacc": "#0b3954",
    "static": "#b85c38",
    "uniform": "#2a9d8f",
    "vstack-b": "#7b2cbf",
    "vstack-o": "#d7263d",
}
ENERGY_COMPONENTS = [
    ("total_dram_energy_nj", "DRAM", "#264653"),
    ("total_l2_energy_nj", "L2", "#2a9d8f"),
    ("total_l1_energy_nj", "L1", "#8ab17d"),
    ("total_reg_energy_nj", "REG", "#e9c46a"),
    ("total_alu_energy_nj", "ALU", "#f4a261"),
    ("total_comm_energy_nj", "COMM", "#e76f51"),
]
TOP_LEVEL_FIELDS = [
    "model",
    "trace_family",
    "input_request_name",
    "trace_scheduler_mode",
    "eviction_policy_cfg",
    "placement_policy_cfg",
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
TRACE_ORDER_HINTS = {
    "tracea": 0,
    "traceb": 1,
    "coder": 2,
    "thinking": 3,
}
NATURAL_RE = re.compile(r"(\d+)")
DATE_TAG_RE = re.compile(r"^\d{6}$")

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def default_output_dir() -> Path:
    return Path("figure") / datetime.now().strftime("%y%m%d-%H%M")


def requested_font_name() -> str:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except ValueError:
        return "Liberation Sans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate trace summary YAMLs and render generation-result figures "
            "for energy, throughput, TTFT, and E2E latency."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing S-*.yaml trace summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where figures and the CSV summary are written. Default: figure/<yymmdd>-<hhmm>/",
    )
    parser.add_argument(
        "--baseline-mode",
        default="attacc",
        choices=MODE_ORDER,
        help="Mode used as the denominator for normalized throughput metrics.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Matplotlib output format, for example png, pdf, or svg.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI for raster formats.",
    )
    return parser.parse_args()


def natural_key(text: object) -> List[object]:
    parts = NATURAL_RE.split(str(text))
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def trace_sort_key(trace_name: str) -> Tuple[int, List[object]]:
    lowered = str(trace_name).lower()
    return (TRACE_ORDER_HINTS.get(lowered, len(TRACE_ORDER_HINTS)), natural_key(trace_name))


def parse_scalar(value: str):
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


def load_top_level_yaml(path: Path) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.lstrip() != line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in TOP_LEVEL_FIELDS:
            parsed[key] = parse_scalar(value)
    return parsed


def date_tag_from_rel_parts(rel_parts: Tuple[str, ...]) -> Optional[str]:
    if not rel_parts:
        return None
    return rel_parts[0] if DATE_TAG_RE.match(rel_parts[0]) else None


def parse_summary_timestamp(path: Path, rel_parts: Tuple[str, ...]) -> datetime:
    date_tag = date_tag_from_rel_parts(rel_parts)
    if date_tag is None:
        raise ValueError(f"cannot infer date tag from path: {path}")
    time_tag = path.stem.rsplit("-", 1)[-1]
    return datetime.strptime(date_tag + time_tag, "%y%m%d%H%M%S")


def infer_mode(rel_parts: Tuple[str, ...], summary: Dict[str, object], filename: str) -> str:
    if len(rel_parts) >= 3 and rel_parts[1] in MODE_ORDER:
        return rel_parts[1]

    die_tag = filename.split("-", 2)[1]
    scheduler = str(summary.get("trace_scheduler_mode") or "").lower()
    placement = str(summary.get("placement_policy_cfg") or "").lower()
    eviction = str(summary.get("eviction_policy_cfg") or "").lower()

    if die_tag == "attacc" and scheduler == "static":
        return "static"
    if die_tag == "attacc":
        return "attacc"
    if die_tag == "uniform":
        return "uniform"
    if die_tag == "vstack":
        if placement == "hotset_broadcast" or eviction == "aware":
            return "vstack-o"
        return "vstack-b"
    return die_tag


def normalize_trace_family(raw_name: str) -> str:
    stem = str(raw_name or "").strip()
    if not stem:
        return "unknown"
    lowered = stem.lower()
    for candidate in ("traceA", "traceB", "coder", "thinking", "example"):
        if candidate.lower() in lowered:
            return candidate
    return stem


def infer_trace_family(rel_parts: Tuple[str, ...], summary: Dict[str, object]) -> str:
    if len(rel_parts) >= 3 and rel_parts[1] in MODE_ORDER:
        return rel_parts[2]
    if summary.get("trace_family"):
        return str(summary["trace_family"])
    if len(rel_parts) >= 2:
        return normalize_trace_family(rel_parts[1])
    return normalize_trace_family(str(summary.get("input_request_name") or "unknown"))


def metric_or_nan(summary: Dict[str, object], key: str, *fallback_keys: str) -> float:
    for candidate in (key, *fallback_keys):
        value = summary.get(candidate)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return math.nan


def safe_adjust(metric: float, queue_delay: float) -> float:
    if not np.isfinite(metric) or not np.isfinite(queue_delay):
        return math.nan
    return max(float(metric) - float(queue_delay), 0.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def discover_latest_summaries(results_root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    latest: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    dropped: List[Dict[str, object]] = []

    for path in sorted(results_root.rglob("S-*.yaml")):
        rel_parts = path.relative_to(results_root).parts
        summary = load_top_level_yaml(path)
        model = str(summary.get("model") or "").strip()
        if not model:
            continue

        record = {
            "path": path,
            "relative_parts": rel_parts,
            "timestamp": parse_summary_timestamp(path, rel_parts),
            "model": model,
            "trace_family": infer_trace_family(rel_parts, summary),
            "mode": infer_mode(rel_parts, summary, path.name),
            "input_request_name": str(summary.get("input_request_name") or ""),
            "trace_scheduler_mode": str(summary.get("trace_scheduler_mode") or ""),
            "eviction_policy_cfg": str(summary.get("eviction_policy_cfg") or ""),
            "placement_policy_cfg": str(summary.get("placement_policy_cfg") or ""),
            "total_energy_nj": metric_or_nan(summary, "total_energy_nj"),
            "total_dram_energy_nj": metric_or_nan(summary, "total_dram_energy_nj"),
            "total_l2_energy_nj": metric_or_nan(summary, "total_l2_energy_nj"),
            "total_l1_energy_nj": metric_or_nan(summary, "total_l1_energy_nj"),
            "total_reg_energy_nj": metric_or_nan(summary, "total_reg_energy_nj"),
            "total_alu_energy_nj": metric_or_nan(summary, "total_alu_energy_nj"),
            "total_comm_energy_nj": metric_or_nan(summary, "total_comm_energy_nj"),
            "avg_ttft_s": metric_or_nan(summary, "avg_ttft_s"),
            "avg_tbt_s": metric_or_nan(summary, "avg_tbt_s"),
            "avg_e2e_latency_s": metric_or_nan(summary, "avg_e2e_latency_s", "avg_latency_s"),
            "avg_queue_delay_s": metric_or_nan(summary, "avg_queue_delay_s"),
            "throughput_tok_per_s": metric_or_nan(summary, "throughput_tok_per_s"),
        }

        key = (record["model"], record["trace_family"], record["mode"])
        existing = latest.get(key)
        if existing is None or record["timestamp"] > existing["timestamp"]:
            if existing is not None:
                dropped.append(existing)
            latest[key] = record
        else:
            dropped.append(record)

    return list(latest.values()), dropped


def build_complete_dataframe(records: Iterable[Dict[str, object]], throughput_baseline_mode: str) -> pd.DataFrame:
    record_df = pd.DataFrame(records)
    if record_df.empty:
        raise ValueError("no trace summary YAMLs were found under the results root")

    record_df = record_df[~record_df["trace_family"].isin(SKIPPED_TRACE_FAMILIES)].copy()
    if record_df.empty:
        raise ValueError("no eligible trace summary YAMLs remained after filtering trace families")

    selected = {
        (row["model"], row["trace_family"], row["mode"]): row
        for row in record_df.to_dict(orient="records")
    }
    traces_per_model: Dict[str, List[str]] = {}
    for model, trace_family in (
        record_df[["model", "trace_family"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        traces_per_model.setdefault(model, []).append(trace_family)

    for model, traces in traces_per_model.items():
        traces_per_model[model] = sorted(set(traces), key=trace_sort_key)

    ordered_models = sorted(traces_per_model.keys(), key=natural_key)
    rows: List[Dict[str, object]] = []

    for model in ordered_models:
        for trace_family in traces_per_model[model]:
            throughput_baseline = selected.get((model, trace_family, throughput_baseline_mode))
            energy_baseline = selected.get((model, trace_family, ENERGY_BASELINE_MODE))
            e2e_latency_baseline = selected.get((model, trace_family, E2E_LATENCY_BASELINE_MODE))
            baseline_queue = (
                float(e2e_latency_baseline["avg_queue_delay_s"])
                if e2e_latency_baseline is not None
                else math.nan
            )
            baseline_ttft = (
                float(e2e_latency_baseline["avg_ttft_s"]) if e2e_latency_baseline is not None else math.nan
            )
            baseline_e2e_latency = (
                float(e2e_latency_baseline["avg_e2e_latency_s"])
                if e2e_latency_baseline is not None
                else math.nan
            )
            baseline_throughput = (
                float(throughput_baseline["throughput_tok_per_s"])
                if throughput_baseline is not None
                else math.nan
            )
            baseline_total_energy = (
                float(energy_baseline["total_energy_nj"]) if energy_baseline is not None else math.nan
            )
            baseline_ttft_adjusted = safe_adjust(baseline_ttft, baseline_queue)
            baseline_e2e_latency_adjusted = safe_adjust(baseline_e2e_latency, baseline_queue)

            for mode in MODE_ORDER:
                record = selected.get((model, trace_family, mode))
                row: Dict[str, object] = {
                    "model": model,
                    "trace_family": trace_family,
                    "mode": mode,
                    "has_summary": record is not None,
                }
                if record is None:
                    row.update(
                        {
                            "input_request_name": "",
                            "trace_scheduler_mode": "",
                            "eviction_policy_cfg": "",
                            "placement_policy_cfg": "",
                            "source_path": "",
                            "source_timestamp": "",
                        }
                    )
                    for key in (
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
                        "avg_queue_delay_s",
                        "throughput_tok_per_s",
                    ):
                        row[key] = math.nan
                else:
                    row.update(
                        {
                            "input_request_name": record["input_request_name"],
                            "trace_scheduler_mode": record["trace_scheduler_mode"],
                            "eviction_policy_cfg": record["eviction_policy_cfg"],
                            "placement_policy_cfg": record["placement_policy_cfg"],
                            "source_path": str(record["path"]),
                            "source_timestamp": record["timestamp"].isoformat(),
                        }
                    )
                    for key in (
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
                        "avg_queue_delay_s",
                        "throughput_tok_per_s",
                    ):
                        row[key] = float(record[key])

                row["ttft_minus_queue_s"] = safe_adjust(
                    float(row["avg_ttft_s"]), float(row["avg_queue_delay_s"])
                )
                row["e2e_latency_minus_queue_s"] = safe_adjust(
                    float(row["avg_e2e_latency_s"]), float(row["avg_queue_delay_s"])
                )
                row["energy_normalized"] = safe_ratio(
                    float(row["total_energy_nj"]), baseline_total_energy
                )
                for key in (
                    "total_dram_energy_nj",
                    "total_l2_energy_nj",
                    "total_l1_energy_nj",
                    "total_reg_energy_nj",
                    "total_alu_energy_nj",
                    "total_comm_energy_nj",
                ):
                    normalized_key = key.replace("_nj", "_normalized")
                    row[normalized_key] = safe_ratio(float(row[key]), baseline_total_energy)
                row["throughput_normalized"] = safe_ratio(
                    float(row["throughput_tok_per_s"]), baseline_throughput
                )
                row["ttft_raw_normalized"] = safe_ratio(float(row["avg_ttft_s"]), baseline_ttft)
                row["ttft_minus_queue_normalized"] = safe_ratio(
                    float(row["ttft_minus_queue_s"]), baseline_ttft_adjusted
                )
                row["e2e_latency_raw_normalized"] = safe_ratio(
                    float(row["avg_e2e_latency_s"]), baseline_e2e_latency
                )
                row["e2e_latency_minus_queue_normalized"] = safe_ratio(
                    float(row["e2e_latency_minus_queue_s"]), baseline_e2e_latency_adjusted
                )
                rows.append(row)

    return pd.DataFrame(rows)


def assign_plot_positions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]]]:
    trace_gap = 0.8
    model_gap = 1.4
    plotted = df.copy()
    x_positions: List[float] = []
    trace_groups: List[Dict[str, object]] = []
    model_groups: List[Dict[str, object]] = []
    x_cursor = 0.0

    sorted_models = sorted(plotted["model"].unique(), key=natural_key)
    row_order: List[int] = []
    for model in sorted_models:
        model_mask = plotted["model"] == model
        traces = sorted(plotted.loc[model_mask, "trace_family"].unique(), key=trace_sort_key)
        model_start = x_cursor
        for trace_family in traces:
            trace_start = x_cursor
            trace_mask = model_mask & (plotted["trace_family"] == trace_family)
            trace_rows = plotted.loc[trace_mask].copy()
            trace_rows["mode_rank"] = trace_rows["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
            trace_rows = trace_rows.sort_values("mode_rank")
            for row_index in trace_rows.index:
                row_order.append(row_index)
                x_positions.append(x_cursor)
                x_cursor += 1.0
            trace_end = x_cursor - 1.0
            trace_groups.append(
                {
                    "label": trace_family,
                    "center": (trace_start + trace_end) / 2.0,
                    "after": trace_end + 0.5,
                }
            )
            x_cursor += trace_gap
        model_end = x_cursor - trace_gap - 1.0
        model_groups.append(
            {
                "label": model,
                "center": (model_start + model_end) / 2.0,
                "after": model_end + 0.5,
            }
        )
        x_cursor += model_gap

    plotted = plotted.loc[row_order].reset_index(drop=True)
    plotted["x"] = x_positions
    return plotted, trace_groups, model_groups


def choose_energy_unit(max_energy_nj: float) -> Tuple[float, str]:
    units = [
        (1.0, "nJ"),
        (1e3, "uJ"),
        (1e6, "mJ"),
        (1e9, "J"),
        (1e12, "kJ"),
        (1e15, "MJ"),
    ]
    for scale, label in reversed(units):
        if max_energy_nj >= scale:
            return scale, label
    return 1.0, "nJ"


def apply_common_axis_style(
    axes,
    plotted: pd.DataFrame,
    trace_groups: List[Dict[str, object]],
    model_groups: List[Dict[str, object]],
    title: str,
    ylabel: str,
    show_reference_line: bool = False,
) -> None:
    axis_list = list(axes) if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    top_axis = axis_list[0]
    bottom_axis = axis_list[-1]

    top_axis.set_title(title, fontsize=15, weight="bold", pad=14)
    bottom_axis.set_ylabel(ylabel)
    bottom_axis.set_xticks(plotted["x"].tolist())
    bottom_axis.set_xticklabels(
        plotted["mode"].tolist(),
        rotation=MODE_TICK_ROTATION,
        fontsize=9,
        ha="right",
        rotation_mode="anchor",
    )

    for axis in axis_list:
        axis.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if show_reference_line:
            axis.axhline(1.0, color="#444444", linestyle=":", linewidth=1.1)

    for axis in axis_list[:-1]:
        axis.tick_params(labelbottom=False, bottom=False)

    text_transform = transforms.blended_transform_factory(bottom_axis.transData, bottom_axis.transAxes)
    for group in trace_groups[:-1]:
        for axis in axis_list:
            axis.axvline(group["after"], color="#d0d0d0", linewidth=0.8, linestyle="-")
    for group in model_groups[:-1]:
        for axis in axis_list:
            axis.axvline(group["after"] + 0.2, color="#8a8a8a", linewidth=1.1, linestyle="-")
    for group in trace_groups:
        bottom_axis.text(
            group["center"],
            -0.23,
            group["label"],
            ha="center",
            va="top",
            fontsize=10,
            rotation=12,
            transform=text_transform,
        )
    for group in model_groups:
        bottom_axis.text(
            group["center"],
            -0.42,
            group["label"],
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
            transform=text_transform,
        )


def figure_size_for_slots(slot_count: int) -> Tuple[float, float]:
    width = max(13.0, slot_count * 0.34)
    return width, 7.6


def plot_break_marks(top_axis, bottom_axis) -> None:
    kwargs = dict(color="#444444", clip_on=False, linewidth=1.0)
    top_axis.plot(
        (-BREAK_MARK_SIZE, +BREAK_MARK_SIZE),
        (-BREAK_MARK_SIZE, +BREAK_MARK_SIZE),
        transform=top_axis.transAxes,
        **kwargs,
    )
    top_axis.plot(
        (1 - BREAK_MARK_SIZE, 1 + BREAK_MARK_SIZE),
        (-BREAK_MARK_SIZE, +BREAK_MARK_SIZE),
        transform=top_axis.transAxes,
        **kwargs,
    )
    bottom_axis.plot(
        (-BREAK_MARK_SIZE, +BREAK_MARK_SIZE),
        (1 - BREAK_MARK_SIZE, 1 + BREAK_MARK_SIZE),
        transform=bottom_axis.transAxes,
        **kwargs,
    )
    bottom_axis.plot(
        (1 - BREAK_MARK_SIZE, 1 + BREAK_MARK_SIZE),
        (1 - BREAK_MARK_SIZE, 1 + BREAK_MARK_SIZE),
        transform=bottom_axis.transAxes,
        **kwargs,
    )


def normalized_break_limits(values: np.ndarray) -> Optional[Tuple[float, float, float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    max_value = float(np.max(finite))
    if max_value <= BREAK_THRESHOLD:
        return None

    small = finite[finite <= BREAK_THRESHOLD]
    large = finite[finite > BREAK_THRESHOLD]
    if large.size == 0:
        return None

    if small.size > 0:
        lower_max = min(BREAK_THRESHOLD, max(1.25, float(np.max(small)) * 1.12))
    else:
        lower_max = 2.0

    upper_min = max(BREAK_THRESHOLD, float(np.min(large)) * 0.90)
    if upper_min <= lower_max:
        upper_min = lower_max + max(1.0, 0.08 * max_value)
    upper_max = max_value * 1.05
    if upper_min >= upper_max:
        return None
    return lower_max, upper_min, upper_max


def draw_mode_bars(ax, plotted: pd.DataFrame, metric_column: str) -> None:
    for mode in MODE_ORDER:
        mode_rows = plotted[plotted["mode"] == mode]
        values = mode_rows[metric_column].to_numpy(dtype=float)
        mask = np.isfinite(values)
        ax.bar(
            mode_rows.loc[mask, "x"],
            values[mask],
            width=0.82,
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.5,
        )


def plot_energy_breakdown(
    plotted: pd.DataFrame,
    trace_groups: List[Dict[str, object]],
    model_groups: List[Dict[str, object]],
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=figure_size_for_slots(len(plotted)))
    bottoms = np.zeros(len(plotted), dtype=float)

    for column, label, color in ENERGY_COMPONENTS:
        normalized_column = column.replace("_nj", "_normalized")
        heights = plotted[normalized_column].to_numpy(dtype=float)
        mask = np.isfinite(heights)
        ax.bar(
            plotted.loc[mask, "x"],
            heights[mask],
            width=0.82,
            bottom=bottoms[mask],
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottoms[mask] += heights[mask]

    apply_common_axis_style(
        [ax],
        plotted,
        trace_groups,
        model_groups,
        title="Generation Energy Breakdown (Normalized to VStack-O)",
        ylabel="Normalized Energy",
        show_reference_line=True,
    )
    ax.legend(loc="upper left", ncol=len(ENERGY_COMPONENTS), frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.38, top=0.88, left=0.08, right=0.99)
    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_metric(
    plotted: pd.DataFrame,
    trace_groups: List[Dict[str, object]],
    model_groups: List[Dict[str, object]],
    metric_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    values = plotted[metric_column].to_numpy(dtype=float)
    broken_limits = normalized_break_limits(values)

    if broken_limits is None:
        fig, ax = plt.subplots(figsize=figure_size_for_slots(len(plotted)))
        draw_mode_bars(ax, plotted, metric_column)
        apply_common_axis_style(
            [ax],
            plotted,
            trace_groups,
            model_groups,
            title=title,
            ylabel=ylabel,
            show_reference_line=True,
        )
        fig.subplots_adjust(bottom=0.38, top=0.90, left=0.08, right=0.99)
    else:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(figure_size_for_slots(len(plotted))[0], 8.6),
            gridspec_kw={"height_ratios": [1.25, 3.4], "hspace": 0.05},
        )
        draw_mode_bars(ax_top, plotted, metric_column)
        draw_mode_bars(ax_bottom, plotted, metric_column)
        lower_max, upper_min, upper_max = broken_limits
        ax_bottom.set_ylim(0.0, lower_max)
        ax_top.set_ylim(upper_min, upper_max)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        plot_break_marks(ax_top, ax_bottom)
        apply_common_axis_style(
            [ax_top, ax_bottom],
            plotted,
            trace_groups,
            model_groups,
            title=title,
            ylabel=ylabel,
            show_reference_line=True,
        )
        fig.subplots_adjust(bottom=0.36, top=0.90, left=0.08, right=0.99)

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    ordered_columns = [
        "model",
        "trace_family",
        "mode",
        "has_summary",
        "input_request_name",
        "trace_scheduler_mode",
        "eviction_policy_cfg",
        "placement_policy_cfg",
        "source_path",
        "source_timestamp",
        "total_energy_nj",
        "total_dram_energy_nj",
        "total_l2_energy_nj",
        "total_l1_energy_nj",
        "total_reg_energy_nj",
        "total_alu_energy_nj",
        "total_comm_energy_nj",
        "energy_normalized",
        "total_dram_energy_normalized",
        "total_l2_energy_normalized",
        "total_l1_energy_normalized",
        "total_reg_energy_normalized",
        "total_alu_energy_normalized",
        "total_comm_energy_normalized",
        "throughput_tok_per_s",
        "throughput_normalized",
        "avg_queue_delay_s",
        "avg_ttft_s",
        "avg_tbt_s",
        "ttft_raw_normalized",
        "ttft_minus_queue_s",
        "ttft_minus_queue_normalized",
        "avg_e2e_latency_s",
        "e2e_latency_raw_normalized",
        "e2e_latency_minus_queue_s",
        "e2e_latency_minus_queue_normalized",
    ]
    df.to_csv(output_path, columns=ordered_columns, index=False)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise SystemExit(f"results root does not exist: {results_root}")

    latest_records, dropped_records = discover_latest_summaries(results_root)
    complete_df = build_complete_dataframe(latest_records, throughput_baseline_mode=args.baseline_mode)
    plotted, trace_groups, model_groups = assign_plot_positions(complete_df)

    csv_path = output_dir / "generation_metrics_dedup.csv"
    write_csv(plotted.drop(columns=["x"]), csv_path)

    plot_energy_breakdown(
        plotted,
        trace_groups,
        model_groups,
        output_dir / f"generation_energy_breakdown.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "throughput_normalized",
        "Generation Throughput (Normalized to AttAcc)",
        "Normalized Throughput",
        output_dir / f"generation_throughput_normalized.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "ttft_raw_normalized",
        "Generation TTFT (Raw, Normalized to VStack-O)",
        "Normalized TTFT",
        output_dir / f"generation_ttft_normalized_raw.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "ttft_minus_queue_normalized",
        "Generation TTFT (Avg Queue Delay Removed, Then Normalized to VStack-O)",
        "Normalized TTFT",
        output_dir / f"generation_ttft_normalized_minus_queue.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "e2e_latency_raw_normalized",
        "Generation E2E Latency (Raw, Normalized to VStack-O)",
        "Normalized E2E Latency",
        output_dir / f"generation_e2e_latency_normalized_raw.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "e2e_latency_minus_queue_normalized",
        "Generation E2E Latency (Avg Queue Delay Removed, Then Normalized to VStack-O)",
        "Normalized E2E Latency",
        output_dir / f"generation_e2e_latency_normalized_minus_queue.{args.format}",
        args.format,
        args.dpi,
    )

    print(f"selected summaries: {len(latest_records)}")
    print(f"dropped older duplicates: {len(dropped_records)}")
    print(f"font request: {requested_font_name()}")
    print(f"csv: {csv_path}")
    print(f"figures: {output_dir}")


if __name__ == "__main__":
    main()
