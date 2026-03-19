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
    from matplotlib import transforms
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for tools/plot_generation_results.py. "
        "Install it in the attacc_baseline environment first."
    ) from exc


MODE_ORDER = ["attacc", "static", "uniform", "vstack-b", "vstack-o"]
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
    "avg_latency_s",
    "avg_queue_delay_s",
    "throughput_tok_per_s",
]
TRACE_ORDER_HINTS = {
    "tracea": 0,
    "traceb": 1,
    "coder": 2,
    "thinking": 3,
    "example": 4,
}
NATURAL_RE = re.compile(r"(\d+)")
DATE_TAG_RE = re.compile(r"^\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate trace summary YAMLs and render generation-result figures "
            "for energy, throughput, TTFT, and latency."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing S-*.yaml trace summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where figures and the CSV summary are written.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="attacc",
        choices=MODE_ORDER,
        help="Mode used as the denominator for normalized metrics.",
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


def metric_or_nan(summary: Dict[str, object], key: str) -> float:
    value = summary.get(key)
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
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
            "avg_latency_s": metric_or_nan(summary, "avg_latency_s"),
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


def build_complete_dataframe(
    records: Iterable[Dict[str, object]], baseline_mode: str
) -> pd.DataFrame:
    record_df = pd.DataFrame(records)
    if record_df.empty:
        raise ValueError("no trace summary YAMLs were found under the results root")

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
            baseline = selected.get((model, trace_family, baseline_mode))
            baseline_queue = (
                float(baseline["avg_queue_delay_s"])
                if baseline is not None
                else math.nan
            )
            baseline_ttft = float(baseline["avg_ttft_s"]) if baseline is not None else math.nan
            baseline_latency = (
                float(baseline["avg_latency_s"]) if baseline is not None else math.nan
            )
            baseline_throughput = (
                float(baseline["throughput_tok_per_s"]) if baseline is not None else math.nan
            )
            baseline_ttft_adjusted = safe_adjust(baseline_ttft, baseline_queue)
            baseline_latency_adjusted = safe_adjust(baseline_latency, baseline_queue)

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
                        "avg_latency_s",
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
                        "avg_latency_s",
                        "avg_queue_delay_s",
                        "throughput_tok_per_s",
                    ):
                        row[key] = float(record[key])

                row["ttft_minus_queue_s"] = safe_adjust(
                    float(row["avg_ttft_s"]), float(row["avg_queue_delay_s"])
                )
                row["latency_minus_queue_s"] = safe_adjust(
                    float(row["avg_latency_s"]), float(row["avg_queue_delay_s"])
                )
                row["throughput_normalized"] = safe_ratio(
                    float(row["throughput_tok_per_s"]), baseline_throughput
                )
                row["ttft_raw_normalized"] = safe_ratio(float(row["avg_ttft_s"]), baseline_ttft)
                row["ttft_minus_queue_normalized"] = safe_ratio(
                    float(row["ttft_minus_queue_s"]), baseline_ttft_adjusted
                )
                row["latency_raw_normalized"] = safe_ratio(
                    float(row["avg_latency_s"]), baseline_latency
                )
                row["latency_minus_queue_normalized"] = safe_ratio(
                    float(row["latency_minus_queue_s"]), baseline_latency_adjusted
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
    ax,
    plotted: pd.DataFrame,
    trace_groups: List[Dict[str, object]],
    model_groups: List[Dict[str, object]],
    title: str,
    ylabel: str,
    show_reference_line: bool = False,
) -> None:
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks(plotted["x"].tolist())
    ax.set_xticklabels(plotted["mode"].tolist(), rotation=0, fontsize=9)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_reference_line:
        ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1.1)

    text_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for group in trace_groups[:-1]:
        ax.axvline(group["after"], color="#d0d0d0", linewidth=0.8, linestyle="-")
    for group in model_groups[:-1]:
        ax.axvline(group["after"] + 0.2, color="#8a8a8a", linewidth=1.1, linestyle="-")
    for group in trace_groups:
        ax.text(
            group["center"],
            -0.16,
            group["label"],
            ha="center",
            va="top",
            fontsize=10,
            transform=text_transform,
        )
    for group in model_groups:
        ax.text(
            group["center"],
            -0.30,
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
    max_energy = 0.0
    for column, label, color in ENERGY_COMPONENTS:
        max_energy = max(
            max_energy,
            np.nanmax(plotted[column].to_numpy(dtype=float)) if not plotted[column].isna().all() else 0.0,
        )
    scale, unit = choose_energy_unit(max_energy)

    for column, label, color in ENERGY_COMPONENTS:
        heights = plotted[column].to_numpy(dtype=float) / scale
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
        ax,
        plotted,
        trace_groups,
        model_groups,
        title="Generation Total Energy Breakdown",
        ylabel=f"Energy ({unit})",
    )
    ax.legend(loc="upper left", ncol=len(ENERGY_COMPONENTS), frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.08, right=0.99)
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
    fig, ax = plt.subplots(figsize=figure_size_for_slots(len(plotted)))
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

    apply_common_axis_style(
        ax,
        plotted,
        trace_groups,
        model_groups,
        title=title,
        ylabel=ylabel,
        show_reference_line=True,
    )
    fig.subplots_adjust(bottom=0.28, top=0.90, left=0.08, right=0.99)
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
        "throughput_tok_per_s",
        "throughput_normalized",
        "avg_queue_delay_s",
        "avg_ttft_s",
        "ttft_raw_normalized",
        "ttft_minus_queue_s",
        "ttft_minus_queue_normalized",
        "avg_latency_s",
        "latency_raw_normalized",
        "latency_minus_queue_s",
        "latency_minus_queue_normalized",
    ]
    df.to_csv(output_path, columns=ordered_columns, index=False)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise SystemExit(f"results root does not exist: {results_root}")

    latest_records, dropped_records = discover_latest_summaries(results_root)
    complete_df = build_complete_dataframe(latest_records, baseline_mode=args.baseline_mode)
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
        "Generation TTFT (Raw, Normalized to AttAcc)",
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
        "Generation TTFT (Avg Queue Delay Removed, Then Normalized)",
        "Normalized TTFT",
        output_dir / f"generation_ttft_normalized_minus_queue.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "latency_raw_normalized",
        "Generation Latency (Raw, Normalized to AttAcc)",
        "Normalized Latency",
        output_dir / f"generation_latency_normalized_raw.{args.format}",
        args.format,
        args.dpi,
    )
    plot_normalized_metric(
        plotted,
        trace_groups,
        model_groups,
        "latency_minus_queue_normalized",
        "Generation Latency (Avg Queue Delay Removed, Then Normalized)",
        "Normalized Latency",
        output_dir / f"generation_latency_normalized_minus_queue.{args.format}",
        args.format,
        args.dpi,
    )

    print(f"selected summaries: {len(latest_records)}")
    print(f"dropped older duplicates: {len(dropped_records)}")
    print(f"csv: {csv_path}")
    print(f"figures: {output_dir}")


if __name__ == "__main__":
    main()
