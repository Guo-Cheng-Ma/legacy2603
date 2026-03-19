import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from .request_state import RequestState, TraceRequest


BLOCK_SIZE_TOKENS = 16
REQUIRED_TRACE_KEYS = {
    "chat_id",
    "parent_chat_id",
    "timestamp",
    "input_length",
    "output_length",
    "type",
    "turn",
    "hash_ids",
}


class TraceFormatError(ValueError):
    pass


@dataclass(frozen=True)
class TraceQPSMetadata:
    requested_qps: float
    raw_qps: float
    raw_arrival_span_s: float
    timestamp_scale_factor: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceFormatError(message)


def _validate_and_build(
    req_id: int,
    line_no: int,
    row: Dict,
    timestamp_scale_factor: float,
) -> TraceRequest:
    missing = sorted(REQUIRED_TRACE_KEYS - set(row.keys()))
    _require(not missing, f"line {line_no}: missing required keys: {missing}")

    chat_id = row["chat_id"]
    parent_chat_id = row["parent_chat_id"]
    timestamp = row["timestamp"]
    input_length = row["input_length"]
    output_length = row["output_length"]
    request_type = row["type"]
    turn = row["turn"]
    hash_ids = row["hash_ids"]

    _require(isinstance(chat_id, int), f"line {line_no}: chat_id must be int")
    _require(
        isinstance(parent_chat_id, int),
        f"line {line_no}: parent_chat_id must be int",
    )
    _require(
        isinstance(timestamp, (int, float)),
        f"line {line_no}: timestamp must be numeric",
    )
    _require(
        isinstance(input_length, int) and input_length >= 0,
        f"line {line_no}: input_length must be non-negative int",
    )
    _require(
        isinstance(output_length, int) and output_length >= 0,
        f"line {line_no}: output_length must be non-negative int",
    )
    _require(
        isinstance(request_type, str) and len(request_type) > 0,
        f"line {line_no}: type must be non-empty string",
    )
    _require(
        isinstance(turn, int) and turn >= 1,
        f"line {line_no}: turn must be int >= 1",
    )
    _require(isinstance(hash_ids, list), f"line {line_no}: hash_ids must be list")
    _require(
        all(isinstance(hash_id, int) for hash_id in hash_ids),
        f"line {line_no}: every hash_id must be int",
    )

    expected_blocks = int(math.ceil(input_length / BLOCK_SIZE_TOKENS))
    _require(
        len(hash_ids) == expected_blocks,
        (
            f"line {line_no}: hash_ids length mismatch, "
            f"expected {expected_blocks}, got {len(hash_ids)}"
        ),
    )

    return TraceRequest(
        req_id=req_id,
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        timestamp=float(timestamp) * timestamp_scale_factor,
        input_length=input_length,
        output_length=output_length,
        request_type=request_type,
        turn=turn,
        hash_ids=hash_ids,
    )


def _iter_rows(trace_path: Path) -> Iterable[Tuple[int, Dict]]:
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(
                    f"line {line_no}: invalid json ({exc.msg})"
                ) from exc
            _require(isinstance(row, dict), f"line {line_no}: row must be object")
            yield line_no, row


def _derive_trace_qps_metadata(
    rows: List[Tuple[int, Dict]],
    requested_qps: float,
) -> TraceQPSMetadata:
    _require(
        isinstance(requested_qps, (int, float)) and float(requested_qps) > 0.0,
        f"requested_qps must be > 0, got {requested_qps}",
    )
    raw_timestamps: List[float] = []
    for line_no, row in rows:
        timestamp = row.get("timestamp")
        _require(
            isinstance(timestamp, (int, float)),
            f"line {line_no}: timestamp must be numeric",
        )
        raw_timestamps.append(float(timestamp))

    _require(
        len(raw_timestamps) >= 2,
        "trace file must contain at least 2 requests to derive raw qps",
    )
    raw_arrival_span_s = max(raw_timestamps) - min(raw_timestamps)
    _require(
        raw_arrival_span_s > 0.0,
        "trace file must span > 0 seconds to derive raw qps",
    )
    requested_qps = float(requested_qps)
    raw_qps = len(raw_timestamps) / raw_arrival_span_s
    return TraceQPSMetadata(
        requested_qps=requested_qps,
        raw_qps=raw_qps,
        raw_arrival_span_s=raw_arrival_span_s,
        timestamp_scale_factor=raw_qps / requested_qps,
    )


def load_trace_requests(
    trace_path: Union[str, Path],
    requested_qps: float,
) -> Tuple[List[TraceRequest], TraceQPSMetadata]:
    path = Path(trace_path)
    _require(path.exists(), f"trace file not found: {path}")

    rows = list(_iter_rows(path))
    metadata = _derive_trace_qps_metadata(rows, requested_qps=requested_qps)

    requests: List[TraceRequest] = []
    req_id = 0
    for line_no, row in rows:
        requests.append(
            _validate_and_build(
                req_id=req_id,
                line_no=line_no,
                row=row,
                timestamp_scale_factor=metadata.timestamp_scale_factor,
            )
        )
        req_id += 1

    requests.sort(key=lambda request: (request.timestamp, request.req_id))
    return requests, metadata


def load_request_states(
    trace_path: Union[str, Path],
    requested_qps: float,
) -> Tuple[List[RequestState], TraceQPSMetadata]:
    requests, metadata = load_trace_requests(trace_path, requested_qps=requested_qps)
    return [
        RequestState(trace=request)
        for request in requests
    ], metadata
