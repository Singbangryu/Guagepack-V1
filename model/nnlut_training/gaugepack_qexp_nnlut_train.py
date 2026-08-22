#!/usr/bin/env python3
"""Build an E7 QEXP reference artifact from a frozen GaugePack U8 artifact.

This tool never opens validation.  It copies the existing calibrated artifact,
re-fits only the two QEXP pages to the analytic E7 contract [0,127] using the
existing train-calibration traces, scores the new pages on the already-disjoint
train-search traces, updates the page/pipeline/summary contracts to 7-bit QEXP,
and leaves every GELU/LayerNorm/scale artifact untouched.

The resulting artifact is intended for the existing GaugePack E2E harness so
that full_vfu_e7_* uses an actual E7 NN-LUT page instead of analytic semantic
override.  It is a segment-local Direct32 *reference* page; final RTL M18/C48
compilation/certification remains a separate compiler step.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SEGMENTS = 16
BOUNDARIES = 15
QMAX_E7 = 127


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: JSON root must be object")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rne_float_to_int(values: np.ndarray) -> np.ndarray:
    # numpy.rint is round-to-nearest, ties-to-even.
    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


def round_shift_rne(values: np.ndarray, shift: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if shift < 0:
        return values << (-shift)
    if shift == 0:
        return values.copy()
    sign = np.where(values < 0, -1, 1).astype(np.int64)
    magnitude = np.abs(values)
    base = magnitude >> shift
    rem = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (rem > half) | ((rem == half) & ((base & 1) != 0))
    rounded = base + increment.astype(np.int64)
    return rounded * sign


def qexp_zero_threshold(score_scale: float, qmax: int) -> int:
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("score_scale must be finite/positive")
    if qmax <= 0:
        raise ValueError("qmax must be positive")

    def code_at(value: int) -> int:
        return int(np.rint(float(qmax) * math.exp(float(value) * score_scale)))

    candidate = min(-1, int(math.floor(math.log(0.5 / float(qmax)) / score_scale)))
    while code_at(candidate) > 0:
        candidate -= 1
    while candidate < -1 and code_at(candidate + 1) == 0:
        candidate += 1
    return candidate


def qexp_target(x: np.ndarray, score_scale: float, qmax: int = QMAX_E7) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    code = rne_float_to_int(float(qmax) * np.exp(x.astype(np.float64) * score_scale))
    return np.clip(code, 0, qmax).astype(np.int64)


def load_softmax_difference_hist(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if "logits_int32" not in payload.files:
            raise KeyError(f"{path}: expected logits_int32")
        raw = np.asarray(payload["logits_int32"])
    if raw.dtype.kind not in "iu" or raw.ndim != 2:
        raise TypeError(f"{path}: logits_int32 must be rank-2 integer array")
    source = raw.astype(np.int64, copy=False)
    diff = source - source.max(axis=1, keepdims=True)
    x, count = np.unique(diff.reshape(-1), return_counts=True)
    return x.astype(np.int64), count.astype(np.float64)


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    total = float(np.sum(weight))
    if total <= 0.0:
        raise ValueError("empty/nonpositive weight")
    return float(np.sum(values.astype(np.float64) * weight.astype(np.float64)) / total)


def weighted_median_int(values: np.ndarray, weight: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weight[order]
    cumulative = np.cumsum(w)
    index = int(np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left"))
    return int(v[min(index, v.size - 1)])


def segment_limits(boundaries: np.ndarray, x_min: int, x_max: int) -> List[Tuple[int, int]]:
    b = np.asarray(boundaries, dtype=np.int64)
    if b.shape != (BOUNDARIES,) or np.any(np.diff(b) <= 0):
        raise ValueError("expected 15 strictly increasing boundaries")
    lows = [x_min, *b.tolist()]
    highs = [int(v) - 1 for v in b.tolist()] + [x_max]
    limits = [(int(lo), int(hi)) for lo, hi in zip(lows, highs)]
    if any(lo > hi for lo, hi in limits):
        raise ValueError("empty segment")
    return limits


def code_metrics(pred: np.ndarray, ref: np.ndarray, weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
    p = np.asarray(pred, dtype=np.int64).reshape(-1)
    r = np.asarray(ref, dtype=np.int64).reshape(-1)
    if p.shape != r.shape or p.size == 0:
        raise ValueError("metric shape/empty error")
    if weight is None:
        w = np.ones(p.shape, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if w.shape != p.shape:
            raise ValueError("metric weight shape mismatch")
    error = p - r
    absolute = np.abs(error)
    total = float(w.sum())
    return {
        "count": int(p.size),
        "weight_sum": total,
        "exact_code_rate": float(np.sum(w[error == 0]) / total),
        "within_1_code_rate": float(np.sum(w[absolute <= 1]) / total),
        "mae_code": float(np.sum(w * absolute) / total),
        "rmse_code": float(math.sqrt(np.sum(w * error.astype(np.float64) ** 2) / total)),
        "max_abs_code": int(absolute.max(initial=0)),
        "mismatch_weight": float(np.sum(w[error != 0])),
    }


def infer_page(
    x: np.ndarray,
    *,
    x_min: int,
    x_max: int,
    boundaries: np.ndarray,
    origin: np.ndarray,
    multiplier: np.ndarray,
    bias: np.ndarray,
    right_shift: np.ndarray,
    qmin: int,
    qmax: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    x_eval = np.clip(x, x_min, x_max)
    seg = np.searchsorted(boundaries, x_eval, side="right")
    local = x_eval - origin[seg]
    p = multiplier[seg] * local + bias[seg]
    out = np.empty_like(p)
    for shift in np.unique(right_shift):
        selected = right_shift[seg] == shift
        out[selected] = round_shift_rne(p[selected], int(shift))
    return np.clip(out, qmin, qmax).astype(np.int64)


def fit_one_segment(
    *,
    seg_lo: int,
    seg_hi: int,
    origin: int,
    score_scale: float,
    zero_threshold: int,
    cal_x: np.ndarray,
    cal_w: np.ndarray,
    coefficient_bits: int,
    max_fraction_bits: int,
) -> Tuple[int, int, int, Dict[str, Any]]:
    active_lo = max(seg_lo, zero_threshold + 1)
    active_hi = min(seg_hi, -1)
    if active_lo > active_hi:
        return 0, 0, 0, {
            "active_range": None,
            "max_abs_code_exhaustive": 0,
            "calibration_exact_code_rate": 1.0,
        }

    selected = (cal_x >= active_lo) & (cal_x <= active_hi)
    x_data = cal_x[selected]
    w_data = cal_w[selected]

    # Full-domain anchors prevent an occurrence-only fit from becoming pathological.
    full_x = np.arange(active_lo, active_hi + 1, dtype=np.int64)
    full_y = qexp_target(full_x, score_scale)
    if x_data.size:
        y_data = qexp_target(x_data, score_scale)
        anchor_count = min(17, full_x.size)
        anchor_idx = np.linspace(0, full_x.size - 1, anchor_count).round().astype(np.int64)
        anchors = full_x[np.unique(anchor_idx)]
        anchor_y = qexp_target(anchors, score_scale)
        anchor_weight = max(float(w_data.sum()) / 100_000.0, 0.25)
        fit_x = np.concatenate([x_data, anchors])
        fit_y = np.concatenate([y_data, anchor_y])
        fit_w = np.concatenate([w_data, np.full(anchors.shape, anchor_weight, dtype=np.float64)])
    else:
        fit_x = full_x
        fit_y = full_y
        fit_w = np.ones(full_x.shape, dtype=np.float64)

    t = fit_x - int(origin)
    total_w = float(fit_w.sum())
    t_mean = float(np.sum(fit_w * t) / total_w)
    y_mean = float(np.sum(fit_w * fit_y) / total_w)
    denominator = float(np.sum(fit_w * (t - t_mean) ** 2))
    slope = 0.0 if denominator == 0.0 else float(np.sum(fit_w * (t - t_mean) * (fit_y - y_mean)) / denominator)
    intercept = y_mean - slope * t_mean

    coeff_lo = -(1 << (coefficient_bits - 1))
    coeff_hi = (1 << (coefficient_bits - 1)) - 1
    fraction_cap = min(int(max_fraction_bits), 30)
    # Bias is about qmax*2^F; cap F so it fits the template coefficient width.
    while fraction_cap > 0 and QMAX_E7 * (1 << fraction_cap) > coeff_hi:
        fraction_cap -= 1

    best: Optional[Tuple[Tuple[Any, ...], int, int, int, Dict[str, Any]]] = None
    local_full = full_x - int(origin)
    for shift in range(fraction_cap, max(fraction_cap - 9, 0) - 1, -1):
        divisor = 1 << shift
        m0 = int(np.rint(slope * divisor))
        for m in range(m0 - 8, m0 + 9):
            if not coeff_lo <= m <= coeff_hi:
                continue
            residual = fit_y.astype(np.int64) * np.int64(divisor) - np.int64(m) * t.astype(np.int64)
            b_mean = int(np.rint(weighted_mean(residual, fit_w)))
            b_med = weighted_median_int(residual, fit_w)
            b_candidates = {
                b_mean,
                b_med,
                int(np.rint(intercept * divisor)),
            }
            # Small sub-LSB phase search matters around RNE boundaries.
            for base in tuple(b_candidates):
                for offset in (0, -1, 1, -(divisor // 4), divisor // 4):
                    b_candidates.add(base + offset)
            for b in b_candidates:
                if not coeff_lo <= b <= coeff_hi:
                    continue
                pred_full = np.clip(
                    round_shift_rne(np.int64(m) * local_full + np.int64(b), shift),
                    0,
                    QMAX_E7,
                )
                abs_full = np.abs(pred_full - full_y)
                max_abs = int(abs_full.max(initial=0))
                pred_fit = np.clip(
                    round_shift_rne(np.int64(m) * t + np.int64(b), shift),
                    0,
                    QMAX_E7,
                )
                err_fit = pred_fit - fit_y
                abs_fit = np.abs(err_fit)
                mae = float(np.sum(fit_w * abs_fit) / total_w)
                mismatch = float(np.sum(fit_w[err_fit != 0]) / total_w)
                rmse = float(math.sqrt(np.sum(fit_w * err_fit.astype(np.float64) ** 2) / total_w))
                key = (max_abs, mae, mismatch, rmse, -shift, abs(m), abs(b), m, b)
                if best is None or key < best[0]:
                    best = (
                        key,
                        int(m),
                        int(b),
                        int(shift),
                        {
                            "active_range": [active_lo, active_hi],
                            "max_abs_code_exhaustive": max_abs,
                            "fit_mae_code": mae,
                            "fit_mismatch_rate": mismatch,
                            "fit_rmse_code": rmse,
                            "regression_slope": slope,
                            "regression_intercept": intercept,
                        },
                    )
    if best is None:
        raise RuntimeError(f"failed to fit segment [{seg_lo},{seg_hi}]")
    return best[1], best[2], best[3], best[4]


def update_output_quant(metadata: Dict[str, Any]) -> None:
    metadata["qexp_bits"] = 7
    metadata["qexp_qmax"] = QMAX_E7
    quant = metadata.get("output_quant")
    if isinstance(quant, dict):
        quant["qmin"] = 0
        quant["qmax"] = QMAX_E7
        quant["scale"] = 1.0 / float(QMAX_E7)
        quant["zero_point"] = 0
        quant["name"] = quant.get("name", "qexp_output")


def fit_layer_page(
    *,
    page_path: Path,
    pipeline_path: Path,
    calibration_trace: Path,
    search_trace: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    template = read_json(page_path)
    pipeline = read_json(pipeline_path)
    schema = str(template.get("schema", ""))
    if schema != "gaugepack.nnlut.pwl.v2":
        raise ValueError(f"{page_path}: E7 builder requires segment-local PWL v2 template")
    if str(template.get("coordinate_mode", "")) != "segment_local":
        raise ValueError(f"{page_path}: coordinate_mode must be segment_local")
    if int(template.get("qmin", -1)) != 0 or int(template.get("qmax", -1)) != 255:
        raise ValueError(f"{page_path}: expected U8 QEXP template [0,255]")
    metadata = copy.deepcopy(template.get("metadata", {}))
    if metadata.get("task") != "softmax_qexp":
        raise ValueError(f"{page_path}: not a softmax_qexp page")
    score_scale = float(pipeline.get("score_scale_folded", metadata.get("score_scale_folded")))
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError(f"{pipeline_path}: invalid score scale")
    if not math.isclose(float(metadata.get("score_scale_folded", score_scale)), score_scale, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{page_path}: page/pipeline score scale mismatch")

    boundaries = np.asarray(template["boundaries"], dtype=np.int64)
    if boundaries.shape != (BOUNDARIES,) or np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"{page_path}: invalid boundaries")
    x_min = int(template["x_min"])
    x_max = int(template["x_max"])
    origin = np.asarray(template.get("origin", [x_min, *boundaries.tolist()]), dtype=np.int64)
    if origin.shape != (SEGMENTS,):
        raise ValueError(f"{page_path}: invalid origin")
    expected_origin = np.asarray([x_min, *boundaries.tolist()], dtype=np.int64)
    if not np.array_equal(origin, expected_origin):
        raise ValueError(f"{page_path}: origin must equal [x_min,*boundaries]")
    coefficient_bits = int(template.get("coefficient_bits", 32))
    max_fraction_bits = int(max(np.asarray(template.get("right_shift", [24]), dtype=np.int64).max(initial=24), 24))

    zero_threshold = qexp_zero_threshold(score_scale, QMAX_E7)
    cal_x, cal_w = load_softmax_difference_hist(calibration_trace)
    search_x, search_w = load_softmax_difference_hist(search_trace)

    multiplier = np.zeros(SEGMENTS, dtype=np.int64)
    bias = np.zeros(SEGMENTS, dtype=np.int64)
    right_shift = np.zeros(SEGMENTS, dtype=np.int64)
    segment_report: List[Dict[str, Any]] = []
    for index, (lo, hi) in enumerate(segment_limits(boundaries, x_min, x_max)):
        m, b, s, report = fit_one_segment(
            seg_lo=lo,
            seg_hi=hi,
            origin=int(origin[index]),
            score_scale=score_scale,
            zero_threshold=zero_threshold,
            cal_x=cal_x,
            cal_w=cal_w,
            coefficient_bits=coefficient_bits,
            max_fraction_bits=max_fraction_bits,
        )
        multiplier[index] = m
        bias[index] = b
        right_shift[index] = s
        segment_report.append({"segment": index, "range": [lo, hi], "M": m, "B_local": b, "shift": s, **report})

    active_cal = (cal_x > zero_threshold) & (cal_x < 0)
    active_search = (search_x > zero_threshold) & (search_x < 0)
    cal_pred = infer_page(
        cal_x[active_cal], x_min=x_min, x_max=x_max, boundaries=boundaries,
        origin=origin, multiplier=multiplier, bias=bias, right_shift=right_shift,
        qmin=0, qmax=QMAX_E7,
    )
    search_pred = infer_page(
        search_x[active_search], x_min=x_min, x_max=x_max, boundaries=boundaries,
        origin=origin, multiplier=multiplier, bias=bias, right_shift=right_shift,
        qmin=0, qmax=QMAX_E7,
    )
    cal_ref = qexp_target(cal_x[active_cal], score_scale)
    search_ref = qexp_target(search_x[active_search], score_scale)
    calibration_metrics = code_metrics(cal_pred, cal_ref, cal_w[active_cal])
    search_metrics = code_metrics(search_pred, search_ref, search_w[active_search])

    grid_lo = max(x_min, zero_threshold + 1)
    grid_hi = min(x_max, -1)
    grid = np.arange(grid_lo, grid_hi + 1, dtype=np.int64)
    grid_pred = infer_page(
        grid, x_min=x_min, x_max=x_max, boundaries=boundaries,
        origin=origin, multiplier=multiplier, bias=bias, right_shift=right_shift,
        qmin=0, qmax=QMAX_E7,
    )
    grid_ref = qexp_target(grid, score_scale)
    grid_metrics = code_metrics(grid_pred, grid_ref)

    new_page = copy.deepcopy(template)
    old_name = str(template.get("name", page_path.stem))
    new_name = old_name.replace("_u8_", "_u7_").replace("u8", "u7")
    if new_name == old_name:
        new_name = "nnlut_softmax_qexp_u7_direct32"
    new_page["name"] = new_name
    new_page["qmin"] = 0
    new_page["qmax"] = QMAX_E7
    new_page["multiplier"] = multiplier.tolist()
    new_page["bias"] = bias.tolist()
    new_page["right_shift"] = right_shift.tolist()
    new_page["origin"] = origin.tolist()
    metadata["task"] = "softmax_qexp"
    metadata["score_scale_folded"] = score_scale
    metadata["analytic_zero_threshold_inclusive"] = int(zero_threshold)
    metadata["e7_build"] = {
        "method": "train-calibration weighted segment-local Direct32 refit",
        "source_u8_page": page_path.name,
        "validation_used": False,
        "boundaries_reused_from_u8": True,
    }
    update_output_quant(metadata)
    new_page["metadata"] = metadata

    new_stem = page_path.stem.replace("_u8_", "_u7_").replace("u8", "u7")
    if new_stem == page_path.stem:
        new_stem = "nnlut_softmax_qexp_u7_direct32"
    new_json = output_dir / f"{new_stem}.json"
    write_json(new_json, new_page)

    new_pipeline = copy.deepcopy(pipeline)
    new_pipeline["qexp_bits"] = 7
    new_pipeline["qexp_qmax"] = QMAX_E7
    new_pipeline["analytic_zero_threshold_inclusive"] = int(zero_threshold)
    new_pipeline["exp_lut_json"] = new_json.name
    new_pipeline["e7_qexp_reference_page"] = True
    write_json(output_dir / "softmax_pipeline.json", new_pipeline)

    return {
        "source_u8_page": str(page_path),
        "e7_page": str(new_json),
        "score_scale": score_scale,
        "zero_threshold_inclusive": int(zero_threshold),
        "x_min": x_min,
        "x_max": x_max,
        "calibration_metrics": calibration_metrics,
        "disjoint_search_metrics": search_metrics,
        "exhaustive_active_domain_metrics": grid_metrics,
        "segment_fit": segment_report,
        "page_name": new_name,
        "page_basename": new_json.name,
        "multiplier": multiplier.tolist(),
        "bias": bias.tolist(),
        "right_shift": right_shift.tolist(),
        "origin": origin.tolist(),
    }


def to_hex(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{int(value) & mask:0{(bits + 3) // 4}X}"


def write_mem(path: Path, values: Iterable[int], bits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(to_hex(int(v), bits) for v in values) + "\n", encoding="ascii")


def update_export_files(entry: Dict[str, Any], report: Mapping[str, Any], softmax_dir: Path) -> None:
    exported = entry.setdefault("exported_files", {})
    exp = exported.setdefault("exp", {})
    new_json = Path(str(report["e7_page"]))
    arrays = {
        "boundary_mem": ("boundary_i32.mem", report.get("boundaries"), 32),
        "multiplier_mem": ("multiplier_i32.mem", report["multiplier"], 32),
        "bias_mem": ("bias_i32.mem", report["bias"], 32),
        "origin_mem": ("origin_i32.mem", report["origin"], 32),
        "right_shift_mem": ("right_shift_u8.mem", report["right_shift"], 8),
    }
    # Use names consistent with the page basename where possible.
    stem = new_json.stem
    for key, (suffix, values, bits) in arrays.items():
        if values is None:
            continue
        filename = f"{stem}.{suffix}"
        write_mem(softmax_dir / filename, values, bits)
        exp[key] = filename
    exp["json"] = new_json.name
    exported["pipeline_json"] = "softmax_pipeline.json"


def locate_qexp_entries(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = summary.get("calibrated_pages")
    if not isinstance(pages, dict):
        raise ValueError("calibration_summary.json has no calibrated_pages object")
    primary = pages.get("qexp_deferred_pv")
    if not isinstance(primary, list):
        primary = pages.get("softmax_exp_div")
    if not isinstance(primary, list) or not primary:
        raise ValueError("no QEXP page entries found in calibration_summary.json")
    return primary


def find_page_from_entry(root: Path, layer: int, entry: Mapping[str, Any]) -> Path:
    exp = entry.get("exported_files", {}).get("exp", {})
    preferred = Path(str(exp.get("json", ""))).name if isinstance(exp, Mapping) else ""
    softmax_dir = root / "pages" / f"encoder_layer_{layer}" / "softmax"
    if preferred and (softmax_dir / preferred).is_file():
        return softmax_dir / preferred
    matches = sorted(softmax_dir.glob("nnlut_softmax_qexp_u8_direct32.json"))
    if len(matches) != 1:
        matches = sorted(softmax_dir.glob("nnlut_softmax_qexp_u*_direct32.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"layer {layer}: cannot uniquely locate U8 QEXP page in {softmax_dir}")
    return matches[0]



def purge_copied_qexp_pages(output_root: Path, layers: Iterable[int]) -> List[str]:
    """Remove copied source QEXP payloads before installing the E7-only pages.

    The baseline evaluator deliberately rejects an artifact directory containing
    more than one nnlut_softmax_qexp_u*_direct32.json per layer.  Because this
    builder starts from shutil.copytree(base, out), the source U8 page must not
    be left beside the new U7 page.  Purge the full QEXP payload family
    (JSON + MEM sidecars) while preserving softmax_pipeline.json and traces.
    """
    removed: List[str] = []
    for layer in layers:
        softmax_dir = output_root / "pages" / f"encoder_layer_{int(layer)}" / "softmax"
        if not softmax_dir.is_dir():
            raise FileNotFoundError(softmax_dir)
        for path in sorted(softmax_dir.glob("nnlut_softmax_qexp_u*_direct32.*")):
            if path.is_file():
                removed.append(str(path.relative_to(output_root)))
                path.unlink()
    return removed

def build_artifact(base: Path, out: Path, overwrite: bool) -> Dict[str, Any]:
    base = base.resolve()
    out = out.resolve()
    if not (base / "calibration_summary.json").is_file():
        raise FileNotFoundError(base / "calibration_summary.json")
    if not (base / "calibrated_scales.json").is_file():
        raise FileNotFoundError(base / "calibrated_scales.json")
    if out.exists():
        if not overwrite and any(out.iterdir()):
            raise FileExistsError(f"{out} is not empty; pass --overwrite-output")
        if overwrite:
            shutil.rmtree(out)
    shutil.copytree(base, out)

    summary_path = out / "calibration_summary.json"
    summary = read_json(summary_path)
    entries = locate_qexp_entries(summary)
    copied_qexp_payloads_removed = purge_copied_qexp_pages(
        out, [int(entry["layer_index"]) for entry in entries]
    )
    reports: List[Dict[str, Any]] = []

    for entry in entries:
        layer = int(entry["layer_index"])
        source_entry = next(
            item for item in locate_qexp_entries(read_json(base / "calibration_summary.json"))
            if int(item["layer_index"]) == layer
        )
        source_page = find_page_from_entry(base, layer, source_entry)
        source_pipeline = base / "pages" / f"encoder_layer_{layer}" / "softmax" / "softmax_pipeline.json"
        calibration_trace = base / "traces" / "calibration" / f"layer_{layer}_softmax_logits.npz"
        search_trace = base / "traces" / "search" / f"layer_{layer}_softmax_logits.npz"
        softmax_dir = out / "pages" / f"encoder_layer_{layer}" / "softmax"
        report = fit_layer_page(
            page_path=source_page,
            pipeline_path=source_pipeline,
            calibration_trace=calibration_trace,
            search_trace=search_trace,
            output_dir=softmax_dir,
        )
        report["layer_index"] = layer
        report["boundaries"] = read_json(Path(report["e7_page"]))["boundaries"]
        old_metrics = entry.pop("final_search_metrics", None)
        if old_metrics is not None:
            entry["source_u8_final_search_metrics"] = old_metrics
        entry["qexp_code_bits"] = 7
        entry["score_scale"] = report["score_scale"]
        entry["e7_qexp_build_metrics"] = {
            "calibration": report["calibration_metrics"],
            "disjoint_search": report["disjoint_search_metrics"],
            "exhaustive_active_domain": report["exhaustive_active_domain_metrics"],
        }
        update_export_files(entry, report, softmax_dir)
        reports.append(report)

    # Keep legacy alias bit-identical to the primary QEXP list if present.
    pages = summary["calibrated_pages"]
    pages["qexp_deferred_pv"] = entries
    if "softmax_exp_div" in pages:
        pages["softmax_exp_div"] = copy.deepcopy(entries)
    arguments = summary.setdefault("arguments", {})
    arguments["qexp_code_bits"] = 7
    summary["e7_qexp_rebuild"] = {
        "source_artifact": str(base),
        "validation_loaded": False,
        "method": "refit QEXP only from existing train calibration/search traces",
        "layers": [int(item["layer_index"]) for item in reports],
        "note": "all non-QEXP pages/scales are copied unchanged",
    }
    write_json(summary_path, summary)

    build_report = {
        "schema": "gaugepack.e7_qexp_artifact_build.v2",
        "source_artifact": str(base),
        "output_artifact": str(out),
        "validation_used": False,
        "qexp_bits": 7,
        "qexp_qmax": QMAX_E7,
        "copied_qexp_payloads_removed": copied_qexp_payloads_removed,
        "layers": reports,
    }
    write_json(out / "e7_qexp_build_report.json", build_report)
    return build_report


def self_test() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        base = root / "base"
        out = root / "e7"
        soft = base / "pages" / "encoder_layer_0" / "softmax"
        soft.mkdir(parents=True)
        boundaries = np.rint(np.geomspace(1, 99, 16)).astype(np.int64)
        # Build monotonic negative-domain boundaries [-99..-1].
        boundaries = np.linspace(-93, -6, 15).round().astype(np.int64)
        x_min, x_max = -100, -1
        origin = [x_min, *boundaries.tolist()]
        page = {
            "schema": "gaugepack.nnlut.pwl.v2",
            "name": "nnlut_softmax_qexp_u8_direct32",
            "relu_count": 15,
            "segment_count": 16,
            "coordinate_mode": "segment_local",
            "x_min": x_min,
            "x_max": x_max,
            "boundaries": boundaries.tolist(),
            "origin": origin,
            "multiplier": [0] * 16,
            "bias": [1] * 16,
            "right_shift": [0] * 16,
            "qmin": 0,
            "qmax": 255,
            "coefficient_bits": 32,
            "input_policy": "clamp",
            "metadata": {"task": "softmax_qexp", "score_scale_folded": 0.05, "qexp_bits": 8, "qexp_qmax": 255, "analytic_zero_threshold_inclusive": -125},
        }
        write_json(soft / "nnlut_softmax_qexp_u8_direct32.json", page)
        write_json(soft / "softmax_pipeline.json", {"schema": "gaugepack.nnlut.softmax.deferred_pv.v1", "score_scale_folded": 0.05, "qexp_bits": 8, "qexp_qmax": 255, "analytic_zero_threshold_inclusive": -125})
        trace_dir = base / "traces"
        for role in ("calibration", "search"):
            (trace_dir / role).mkdir(parents=True, exist_ok=True)
            logits = rng.integers(-100, 1, size=(128, 64), dtype=np.int32)
            logits[:, 0] = 0
            np.savez_compressed(trace_dir / role / "layer_0_softmax_logits.npz", logits_int32=logits)
        write_json(base / "calibrated_scales.json", {"schema": "test"})
        entry = {"layer_index": 0, "score_scale": 0.05, "value_scale": 0.06, "qexp_code_bits": 8, "task": "softmax", "exported_files": {"exp": {"json": "nnlut_softmax_qexp_u8_direct32.json"}, "pipeline_json": "softmax_pipeline.json"}}
        write_json(base / "calibration_summary.json", {"schema": "test", "arguments": {"qexp_code_bits": 8}, "calibrated_pages": {"qexp_deferred_pv": [entry], "softmax_exp_div": [copy.deepcopy(entry)]}})
        report = build_artifact(base, out, overwrite=True)
        layer = report["layers"][0]
        if layer["disjoint_search_metrics"]["max_abs_code"] > 3:
            raise AssertionError("synthetic E7 fit is unexpectedly poor")
        rebuilt = read_json(out / "calibration_summary.json")
        if int(rebuilt["arguments"]["qexp_code_bits"]) != 7:
            raise AssertionError("summary qexp width was not updated")
        soft_out = out / "pages" / "encoder_layer_0" / "softmax"
        if not (soft_out / layer["page_basename"]).is_file():
            raise AssertionError("E7 page missing")
        qexp_json = sorted(soft_out.glob("nnlut_softmax_qexp_u*_direct32.json"))
        if [path.name for path in qexp_json] != ["nnlut_softmax_qexp_u7_direct32.json"]:
            raise AssertionError(f"E7 artifact must contain exactly one QEXP page, got {qexp_json}")
        if list(soft_out.glob("nnlut_softmax_qexp_u8_direct32.*")):
            raise AssertionError("copied U8 QEXP payload survived E7 rebuild")
    print("SELF-TEST PASS: E7-only QEXP artifact, copied-U8 purge, train/search isolation, page/pipeline/summary rewrite")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--base-artifact-dir", type=Path, default=Path("nnlut_sst2_gaugepack_all_fixed"))
    p.add_argument("--out-dir", type=Path, default=Path("nnlut_sst2_gaugepack_e7_qexp"))
    p.add_argument("--overwrite-output", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        report = build_artifact(args.base_artifact_dir, args.out_dir, args.overwrite_output)
        print("E7 QEXP artifact built without loading validation.")
        for layer in report["layers"]:
            cal = layer["calibration_metrics"]
            sea = layer["disjoint_search_metrics"]
            grid = layer["exhaustive_active_domain_metrics"]
            print(
                f"layer{layer['layer_index']}: search exact={sea['exact_code_rate']:.6f} "
                f"MAE={sea['mae_code']:.6f} max={sea['max_abs_code']} | "
                f"grid max={grid['max_abs_code']}"
            )
        print(f"WROTE {Path(args.out_dir).resolve() / 'e7_qexp_build_report.json'}")
        return 0
    except (AssertionError, FileExistsError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
