#!/usr/bin/env python3
"""Standalone GaugePack v1 terminal GELU NN-LUT trainer.

Frozen arithmetic contract
--------------------------

    a      = FFN1 integer accumulator + folded integer bias
    x_real = a * s_acc
    g_real = GELU_exact(x_real)
    Y      = NARROW_S8(RNE_even(g_real / s_out))

``a`` is carried in an S32 container, while the active calibrated domain must
be safe for the VFU's signed A27 DSP input.  The learned reference model is a
continuous 15-hinge ReLU spline (16 PWL segments):

    f(a) = b + m*a + sum_i delta_m[i] * relu(a-knot[i])

This file stops at the floating PWL artifact.  It does not claim to have
compiled the page into GaugePack's production A27/M18/C48/P48 format.  The
coefficient compiler must quantize the exported boundaries, slopes and
intercepts and rerun the bit-true regression.

Input NPZ schema
----------------

Both calibration and disjoint-search files use:

    x_int32 : raw FFN1 accumulator values including integer bias, shape [N]
    count   : optional occurrence count/weight, shape [N], positive

Repeated x_int32 entries are coalesced.  Calibration is the only source used
to select ``s_out`` and fit the page.  Search is opened only for held-out
metrics.  The SST-2 validation split must not be supplied to this trainer.

Example
-------

    python gaugepack_gelu_nnlut_train.py \
      --calibration-npz layer0_ffn1_acc_calibration.npz \
      --search-npz layer0_ffn1_acc_search.npz \
      --accumulator-scale 0.0003125 \
      --output-scale 0.03125 \
      --site-name layer0_ffn1_gelu \
      --out-dir ./gelu_layer0 \
      --device cuda

Dependencies: Python 3.10+, NumPy, PyTorch 2.x.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - environment-specific.
    raise SystemExit(
        "This trainer requires PyTorch 2.x. Install torch before running it. "
        f"Original import error: {exc}"
    ) from exc


SEED = 20260728
SEGMENT_COUNT = 16
HINGE_COUNT = SEGMENT_COUNT - 1
A27_MIN = -(1 << 26)
A27_MAX = (1 << 26) - 1
GELU_QMIN = -127
GELU_QMAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{requested} requested, but CUDA is unavailable")
    return requested


def json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_int32(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"{name} must contain integer-valued data")
        array = rounded
    result = np.asarray(array, dtype=np.int64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if int(result.min()) < INT32_MIN or int(result.max()) > INT32_MAX:
        raise ValueError(f"{name} is outside signed INT32")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class Histogram:
    x: np.ndarray
    weight: np.ndarray
    source: str

    def validate(self, role: str) -> None:
        x = ensure_int32(f"{role}.x_int32", self.x)
        weight = np.asarray(self.weight, dtype=np.float64).reshape(-1)
        if x.shape != weight.shape:
            raise ValueError(f"{role}: x_int32/count shape mismatch")
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError(f"{role}: count must be finite and positive")
        if int(x.min()) < A27_MIN or int(x.max()) > A27_MAX:
            raise ValueError(
                f"{role}: active GELU domain [{int(x.min())}, {int(x.max())}] "
                f"does not fit signed A27 [{A27_MIN}, {A27_MAX}]"
            )


def coalesce_histogram(
    x: np.ndarray, weight: np.ndarray, source: str
) -> Histogram:
    x64 = ensure_int32(f"{source}.x_int32", x)
    w64 = np.asarray(weight, dtype=np.float64).reshape(-1)
    if x64.shape != w64.shape:
        raise ValueError(f"{source}: x_int32/count shape mismatch")
    if not np.all(np.isfinite(w64)) or np.any(w64 <= 0.0):
        raise ValueError(f"{source}: count must be finite and positive")
    order = np.argsort(x64, kind="stable")
    sorted_x = x64[order]
    sorted_w = w64[order]
    unique_x, first = np.unique(sorted_x, return_index=True)
    unique_w = np.add.reduceat(sorted_w, first)
    result = Histogram(
        x=np.ascontiguousarray(unique_x, dtype=np.int64),
        weight=np.ascontiguousarray(unique_w, dtype=np.float64),
        source=source,
    )
    result.validate(source)
    return result


def load_histogram_npz(path: Path, role: str) -> Histogram:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if "x_int32" not in payload.files:
            raise KeyError(f"{path}: expected key x_int32")
        x = np.asarray(payload["x_int32"])
        weight = (
            np.asarray(payload["count"], dtype=np.float64)
            if "count" in payload.files
            else np.ones(np.asarray(x).size, dtype=np.float64)
        )
    return coalesce_histogram(x, weight, f"{role}:{path}")


def gelu_teacher_real(x_int32: np.ndarray, accumulator_scale: float) -> np.ndarray:
    if not math.isfinite(accumulator_scale) or accumulator_scale <= 0.0:
        raise ValueError("accumulator_scale must be finite and positive")
    x = ensure_int32("GELU teacher input", x_int32)
    real_x = x.astype(np.float64) * float(accumulator_scale)
    if not np.all(np.isfinite(real_x)):
        raise ValueError("GELU real input overflowed float64")
    erf_value = np.vectorize(math.erf, otypes=[np.float64])(
        real_x / math.sqrt(2.0)
    )
    target = 0.5 * real_x * (1.0 + erf_value)
    if not np.all(np.isfinite(target)):
        raise ValueError("GELU teacher produced NaN or infinity")
    return target


def choose_output_scale(
    calibration_x: np.ndarray,
    accumulator_scale: float,
    requested: Optional[float],
) -> float:
    if requested is not None:
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("--output-scale must be finite and positive")
        return float(requested)
    target = gelu_teacher_real(calibration_x, accumulator_scale)
    positive = float(target.max(initial=0.0)) / float(GELU_QMAX)
    negative = -float(target.min(initial=0.0)) / float(-GELU_QMIN)
    return max(positive, negative, np.finfo(np.float64).tiny)


def gelu_target_code(
    x_int32: np.ndarray, accumulator_scale: float, output_scale: float
) -> np.ndarray:
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be finite and positive")
    code = np.rint(
        gelu_teacher_real(x_int32, accumulator_scale) / output_scale
    ).astype(np.int64)
    return np.clip(code, GELU_QMIN, GELU_QMAX)


def weighted_metrics(
    predicted: np.ndarray, reference: np.ndarray, weight: np.ndarray
) -> Dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.int64).reshape(-1)
    ref = np.asarray(reference, dtype=np.int64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if not (pred.shape == ref.shape == w.shape):
        raise ValueError("metric shape mismatch")
    if pred.size == 0 or float(w.sum()) <= 0.0:
        raise ValueError("metrics require at least one positive-weight sample")
    error = pred - ref
    absolute = np.abs(error)
    total = float(w.sum())
    return {
        "unique_x_count": int(pred.size),
        "weighted_sample_count": total,
        "exact_code_rate": float(w[error == 0].sum() / total),
        "within_1_code_rate": float(w[absolute <= 1].sum() / total),
        "mae_code": float(np.sum(w * absolute) / total),
        "rmse_code": float(
            math.sqrt(np.sum(w * error.astype(np.float64) ** 2) / total)
        ),
        "max_abs_code": int(absolute.max(initial=0)),
        "target_code_min": int(ref.min()),
        "target_code_max": int(ref.max()),
        "predicted_code_min": int(pred.min()),
        "predicted_code_max": int(pred.max()),
        "minus_128_count": int(np.count_nonzero(pred == -128)),
    }


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 120
    batch_size: int = 4096
    learning_rate: float = 1.0e-3
    float_loss_weight: float = 0.05
    seed: int = SEED
    device: str = "auto"
    train_dtype: str = "float64"
    log_every: int = 20

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.float_loss_weight < 0.0 or not math.isfinite(
            self.float_loss_weight
        ):
            raise ValueError("float_loss_weight must be finite and non-negative")
        if self.train_dtype not in {"float32", "float64"}:
            raise ValueError("train_dtype must be float32 or float64")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


class OrderedHingePWL(nn.Module):
    """Fifteen ordered ReLU hinges, producing sixteen continuous lines."""

    def __init__(
        self,
        x_min: int,
        x_max: int,
        initial_knots_raw: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if x_min >= x_max or x_max - x_min < HINGE_COUNT:
            raise ValueError(
                f"domain [{x_min}, {x_max}] cannot hold 16 integer segments"
            )
        center = 0.5 * (float(x_min) + float(x_max))
        scale = 0.5 * (float(x_max) - float(x_min))
        self.register_buffer("x_center", torch.tensor(center, dtype=torch.float64))
        self.register_buffer("x_scale", torch.tensor(scale, dtype=torch.float64))

        if initial_knots_raw is None:
            knots_u = np.linspace(-1.0, 1.0, HINGE_COUNT + 2)[1:-1]
        else:
            raw = np.asarray(initial_knots_raw, dtype=np.float64)
            if raw.shape != (HINGE_COUNT,):
                raise ValueError(
                    f"initial_knots_raw must have shape ({HINGE_COUNT},)"
                )
            knots_u = np.clip(
                (raw - center) / scale, -1.0 + 1.0e-6, 1.0 - 1.0e-6
            )
            knots_u.sort()
        points = np.concatenate(([-1.0], knots_u, [1.0]))
        gaps = np.maximum(np.diff(points), 1.0e-6)
        gaps /= gaps.sum()
        self.raw_gap = nn.Parameter(
            torch.tensor(np.log(np.expm1(gaps)), dtype=torch.float64)
        )
        self.base = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.base_slope = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.delta_slope = nn.Parameter(
            torch.zeros(HINGE_COUNT, dtype=torch.float64)
        )

    def knots_u(self) -> torch.Tensor:
        positive_gap = F.softplus(self.raw_gap) + 1.0e-8
        cumulative = torch.cumsum(positive_gap, dim=0)
        fraction = cumulative[:-1] / cumulative[-1]
        return -1.0 + 2.0 * fraction

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        u = (x_raw - self.x_center) / self.x_scale
        hinges = torch.relu(u.unsqueeze(-1) - self.knots_u())
        return (
            self.base
            + self.base_slope * u
            + torch.sum(hinges * self.delta_slope, dim=-1)
        )

    @torch.no_grad()
    def initialize_least_squares(
        self,
        x_raw: np.ndarray,
        y_code: np.ndarray,
        weight: np.ndarray,
        max_points: int = 50_000,
        seed: int = SEED,
    ) -> None:
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        y = np.asarray(y_code, dtype=np.float64).reshape(-1)
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if not (x.shape == y.shape == w.shape):
            raise ValueError("least-squares x/y/weight shape mismatch")
        if x.size > max_points:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(
                x.size, size=max_points, replace=False, p=w / w.sum()
            )
            x, y, w = x[chosen], y[chosen], w[chosen]
        center = float(self.x_center.detach().cpu())
        scale = float(self.x_scale.detach().cpu())
        knots_u = self.knots_u().detach().cpu().numpy().astype(np.float64)
        u = (x - center) / scale
        design = np.concatenate(
            (
                np.ones((x.size, 1), dtype=np.float64),
                u[:, None],
                np.maximum(u[:, None] - knots_u[None, :], 0.0),
            ),
            axis=1,
        )
        root_weight = np.sqrt(w / max(float(w.mean()), 1.0e-12))
        coefficient, *_ = np.linalg.lstsq(
            design * root_weight[:, None], y * root_weight, rcond=None
        )
        dtype, device = self.base.dtype, self.base.device
        self.base.copy_(torch.tensor(coefficient[0], dtype=dtype, device=device))
        self.base_slope.copy_(
            torch.tensor(coefficient[1], dtype=dtype, device=device)
        )
        self.delta_slope.copy_(
            torch.tensor(coefficient[2:], dtype=dtype, device=device)
        )

    @torch.no_grad()
    def raw_lines_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        knots_u = self.knots_u().detach().cpu().numpy().astype(np.float64)
        delta = self.delta_slope.detach().cpu().numpy().astype(np.float64)
        base = float(self.base.detach().cpu())
        initial_slope = float(self.base_slope.detach().cpu())
        center = float(self.x_center.detach().cpu())
        scale = float(self.x_scale.detach().cpu())
        cumulative_delta = np.concatenate(([0.0], np.cumsum(delta)))
        slope_u = initial_slope + cumulative_delta
        cumulative_offset = np.concatenate(
            ([0.0], np.cumsum(delta * knots_u))
        )
        intercept_u = base - cumulative_offset
        slope_raw = slope_u / scale
        intercept_raw = intercept_u - slope_raw * center
        knots_raw = center + scale * knots_u
        return knots_raw, slope_raw, intercept_raw


def initial_knots(x: np.ndarray) -> np.ndarray:
    raw = ensure_int32("knot initialization x", x)
    if int(raw.max()) - int(raw.min()) < HINGE_COUNT:
        raise ValueError("calibration GELU domain is too narrow for 16 segments")
    probabilities = np.linspace(0.0, 1.0, HINGE_COUNT + 2)[1:-1]
    return np.quantile(raw.astype(np.float64), probabilities)


def ste_round_clamp(values: torch.Tensor) -> torch.Tensor:
    hard = torch.clamp(torch.round(values), GELU_QMIN, GELU_QMAX)
    return values + (hard - values).detach()


def predict_numpy(
    model: OrderedHingePWL,
    x: np.ndarray,
    x_min: int,
    x_max: int,
    batch_size: int = 65_536,
) -> np.ndarray:
    source = ensure_int32("prediction x", x)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    output: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, source.size, batch_size):
            part = np.clip(source[begin : begin + batch_size], x_min, x_max)
            tensor = torch.as_tensor(part, dtype=dtype, device=device)
            code = torch.clamp(
                torch.round(model(tensor)), GELU_QMIN, GELU_QMAX
            )
            output.append(code.to(torch.int64).cpu().numpy())
    return np.concatenate(output)


def segment_usage(
    model: OrderedHingePWL, x: np.ndarray, weight: np.ndarray
) -> List[float]:
    knots, _, _ = model.raw_lines_numpy()
    segment = np.searchsorted(knots, np.asarray(x, dtype=np.float64), side="right")
    return np.bincount(
        segment,
        weights=np.asarray(weight, dtype=np.float64),
        minlength=SEGMENT_COUNT,
    ).astype(float).tolist()


@dataclass
class TrainedGELUPage:
    name: str
    model: OrderedHingePWL
    x_min: int
    x_max: int
    accumulator_scale: float
    output_scale: float
    fit_config: FitConfig
    loss_history: List[Dict[str, float]]
    calibration_metrics: Dict[str, Any]
    search_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

    def export(self, directory: Path) -> Dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        knots, slopes, intercepts = self.model.raw_lines_numpy()
        payload = {
            "schema": "gaugepack.gelu.float_pwl_training.v1",
            "name": self.name,
            "task": "gelu",
            "segment_count": SEGMENT_COUNT,
            "hinge_count": HINGE_COUNT,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "knots_float": knots,
            "slope_float": slopes,
            "intercept_float": intercepts,
            "qmin": GELU_QMIN,
            "qmax": GELU_QMAX,
            "accumulator_scale": self.accumulator_scale,
            "output_scale": self.output_scale,
            "input_policy": "clamp_to_calibration_domain",
            "compiler_status": "NOT_COMPILED_M18_C48",
            "deployment_equations": {
                "input": "a=FFN1_integer_accumulator_plus_bias",
                "teacher": "g=GELU_exact(a*s_acc)",
                "output": "Y=NARROW_S8(RNE_even(g/s_out))",
            },
            "fit_config": self.fit_config,
            "loss_history": self.loss_history,
            "calibration_metrics": self.calibration_metrics,
            "search_metrics": self.search_metrics,
            "metadata": self.metadata,
        }
        json_path = directory / f"{self.name}.float_pwl.json"
        csv_path = directory / f"{self.name}.segments.csv"
        checkpoint_path = directory / f"{self.name}.trainer_state.pt"
        save_json(json_path, payload)
        left_edges = np.concatenate(([float(self.x_min)], knots))
        right_edges = np.concatenate((knots, [float(self.x_max)]))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "segment",
                    "x_left_float",
                    "x_right_float",
                    "slope_float",
                    "intercept_float",
                ]
            )
            for index in range(SEGMENT_COUNT):
                writer.writerow(
                    [
                        index,
                        f"{left_edges[index]:.17g}",
                        f"{right_edges[index]:.17g}",
                        f"{slopes[index]:.17g}",
                        f"{intercepts[index]:.17g}",
                    ]
                )
        torch.save(
            {
                "schema": "gaugepack.gelu.trainer_checkpoint.v1",
                "name": self.name,
                "model_state_dict": self.model.state_dict(),
                "x_min": self.x_min,
                "x_max": self.x_max,
                "accumulator_scale": self.accumulator_scale,
                "output_scale": self.output_scale,
                "fit_config": dataclasses.asdict(self.fit_config),
                "frozen_contract": {
                    "active_input": "signed_A27",
                    "output_code_range": [GELU_QMIN, GELU_QMAX],
                    "minus_128_forbidden": True,
                },
            },
            checkpoint_path,
        )
        return {
            "float_pwl_json": str(json_path),
            "segment_csv": str(csv_path),
            "trainer_checkpoint": str(checkpoint_path),
        }


def evaluate_histogram(
    *,
    model: OrderedHingePWL,
    histogram: Histogram,
    accumulator_scale: float,
    output_scale: float,
    x_min: int,
    x_max: int,
) -> Dict[str, Any]:
    reference = gelu_target_code(
        histogram.x, accumulator_scale, output_scale
    )
    predicted = predict_numpy(model, histogram.x, x_min, x_max)
    metrics = weighted_metrics(predicted, reference, histogram.weight)
    ood = (histogram.x < x_min) | (histogram.x > x_max)
    total_weight = float(histogram.weight.sum())
    metrics.update(
        {
            "source": histogram.source,
            "x_observed_min": int(histogram.x.min()),
            "x_observed_max": int(histogram.x.max()),
            "out_of_domain_weight": float(histogram.weight[ood].sum()),
            "out_of_domain_rate": float(
                histogram.weight[ood].sum() / total_weight
            ),
            "segment_usage_weight": segment_usage(
                model,
                np.clip(histogram.x, x_min, x_max),
                histogram.weight,
            ),
        }
    )
    return metrics


def train_gelu_page(
    *,
    name: str,
    calibration: Histogram,
    search: Histogram,
    accumulator_scale: float,
    output_scale: float,
    config: FitConfig,
) -> TrainedGELUPage:
    config.validate()
    calibration.validate("calibration")
    search.validate("search")
    seed_everything(config.seed)
    x, weight = calibration.x, calibration.weight
    target = gelu_target_code(x, accumulator_scale, output_scale)
    x_min, x_max = int(x.min()), int(x.max())
    if x_max - x_min < HINGE_COUNT:
        raise ValueError(
            f"{name}: calibration domain [{x_min}, {x_max}] is too narrow"
        )
    if config.train_dtype == "float32" and max(abs(x_min), abs(x_max)) > (1 << 24):
        print(
            "[warning] GELU input exceeds exact float32 integer resolution; "
            "--train-dtype float64 is strongly recommended"
        )
    device = resolve_device(config.device)
    dtype = torch.float64 if config.train_dtype == "float64" else torch.float32
    model = OrderedHingePWL(
        x_min=x_min,
        x_max=x_max,
        initial_knots_raw=initial_knots(x),
    ).to(device=device, dtype=dtype)
    model.initialize_least_squares(x, target, weight, seed=config.seed)

    x_tensor = torch.as_tensor(x, dtype=dtype, device=device)
    y_tensor = torch.as_tensor(target, dtype=dtype, device=device)
    w_tensor = torch.as_tensor(
        weight / max(float(weight.mean()), 1.0e-12),
        dtype=dtype,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    milestones = sorted(
        {
            max(1, int(config.epochs * 0.60)),
            max(1, int(config.epochs * 0.85)),
        }
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(config.epochs):
        permutation = torch.randperm(x.size, generator=generator)
        total_loss, total_weight = 0.0, 0.0
        for begin in range(0, x.size, config.batch_size):
            chosen = permutation[begin : begin + config.batch_size].to(device)
            xb, yb, wb = x_tensor[chosen], y_tensor[chosen], w_tensor[chosen]
            optimizer.zero_grad(set_to_none=True)
            raw_prediction = model(xb)
            hard_prediction = ste_round_clamp(raw_prediction)
            weight_sum = torch.sum(wb)
            hard_l1 = torch.sum(wb * torch.abs(hard_prediction - yb)) / weight_sum
            float_l1 = torch.sum(wb * torch.abs(raw_prediction - yb)) / weight_sum
            loss = hard_l1 + config.float_loss_weight * float_l1
            loss.backward()
            optimizer.step()
            batch_weight = float(weight_sum.detach().cpu())
            total_loss += float(loss.detach().cpu()) * batch_weight
            total_weight += batch_weight
        scheduler.step()
        epoch_loss = total_loss / max(total_weight, 1.0)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": epoch_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (
            epoch == 0
            or epoch + 1 == config.epochs
            or (epoch + 1) % config.log_every == 0
        ):
            print(
                f"[{name}] epoch {epoch + 1:4d}/{config.epochs} "
                f"loss={epoch_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
    model.eval()
    calibration_metrics = evaluate_histogram(
        model=model,
        histogram=calibration,
        accumulator_scale=accumulator_scale,
        output_scale=output_scale,
        x_min=x_min,
        x_max=x_max,
    )
    search_metrics = evaluate_histogram(
        model=model,
        histogram=search,
        accumulator_scale=accumulator_scale,
        output_scale=output_scale,
        x_min=x_min,
        x_max=x_max,
    )
    search_metrics["elapsed_training_seconds"] = time.perf_counter() - started
    metadata = {
        "direct_raw_int32": True,
        "input_kind": "FFN1_accumulator_plus_integer_bias",
        "active_input_contract": "signed A27 range certificate required",
        "accumulator_scale": accumulator_scale,
        "output_scale": output_scale,
        "output_quant": "NARROW_S8 [-127,127]",
        "minus_128_forbidden": True,
        "output_scale_resolution": "resolved before fit from calibration or command line",
        "search_role": "held_out_metrics_only",
        "validation_used": False,
        "compiler_required_next": (
            "production A27/M18/C48/P48 compile and bit-true regression"
        ),
    }
    return TrainedGELUPage(
        name=name,
        model=model,
        x_min=x_min,
        x_max=x_max,
        accumulator_scale=accumulator_scale,
        output_scale=output_scale,
        fit_config=config,
        loss_history=history,
        calibration_metrics=calibration_metrics,
        search_metrics=search_metrics,
        metadata=metadata,
    )


def prepare_output_directory(path: Path, overwrite_output: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite_output:
        raise FileExistsError(
            f"output directory is not empty: {path}. "
            "Pass --overwrite-output to replace this script's named outputs."
        )
    path.mkdir(parents=True, exist_ok=True)


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    if args.calibration_npz is None or args.search_npz is None:
        raise ValueError("--calibration-npz and --search-npz are both required")
    if args.accumulator_scale is None:
        raise ValueError("--accumulator-scale is required")
    if args.site_name.strip() == "":
        raise ValueError("--site-name must not be empty")
    output_dir = args.out_dir.resolve()
    prepare_output_directory(output_dir, args.overwrite_output)
    calibration = load_histogram_npz(args.calibration_npz, "calibration")
    search = load_histogram_npz(args.search_npz, "search")
    output_scale = choose_output_scale(
        calibration.x, args.accumulator_scale, args.output_scale
    )
    config = FitConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        float_loss_weight=args.float_loss_weight,
        seed=args.seed,
        device=args.device,
        train_dtype=args.train_dtype,
        log_every=args.log_every,
    )
    page = train_gelu_page(
        name=args.site_name,
        calibration=calibration,
        search=search,
        accumulator_scale=args.accumulator_scale,
        output_scale=output_scale,
        config=config,
    )
    exported = page.export(output_dir)
    manifest = {
        "schema": "gaugepack.gelu.training_run.v1",
        "frozen_contract": {
            "input": "FFN1 raw integer accumulator plus integer bias",
            "active_input": "signed A27",
            "teacher": "GELU_exact(a*s_acc)",
            "output_code": "NARROW_S8 [-127,127]",
        },
        "site_name": args.site_name,
        "calibration_source": calibration.source,
        "search_source": search.source,
        "accumulator_scale": args.accumulator_scale,
        "output_scale": output_scale,
        "output_scale_selected_from": (
            "command_line" if args.output_scale is not None else "calibration_only"
        ),
        "calibration_metrics": page.calibration_metrics,
        "search_metrics": page.search_metrics,
        "exported_files": exported,
        "compiler_status": "NOT_COMPILED_M18_C48",
        "arguments": vars(args),
    }
    manifest_path = output_dir / "training_manifest.json"
    save_json(manifest_path, manifest)
    exported["training_manifest"] = str(manifest_path)
    metrics = page.search_metrics
    print(
        f"[{args.site_name}] search MAE={metrics['mae_code']:.6f}, "
        f"exact={100.0 * metrics['exact_code_rate']:.2f}%, "
        f"within1={100.0 * metrics['within_1_code_rate']:.2f}%, "
        f"maxabs={metrics['max_abs_code']}"
    )
    print(f"GELU output scale s_out={output_scale:.17g}")
    print(f"Float PWL artifacts: {output_dir}")
    print("Compiler status: NOT_COMPILED_M18_C48")
    return manifest


def run_self_test() -> None:
    cal_x = np.arange(-800, 1201, dtype=np.int64)
    search_x = np.arange(-780, 1181, 2, dtype=np.int64)
    cal_w = 1.0 + np.exp(-np.abs(cal_x.astype(np.float64)) / 250.0) * 5.0
    search_w = 1.0 + np.exp(-np.abs(search_x.astype(np.float64)) / 300.0) * 3.0
    calibration = coalesce_histogram(cal_x, cal_w, "selftest_calibration")
    search = coalesce_histogram(search_x, search_w, "selftest_search")
    accumulator_scale = 0.01
    output_scale = choose_output_scale(
        calibration.x, accumulator_scale, None
    )
    if int(gelu_target_code(np.asarray([0]), accumulator_scale, output_scale)[0]) != 0:
        raise AssertionError("GELU(0) quantized code must be zero")
    page = train_gelu_page(
        name="selftest_gelu",
        calibration=calibration,
        search=search,
        accumulator_scale=accumulator_scale,
        output_scale=output_scale,
        config=FitConfig(
            epochs=3,
            batch_size=512,
            learning_rate=1.0e-3,
            float_loss_weight=0.05,
            seed=SEED,
            device="cpu",
            train_dtype="float64",
            log_every=1,
        ),
    )
    knots, slopes, intercepts = page.model.raw_lines_numpy()
    if knots.size != HINGE_COUNT or np.any(np.diff(knots) <= 0.0):
        raise AssertionError("ordered 15-knot contract failed")
    if slopes.size != SEGMENT_COUNT or intercepts.size != SEGMENT_COUNT:
        raise AssertionError("16-segment line export failed")
    probe = predict_numpy(page.model, search.x, page.x_min, page.x_max)
    if np.any((probe < GELU_QMIN) | (probe > GELU_QMAX)):
        raise AssertionError("GELU output escaped NARROW_S8")
    if np.any(probe == -128):
        raise AssertionError("illegal -128 GELU code appeared")
    with tempfile.TemporaryDirectory(prefix="gaugepack_gelu_selftest_") as tmp:
        exported = page.export(Path(tmp))
        payload = json.loads(
            Path(exported["float_pwl_json"]).read_text(encoding="utf-8")
        )
        if payload.get("compiler_status") != "NOT_COMPILED_M18_C48":
            raise AssertionError("compiler status was not preserved")
    print(
        "Self-test passed: raw accumulator teacher, exact GELU, "
        "15 ordered hinges, 16 float lines, and NARROW_S8 output."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one GaugePack FFN1-accumulator -> terminal S8 GELU "
            "16-segment NN-LUT reference page."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calibration-npz", type=Path)
    parser.add_argument("--search-npz", type=Path)
    parser.add_argument("--site-name", default="gelu_site")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("gaugepack_gelu_nnlut_artifact")
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--accumulator-scale",
        "--input-scale",
        dest="accumulator_scale",
        type=float,
    )
    parser.add_argument("--output-scale", type=float)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--float-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train-dtype", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        else:
            run_training(args)
        return 0
    except (
        ValueError,
        KeyError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
    ) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
