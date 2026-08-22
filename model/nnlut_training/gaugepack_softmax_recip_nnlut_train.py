#!/usr/bin/env python3
"""Standalone GaugePack Softmax reciprocal NN-LUT fitter/compiler.

This file reproduces the FINAL GaugePack v1 Softmax reciprocal page:

    QEXP code E       : U7, 0..127
    non-zero row sum L: U13, 127..8128 (= 64 * 127)
    numerator N       : signed S21 bound, |N| <= 127 * L
    target reciprocal : R ~= 2^23 / L

The current reciprocal page is a 15-ReLU / 16-segment piecewise-linear
NN-LUT, but it is NOT gradient-trained.  Each segment is fitted analytically
with a relative-minimax line and then quantized/search-refined under the
actual VFU DSP contract:

    seg = searchsorted(boundaries, L, side="right")
    P   = L * M18[seg] + C48[seg]
    R   = RNE_even(P / 2^G), G = 8

The final context operation is:

    context = NARROW_S8(RNE_even(N * R / 2^23))

Padding queries use L=0 and bypass the reciprocal page with context=0.

Only NumPy is required.  The default run fits the page, checks it against the
frozen coefficient snapshot, exhaustively certifies every legal non-zero L,
runs random signed (N,L) comparison vectors, and exports JSON/CSV/MEM files.

Examples
--------

    python gaugepack_softmax_recip_nnlut_train.py --self-test

    python gaugepack_softmax_recip_nnlut_train.py \
        --out-dir ./gaugepack_recip_artifact

Optional workload-pair audit (NPZ keys: numerator, denominator):

    python gaugepack_softmax_recip_nnlut_train.py \
        --pairs-npz ./softmax_pairs.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Frozen GaugePack v1 arithmetic contract
# ---------------------------------------------------------------------------

SEGMENT_COUNT = 16
BOUNDARY_COUNT = SEGMENT_COUNT - 1

QEXP_MAX = 127
SEQUENCE_LENGTH = 64
L_MIN_NONZERO = QEXP_MAX
L_MAX = SEQUENCE_LENGTH * QEXP_MAX
L_BITS = 13
N_BITS = 21

RECIP_FRACTION_BITS = 23
RECIP_TARGET_SCALE = 1 << RECIP_FRACTION_BITS

A27_MIN = -(1 << 26)
A27_MAX = (1 << 26) - 1
M18_MIN = -(1 << 17)
M18_MAX = (1 << 17) - 1
C48_MIN = -(1 << 47)
C48_MAX = (1 << 47) - 1
P48_MIN = C48_MIN
P48_MAX = C48_MAX

NARROW_S8_MIN = -127
NARROW_S8_MAX = 127

FROZEN_G = 8
FROZEN_BOUNDARIES = np.asarray(
    [
        165,
        214,
        277,
        360,
        466,
        605,
        784,
        1016,
        1318,
        1709,
        2216,
        2874,
        3727,
        4833,
        6268,
    ],
    dtype=np.int64,
)
FROZEN_M18 = np.asarray(
    [
        -102266,
        -60607,
        -36064,
        -21412,
        -12724,
        -7567,
        -4495,
        -2676,
        -1591,
        -945,
        -562,
        -335,
        -199,
        -118,
        -70,
        -42,
    ],
    dtype=np.int64,
)
FROZEN_C48 = np.asarray(
    [
        29759309,
        22909425,
        17671643,
        13618709,
        10497188,
        8095996,
        6239535,
        4814319,
        3712313,
        2861019,
        2206350,
        1703462,
        1312912,
        1010997,
        778682,
        603164,
    ],
    dtype=np.int64,
)


def as_i64(values: np.ndarray | Sequence[int] | int) -> np.ndarray:
    """Convert scalar/sequence/array input to a NumPy signed-int64 array."""

    return np.asarray(values, dtype=np.int64)


def round_shift_rne(
    values: np.ndarray | Sequence[int] | int, shift: int
) -> np.ndarray:
    """Signed arithmetic division by 2**shift, nearest with ties to even.

    The floor-quotient formulation is deliberate: it also handles negative
    half-way values exactly like the intended RTL RNE-even block.
    """

    if shift < 0 or shift > 62:
        raise ValueError(f"shift must be in [0, 62], got {shift}")
    source = as_i64(values)
    if shift == 0:
        return source.copy()

    divisor = np.int64(1 << shift)
    half = np.int64(1 << (shift - 1))
    quotient = np.floor_divide(source, divisor)
    remainder = source - quotient * divisor
    increment = (remainder > half) | (
        (remainder == half) & ((quotient & np.int64(1)) != 0)
    )
    return quotient + increment.astype(np.int64)


def round_unsigned_ratio_rne(
    numerator: np.ndarray | Sequence[int] | int,
    denominator: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Exact RNE-even for non-negative integer numerator/positive denominator."""

    n, d = np.broadcast_arrays(as_i64(numerator), as_i64(denominator))
    if np.any(n < 0) or np.any(d <= 0):
        raise ValueError("unsigned ratio requires numerator >= 0 and denominator > 0")

    quotient = n // d
    remainder = n - quotient * d
    twice_remainder = remainder << 1
    increment = (twice_remainder > d) | (
        (twice_remainder == d) & ((quotient & np.int64(1)) != 0)
    )
    return quotient + increment.astype(np.int64)


def narrow_s8(values: np.ndarray | Sequence[int] | int) -> np.ndarray:
    """GaugePack signed activation clamp; -128 is intentionally forbidden."""

    return np.clip(as_i64(values), NARROW_S8_MIN, NARROW_S8_MAX)


def exact_context_rne(
    numerator: np.ndarray | Sequence[int] | int,
    denominator: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Exact signed RNE-even N/L oracle with the frozen L=0 -> 0 policy."""

    n, l = np.broadcast_arrays(as_i64(numerator), as_i64(denominator))
    if np.any(l < 0):
        raise ValueError("Softmax denominator must be non-negative")

    result = np.zeros_like(n)
    active = l != 0
    if np.any(active):
        magnitude = round_unsigned_ratio_rne(np.abs(n[active]), l[active])
        result[active] = np.where(n[active] < 0, -magnitude, magnitude)
    return narrow_s8(result)


def geometric_boundaries(x_min: int, x_max: int) -> np.ndarray:
    """Split a positive domain into 16 approximately equal log-width regions."""

    if x_min <= 0 or x_max <= x_min:
        raise ValueError("reciprocal domain must be positive and non-empty")

    ratio = (float(x_max) / float(x_min)) ** (1.0 / SEGMENT_COUNT)
    boundaries: List[int] = []
    previous = x_min
    for index in range(1, SEGMENT_COUNT):
        boundary = int(math.ceil(float(x_min) * ratio**index))
        boundary = max(boundary, previous + 1)
        boundary = min(boundary, x_max - (SEGMENT_COUNT - 1 - index))
        boundaries.append(boundary)
        previous = boundary

    result = np.asarray(boundaries, dtype=np.int64)
    if result.size != BOUNDARY_COUNT or np.any(np.diff(result) <= 0):
        raise AssertionError("failed to construct 16 ordered reciprocal segments")
    return result


def segment_limits(
    boundaries: np.ndarray, x_min: int, x_max: int
) -> List[Tuple[int, int]]:
    """Convert boundary starts into 16 inclusive [lo, hi] intervals."""

    result: List[Tuple[int, int]] = []
    left = int(x_min)
    for boundary in as_i64(boundaries):
        result.append((left, int(boundary) - 1))
        left = int(boundary)
    result.append((left, int(x_max)))
    return result


def relative_minimax_line(
    x_lo: int, x_hi: int, target_scale: int
) -> Tuple[float, float]:
    """Return slope/intercept for a relative-minimax fit of K/x on [a,b]."""

    a = float(x_lo)
    b = float(x_hi)
    k = float(target_scale)
    denominator = a * a + 6.0 * a * b + b * b
    slope = -8.0 * k / denominator
    intercept = -slope * (a + b)
    return slope, intercept


@dataclass(frozen=True)
class ReciprocalPage:
    """One compiled 15-boundary/16-segment NN-LUT coefficient page."""

    name: str
    x_min: int
    x_max: int
    target_scale: int
    boundaries: np.ndarray
    multiplier_m18: np.ndarray
    bias_c48: np.ndarray
    coefficient_shift: int

    def validate(self) -> None:
        boundaries = as_i64(self.boundaries)
        multiplier = as_i64(self.multiplier_m18)
        bias = as_i64(self.bias_c48)

        if boundaries.size != BOUNDARY_COUNT:
            raise ValueError("page requires exactly 15 boundaries")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("page boundaries must be strictly increasing")
        if int(boundaries[0]) <= self.x_min or int(boundaries[-1]) > self.x_max:
            raise ValueError("page boundary escaped its input domain")
        if multiplier.size != SEGMENT_COUNT or bias.size != SEGMENT_COUNT:
            raise ValueError("page requires exactly 16 M/C pairs")
        if self.x_min < A27_MIN or self.x_max > A27_MAX:
            raise ValueError("page input does not fit signed A27")
        if np.any((multiplier < M18_MIN) | (multiplier > M18_MAX)):
            raise ValueError("page multiplier does not fit signed M18")
        if np.any((bias < C48_MIN) | (bias > C48_MAX)):
            raise ValueError("page bias does not fit signed C48")
        if self.coefficient_shift < 0 or self.coefficient_shift > 47:
            raise ValueError("coefficient shift is outside the P48 post range")

        for index, (lo, hi) in enumerate(
            segment_limits(boundaries, self.x_min, self.x_max)
        ):
            endpoints = (
                int(multiplier[index]) * lo + int(bias[index]),
                int(multiplier[index]) * hi + int(bias[index]),
            )
            if min(endpoints) < P48_MIN or max(endpoints) > P48_MAX:
                raise ValueError(f"segment {index} overflows signed P48")

    def segment_index(
        self, denominator: np.ndarray | Sequence[int] | int
    ) -> np.ndarray:
        """RTL-equivalent boundary lookup; boundary value starts next segment."""

        l = as_i64(denominator)
        if np.any((l < self.x_min) | (l > self.x_max)):
            raise ValueError("non-zero denominator is outside the certified page domain")
        return np.searchsorted(as_i64(self.boundaries), l, side="right")

    def infer_reciprocal(
        self, denominator: np.ndarray | Sequence[int] | int
    ) -> np.ndarray:
        """Bit-exact L -> R path, excluding the external L=0 bypass."""

        self.validate()
        l = as_i64(denominator)
        segment = self.segment_index(l)
        p_value = (
            l * as_i64(self.multiplier_m18)[segment]
            + as_i64(self.bias_c48)[segment]
        )
        reciprocal = round_shift_rne(p_value, self.coefficient_shift)
        if np.any((reciprocal < 0) | (reciprocal > M18_MAX)):
            raise ValueError("reciprocal code is outside non-negative signed M18")
        return reciprocal.astype(np.int64)

    def infer_context(
        self,
        numerator: np.ndarray | Sequence[int] | int,
        denominator: np.ndarray | Sequence[int] | int,
    ) -> np.ndarray:
        """Bit-exact frozen N,L -> context path, including L=0 bypass."""

        n, l = np.broadcast_arrays(as_i64(numerator), as_i64(denominator))
        if np.any(l < 0):
            raise ValueError("Softmax denominator must be non-negative")

        output = np.zeros_like(n)
        active = l != 0
        if np.any(active):
            reciprocal = self.infer_reciprocal(l[active])
            product = n[active] * reciprocal
            output[active] = round_shift_rne(product, RECIP_FRACTION_BITS)
        return narrow_s8(output)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready page description, including explicit DSP equations."""

        self.validate()
        limits = segment_limits(self.boundaries, self.x_min, self.x_max)
        segments = []
        for index, (lo, hi) in enumerate(limits):
            segments.append(
                {
                    "index": index,
                    "l_lo_inclusive": lo,
                    "l_hi_inclusive": hi,
                    "multiplier_m18": int(self.multiplier_m18[index]),
                    "bias_c48": int(self.bias_c48[index]),
                }
            )
        return {
            "name": self.name,
            "kind": "raw_geometric_relative_minimax_pwl",
            "input_kind": "raw_rowsum_l",
            "x_min": self.x_min,
            "x_max": self.x_max,
            "target_scale": self.target_scale,
            "boundaries": self.boundaries.tolist(),
            "multiplier_m18": self.multiplier_m18.tolist(),
            "bias_c48": self.bias_c48.tolist(),
            "coefficient_shift_g": self.coefficient_shift,
            "reciprocal_equation": "R=RNE_even((L*M[seg]+C[seg])/2^G)",
            "context_equation": "Y=NARROW_S8(RNE_even(N*R/2^23))",
            "segments": segments,
        }


def quantized_line_candidate(
    *,
    slope: float,
    intercept: float,
    coefficient_shift: int,
    x_lo: int,
    x_hi: int,
    target_scale: int,
) -> Tuple[int, int]:
    """Search a small integer M/C neighborhood under M18/C48/P48 limits."""

    scale = 1 << coefficient_shift
    multiplier_center = int(np.rint(slope * scale))
    x_values = np.arange(x_lo, x_hi + 1, dtype=np.int64)
    midpoint = (x_lo + x_hi) // 2

    best: Optional[Tuple[float, float, int, int]] = None
    for multiplier in range(multiplier_center - 2, multiplier_center + 3):
        if multiplier < M18_MIN or multiplier > M18_MAX:
            continue

        scaled_line_midpoint = (
            slope * float(midpoint) + intercept
        ) * float(scale)
        bias_center = int(
            np.rint(scaled_line_midpoint - float(multiplier * midpoint))
        )
        for bias in range(bias_center - 8, bias_center + 9):
            if bias < C48_MIN or bias > C48_MAX:
                continue

            p_value = multiplier * x_values + bias
            if int(p_value.min()) < P48_MIN or int(p_value.max()) > P48_MAX:
                continue

            reciprocal = round_shift_rne(p_value, coefficient_shift)
            signed_relative_error = (
                reciprocal.astype(np.float64)
                * x_values.astype(np.float64)
                / float(target_scale)
                - 1.0
            )
            score = (
                float(np.max(np.abs(signed_relative_error))),
                float(np.mean(np.abs(signed_relative_error))),
                multiplier,
                bias,
            )
            if best is None or score < best:
                best = score

    if best is None:
        raise ValueError("no quantized M18/C48 line candidate fits")
    return int(best[2]), int(best[3])


def select_highest_feasible_shift(
    limits: Sequence[Tuple[int, int]],
    lines: Sequence[Tuple[float, float]],
) -> int:
    """Choose the largest G for which all 16 initial M/C fits are legal."""

    for coefficient_shift in range(30, -1, -1):
        scale = float(1 << coefficient_shift)
        feasible = True
        for (lo, hi), (slope, intercept) in zip(limits, lines):
            multiplier = int(np.rint(slope * scale))
            midpoint = (lo + hi) // 2
            bias = int(
                np.rint(
                    (slope * float(midpoint) + intercept) * scale
                    - float(multiplier * midpoint)
                )
            )
            endpoints = (multiplier * lo + bias, multiplier * hi + bias)
            feasible &= M18_MIN <= multiplier <= M18_MAX
            feasible &= C48_MIN <= bias <= C48_MAX
            feasible &= min(endpoints) >= P48_MIN and max(endpoints) <= P48_MAX
        if feasible:
            return coefficient_shift
    raise ValueError("reciprocal page cannot fit the A27/M18/C48 contract")


def fit_frozen_raw_v1() -> ReciprocalPage:
    """Fit/compile the final geometric raw-v1 reciprocal page from scratch."""

    boundaries = geometric_boundaries(L_MIN_NONZERO, L_MAX)
    limits = segment_limits(boundaries, L_MIN_NONZERO, L_MAX)
    lines = [
        relative_minimax_line(lo, hi, RECIP_TARGET_SCALE) for lo, hi in limits
    ]
    coefficient_shift = select_highest_feasible_shift(limits, lines)

    multiplier = np.empty(SEGMENT_COUNT, dtype=np.int64)
    bias = np.empty(SEGMENT_COUNT, dtype=np.int64)
    for index, ((lo, hi), (slope, intercept)) in enumerate(zip(limits, lines)):
        multiplier[index], bias[index] = quantized_line_candidate(
            slope=slope,
            intercept=intercept,
            coefficient_shift=coefficient_shift,
            x_lo=lo,
            x_hi=hi,
            target_scale=RECIP_TARGET_SCALE,
        )

    page = ReciprocalPage(
        name="gaugepack_sm_recip_raw_geometric_minimax_v1",
        x_min=L_MIN_NONZERO,
        x_max=L_MAX,
        target_scale=RECIP_TARGET_SCALE,
        boundaries=boundaries,
        multiplier_m18=multiplier,
        bias_c48=bias,
        coefficient_shift=coefficient_shift,
    )
    page.validate()
    return page


def assert_frozen_snapshot(page: ReciprocalPage) -> None:
    """Fail loudly if fitter behavior drifts from the architecture snapshot."""

    checks = {
        "G": page.coefficient_shift == FROZEN_G,
        "boundaries": np.array_equal(page.boundaries, FROZEN_BOUNDARIES),
        "M18": np.array_equal(page.multiplier_m18, FROZEN_M18),
        "C48": np.array_equal(page.bias_c48, FROZEN_C48),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            "fitted page does not match the frozen GaugePack v1 snapshot: "
            + ", ".join(failed)
        )


def legal_domain_certificate(page: ReciprocalPage) -> Dict[str, Any]:
    """Exhaustively certify all 8002 legal non-zero rowsum values."""

    l = np.arange(L_MIN_NONZERO, L_MAX + 1, dtype=np.int64)
    reciprocal = page.infer_reciprocal(l)
    segment = page.segment_index(l)
    p_value = (
        l * page.multiplier_m18[segment] + page.bias_c48[segment]
    )

    error_numerator = np.abs(reciprocal * l - RECIP_TARGET_SCALE)
    worst_index = int(np.argmax(error_numerator))
    worst_gain_numerator = int(error_numerator[worst_index])
    context_error_numerator = QEXP_MAX * worst_gain_numerator
    context_error_denominator = RECIP_TARGET_SCALE
    certified_code_bound = (
        0
        if context_error_numerator == 0
        else context_error_numerator // context_error_denominator + 1
    )

    max_numerator_magnitude = QEXP_MAX * L_MAX
    s21_max = (1 << (N_BITS - 1)) - 1
    if max_numerator_magnitude > s21_max:
        raise AssertionError("derived numerator bound does not fit signed S21")

    return {
        "denominator_count": int(l.size),
        "legal_nonzero_l": [L_MIN_NONZERO, L_MAX],
        "l_zero_policy": "bypass reciprocal; context=0",
        "rowsum_bits": L_BITS,
        "numerator_bits": N_BITS,
        "max_numerator_magnitude": max_numerator_magnitude,
        "s21_positive_max": s21_max,
        "runtime_reciprocal_min": int(reciprocal.min()),
        "runtime_reciprocal_max": int(reciprocal.max()),
        "runtime_reciprocal_fits_nonnegative_s18": bool(
            int(reciprocal.min()) >= 0 and int(reciprocal.max()) <= M18_MAX
        ),
        "p48_min_observed": int(p_value.min()),
        "p48_max_observed": int(p_value.max()),
        "p48_fits": bool(
            int(p_value.min()) >= P48_MIN and int(p_value.max()) <= P48_MAX
        ),
        "max_abs_relative_reciprocal_error": (
            worst_gain_numerator / float(RECIP_TARGET_SCALE)
        ),
        "max_real_context_error_bound_code": (
            context_error_numerator / float(context_error_denominator)
        ),
        "certified_max_context_code_error": int(certified_code_bound),
        "worst_denominator": int(l[worst_index]),
        "worst_reciprocal": int(reciprocal[worst_index]),
        "exact_bound_fraction": {
            "numerator": int(context_error_numerator),
            "denominator": int(context_error_denominator),
        },
        "reciprocal_coefficient_shift_g": page.coefficient_shift,
        "final_context_shift": RECIP_FRACTION_BITS,
        "no_clz": True,
        "no_normalization": True,
        "no_row_dependent_shift": True,
    }


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
    """Compute integer context-code error statistics."""

    ref, pred = np.broadcast_arrays(as_i64(reference), as_i64(prediction))
    error = pred - ref
    absolute = np.abs(error)
    count = int(error.size)
    divisor = max(count, 1)
    return {
        "count": count,
        "exact_code_rate": float(np.count_nonzero(error == 0) / divisor),
        "within_1_code_rate": float(np.count_nonzero(absolute <= 1) / divisor),
        "within_2_code_rate": float(np.count_nonzero(absolute <= 2) / divisor),
        "mae_code": float(absolute.mean()) if count else 0.0,
        "rmse_code": (
            float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
            if count
            else 0.0
        ),
        "max_abs_code": int(absolute.max(initial=0)),
        "mismatch_count": int(np.count_nonzero(error)),
    }


def validate_pairs(
    page: ReciprocalPage,
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    label: str,
) -> Dict[str, Any]:
    """Validate a legal pair set against exact signed RNE N/L."""

    n, l = np.broadcast_arrays(as_i64(numerator), as_i64(denominator))
    legal_l = (l == 0) | ((l >= L_MIN_NONZERO) & (l <= L_MAX))
    if np.any(~legal_l):
        bad = int(l[~legal_l][0])
        raise ValueError(f"{label}: illegal denominator {bad}")
    if np.any(np.abs(n) > QEXP_MAX * l):
        raise ValueError(f"{label}: numerator violates |N| <= 127*L")

    reference = exact_context_rne(n, l)
    prediction = page.infer_context(n, l)
    metrics = error_metrics(reference, prediction)
    metrics["label"] = label
    metrics["denominator_min"] = int(l.min(initial=0))
    metrics["denominator_max"] = int(l.max(initial=0))
    metrics["numerator_min"] = int(n.min(initial=0))
    metrics["numerator_max"] = int(n.max(initial=0))
    return metrics


def random_pair_audit(
    page: ReciprocalPage, count: int, seed: int
) -> Dict[str, Any]:
    """Generate deterministic random legal E7 Softmax (N,L) test pairs."""

    if count <= 0:
        raise ValueError("random pair count must be positive")
    generator = np.random.default_rng(seed)
    l = generator.integers(
        L_MIN_NONZERO, L_MAX + 1, size=count, dtype=np.int64
    )
    bound = QEXP_MAX * l
    # Uniformly choose one integer from each pair's variable legal interval.
    unit = generator.random(count)
    n = np.floor(unit * (2.0 * bound.astype(np.float64) + 1.0)).astype(np.int64)
    n -= bound

    # Add exact endpoints, every segment boundary neighbor, and L=0 bypasses.
    probe_l: List[int] = [0, L_MIN_NONZERO, L_MAX]
    for boundary in page.boundaries.tolist():
        probe_l.extend([boundary - 1, boundary, min(boundary + 1, L_MAX)])
    probe_l_array = np.asarray(probe_l, dtype=np.int64)
    probe_n_parts = [
        np.zeros_like(probe_l_array),
        QEXP_MAX * probe_l_array,
        -QEXP_MAX * probe_l_array,
        probe_l_array // 2,
        -(probe_l_array // 2),
    ]
    l = np.concatenate([l, *([probe_l_array] * len(probe_n_parts))])
    n = np.concatenate([n, *probe_n_parts])

    return validate_pairs(page, n, l, label=f"random_seed_{seed}")


def load_pair_npz(
    path: Path, numerator_key: str, denominator_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load optional real workload pairs from one NPZ file."""

    with np.load(path, allow_pickle=False) as bundle:
        if numerator_key not in bundle or denominator_key not in bundle:
            raise KeyError(
                f"{path} must contain {numerator_key!r} and {denominator_key!r}"
            )
        numerator = np.asarray(bundle[numerator_key], dtype=np.int64).reshape(-1)
        denominator = np.asarray(bundle[denominator_key], dtype=np.int64).reshape(-1)
    if numerator.size != denominator.size:
        raise ValueError("NPZ numerator and denominator arrays must have equal size")
    return numerator, denominator


def twos_complement_hex(value: int, bits: int) -> str:
    """Return one fixed-width two's-complement hexadecimal memory word."""

    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if value < minimum or value > maximum:
        raise ValueError(f"{value} does not fit signed {bits}-bit")
    encoded = value & ((1 << bits) - 1)
    digits = (bits + 3) // 4
    return f"{encoded:0{digits}X}"


def unsigned_hex(value: int, bits: int) -> str:
    """Return one fixed-width unsigned hexadecimal memory word."""

    if value < 0 or value > (1 << bits) - 1:
        raise ValueError(f"{value} does not fit unsigned {bits}-bit")
    digits = (bits + 3) // 4
    return f"{value:0{digits}X}"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical compact JSON for artifact identity/regression."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_artifact(
    page: ReciprocalPage,
    certificate: Mapping[str, Any],
    random_metrics: Mapping[str, Any],
    workload_metrics: Optional[Mapping[str, Any]],
    out_dir: Path,
    prefix: str,
) -> Dict[str, str]:
    """Export the fitted page for software reference and RTL initialization."""

    out_dir.mkdir(parents=True, exist_ok=True)
    page_dict = page.to_dict()
    artifact = {
        "schema": "gaugepack.softmax.reciprocal.raw_v1.v1",
        "status": "FINAL_FROZEN_BASELINE",
        "fitter": "geometric 16-segment relative-minimax + integer M/C search",
        "gradient_training": False,
        "qexp_contract": {
            "code": "E7/U7",
            "e_min": 0,
            "e_max": QEXP_MAX,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "integer_contract": {
            "L": "U13; 0 or 127..8128",
            "N": "S21 bound; |N|<=127L",
            "R": "nonnegative S18 container",
            "reciprocal_dsp": "A27 * M18 + C48 -> P48",
            "context_dsp": "S21 * nonnegative-S18 -> P48",
            "narrow": "[-127,127]; -128 forbidden",
        },
        "page": page_dict,
        "certificate": dict(certificate),
        "random_pair_audit": dict(random_metrics),
        "workload_pair_audit": (
            None if workload_metrics is None else dict(workload_metrics)
        ),
        "frozen_snapshot_match": True,
    }
    artifact["page_sha256"] = canonical_sha256(page_dict)

    json_path = out_dir / f"{prefix}.json"
    boundary_path = out_dir / f"{prefix}.boundary_u13.mem"
    multiplier_path = out_dir / f"{prefix}.multiplier_s18.mem"
    bias_path = out_dir / f"{prefix}.bias_s48.mem"
    csv_path = out_dir / f"{prefix}.segments.csv"

    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    boundary_path.write_text(
        "\n".join(unsigned_hex(int(value), L_BITS) for value in page.boundaries)
        + "\n",
        encoding="ascii",
    )
    multiplier_path.write_text(
        "\n".join(twos_complement_hex(int(value), 18) for value in page.multiplier_m18)
        + "\n",
        encoding="ascii",
    )
    bias_path.write_text(
        "\n".join(twos_complement_hex(int(value), 48) for value in page.bias_c48)
        + "\n",
        encoding="ascii",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "l_lo", "l_hi", "multiplier_m18", "bias_c48"])
        for segment in page_dict["segments"]:
            writer.writerow(
                [
                    segment["index"],
                    segment["l_lo_inclusive"],
                    segment["l_hi_inclusive"],
                    segment["multiplier_m18"],
                    segment["bias_c48"],
                ]
            )

    return {
        "json": str(json_path),
        "boundary_mem": str(boundary_path),
        "multiplier_mem": str(multiplier_path),
        "bias_mem": str(bias_path),
        "segment_csv": str(csv_path),
    }


def self_test() -> Dict[str, Any]:
    """Fast deterministic checks for fitting, RNE, boundary, and export logic."""

    tied = np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.int64)
    expected_shift = np.asarray([-4, -2, -2, 0, 0, 2, 2, 4], dtype=np.int64)
    if not np.array_equal(round_shift_rne(tied, 1), expected_shift):
        raise AssertionError("signed RNE-even shift self-test failed")

    numerator = np.asarray([-5, -3, -1, 1, 3, 5], dtype=np.int64)
    denominator = np.full_like(numerator, 2)
    expected_divide = np.asarray([-2, -2, 0, 0, 2, 2], dtype=np.int64)
    if not np.array_equal(
        exact_context_rne(numerator, denominator), expected_divide
    ):
        raise AssertionError("signed exact divide RNE-even self-test failed")

    page = fit_frozen_raw_v1()
    assert_frozen_snapshot(page)

    for boundary_index, boundary in enumerate(page.boundaries.tolist()):
        left_segment = int(page.segment_index(boundary - 1))
        boundary_segment = int(page.segment_index(boundary))
        if left_segment != boundary_index or boundary_segment != boundary_index + 1:
            raise AssertionError("boundary side='right' semantics changed")

    if int(page.infer_context(123, 0)) != 0:
        raise AssertionError("L=0 bypass policy failed")

    certificate = legal_domain_certificate(page)
    if certificate["certified_max_context_code_error"] != 2:
        raise AssertionError("frozen legal-domain max-code certificate changed")
    expected_bound = 1.1926809549331665
    if not math.isclose(
        certificate["max_real_context_error_bound_code"],
        expected_bound,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("frozen real context-error certificate changed")

    random_metrics = random_pair_audit(page, count=50_000, seed=20260818)
    if int(random_metrics["max_abs_code"]) > 2:
        raise AssertionError("random context error escaped certified max-2 bound")

    with tempfile.TemporaryDirectory(prefix="gaugepack_recip_selftest_") as temp:
        paths = export_artifact(
            page,
            certificate,
            random_metrics,
            None,
            Path(temp),
            "selftest_recip",
        )
        reloaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        if reloaded["page"] != page.to_dict():
            raise AssertionError("JSON export/reload changed the page")
        if len(Path(paths["boundary_mem"]).read_text().splitlines()) != 15:
            raise AssertionError("boundary MEM export line count changed")
        if len(Path(paths["multiplier_mem"]).read_text().splitlines()) != 16:
            raise AssertionError("multiplier MEM export line count changed")
        if len(Path(paths["bias_mem"]).read_text().splitlines()) != 16:
            raise AssertionError("bias MEM export line count changed")

    return {
        "status": "PASS",
        "frozen_snapshot_match": True,
        "certificate": certificate,
        "random_pair_audit": random_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit, certify, and export the frozen GaugePack E7 Softmax "
            "raw reciprocal 16-segment NN-LUT page."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("gaugepack_recip_artifact")
    )
    parser.add_argument(
        "--prefix", default="gaugepack_sm_recip_raw_geometric_minimax_v1"
    )
    parser.add_argument("--random-pairs", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--pairs-npz",
        type=Path,
        help="optional workload audit NPZ; it is never used to fit v1",
    )
    parser.add_argument("--numerator-key", default="numerator")
    parser.add_argument("--denominator-key", default="denominator")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="fit and validate but do not write JSON/CSV/MEM files",
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.self_test:
        return self_test()
    if args.random_pairs <= 0:
        raise ValueError("--random-pairs must be positive")

    page = fit_frozen_raw_v1()
    assert_frozen_snapshot(page)
    certificate = legal_domain_certificate(page)
    random_metrics = random_pair_audit(page, args.random_pairs, args.seed)
    if int(random_metrics["max_abs_code"]) > int(
        certificate["certified_max_context_code_error"]
    ):
        raise AssertionError("observed error exceeds the legal-domain certificate")

    workload_metrics: Optional[Dict[str, Any]] = None
    if args.pairs_npz is not None:
        numerator, denominator = load_pair_npz(
            args.pairs_npz, args.numerator_key, args.denominator_key
        )
        workload_metrics = validate_pairs(
            page,
            numerator,
            denominator,
            label=str(args.pairs_npz),
        )

    exported: Dict[str, str] = {}
    if not args.no_export:
        exported = export_artifact(
            page,
            certificate,
            random_metrics,
            workload_metrics,
            args.out_dir,
            args.prefix,
        )

    return {
        "status": "PASS",
        "frozen_snapshot_match": True,
        "page": page.to_dict(),
        "certificate": certificate,
        "random_pair_audit": random_metrics,
        "workload_pair_audit": workload_metrics,
        "exported": exported,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
