#!/usr/bin/env python3
"""
GaugePack FINAL-frozen arithmetic E2E inference model (2026-08-19).

Purpose
-------
Put the *model arithmetic* of the current GaugePack PMPU/VFU design in one
readable Python file.  This is NOT cycle-accurate RTL and does NOT model
PACK2 scheduling, scratch-bank timing, ready/valid, hidden-major addressing,
or DSP pipeline latency.

Frozen arithmetic represented here
----------------------------------
PMPU / matrix side
  * Q/K/V, O-projection, FFN2: mathematical GEMM+bias from checkpoint weights.
    Their final W4 compiler scales are intentionally NOT invented here.
  * FFN1->GELU uses the existing calibrated integer accumulator contract
    (S8 activation x calibrated S8 weight) because the current GELU artifact
    was trained against that exact code domain.
  * QK and E7*V (PV) are exact small integer matmuls.

VFU / nonlinear side
  * GELU: calibrated 16-segment NN-LUT/PWL page.
  * Softmax FINAL:
        score = Q @ K^T
        d = score - rowmax
        E = E7 QEXP page, E in [0,127]
        L = sum(E)
        R ~= 2^23/L using frozen raw geometric 16-segment minimax v1
        N = E @ V
        context = narrow_s8(RNE_even(N*R / 2^23))
    No U8 XOR/Vsum bridge, no exact divider, no CLZ normalization.
  * LayerNorm FINAL:
        skip remains in its native S8 scale
        main alone is requantized to that scale
        z = main8 + skip8            (S9, no saturation)
        Pmom = sum(z * (2^23 + z))   (G23 MomentPack)
        S,Q decoded from Pmom
        D = 128*Q - S^2              (epsilon code is frozen to zero)
        D27 = RNE_even(D / 2^4)
        rho = site-specific RSQRT page
        n = 128*z - S
        T = n*rho                     (raw; no intermediate requant)
        y = narrow_s8(RNE_even((T*M_gamma + C_beta)/2^F))
    There is NO D==0 special/zero-detection path in this model.  A constant row
    has n==0, therefore T==0 for any finite clamped page output and affine
    naturally returns beta.

Expected inputs on the user's current project
---------------------------------------------
  --model-path ./bert-tiny-sst2
  --base-artifact-dir ./nnlut_sst2_gaugepack_e7_qexp
  --ln-artifact-dir ./gaugepack_ln_main_to_skip_artifacts

The script can evaluate SST-2 validation or a single --text string.

Important scope note
--------------------
This file intentionally refuses to fake compiler-dependent W4 requant scales.
When the final PMPU compiler emits those coefficients, the generic GEMM helpers
can be replaced by integer W4 GEMMs without changing the frozen VFU equations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This E2E file requires PyTorch") from exc

# HuggingFace/datasets are loaded lazily so --self-test works in a minimal
# PyTorch environment.
load_dataset = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
DataCollatorWithPadding = None

def require_hf_runtime() -> None:
    global load_dataset, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    if load_dataset is not None:
        return
    try:
        from datasets import load_dataset as _load_dataset
        from transformers import (
            AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
            AutoTokenizer as _AutoTokenizer,
            DataCollatorWithPadding as _DataCollatorWithPadding,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Full E2E requires transformers and datasets; install them in the project venv"
        ) from exc
    load_dataset = _load_dataset
    AutoModelForSequenceClassification = _AutoModelForSequenceClassification
    AutoTokenizer = _AutoTokenizer
    DataCollatorWithPadding = _DataCollatorWithPadding


# =============================================================================
# 0. Frozen integer constants
# =============================================================================

NARROW_MIN = -127
NARROW_MAX = 127
HIDDEN = 128
HEADS = 2
HEAD_DIM = 64
INTERMEDIATE = 512
MOMENT_GAP = 23
MOMENT_BASE = 1 << MOMENT_GAP
D_SHIFT = 4
SOFTMAX_RECIP_FRAC = 23

# Softmax FINAL raw geometric minimax reciprocal v1.
# R ~= 2^23 / L.  PWL equation: R = RNE((L*M[seg] + C[seg]) / 2^8)
RAW_RECIP_BOUNDARIES = (
    165, 214, 277, 360, 466, 605, 784, 1016,
    1318, 1709, 2216, 2874, 3727, 4833, 6268,
)
RAW_RECIP_M = (
    -102266, -60607, -36064, -21412,
    -12724, -7567, -4495, -2676,
    -1591, -945, -562, -335,
    -199, -118, -70, -42,
)
RAW_RECIP_C = (
    29759309, 22909425, 17671643, 13618709,
    10497188, 8095996, 6239535, 4814319,
    3712313, 2861019, 2206350, 1703462,
    1312912, 1010997, 778682, 603164,
)
RAW_RECIP_COEF_SHIFT = 8


# =============================================================================
# 1. Generic integer helpers
# =============================================================================


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: JSON root must be an object")
    return obj


def round_shift_rne(values: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    """Signed arithmetic /2**shift, round-nearest ties-to-even."""
    source = values.to(torch.int64)
    if isinstance(shift, int):
        amount = torch.full_like(source, int(shift))
    else:
        amount = torch.broadcast_to(shift.to(torch.int64), source.shape)
    if bool(torch.any((amount < 0) | (amount > 62)).item()):
        raise ValueError("RNE shift must be in [0,62]")
    out = torch.empty_like(source)
    for s in torch.unique(amount).tolist():
        active = amount == int(s)
        if int(s) == 0:
            out[active] = source[active]
            continue
        div = 1 << int(s)
        half = 1 << (int(s) - 1)
        q = torch.div(source[active], div, rounding_mode="floor")
        r = source[active] - q * div
        inc = (r > half) | ((r == half) & ((q & 1) != 0))
        out[active] = q + inc.to(torch.int64)
    return out


def quantize_narrow_s8(values: torch.Tensor, scale: float) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid S8 scale {scale}")
    return torch.clamp(
        torch.round(values.to(torch.float64) / float(scale)),
        NARROW_MIN,
        NARROW_MAX,
    ).to(torch.int64)


def narrow_s8(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp(values.to(torch.int64), NARROW_MIN, NARROW_MAX)


def exact_small_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    reduction: int,
    max_term: int,
) -> torch.Tensor:
    """Integer GEMM via FP32 only when every sum is exactly representable."""
    if int(reduction) * int(max_term) >= (1 << 24):
        raise OverflowError("exact_small_matmul FP32 exactness bound exceeded")
    return torch.round(
        torch.matmul(left.to(torch.float32), right.to(torch.float32))
    ).to(torch.int64)


def semantic_gemm(x: torch.Tensor, linear: torch.nn.Module) -> torch.Tensor:
    """PMPU mathematical operation only: GEMM+bias, no PACK/W4 cycle model."""
    return F.linear(x, linear.weight, linear.bias)


# =============================================================================
# 2. Portable 16-segment NN-LUT/PWL reader
# =============================================================================


@dataclass(frozen=True)
class PWLPage:
    path: Path
    schema: str
    name: str
    x_min: int
    x_max: int
    boundaries: torch.Tensor
    multiplier: torch.Tensor
    bias: torch.Tensor
    right_shift: torch.Tensor
    origin: torch.Tensor
    qmin: int
    qmax: int
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "PWLPage":
        payload = read_json(path)
        schema = str(payload.get("schema", ""))
        if schema not in {"gaugepack.nnlut.pwl.v1", "gaugepack.nnlut.pwl.v2"}:
            raise ValueError(f"{path}: unsupported PWL schema {schema!r}")
        b = np.asarray(payload["boundaries"], dtype=np.int64)
        m = np.asarray(payload["multiplier"], dtype=np.int64)
        c = np.asarray(payload["bias"], dtype=np.int64)
        s = np.asarray(payload["right_shift"], dtype=np.int64)
        segments = b.size + 1
        if segments != 16 or b.size != 15 or np.any(np.diff(b) <= 0):
            raise ValueError(f"{path}: expected 16 ordered segments")
        if schema.endswith("v2"):
            if str(payload.get("coordinate_mode", "")) != "segment_local":
                raise ValueError(f"{path}: v2 page must be segment_local")
            o = np.asarray(payload["origin"], dtype=np.int64)
        else:
            o = np.zeros(segments, dtype=np.int64)
        for name, arr in (("M", m), ("C", c), ("shift", s), ("origin", o)):
            if arr.shape != (16,):
                raise ValueError(f"{path}: {name} must have 16 entries")
        meta = payload.get("metadata", {})
        if not isinstance(meta, Mapping):
            raise TypeError(f"{path}: metadata must be object")
        return cls(
            path=path.resolve(),
            schema=schema,
            name=str(payload.get("name", path.stem)),
            x_min=int(payload["x_min"]),
            x_max=int(payload["x_max"]),
            boundaries=torch.as_tensor(b, dtype=torch.int64),
            multiplier=torch.as_tensor(m, dtype=torch.int64),
            bias=torch.as_tensor(c, dtype=torch.int64),
            right_shift=torch.as_tensor(s, dtype=torch.int64),
            origin=torch.as_tensor(o, dtype=torch.int64),
            qmin=int(payload["qmin"]),
            qmax=int(payload["qmax"]),
            metadata=dict(meta),
        )

    def to(self, device: torch.device | str) -> "PWLPage":
        return dataclasses.replace(
            self,
            boundaries=self.boundaries.to(device),
            multiplier=self.multiplier.to(device),
            bias=self.bias.to(device),
            right_shift=self.right_shift.to(device),
            origin=self.origin.to(device),
        )

    def infer(self, values: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(values.to(torch.int64), self.x_min, self.x_max)
        seg = torch.bucketize(x, self.boundaries, right=True)
        p = self.multiplier[seg] * (x - self.origin[seg]) + self.bias[seg]
        y = round_shift_rne(p, self.right_shift[seg])
        return torch.clamp(y, self.qmin, self.qmax)

    @property
    def output_scale(self) -> float:
        q = self.metadata.get("output_quant", {})
        if not isinstance(q, Mapping) or "scale" not in q:
            raise ValueError(f"{self.path}: missing metadata.output_quant.scale")
        return float(q["scale"])


# =============================================================================
# 3. Artifact loading: GELU + E7 QEXP
# =============================================================================


@dataclass(frozen=True)
class GeluArtifact:
    layer: int
    input_scale: float
    weight_scale: float
    accumulator_scale: float
    page: PWLPage


@dataclass(frozen=True)
class AttentionArtifact:
    layer: int
    query_scale: float
    key_scale: float
    value_scale: float
    score_scale: float
    head_dim: int
    page: PWLPage


@dataclass(frozen=True)
class BaseArtifacts:
    root: Path
    gelu: Mapping[int, GeluArtifact]
    attention: Mapping[int, AttentionArtifact]

    @classmethod
    def load(cls, root: Path) -> "BaseArtifacts":
        root = root.resolve()
        scales = read_json(root / "calibrated_scales.json")
        ffn_rows = {int(x["layer_index"]): x for x in scales.get("ffn_gelu", [])}
        att_rows = {
            int(x["layer_index"]): x for x in scales.get("attention_softmax", [])
        }
        if not ffn_rows or not att_rows:
            raise ValueError("calibrated_scales.json lacks ffn_gelu/attention_softmax")

        gelu: Dict[int, GeluArtifact] = {}
        attention: Dict[int, AttentionArtifact] = {}
        for layer in sorted(ffn_rows):
            gd = root / "pages" / f"encoder_layer_{layer}" / "gelu"
            gpages = list(gd.glob("nnlut_gelu_direct32.json"))
            if len(gpages) != 1:
                raise ValueError(f"layer {layer}: expected one GELU page in {gd}")
            gp = PWLPage.load(gpages[0])
            row = ffn_rows[layer]
            gelu[layer] = GeluArtifact(
                layer=layer,
                input_scale=float(row["ffn_input_scale"]),
                weight_scale=float(row["ffn_weight_scale"]),
                accumulator_scale=float(row["gelu_accumulator_scale"]),
                page=gp,
            )

            ad = root / "pages" / f"encoder_layer_{layer}" / "softmax"
            qpages = list(ad.glob("nnlut_softmax_qexp_u7_direct32.json"))
            if len(qpages) != 1:
                raise ValueError(
                    f"layer {layer}: FINAL model requires exactly one E7 QEXP page in {ad}"
                )
            qp = PWLPage.load(qpages[0])
            if qp.qmin != 0 or qp.qmax != 127:
                raise ValueError(f"layer {layer}: QEXP is not E7 [0,127]")
            arow = att_rows[layer]
            score_scale = float(arow["score_accumulator_scale"])
            derived = (
                float(arow["query_scale"])
                * float(arow["key_scale"])
                / math.sqrt(float(arow["head_dim"]))
            )
            if not math.isclose(score_scale, derived, rel_tol=1e-8):
                raise ValueError(f"layer {layer}: attention score scale mismatch")
            attention[layer] = AttentionArtifact(
                layer=layer,
                query_scale=float(arow["query_scale"]),
                key_scale=float(arow["key_scale"]),
                value_scale=float(arow["value_scale"]),
                score_scale=score_scale,
                head_dim=int(arow["head_dim"]),
                page=qp,
            )
        return cls(root=root, gelu=gelu, attention=attention)

    def to(self, device: torch.device | str) -> "BaseArtifacts":
        return BaseArtifacts(
            root=self.root,
            gelu={k: dataclasses.replace(v, page=v.page.to(device)) for k, v in self.gelu.items()},
            attention={k: dataclasses.replace(v, page=v.page.to(device)) for k, v in self.attention.items()},
        )


# =============================================================================
# 4. Artifact loading: FINAL main->skip MomentPack LayerNorm
# =============================================================================


def _get_alias(obj: Mapping[str, Any], *names: str, default: Any = dataclasses.MISSING) -> Any:
    for n in names:
        if n in obj:
            return obj[n]
    if default is not dataclasses.MISSING:
        return default
    raise KeyError(f"missing required field; aliases={names}")


def _norm_site_name(layer: int, raw: str, kind: str) -> str:
    aliases = {
        "attention": "attention_output_ln",
        "attention_output": "attention_output_ln",
        "attention_output_ln": "attention_output_ln",
        "ffn": "ffn_output_ln",
        "output": "ffn_output_ln",
        "ffn_output": "ffn_output_ln",
        "ffn_output_ln": "ffn_output_ln",
    }
    prefix = f"layer_{layer}_"
    raw = raw.strip()
    if raw.startswith(prefix):
        suffix = aliases.get(raw[len(prefix):], raw[len(prefix):])
    else:
        suffix = aliases.get(raw or kind, raw or kind)
    return prefix + suffix


def _resolve_manifest(root: Path) -> Path:
    names = ("layernorm_momentpack_manifest.json", "manifest.json")
    hits = [root / n for n in names if (root / n).is_file()]
    if len(hits) != 1:
        raise ValueError(f"{root}: expected one MomentPack manifest, found {hits}")
    return hits[0]


def _resolve_page(root: Path, value: Any) -> Path:
    if isinstance(value, Mapping):
        value = _get_alias(value, "json", "path", "file")
    p = Path(str(value))
    if not p.is_absolute():
        q = root / p
        if q.is_file():
            return q.resolve()
        hits = list(root.rglob(p.name))
        if len(hits) == 1:
            return hits[0].resolve()
    if p.is_file():
        return p.resolve()
    raise FileNotFoundError(value)


@dataclass(frozen=True)
class LNSite:
    layer: int
    site_name: str
    kind: str
    branch_scale: float
    output_scale: float
    rsqrt_scale: float
    d_shift: int
    gap_bits: int
    fraction_bits: int
    gamma_multiplier: torch.Tensor
    beta_bias: torch.Tensor
    page: PWLPage


@dataclass(frozen=True)
class LNArtifacts:
    root: Path
    sites: Mapping[Tuple[int, str], LNSite]

    @classmethod
    def load(cls, root: Path) -> "LNArtifacts":
        root = root.resolve()
        manifest = read_json(_resolve_manifest(root))
        raw_sites = manifest.get("sites")
        if not isinstance(raw_sites, list) or not raw_sites:
            raise ValueError("MomentPack manifest has no sites")
        sites: Dict[Tuple[int, str], LNSite] = {}
        for raw in raw_sites:
            layer = int(raw["layer_index"])
            kind = str(raw.get("kind", ""))
            name = _norm_site_name(
                layer, str(_get_alias(raw, "site_name", "name", default="")), kind
            )
            if int(raw.get("hidden_size", 128)) != HIDDEN:
                raise ValueError(f"{name}: hidden size != 128")
            d_shift = int(_get_alias(raw, "d_shift", "d_pre_shift", default=4))
            gap = int(_get_alias(raw, "moment_gap_bits", "moment_gap", default=23))
            if d_shift != D_SHIFT or gap != MOMENT_GAP:
                raise ValueError(f"{name}: FINAL requires D_SHIFT=4, G23")
            # Latest frozen model has integer epsilon_D == 0.  Old manifests may
            # still carry the source epsilon real value; only code-domain value matters.
            epsilon_d_code = int(_get_alias(raw, "epsilon_d_code", default=0))
            if epsilon_d_code != 0:
                raise ValueError(
                    f"{name}: FINAL frozen model expects epsilon_d_code=0, got {epsilon_d_code}"
                )
            page = PWLPage.load(_resolve_page(root, raw["rsqrt_page"]))
            if page.qmin != 0 or page.qmax != 255:
                raise ValueError(f"{name}: RSQRT page must output U8")
            site = LNSite(
                layer=layer,
                site_name=name,
                kind=kind,
                branch_scale=float(raw["branch_scale"]),
                output_scale=float(raw["output_scale"]),
                rsqrt_scale=float(_get_alias(raw, "rsqrt_scale", "rsqrt_output_scale")),
                d_shift=d_shift,
                gap_bits=gap,
                fraction_bits=int(_get_alias(raw, "affine_fraction_bits", "fraction_bits")),
                gamma_multiplier=torch.as_tensor(raw["gamma_multiplier"], dtype=torch.int64),
                beta_bias=torch.as_tensor(raw["beta_bias"], dtype=torch.int64),
                page=page,
            )
            if site.gamma_multiplier.numel() != HIDDEN or site.beta_bias.numel() != HIDDEN:
                raise ValueError(f"{name}: affine coefficient length != 128")
            sites[(layer, name)] = site
        expected = {
            (l, f"layer_{l}_attention_output_ln") for l in range(2)
        } | {(l, f"layer_{l}_ffn_output_ln") for l in range(2)}
        if set(sites) != expected:
            raise ValueError(f"LayerNorm site set mismatch: {sorted(sites)}")
        return cls(root=root, sites=sites)

    def to(self, device: torch.device | str) -> "LNArtifacts":
        return LNArtifacts(
            root=self.root,
            sites={
                k: dataclasses.replace(
                    v,
                    gamma_multiplier=v.gamma_multiplier.to(device),
                    beta_bias=v.beta_bias.to(device),
                    page=v.page.to(device),
                )
                for k, v in self.sites.items()
            },
        )


# =============================================================================
# 5. Frozen VFU arithmetic
# =============================================================================


def raw_reciprocal_v1(rowsum: torch.Tensor) -> torch.Tensor:
    """FINAL no-CLZ reciprocal: R ~= 2^23/L, positive S18 for active rows."""
    l = rowsum.to(torch.int64)
    active = l > 0
    # Invalid/padding rows may have L=0.  They are architectural mask cases,
    # not a reciprocal zero-detection datapath.
    safe = torch.where(active, l, torch.full_like(l, 127))
    boundaries = torch.as_tensor(RAW_RECIP_BOUNDARIES, dtype=torch.int64, device=l.device)
    m = torch.as_tensor(RAW_RECIP_M, dtype=torch.int64, device=l.device)
    c = torch.as_tensor(RAW_RECIP_C, dtype=torch.int64, device=l.device)
    seg = torch.bucketize(safe, boundaries, right=True)
    p = safe * m[seg] + c[seg]
    r = round_shift_rne(p, RAW_RECIP_COEF_SHIFT)
    r = torch.where(active, r, torch.zeros_like(r))
    if bool(torch.any((r < 0) | (r > 131071)).item()):
        raise AssertionError("raw reciprocal escaped positive S18")
    return r


def frozen_softmax_context(
    query_real: torch.Tensor,
    key_real: torch.Tensor,
    value_real: torch.Tensor,
    attention_mask: torch.Tensor,
    art: AttentionArtifact,
) -> torch.Tensor:
    """FINAL E7 QEXP + deferred PV + raw reciprocal v1."""
    q = quantize_narrow_s8(query_real, art.query_scale)
    k = quantize_narrow_s8(key_real, art.key_scale)
    v = quantize_narrow_s8(value_real, art.value_scale)

    bsz, tokens, hidden = q.shape
    if hidden != HEADS * HEAD_DIM or art.head_dim != HEAD_DIM:
        raise ValueError("attention hidden/head shape mismatch")

    def heads(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(bsz, tokens, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    qh, kh, vh = heads(q), heads(k), heads(v)
    score = exact_small_matmul(
        qh, kh.transpose(-1, -2), reduction=HEAD_DIM, max_term=127 * 127
    )

    valid_token = attention_mask.to(torch.bool)
    pair_valid = (
        valid_token[:, None, :, None]
        & valid_token[:, None, None, :]
    )
    key_valid = valid_token[:, None, None, :]
    masked = torch.where(key_valid, score, torch.full_like(score, -(1 << 60)))
    rowmax = masked.max(dim=-1, keepdim=True).values
    d = score - rowmax

    # Latest QEXP page itself carries its endpoint/low-tail mapping.
    # No separate d==0 / zero-threshold detector is inserted here.
    e = art.page.infer(d)
    e = torch.where(pair_valid, e, torch.zeros_like(e))
    if bool(torch.any((e < 0) | (e > 127)).item()):
        raise AssertionError("E7 QEXP escaped [0,127]")

    l = e.sum(dim=-1, dtype=torch.int64)               # [B,H,Q]
    r = raw_reciprocal_v1(l)                          # [B,H,Q]
    n = exact_small_matmul(
        e, vh, reduction=tokens, max_term=127 * 127
    )                                                  # [B,H,Q,D]
    p = n * r.unsqueeze(-1)
    context_code = narrow_s8(round_shift_rne(p, SOFTMAX_RECIP_FRAC))
    context_code = torch.where(
        valid_token[:, None, :, None], context_code, torch.zeros_like(context_code)
    )
    context = context_code.to(torch.float64) * art.value_scale
    return context.permute(0, 2, 1, 3).contiguous().reshape(bsz, tokens, hidden)


def frozen_gelu(
    hidden_real: torch.Tensor,
    weight_code: torch.Tensor,
    bias_code: torch.Tensor,
    art: GeluArtifact,
) -> torch.Tensor:
    """Current artifact-backed PMPU FFN1 accumulator -> terminal GELU code."""
    x = quantize_narrow_s8(hidden_real, art.input_scale)
    acc = exact_small_matmul(
        x, weight_code.transpose(0, 1), reduction=x.shape[-1], max_term=127 * 127
    ) + bias_code
    code = art.page.infer(acc)
    return (code.to(torch.float64) * art.page.output_scale).to(hidden_real.dtype)


def momentpack(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """G23 MomentPack exactly as the frozen DSP arithmetic, but vectorized."""
    z = z.to(torch.int64)
    if z.shape[-1] != HIDDEN:
        raise ValueError("MomentPack requires hidden=128")
    if bool(torch.any((z < -254) | (z > 254)).item()):
        raise AssertionError("z escaped S9 residual contract")
    packed = torch.sum(z * (MOMENT_BASE + z), dim=-1, dtype=torch.int64)
    s = torch.div(packed, MOMENT_BASE, rounding_mode="floor")
    q = packed - s * MOMENT_BASE
    direct_s = z.sum(dim=-1, dtype=torch.int64)
    direct_q = (z * z).sum(dim=-1, dtype=torch.int64)
    if not torch.equal(s, direct_s) or not torch.equal(q, direct_q):
        raise AssertionError("MomentPack fields != direct S/Q")
    if bool(torch.any(q >= MOMENT_BASE).item()):
        raise AssertionError("Q carried into MomentPack S field")
    d = HIDDEN * q - s * s
    if bool(torch.any(d < 0).item()):
        raise AssertionError("LayerNorm D became negative")
    return packed, s, q, d


def frozen_layernorm(
    main_real: torch.Tensor,
    skip_real: torch.Tensor,
    site: LNSite,
) -> torch.Tensor:
    """FINAL main-only RQ + S9 residual + MomentPack + RSQRT + folded affine."""
    # Hardware interpretation:
    #   main PMPU result -> one RQ to native skip scale.
    # Python has skip_real as a float tensor, so quantizing it here reconstructs
    # the native S8 code already held by the skip producer; it is NOT a second
    # hardware skip-requant pass.
    main8 = quantize_narrow_s8(main_real, site.branch_scale)
    skip8 = quantize_narrow_s8(skip_real, site.branch_scale)
    z = main8 + skip8                                     # S9, no saturation

    _, s, _, d_full = momentpack(z)
    d27 = round_shift_rne(d_full, site.d_shift)

    # FINAL: no epsilon add and no D==0 special path.
    rho = site.page.infer(d27)                             # U8 finite clamped page output
    n = HIDDEN * z - s.unsqueeze(-1)                      # 128z-S
    t = n * rho.unsqueeze(-1)                             # raw S25-ish stage
    p = t * site.gamma_multiplier + site.beta_bias
    out = narrow_s8(round_shift_rne(p, site.fraction_bits))
    return (out.to(torch.float64) * site.output_scale).to(main_real.dtype)


# =============================================================================
# 6. Full manual BERT-Tiny encoder using the frozen arithmetic
# =============================================================================


class GaugePackFrozenModel:
    def __init__(
        self,
        hf_model: torch.nn.Module,
        base: BaseArtifacts,
        ln: LNArtifacts,
        device: torch.device | str,
    ) -> None:
        self.model = hf_model.eval().to(device)
        self.device = torch.device(device)
        self.base = base.to(self.device)
        self.ln = ln.to(self.device)
        cfg = self.model.config
        required = {
            "hidden_size": HIDDEN,
            "num_attention_heads": HEADS,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": 2,
        }
        for name, expected in required.items():
            if int(getattr(cfg, name, -1)) != expected:
                raise ValueError(f"model {name} != frozen GaugePack value {expected}")
        if set(self.base.gelu) != {0, 1} or set(self.base.attention) != {0, 1}:
            raise ValueError("base artifacts must contain exactly encoder layers 0 and 1")

        # Freeze the artifact-backed FFN1 integer weights once.  This matches the
        # current calibrated GELU input domain and avoids re-quantizing weights
        # on every validation batch.
        self.ffn1_weight_code: Dict[int, torch.Tensor] = {}
        self.ffn1_bias_code: Dict[int, torch.Tensor] = {}
        for li, layer in enumerate(self.model.bert.encoder.layer):
            art = self.base.gelu[li]
            dense = layer.intermediate.dense
            self.ffn1_weight_code[li] = quantize_narrow_s8(
                dense.weight.detach(), art.weight_scale
            )
            if dense.bias is None:
                bias = torch.zeros(dense.out_features, dtype=torch.int64, device=self.device)
            else:
                bias = torch.round(
                    dense.bias.detach().to(torch.float64) / art.accumulator_scale
                ).to(torch.int64)
            self.ffn1_bias_code[li] = bias

    def _ln_site(self, layer: int, kind: str) -> LNSite:
        return self.ln.sites[(layer, f"layer_{layer}_{kind}")]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        # Embedding/pooler/classifier are PS-side in the current project scope.
        hidden = self.model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        for li, layer in enumerate(self.model.bert.encoder.layer):
            # ---------------- Attention / PMPU Q,K,V + QK ----------------
            q_real = semantic_gemm(hidden, layer.attention.self.query)
            k_real = semantic_gemm(hidden, layer.attention.self.key)
            v_real = semantic_gemm(hidden, layer.attention.self.value)
            context = frozen_softmax_context(
                q_real, k_real, v_real, attention_mask, self.base.attention[li]
            ).to(hidden.dtype)

            # PMPU O projection, then FINAL main->skip LayerNorm.
            attn_main = semantic_gemm(context, layer.attention.output.dense)
            hidden = frozen_layernorm(
                attn_main,
                hidden,
                self._ln_site(li, "attention_output_ln"),
            )

            # ---------------- FFN / PMPU FFN1 + VFU GELU ----------------
            inter = frozen_gelu(
                hidden,
                self.ffn1_weight_code[li],
                self.ffn1_bias_code[li],
                self.base.gelu[li],
            )

            # PMPU FFN2, then FINAL main->skip LayerNorm.
            ffn_main = semantic_gemm(inter, layer.output.dense)
            hidden = frozen_layernorm(
                ffn_main,
                hidden,
                self._ln_site(li, "ffn_output_ln"),
            )

        if getattr(self.model.bert, "pooler", None) is not None:
            pooled = self.model.bert.pooler(hidden)
        else:
            pooled = hidden[:, 0]
        if hasattr(self.model, "dropout"):
            pooled = self.model.dropout(pooled)  # eval(): identity
        return self.model.classifier(pooled)


# =============================================================================
# 7. Dataset / CLI
# =============================================================================


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return name


def make_validation_loader(args: argparse.Namespace, tokenizer: Any) -> DataLoader:
    require_hf_runtime()
    if args.validation_file:
        ds = load_dataset("csv", data_files={"validation": args.validation_file})["validation"]
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config, split="validation")

    def tokenize(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
        )

    ds = ds.map(tokenize, batched=True)
    keep = [x for x in ("input_ids", "attention_mask", "token_type_ids", "label") if x in ds.column_names]
    ds.set_format(type="torch", columns=keep)
    collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=args.max_length)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )


@torch.no_grad()
def evaluate_sst2(gp: GaugePackFrozenModel, loader: DataLoader, *, compare_fp32: bool) -> None:
    correct = 0
    total = 0
    fp_correct = 0
    agreement = 0
    for bi, batch in enumerate(loader):
        labels = batch.pop("labels", batch.pop("label", None))
        if labels is None:
            raise KeyError("validation batch has no label(s)")
        labels = labels.to(gp.device)
        inputs = {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}}
        logits = gp.forward(**inputs)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        if compare_fp32:
            fp_inputs = {k: v.to(gp.device) for k, v in inputs.items()}
            fp_logits = gp.model(**fp_inputs).logits
            fp_pred = fp_logits.argmax(dim=-1)
            fp_correct += int((fp_pred == labels).sum().item())
            agreement += int((fp_pred == pred).sum().item())
        if bi == 0 or (bi + 1) % 20 == 0:
            print(f"[GaugePack] batches={bi+1} samples={total}", flush=True)

    print("\n=== GaugePack FINAL arithmetic E2E ===")
    print(f"GaugePack : {correct}/{total}  acc={correct/max(total,1):.6f}")
    if compare_fp32:
        print(f"FP32      : {fp_correct}/{total}  acc={fp_correct/max(total,1):.6f}")
        print(f"agreement : {agreement}/{total}  rate={agreement/max(total,1):.6f}")


def run_text(gp: GaugePackFrozenModel, tokenizer: Any, text: str, max_length: int) -> None:
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    logits = gp.forward(**batch)
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    pred = int(logits.argmax(dim=-1).item())
    print("text:", text)
    print("logits:", logits[0].detach().cpu().tolist())
    print("probs :", probs[0].detach().cpu().tolist())
    print("pred  :", pred)


def self_test() -> None:
    # Signed RNE including ties.
    x = torch.tensor([3, 1, -1, -3, 5, -5], dtype=torch.int64)
    got = round_shift_rne(x, 1).tolist()
    expected = [2, 0, 0, -2, 2, -2]
    if got != expected:
        raise AssertionError((got, expected))

    # MomentPack exact S/Q split over random legal S9 residuals.
    g = torch.Generator().manual_seed(20260819)
    z = torch.randint(-254, 255, (32, 128), generator=g, dtype=torch.int64)
    _, s, q, d = momentpack(z)
    if not torch.equal(s, z.sum(-1)):
        raise AssertionError("MomentPack S self-test")
    if not torch.equal(q, (z*z).sum(-1)):
        raise AssertionError("MomentPack Q self-test")
    if bool(torch.any(d < 0).item()):
        raise AssertionError("MomentPack D self-test")

    # Frozen reciprocal legal-domain basic contract.
    l = torch.arange(127, 8129, dtype=torch.int64)
    r = raw_reciprocal_v1(l)
    if bool(torch.any((r < 0) | (r > 131071)).item()):
        raise AssertionError("reciprocal S18 self-test")
    # Bound by the frozen design certificate: quotient perturbation <=2 codes.
    # Sample boundary-sensitive N=127*L and N=L/2 cases.
    n = torch.stack((127*l, torch.div(l, 2, rounding_mode="floor")), dim=0)
    cand = narrow_s8(round_shift_rne(n * r.unsqueeze(0), 23))
    exact = torch.round(n.to(torch.float64) / l.to(torch.float64)).to(torch.int64)
    exact = narrow_s8(exact)
    if int(torch.max(torch.abs(cand-exact)).item()) > 2:
        raise AssertionError("raw reciprocal context bound self-test")

    print("SELF-TEST PASS: RNE, G23 MomentPack, E7 raw reciprocal v1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-file GaugePack FINAL frozen arithmetic E2E inference"
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model-path", default="./bert-tiny-sst2")
    p.add_argument("--base-artifact-dir", type=Path, default=Path("nnlut_sst2_gaugepack_e7_qexp"))
    p.add_argument("--ln-artifact-dir", type=Path, default=Path("gaugepack_ln_main_to_skip_artifacts"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--dataset-name", default="nyu-mll/glue")
    p.add_argument("--dataset-config", default="sst2")
    p.add_argument("--validation-file")
    p.add_argument("--compare-fp32", action="store_true")
    p.add_argument("--text", help="run one sentence instead of SST-2 validation")
    p.add_argument("--seed", type=int, default=20260728)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.max_length != 64:
        raise ValueError("current GaugePack BERT-tiny artifact is frozen at max_length=64")
    seed_all(args.seed)
    require_hf_runtime()
    device = resolve_device(args.device)
    print(f"device={device}")
    print(f"model={args.model_path}")
    print(f"base_artifact={args.base_artifact_dir}")
    print(f"ln_artifact={args.ln_artifact_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    base = BaseArtifacts.load(args.base_artifact_dir)
    ln = LNArtifacts.load(args.ln_artifact_dir)
    gp = GaugePackFrozenModel(model, base, ln, device)

    if args.text is not None:
        run_text(gp, tokenizer, args.text, args.max_length)
    else:
        loader = make_validation_loader(args, tokenizer)
        evaluate_sst2(gp, loader, compare_fp32=args.compare_fp32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
