#!/usr/bin/env python3
"""Unified entry point for the four frozen GaugePack v1 NN-LUT flows.

The operator implementations remain in their individual files so there is one
canonical implementation per arithmetic contract.  This entry point can:

* forward one command to ``gelu``, ``qexp``, ``reciprocal`` or ``rsqrt``;
* run all four deterministic self-tests; or
* execute an ordered JSON job plan.

Examples
--------

    python gaugepack_all_nnlut_train.py --self-test

    python gaugepack_all_nnlut_train.py gelu \
      --calibration-npz ffn1_calibration.npz \
      --search-npz ffn1_search.npz \
      --accumulator-scale 0.0003125

    python gaugepack_all_nnlut_train.py --plan nnlut_jobs.json \
      --report nnlut_jobs_report.json

Plan schema::

    {
      "jobs": [
        {"operator": "gelu", "args": ["--self-test"]},
        {"operator": "qexp", "args": ["--self-test"]},
        {"operator": "reciprocal", "args": ["--self-test"]},
        {"operator": "rsqrt", "args": ["--self-test"]}
      ]
    }
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


OPERATOR_MODULES: Mapping[str, str] = {
    "gelu": "gaugepack_gelu_nnlut_train",
    "qexp": "gaugepack_qexp_nnlut_train",
    "reciprocal": "gaugepack_softmax_recip_nnlut_train",
    "rsqrt": "gaugepack_rsqrt_nnlut_train",
}


def invoke(operator: str, arguments: Sequence[str]) -> Dict[str, Any]:
    if operator not in OPERATOR_MODULES:
        raise ValueError(
            f"unknown operator {operator!r}; choose from {sorted(OPERATOR_MODULES)}"
        )
    module_name = OPERATOR_MODULES[operator]
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise RuntimeError(f"{module_name} does not expose callable main(argv)")
    started = time.perf_counter()
    try:
        result = entry(list(arguments))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            raise RuntimeError(
                f"{operator} exited with status {code}"
            ) from exc
        result = code
    status = 0 if result is None else int(result)
    if status != 0:
        raise RuntimeError(f"{operator} returned status {status}")
    return {
        "operator": operator,
        "module": module_name,
        "arguments": list(arguments),
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
    }


def load_plan(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("plan root must be an object containing a jobs array")
    jobs: List[Dict[str, Any]] = []
    for index, raw in enumerate(payload["jobs"]):
        if not isinstance(raw, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        operator = raw.get("operator")
        arguments = raw.get("args", [])
        if not isinstance(operator, str) or operator not in OPERATOR_MODULES:
            raise ValueError(
                f"jobs[{index}].operator must be one of {sorted(OPERATOR_MODULES)}"
            )
        if not isinstance(arguments, list) or not all(
            isinstance(item, str) for item in arguments
        ):
            raise ValueError(f"jobs[{index}].args must be an array of strings")
        jobs.append({"operator": operator, "args": list(arguments)})
    if not jobs:
        raise ValueError("plan jobs array must not be empty")
    return jobs


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_jobs(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    results = [
        invoke(str(job["operator"]), list(job.get("args", []))) for job in jobs
    ]
    return {
        "schema": "gaugepack.nnlut.unified_run.v1",
        "status": "PASS",
        "operator_order": [result["operator"] for result in results],
        "jobs": results,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_used": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for GaugePack GELU/QEXP/reciprocal/RSQRT NN-LUT flows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "operator",
        nargs="?",
        choices=sorted(OPERATOR_MODULES),
        help="individual operator to run; remaining arguments are forwarded",
    )
    parser.add_argument(
        "operator_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded unchanged to the selected operator",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run all four individual deterministic self-tests",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="ordered JSON job plan; mutually exclusive with an operator",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON summary for --self-test or --plan",
    )
    parser.add_argument(
        "--list-operators",
        action="store_true",
        help="print the supported operator names",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = sum(
        int(value)
        for value in (
            args.operator is not None,
            args.plan is not None,
            args.self_test,
            args.list_operators,
        )
    )
    if selected != 1:
        parser.error(
            "choose exactly one of an operator, --plan, --self-test, or --list-operators"
        )

    try:
        if args.list_operators:
            print("\n".join(sorted(OPERATOR_MODULES)))
            return 0
        if args.operator is not None:
            invoke(args.operator, args.operator_args)
            return 0
        if args.self_test:
            jobs = [
                {"operator": operator, "args": ["--self-test"]}
                for operator in ("gelu", "qexp", "reciprocal", "rsqrt")
            ]
        else:
            jobs = load_plan(args.plan)
        report = run_jobs(jobs)
        if args.report is not None:
            write_report(args.report, report)
            print(f"Unified report: {args.report.resolve()}")
        print(
            "Unified run PASS: "
            + ", ".join(str(item["operator"]) for item in report["jobs"])
        )
        return 0
    except (
        FileNotFoundError,
        ImportError,
        json.JSONDecodeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
