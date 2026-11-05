#!/usr/bin/env python3
"""Example: fit the biomass composition to an experimental TGA/DTG curve.

Usage
-----
    python examples/run_optimization.py
    python examples/run_optimization.py data/experimental/oak.csv --tipo hardwood
    python examples/run_optimization.py data/experimental/corn_stover.csv --tipo grass --top 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer import optimize_tga, plot_fit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit biomass composition from an experimental TGA/DTG curve."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=str(ROOT / "data" / "experimental" / "pine.csv"),
        help="CSV file with TGA data (default: data/experimental/pine.csv)",
    )
    parser.add_argument(
        "--tipo",
        choices=["softwood", "hardwood", "grass"],
        default="softwood",
        help="Biomass type used to select the hemicellulose component (default: softwood)",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="Optional JSON file with custom bounds",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="Number of distinct best solutions to search for (default: 1)",
    )
    parser.add_argument(
        "--ash",
        type=float,
        default=None,
        help="Fix the ash fraction (e.g. 0.05 for 5%%)",
    )
    parser.add_argument(
        "--moist-e-shift",
        type=float,
        default=0.0,
        help="Shift the moisture E/R activation energy",
    )
    parser.add_argument(
        "--moist-a-shift",
        type=float,
        default=0.0,
        help="Shift the moisture ln(A) pre-exponential factor",
    )
    parser.add_argument(
        "--xygr-e-shift",
        type=float,
        default=0.0,
        help="Shift the grass hemicellulose (XYGR) E/R activation energy",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="Maximum Differential Evolution iterations (default: 50)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=15,
        help="Differential Evolution population size (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for Differential Evolution (default: 1; -1 uses all cores)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results" / "ajuste_tga_dtg.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=str(ROOT / "results" / "resultado.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bounds = None
    if args.bounds:
        with open(args.bounds, "r", encoding="utf-8") as f:
            bounds = json.load(f)

    print(f"Experimental file : {args.filename}")
    print(f"Biomass type      : {args.tipo.upper()}")
    print(f"Searching top     : {args.top} solution(s)\n")

    result = optimize_tga(
        filename=args.filename,
        biomass_type=args.tipo,
        bounds=bounds,
        top_n=args.top,
        fixed_ash=args.ash,
        moist_e_shift=args.moist_e_shift,
        moist_a_shift=args.moist_a_shift,
        xygr_e_shift=args.xygr_e_shift,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=args.workers,
    )

    print(f"Detected heating rate: {result['data']['heating_rate']:.2f} °C/min")
    print(f"Best normalized error: {result['error']:.6f}\n")

    print("Best composition:")
    for name, value in result["composition"].items():
        print(f"  {name:<6}: {value * 100:6.2f}%")

    if args.top > 1:
        print(f"\nSummary over the top {args.top} solutions:")
        keys = result["components"] + ["ASH"]
        values = {k: [comp[k] * 100 for _, comp, _ in result["all_results"]] for k in keys}
        print(f"{'Component':<10} | {'Mean':>8} | {'Range [min - max]':<20}")
        print("-" * 46)
        for key in keys:
            vals = values[key]
            print(
                f"{key:<10} | {np.mean(vals):>7.2f}% | "
                f"[{np.min(vals):>5.2f}% - {np.max(vals):>5.2f}%]"
            )

    plot_path = plot_fit(result, args.output)
    print(f"\nPlot saved to: {plot_path}")

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "filename": result["filename"],
        "biomass_type": result["biomass_type"],
        "error": result["error"],
        "composition": result["composition"],
        "heating_rate": result["data"]["heating_rate"],
        "top_n": args.top,
        "solutions": [
            {"error": err, "composition": comp}
            for err, comp, _ in result["all_results"]
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
