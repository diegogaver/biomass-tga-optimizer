"""Stochastic optimizer for biomass composition from TGA/DTG curves.

The optimizer fits the initial mass fractions of the reference components
(cellulose, hemicellulose, lignins, extractives and moisture) by minimizing a
normalized combination of the TGA and DTG residuals. Differential Evolution is
used because the objective is non-convex and the simulator is evaluated as a
black box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from .simulator import detailed_biomass_simulator


DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "CELL": (0.30, 0.60),
    "GMSW": (0.10, 0.40),
    "XYHW": (0.10, 0.40),
    "XYGR": (0.10, 0.40),
    "LIGO": (0.00, 0.35),
    "LIGC": (0.00, 0.35),
    "LIGH": (0.00, 0.35),
    "TANN": (0.00, 0.15),
    "TGL": (0.00, 0.15),
    "MOIST": (0.00, 0.15),
}

COMPONENTS_BY_TYPE: Dict[str, List[str]] = {
    "softwood": ["CELL", "GMSW", "LIGO", "LIGC", "TANN", "TGL", "MOIST"],
    "hardwood": ["CELL", "XYHW", "LIGO", "LIGC", "TANN", "TGL", "MOIST"],
    "grass": ["CELL", "XYGR", "LIGO", "LIGC", "TANN", "TGL", "MOIST"],
}


def _read_csv(filename: str | Path) -> pd.DataFrame:
    filename = Path(filename)
    try:
        return pd.read_csv(filename)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(filename, encoding="utf-16")
        except UnicodeDecodeError:
            return pd.read_csv(filename, encoding="latin1")


def load_experimental_data(
    filename: str | Path,
    temp_col: Optional[str] = None,
    mass_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> dict:
    """Load and clean a TGA/DTG experimental curve.

    Columns are auto-detected when not provided. Mass values given as
    percentages are normalized to fractions.
    """
    df = _read_csv(filename)
    cols = list(df.columns)

    temp_col = temp_col or next((c for c in cols if "temp" in c.lower()), cols[0])
    mass_col = mass_col or next(
        (
            c
            for c in cols
            if any(token in c.lower() for token in ("masa", "mass", "frac", "tg"))
        ),
        cols[-1],
    )
    time_col = time_col or next(
        (c for c in cols if "time" in c.lower() or "tiempo" in c.lower()), None
    )

    df = df.dropna(subset=[temp_col, mass_col])
    df = df.drop_duplicates(subset=[temp_col])
    df = df.sort_values(by=[temp_col])
    df = df[df[temp_col] >= 30.0]

    t_exp = df[temp_col].to_numpy(dtype=float)
    m_exp = df[mass_col].to_numpy(dtype=float)

    if np.nanmax(m_exp) > 2.0:
        m_exp = m_exp / 100.0

    T_start = float(np.min(t_exp))
    T_end = float(np.max(t_exp))

    heating_rate = 10.0
    if time_col:
        time_vals = df[time_col].to_numpy(dtype=float)
        dt = time_vals[-1] - time_vals[0]
        if dt > 0:
            heating_rate = (T_end - T_start) / dt

    dtg_raw = -np.gradient(m_exp, t_exp)
    dtg_raw = np.nan_to_num(dtg_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if len(dtg_raw) > 50:
        window = min(len(dtg_raw) // 20, 301)
        if window % 2 == 0:
            window += 1
        window = max(5, window)
        dtg_exp = savgol_filter(dtg_raw, window_length=window, polyorder=3)
    else:
        dtg_exp = dtg_raw

    return {
        "t_exp": t_exp,
        "m_exp": m_exp,
        "dtg_exp": dtg_exp,
        "var_m": float(np.var(m_exp)),
        "var_dtg": float(np.var(dtg_exp)),
        "heating_rate": float(heating_rate),
        "T_start": T_start,
        "T_end": T_end,
    }


def objective_function(
    x: Sequence[float],
    t_exp: np.ndarray,
    m_exp: np.ndarray,
    dtg_exp: np.ndarray,
    var_m: float,
    var_dtg: float,
    heating_rate: float,
    T_start: float,
    T_end: float,
    components: Sequence[str],
    previous_solutions: Optional[Iterable[Sequence[float]]] = None,
    fixed_ash: Optional[float] = None,
    moist_e_shift: float = 0.0,
    moist_a_shift: float = 0.0,
    xygr_e_shift: float = 0.0,
) -> float:
    """Normalized TGA + DTG objective with physical constraints."""
    x = np.asarray(x, dtype=float)
    sum_x = float(np.sum(x))
    penalty = 0.0

    if fixed_ash is not None:
        ash_frac = fixed_ash
        target_sum = 1.0 - fixed_ash
        diff = abs(sum_x - target_sum)
        if diff > 1e-3:
            penalty += diff * 1e6
    else:
        ash_frac = 1.0 - sum_x
        if sum_x > 1.0 or ash_frac > 0.15 or ash_frac < 0.0:
            penalty += abs(1.0 - sum_x) * 1e5

    lig_sum = sum(x[i] for i, c in enumerate(components) if c in ("LIGO", "LIGC", "LIGH"))
    if lig_sum < 0.15:
        penalty += (0.15 - lig_sum) * 1e5
    elif lig_sum > 0.35:
        penalty += (lig_sum - 0.35) * 1e5

    if previous_solutions:
        for prev_x in previous_solutions:
            dist = np.linalg.norm(x - np.asarray(prev_x, dtype=float))
            if dist < 0.05:
                penalty += (0.05 - dist) * 1e4

    if penalty > 1000:
        return penalty

    comp = {c: float(x[i]) for i, c in enumerate(components)}
    comp["ASH"] = ash_frac

    t_sim, m_sim = detailed_biomass_simulator(
        comp,
        heating_rate=heating_rate,
        T_start=T_start,
        T_end=T_end,
        moist_e_shift=moist_e_shift,
        moist_a_shift=moist_a_shift,
        xygr_e_shift=xygr_e_shift,
    )

    f_interp = interp1d(t_sim, m_sim, kind="linear", fill_value="extrapolate")
    m_sim_interp = f_interp(t_exp)
    dtg_sim = -np.gradient(m_sim_interp, t_exp)

    mse_m = np.mean((m_exp - m_sim_interp) ** 2) / (var_m + 1e-10)
    mse_dtg = np.mean((dtg_exp - dtg_sim) ** 2) / (var_dtg + 1e-10)
    return 0.5 * mse_m + 0.5 * mse_dtg + penalty


def optimize_tga(
    filename: str | Path,
    biomass_type: str = "softwood",
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    top_n: int = 1,
    fixed_ash: Optional[float] = None,
    moist_e_shift: float = 0.0,
    moist_a_shift: float = 0.0,
    xygr_e_shift: float = 0.0,
    maxiter: int = 50,
    popsize: int = 15,
    tol: float = 0.01,
    seed: int = 42,
    workers: int = 1,
) -> dict:
    """Run Differential Evolution and return the best compositions."""
    if biomass_type not in COMPONENTS_BY_TYPE:
        raise ValueError(f"biomass_type must be one of {sorted(COMPONENTS_BY_TYPE)}")

    components = COMPONENTS_BY_TYPE[biomass_type]
    bounds_dict = DEFAULT_BOUNDS.copy()
    if bounds:
        bounds_dict.update({k: tuple(v) for k, v in bounds.items()})

    bounds_list = [bounds_dict.get(c, (0.0, 1.0)) for c in components]
    data = load_experimental_data(filename)

    previous_solutions: List[np.ndarray] = []
    results: List[Tuple[float, dict, np.ndarray]] = []

    for i in range(top_n):
        result = differential_evolution(
            objective_function,
            bounds_list,
            args=(
                data["t_exp"],
                data["m_exp"],
                data["dtg_exp"],
                data["var_m"],
                data["var_dtg"],
                data["heating_rate"],
                data["T_start"],
                data["T_end"],
                components,
                previous_solutions,
                fixed_ash,
                moist_e_shift,
                moist_a_shift,
                xygr_e_shift,
            ),
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            seed=seed + i,
            workers=workers,
        )

        best_x = result.x
        ash_val = fixed_ash if fixed_ash is not None else 1.0 - float(np.sum(best_x))
        best_comp = {c: float(best_x[j]) for j, c in enumerate(components)}
        best_comp["ASH"] = ash_val

        results.append((float(result.fun), best_comp, best_x))
        previous_solutions.append(best_x)

    results.sort(key=lambda item: item[0])
    best_error, best_comp, best_x = results[0]

    return {
        "error": best_error,
        "composition": best_comp,
        "x": best_x,
        "all_results": results,
        "components": components,
        "biomass_type": biomass_type,
        "data": data,
        "filename": str(filename),
    }


def plot_fit(result: dict, output_path: str | Path = "ajuste_tga_dtg.png") -> Path:
    """Plot experimental data against the best simulated TGA/DTG curve."""
    data = result["data"]
    comp = result["composition"]
    heating_rate = data["heating_rate"]
    T_start = data["T_start"]
    T_end = data["T_end"]

    t_opt, m_opt = detailed_biomass_simulator(
        comp, heating_rate=heating_rate, T_start=T_start, T_end=T_end
    )
    dtg_opt = -np.gradient(m_opt, t_opt)

    t_exp = data["t_exp"]
    m_exp = data["m_exp"]
    dtg_exp = data["dtg_exp"]
    step = max(1, len(t_exp) // 200)

    fig, ax1 = plt.subplots(figsize=(12, 6), sharex=True)
    ax1.scatter(
        t_exp[::step],
        (m_exp * 100)[::step],
        color="black",
        label="TGA experimental",
        alpha=0.6,
        marker=".",
    )
    ax1.plot(t_opt, m_opt * 100, color="red", label="TGA simulated", linewidth=2)
    ax1.set_ylabel("Solid mass (%)")
    ax1.grid(True, alpha=0.3)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.plot(t_exp, dtg_exp * 100, color="gray", label="DTG experimental", alpha=0.8, linestyle="--")
    ax2.plot(t_opt, dtg_opt * 100, color="blue", label="DTG simulated", linewidth=2)
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("DTG (%/°C)")
    ax2.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10)

    comp_text = "Optimal composition:\n" + "\n".join(
        f"{k}: {v * 100:.1f}%" for k, v in comp.items()
    )
    ax1.text(
        0.02,
        0.05,
        comp_text,
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
