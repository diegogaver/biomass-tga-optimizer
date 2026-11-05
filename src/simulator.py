"""ODE simulator for solid-phase biomass pyrolysis.

The mechanism is read from a JSON file produced by :mod:`src.parser`. The
system is integrated with ``scipy.integrate.solve_ivp`` using the BDF method
and an analytical Jacobian, which is significantly faster than finite
differences for stiff solid-phase kinetics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_MECHANISM = (
    Path(__file__).resolve().parent.parent / "data" / "kinetics" / "cineticas_solidas.json"
)


class SimpleMecanismo:
    """Minimal mechanism container loaded from a JSON file."""

    def __init__(self, json_path: str | Path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.solid_species = data["solid_species"]
        self.reactions = data["reactions"]


_MECHANISM_CACHE: Dict[Path, SimpleMecanismo] = {}


def _load_mechanism(path: str | Path) -> SimpleMecanismo:
    path = Path(path).resolve()
    if path not in _MECHANISM_CACHE:
        _MECHANISM_CACHE[path] = SimpleMecanismo(path)
    return _MECHANISM_CACHE[path]


def detailed_biomass_simulator(
    initial_fractions: Dict[str, float],
    heating_rate: float = 10.0,
    T_start: float = 30.0,
    T_end: float = 600.0,
    mechanism_path: Optional[str | Path] = None,
    moist_e_shift: float = 0.0,
    moist_a_shift: float = 0.0,
    xygr_e_shift: float = 0.0,
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the remaining solid mass during a linear heating ramp.

    Parameters
    ----------
    initial_fractions:
        Initial mass fractions keyed by solid species name (``CELL``, ``GMSW``,
        ``LIGO``, ``ASH``, ...).
    heating_rate:
        Heating rate in degrees Celsius per minute.
    T_start, T_end:
        Temperature range in degrees Celsius.
    mechanism_path:
        Optional path to the JSON mechanism. Defaults to
        ``data/kinetics/cineticas_solidas.json``.
    moist_e_shift, moist_a_shift, xygr_e_shift:
        Calibration shifts applied to the activation energy (``E/R``) and
        pre-exponential factor (``ln A``) of the moisture and grass
        hemicellulose reactions.
    n_points:
        Number of output points.

    Returns
    -------
    temperatures, total_mass:
        Arrays with temperature (degrees Celsius) and total solid mass fraction.
    """
    beta = heating_rate / 60.0  # degrees Celsius per second
    T0 = T_start + 273.15

    mechanism = _load_mechanism(mechanism_path or DEFAULT_MECHANISM)
    species_list = mechanism.solid_species
    species_idx = {sp: i for i, sp in enumerate(species_list)}
    num_species = len(species_list)
    num_rxns = len(mechanism.reactions)

    S = np.zeros((num_species, num_rxns))
    reactant_idx = np.zeros(num_rxns, dtype=int)
    lnA_arr = np.zeros(num_rxns)
    beta_arr = np.zeros(num_rxns)
    E_R_arr = np.zeros(num_rxns)

    for j, rxn in enumerate(mechanism.reactions):
        reactant = rxn["reactant"]
        if reactant not in species_idx:
            # Gas-phase reactant: disable this reaction by making k effectively 0.
            lnA_arr[j] = -1e6
            continue

        r_idx = species_idx[reactant]
        reactant_idx[j] = r_idx
        lnA_arr[j] = rxn["lnA"]
        beta_arr[j] = rxn["beta"]
        E_R_arr[j] = rxn["E_R"]

        S[r_idx, j] -= 1.0
        for prod, coeff in rxn["solid_products"].items():
            p_idx = species_idx[prod]
            S[p_idx, j] += coeff

    # Apply calibration shifts once.
    eff_lnA = lnA_arr.copy()
    eff_E_R = E_R_arr.copy()
    for j, rxn in enumerate(mechanism.reactions):
        if rxn["reactant"] == "MOIST":
            eff_E_R[j] += moist_e_shift
            eff_lnA[j] += moist_a_shift
        elif rxn["reactant"] == "XYGR":
            eff_E_R[j] += xygr_e_shift

    M_sel = np.zeros((num_rxns, num_species))
    for j in range(num_rxns):
        M_sel[j, reactant_idx[j]] = 1.0

    def rate_constants(T_K: float) -> np.ndarray:
        k = np.exp(eff_lnA) * (T_K ** beta_arr) * np.exp(-eff_E_R / T_K)
        if T_K < 200:
            k = np.where(eff_E_R > 10000, 0.0, k)
        return k

    def odes(t: float, y: np.ndarray) -> np.ndarray:
        T_K = T0 + beta * t
        k_rxn = rate_constants(T_K)

        if y.ndim == 1:
            v = k_rxn * y[reactant_idx]
        else:
            v = k_rxn[:, np.newaxis] * y[reactant_idx, :]
        return S @ v

    def jacobian(t: float, y: np.ndarray) -> np.ndarray:
        T_K = T0 + beta * t
        k_rxn = rate_constants(T_K)
        return (S * k_rxn) @ M_sel

    y0 = np.zeros(num_species)
    for sp, val in initial_fractions.items():
        if sp in species_idx:
            y0[species_idx[sp]] = val

    total_in = np.sum(y0)
    if total_in > 0:
        y0 = y0 / total_in

    t_span = (0.0, (T_end - T_start) / beta)
    sol = solve_ivp(
        odes,
        t_span,
        y0,
        method="BDF",
        jac=jacobian,
        vectorized=True,
        t_eval=np.linspace(t_span[0], t_span[1], n_points),
        rtol=1e-2,
        atol=1e-4,
    )

    masa_total = np.sum(sol.y, axis=0)
    temps = T0 + beta * sol.t - 273.15
    return temps, masa_total


if __name__ == "__main__":
    composition = {
        "CELL": 0.41,
        "GMSW": 0.14,
        "LIGO": 0.26,
        "LIGC": 0.07,
        "TANN": 0.05,
        "ASH": 0.07,
    }
    temperatures, mass = detailed_biomass_simulator(composition)
    print(f"Simulated {len(temperatures)} points.")
    print(f"Final solid mass: {mass[-1] * 100:.2f}%")
