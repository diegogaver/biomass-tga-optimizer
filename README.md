## Citation

If you use this software in your research, please cite our paper: > **Racero-Galaraga, D., & Cressoni De Conti, A.** (2026). *Development and validation of software for the determination of lignocellulosic biomass components via TGA, based on a multistep kinetic model*. Biochemical Engineering Journal (Under Review).
# Biomass TGA Optimizer

A Python tool to estimate the initial composition of biomass (cellulose, hemicellulose, lignins, extractives, moisture, and ash) from experimental **TGA/DTG** curves and a solid-phase kinetic mechanism.

The project combines:

- A **parser** that converts the molar stoichiometry of an OpenSMOKE/PoliMi mechanism into mass yields.
- A stiff **ODE simulator** (`BDF` with an analytical Jacobian).
- A **stochastic optimizer** based on *Differential Evolution*.

## Structure

```text
biomass-tga-optimizer/
│
├── data/
│   ├── experimental/         # Test TGA/DTG curves (pine, oak, corn_stover)
│   └── kinetics/             # Kinetic mechanism and preprocessed JSON
│
├── src/
│   ├── __init__.py
│   ├── parser.py             # Molar stoichiometry -> mass conversion
│   ├── simulator.py          # ODE solver (BDF + analytical Jacobian)
│   └── optimizer.py          # Differential Evolution
│
├── examples/
│   └── run_optimization.py   # Ready-to-run example script
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


```

#### Installation

Using a virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Softwood (pino)
python examples/run_optimization.py

# Hardwood (roble)
python examples/run_optimization.py data/experimental/oak.csv --tipo hardwood

# Grass (rastrojo de maíz), buscando las 3 mejores soluciones
python examples/run_optimization.py data/experimental/corn_stover.csv --tipo grass --top 3
```

Results are saved by default to:

- `results/ajuste_tga_dtg.png` — experimental vs. simulated TGA/DTG plot.
    
- `results/resultado.json` — optimal composition and normalized error.

### Main Options

|**Option**|**Description**|
|---|---|
|`--tipo {softwood,hardwood,grass}`|Selects the reference hemicellulose.|
|`--top N`|Searches for `N` distinct solutions (niching).|
|`--ash F`|Fixes the ash fraction (e.g., `0.05`).|
|`--bounds file.json`|Custom physical boundaries.|
|`--maxiter`, `--popsize`, `--seed`|Optimizer control settings.|
|`--output`, `--json`|Output file paths.|

## Experimental Data Format

The CSV must contain at least a temperature column and a mass column. Columns are automatically detected (`temp`/`Temperatura`, `masa`/`mass`/`TG`), and a time column is supported to infer the heating rate.

Example:

```csv
tiempo,Temp_C,Masa_Restante_TG
0.0037,20.92,1.0000
0.0042,20.91,0.9999
...
```

If the mass is given in percentage (0–100), it is automatically normalized to a fraction.

## Regenerating the Kinetic Mechanism

The simulator uses `data/kinetics/cineticas_solidas.json`. To regenerate it from original OpenSMOKE files:

```bash
python -m src.parser \
  --xml data/kinetics/kinetics.solid.xml \
  --out data/kinetics/Reaction_Tables_Solid.out \
  --json data/kinetics/cineticas_solidas.json
```

## Usage as a Library

```python
from src.optimizer import optimize_tga, plot_fit
from src.simulator import detailed_biomass_simulator

result = optimize_tga("data/experimental/pine.csv", biomass_type="softwood", top_n=1)
print(result["composition"])
plot_fit(result, "results/ajuste.png")

# Simulación directa con una composición dada
t, mass = detailed_biomass_simulator(
    {"CELL": 0.44, "GMSW": 0.15, "LIGO": 0.27, "LIGC": 0.07, "ASH": 0.07}
)
```

## How Fitting Works

1. The experimental curve is loaded, and its smoothed DTG is calculated.
    
2. For each candidate composition, the kinetic mechanism is integrated.
    
3. The simulated mass is interpolated over the experimental temperature.
    
4. The error combines the normalized MSE of TGA and DTG at 50% each.
    
5. Physical constraints (sum of fractions, lignin content, maximum ash) are applied via penalties.
    
6. Differential Evolution globally minimizes the error.
    

## License

Distributed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.