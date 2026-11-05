# Biomass TGA Optimizer

Herramienta en Python para estimar la composición inicial de biomasa (celulosa,
hemicelulosa, ligninas, extractivos, humedad y ceniza) a partir de curvas
experimentales de **TGA/DTG** y un mecanismo cinético de fase sólida.

El proyecto combina:

- Un **parser** que convierte la estequiometría molar de un mecanismo
  OpenSMOKE/PoliMi en rendimientos másicos.
- Un **simulador ODE** rígido (`BDF` con Jacobiano analítico).
- Un **optimizador estocástico** basado en *Differential Evolution*.

## Estructura

```
biomass-tga-optimizer/
│
├── data/
│   ├── experimental/         # Curvas TGA/DTG de prueba (pine, oak, corn_stover)
│   └── kinetics/             # Mecanismo cinético y JSON preprocesado
│
├── src/
│   ├── __init__.py
│   ├── parser.py             # Estequiometría molar -> masa
│   ├── simulator.py          # Solver ODE (BDF + Jacobiano analítico)
│   └── optimizer.py          # Differential Evolution
│
├── examples/
│   └── run_optimization.py   # Script de ejemplo listo para ejecutar
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Instalación

Se recomienda usar un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
# Softwood (pino)
python examples/run_optimization.py

# Hardwood (roble)
python examples/run_optimization.py data/experimental/oak.csv --tipo hardwood

# Grass (rastrojo de maíz), buscando las 3 mejores soluciones
python examples/run_optimization.py data/experimental/corn_stover.csv --tipo grass --top 3
```

Los resultados se guardan por defecto en:

- `results/ajuste_tga_dtg.png` — gráfica TGA/DTG experimental vs. simulada.
- `results/resultado.json` — composición óptima y error normalizado.

### Opciones principales

| Opción | Descripción |
| --- | --- |
| `--tipo {softwood,hardwood,grass}` | Selecciona la hemicelulosa de referencia. |
| `--top N` | Busca `N` soluciones distintas (niching). |
| `--ash F` | Fija la fracción de ceniza (ej. `0.05`). |
| `--bounds archivo.json` | Límites físicos personalizados. |
| `--maxiter`, `--popsize`, `--seed` | Control del optimizador. |
| `--output`, `--json` | Rutas de salida. |

## Formato de los datos experimentales

El CSV debe contener, como mínimo, una columna de temperatura y una de masa.
Las columnas se detectan automáticamente (`temp`/`Temperatura`, `masa`/`mass`/
`TG`) y se admite una columna de tiempo para inferir la tasa de calentamiento.

Ejemplo:

```csv
tiempo,Temp_C,Masa_Restante_TG
0.0037,20.92,1.0000
0.0042,20.91,0.9999
...
```

Si la masa está en porcentaje (0–100) se normaliza automáticamente a fracción.

## Regenerar el mecanismo cinético

El simulador usa `data/kinetics/cineticas_solidas.json`. Para regenerarlo desde
los archivos originales de OpenSMOKE:

```bash
python -m src.parser \
  --xml data/kinetics/kinetics.solid.xml \
  --out data/kinetics/Reaction_Tables_Solid.out \
  --json data/kinetics/cineticas_solidas.json
```

## Uso como librería

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

## Cómo funciona el ajuste

1. Se carga la curva experimental y se calcula su DTG suavizada.
2. Para cada composición candidata se integra el mecanismo cinético.
3. Se interpola la masa simulada sobre la temperatura experimental.
4. El error combina el MSE normalizado de TGA y DTG al 50 % cada uno.
5. Se aplican restricciones físicas (suma de fracciones, contenido de lignina,
   ceniza máxima) mediante penalizaciones.
6. Differential Evolution minimiza el error de forma global.

## Licencia

Distribuido bajo la licencia MIT. Ver [LICENSE](LICENSE).
