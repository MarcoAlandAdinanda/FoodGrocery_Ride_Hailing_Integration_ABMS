# Minimal Portable Backup

This is the minimal source package for running and verifying the food-grocery
agent-based simulation. It contains no Git metadata or configuration, virtual
environment, cache, experiment output, notebook, report, or raw calibration
dataset.

## Contents

- `src/`: simulation engine, parameters, experiment controller, visualizer, and
  supporting source code.
- `tests/`: automated regression tests.
- `dataset/food_hourly_demand_profile.csv`: default demand input.
- `source/Integrated_Food_Grocery_Delivery_Model_Documentation.pdf`: formal model
  specification referenced and hashed by the engine metadata.
- `requirements-simulation.txt`: validated Python dependencies.
- `SIMULATION_README.md`: commands, model behavior, and output definitions.
- `about_research.md`: short research context.

## Run

The project was validated with Python 3.12.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-simulation.txt
python -B src/simulation_engine.py --p-integration 0.44
```

Run the paired experiment controller with:

```powershell
python -B src/simulation_experiments.py --p-integrated 0.44 --replications 30 --sensitivity --summary-only
```

Verify the transferred package with:

```powershell
python -m unittest discover -s tests -q
```

Generated output is written below `output/` and is intentionally absent from
this backup.
