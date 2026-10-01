# Data

Experiment tables produced by `main.py`.

| Path | Description |
|------|-------------|
| `sim/sim_grid_coher_03.csv` | Simulation grid with coherent dephasing (0.3 rad/µs), 27 rows |
| `fake/fake_backends.csv` | Fake Brisbane / Kyoto / Sherbrooke, 3 rows |
| `real/round1/` | First IBM hardware round (Fez, Kingston, Marrakesh; triples 0-1-2 and 14-15-16) |
| `real/round2/` | Second round: connected boundary triples only |
| `combined/combined_results.csv` | sim + fake + round1 (36 rows) |
| `combined/combined_results_r2.csv` | **Final table**: + round2 (**47 rows**) |

Re-analyze:

```bash
./venv/bin/python main.py --analyze data/combined/combined_results_r2.csv
```
