# Quantum Error Mitigation Lab — Direction 2

Adaptive selection of **BARE / DD / ZNE / ZNE+DD** for a 3-qubit Grover circuit, based on device features (`T1`, `T2`, `ratio`, `R_coh`, idle windows).

Measurement and decision are **separated**:
1. A harness runs **all four strategies** unconditionally (real IBM hardware or noisy simulation).
2. Each run writes a CSV row: hardware features → success probabilities (+ SEM).
3. A thin policy `features → strategy` is fitted and evaluated (accuracy, regret) on the collected table.

## Repository layout

```
quantum_research/
├── main.py                 # Main lab (sim / fake / real / analyze)
├── requirements.txt
├── README.md               # This file (English)
├── docs/
│   ├── RESULTS_DIRECTION2.md   # Full results write-up (Russian)
│   └── BACKGROUND_RU.md        # Earlier project background (Russian)
├── data/
│   ├── sim/                # Simulation grid
│   ├── fake/               # Fake-backend runs
│   ├── real/
│   │   ├── round1/         # First hardware round
│   │   └── round2/         # Boundary triples (connected layouts)
│   └── combined/           # Merged tables for threshold fitting
├── scripts/
│   ├── run_ibm.sh          # Interactive IBM submit helper
│   ├── fetch_kingston_r2.py
│   └── wait_kingston_r2.sh
├── legacy/                 # Older adaptive controller & explorers
└── logs/                   # Runtime logs (gitignored)
```

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Analyze existing results

```bash
./venv/bin/python main.py --analyze data/combined/combined_results_r2.csv
```

Fitted policy (after round 2, 47 rows):

```
ratio < 0.60  and  idle > 0  →  DD candidate
R_coh < 70                   →  ZNE candidate
both                         →  ZNE+DD
else                         →  BARE
```

Accuracy ≈ **72%**, mean regret ≈ **0.39 pp**.

### Simulation grid

```bash
./venv/bin/python main.py --sim-grid --out data/sim/sim_grid.csv --shots 10000
```

### Fake backends

```bash
./venv/bin/python main.py --fake-grid --out data/fake/fake_backends.csv
```

### Real IBM hardware

```bash
# List backends
./venv/bin/python main.py --list-backends

# Dry-run (transpile only, no queue)
./venv/bin/python main.py --dry-run --backend ibm_fez --triples "0,1,2; 3,4,5"

# Submit
./venv/bin/python main.py --real --backend ibm_fez \
  --triples "0,1,2; 3,4,5" --shots 4000 \
  --out data/real/round2/ibm_fez.csv

# Or interactive helper
./scripts/run_ibm.sh ibm_fez "0,1,2" data/real/round2/ibm_fez.csv
```

Token resolution order: `--token` → `QISKIT_IBM_TOKEN` → `IBM_QUANTUM_TOKEN` → saved Qiskit account.

### Combine CSVs and refit

```bash
./venv/bin/python main.py --combine \
  data/sim/sim_grid_coher_03.csv \
  data/fake/fake_backends.csv \
  data/real/round1/real_all_backends.csv \
  data/real/round2/real_r2_fez.csv \
  data/real/round2/real_r2_marrakesh.csv \
  data/real/round2/real_r2_kingston.csv \
  --out data/combined/combined_results_r2.csv

./venv/bin/python main.py --analyze data/combined/combined_results_r2.csv
```

## Key results (hardware)

**Star result — `ibm_fez` qubits 0-1-2** (ratio≈0.41, R_coh≈62, idle≈460 ns):

| Round | BARE | DD | ZNE | ZNE+DD |
|------:|-----:|---:|----:|-------:|
| 1 | 19.2% | 35.8% | 22.1% | **48.0%** |
| 2 | 29.6% | 39.1% | 38.8% | **52.3%** |

DD alone helps; ZNE alone does little; **together** they more than double bare success — consistent with DD converting slow 1/f noise into something ZNE can extrapolate.

Other findings:
- Short idle (~500 ns) on Kingston/Marrakesh often makes pure **ZNE** better than DD.
- Disconnected triples (e.g. 14-15-16) collapse via SWAP inflation — not a mitigation test.
- Real chips need a higher `R_coh` threshold (~70) than simulation (~35).

Full write-up: [`docs/RESULTS_DIRECTION2.md`](docs/RESULTS_DIRECTION2.md).

## Data dictionary (CSV columns)

| Column | Meaning |
|--------|---------|
| `backend`, `triple` | Device and physical qubit triple |
| `t1_us`, `t2_us`, `ratio` | Coherence; `ratio = T2/(2·T1)` |
| `twoq_err`, `r_coh`, `idle_ns` | 2Q error, `T2/T_circ`, max idle window |
| `p_bare`, `p_dd`, `p_zne`, `p_zne_dd` | Success probabilities (+ `sem_*`) |
| `best` | Empirical oracle (argmax over strategies) |

## Requirements

- Python 3.10+
- Qiskit 2.x, qiskit-aer, qiskit-ibm-runtime (see `requirements.txt`)

## Legacy code

`legacy/` holds earlier prototypes (feature-based adaptive controller, Fake Kyoto/Sherbrooke helpers, `rcoh` explorer). Prefer `main.py` for new experiments.
