#!/usr/bin/env bash
# Wait for Kingston round-2 jobs, then fetch / combine / analyze.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG=logs/real_r2_kingston_wait.log
CSV=data/real/round2/real_r2_kingston.csv
mkdir -p logs data/real/round2 data/combined

echo "START $(date)" | tee "$LOG"

while true; do
  if [[ -f "$CSV" ]]; then
    echo "CSV READY $(date)" | tee -a "$LOG"
    break
  fi

  ./venv/bin/python -u -c '
from qiskit_ibm_runtime import QiskitRuntimeService
import sys
s = QiskitRuntimeService()
jobs = {
    "44-45-46": "d9cbg741osis73bjl790",
    "23-24-25": "d9cbvovngvls73a9977g",
    "0-1-2": "d9cc01nngvls73a997h0",
}
st = {k: str(s.job(j).status()) for k, j in jobs.items()}
print(st, flush=True)
sys.exit(0 if not all("DONE" in v for v in st.values()) else 42)
' >>"$LOG" 2>&1
  rc=$?

  if [[ $rc -eq 42 ]]; then
    echo "ALL DONE $(date)" | tee -a "$LOG"
    ./venv/bin/python -u scripts/fetch_kingston_r2.py >>"$LOG" 2>&1
    ./venv/bin/python main.py --combine \
      data/sim/sim_grid_coher_03.csv \
      data/fake/fake_backends.csv \
      data/real/round1/real_all_backends.csv \
      data/real/round2/real_r2_fez.csv \
      data/real/round2/real_r2_marrakesh.csv \
      data/real/round2/real_r2_kingston.csv \
      --out data/combined/combined_results_r2.csv >>"$LOG" 2>&1
    ./venv/bin/python main.py --analyze data/combined/combined_results_r2.csv >>"$LOG" 2>&1
    echo "ANALYZE DONE $(date)" | tee -a "$LOG"
    break
  fi

  echo "$(date +%H:%M:%S) waiting (kingston queue)" | tee -a "$LOG"
  sleep 120
done

echo "EXIT $(date)" | tee -a "$LOG"
