#!/usr/bin/env bash
# Run the mitigation harness on real IBM Quantum hardware (SamplerV2).
#
# Setup (once):
#   export QISKIT_IBM_TOKEN="your_token_here"
#   # or save account via QiskitRuntimeService.save_account(...)
#
# Usage (from repo root or this directory):
#   ./scripts/run_ibm.sh
#   ./scripts/run_ibm.sh ibm_fez "0,1,2; 3,4,5" data/real/round2/ibm_fez.csv

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
BACKEND="${1:-ibm_sherbrooke}"
TRIPLES="${2:-0,1,2}"
OUT="${3:-data/real/${BACKEND#ibm_}.csv}"
SHOTS="${SHOTS:-4000}"
DD="${DD:-CPMG2}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

mkdir -p "$(dirname "$OUT")"

echo "=== Preflight (dry-run) ==="
"$PYTHON" main.py --dry-run \
  --backend "$BACKEND" \
  --triples "$TRIPLES" \
  --dd "$DD"

echo ""
read -r -p "Submit to IBM? [y/N] " confirm
case "$confirm" in
  y|Y) ;;
  *) echo "Cancelled."; exit 0 ;;
esac

echo "=== Real run ==="
"$PYTHON" main.py --real \
  --backend "$BACKEND" \
  --triples "$TRIPLES" \
  --dd "$DD" \
  --shots "$SHOTS" \
  --out "$OUT"

echo ""
echo "Done: $OUT"
echo "Analyze: $PYTHON main.py --analyze $OUT"
