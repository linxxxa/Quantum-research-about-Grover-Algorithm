#!/usr/bin/env python3
"""Submit Kingston round-2 triples, wait for DONE, write real_r2_kingston.csv."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("lab", ROOT / "main.py")
lab = importlib.util.module_from_spec(spec)
sys.modules["lab"] = lab
spec.loader.exec_module(lab)

TRIPLES = [(44, 45, 46), (23, 24, 25), (0, 1, 2)]
SHOTS = 4000
OUT = ROOT / "data" / "real" / "round2" / "real_r2_kingston.csv"
STATE = ROOT / "data" / "real" / "round2" / "real_r2_kingston_jobs.json"
EXISTING_44 = "d9cbg741osis73bjl790"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    grover = lab.build_grover_3q(iterations=1)

    jobs: dict[str, str] = {}
    if STATE.exists():
        jobs = json.loads(STATE.read_text())

    try:
        j = service.job(EXISTING_44)
        print(f"reuse {EXISTING_44} status={j.status()}", flush=True)
        jobs["44-45-46"] = EXISTING_44
    except Exception as exc:
        print(f"reuse fail: {exc}", flush=True)

    def submit(triple: tuple[int, int, int]) -> str:
        built = lab.build_strategy_circuits(
            grover, backend, triple=triple, dd_name="CPMG2")
        # Без Batch: иначе __exit__ блокируется, пока job в очереди Kingston.
        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        job = sampler.run(
            [(c,) for c in [built.c1, built.c1_dd, built.c3, built.c3_dd]],
            shots=SHOTS,
        )
        jid = job.job_id()
        print(f"submitted {triple} -> {jid}", flush=True)
        return jid

    for t in TRIPLES:
        key = "-".join(map(str, t))
        if key in jobs:
            print(f"already have job for {key}: {jobs[key]}", flush=True)
            continue
        jobs[key] = submit(t)
        STATE.write_text(json.dumps(jobs, indent=2))

    STATE.write_text(json.dumps(jobs, indent=2))
    print("jobs:", jobs, flush=True)

    while True:
        statuses = {k: str(service.job(jid).status()) for k, jid in jobs.items()}
        print(f"status {statuses}", flush=True)
        if all("DONE" in s or s in ("ERROR", "CANCELLED") for s in statuses.values()):
            break
        time.sleep(45)

    rows = []
    for t in TRIPLES:
        key = "-".join(map(str, t))
        job = service.job(jobs[key])
        if "DONE" not in str(job.status()):
            print(f"skip {key} status={job.status()}", flush=True)
            continue
        probs = [
            lab._prob_and_sem(lab._counts_from_pub(pub), SHOTS)
            for pub in job.result()
        ]
        (p1, s1), (p1d, s1d), (p3, s3), (p3d, s3d) = probs
        p_zne, s_zne = lab.richardson_zne(p1, s1, p3, s3)
        p_zne_dd, s_zne_dd = lab.richardson_zne(p1d, s1d, p3d, s3d)
        built = lab.build_strategy_circuits(
            grover, backend, triple=t, dd_name="CPMG2")
        feats = lab.extract_features(backend, t, built.t_circ_s, built.idle_ns)
        row = lab.Row(
            features=feats, dd_name="CPMG2",
            p_bare=p1, sem_bare=s1, p_dd=p1d, sem_dd=s1d,
            p_zne=p_zne, sem_zne=s_zne, p_zne_dd=p_zne_dd, sem_zne_dd=s_zne_dd,
            p_theory=lab.grover_theoretical_max(),
        )
        lab._print_row(row)
        rows.append(row)

    lab.write_csv(rows, str(OUT))
    print(f"Wrote {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
