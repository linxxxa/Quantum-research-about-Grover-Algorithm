"""

Universal adaptive Grover error-mitigation controller.


INPUT:  любой v2-совместимый бэкенд (Fake* или реальный IBM Quantum).

OUTPUT: одна из 4 стратегий — BARE | ZNE | DD | ZNE+DD.


Реальный бэкенд подключается через QiskitRuntimeService:


    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum", token="<YOUR_TOKEN>")

    backend = service.backend("ibm_brisbane")          # или любой другой

    ctrl = AdaptiveMitigationController(backend)


Для локального запуска на Fake-бэкенде:


    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    ctrl = AdaptiveMitigationController(FakeBrisbane())

"""


from __future__ import annotations


import warnings

from collections import defaultdict

from dataclasses import dataclass, field

from datetime import datetime, timezone

from enum import Enum

from typing import Optional



import numpy as np


from qiskit import QuantumCircuit, transpile

from qiskit.circuit.library import XGate, YGate

from qiskit.transpiler import InstructionDurations, PassManager

from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

from qiskit_aer import AerSimulator

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")

warnings.filterwarnings("ignore", category=UserWarning, module="qiskit")



# ---------------------------------------------------------------------------

# Enums & dataclasses

# ---------------------------------------------------------------------------


class Strategy(Enum):

    BARE   = "BARE"

    ZNE    = "ZNE"

    DD     = "DD"

    ZNE_DD = "ZNE+DD"



@dataclass

class Plan:

    strategy:    Strategy

    dd_seq:      Optional[list] = None

    dd_name:     str            = "None"

    zne_scale:   int            = 1

    diagnostics: dict           = field(default_factory=dict)



# ---------------------------------------------------------------------------

# Circuit builder

# ---------------------------------------------------------------------------


def build_grover_3q(iterations: int = 1) -> QuantumCircuit:

    """3-qubit Grover circuit targeting |111⟩."""

    qc = QuantumCircuit(3)

    qc.h([0, 1, 2])

    for _ in range(iterations):

        qc.barrier()

        qc.ccz(0, 1, 2)

        qc.barrier()

        qc.h([0, 1, 2])

        qc.x([0, 1, 2])

        qc.ccz(0, 1, 2)

        qc.x([0, 1, 2])

        qc.h([0, 1, 2])

    qc.measure_all()

    return qc



# ---------------------------------------------------------------------------

# Controller

# ---------------------------------------------------------------------------


class AdaptiveMitigationController:

    RATIO_DEPHASING = 0.85   # t2 / (2*t1) < this → dephasing dominates → use DD

    COH_TIGHT       = 25     # t2 / t_circ < this → circuit is long vs coherence → use ZNE

    DRIFT_HOURS     = 12     # calibration age threshold

    BASIS           = ["rz", "sx", "x", "y", "cx", "id"]


    def __init__(self, backend, *, drift_mode: str | bool = "auto", seed: int = 42):

        self.backend        = backend

        self.target         = backend.target

        self.dt             = self._resolve_dt()

        self.seed           = seed

        self.t_pi_ns        = self._discover_t_pi_ns()

        self.drift_detected = self._check_drift(drift_mode)


    # ---- helpers -----------------------------------------------------------


    def _resolve_dt(self) -> float:

        """Return dt in seconds; fall back to 0.222 ns if not available."""

        try:

            dt = self.target.dt

            if dt and dt > 0:

                return float(dt)

        except AttributeError:

            pass

        return 0.222e-9


    def _discover_t_pi_ns(self) -> float:

        """Estimate π-pulse duration in nanoseconds from the target."""

        for name, multiplier in (("x", 1.0), ("sx", 2.0), ("rx", 1.0)):

            try:

                op  = self.target[name]

                key = next(iter(op))

                dur = op[key].duration

                if dur:

                    return dur * 1e9 * multiplier

            except (KeyError, AttributeError, StopIteration):

                continue

        return 35.0


    def _check_drift(self, mode: str | bool) -> bool:

        """Return True when calibration data is stale."""

        if isinstance(mode, bool):

            return mode

        # "auto" path — works for both Fake* and real backends

        try:

            props = self.backend.properties()

            if props is None:

                return False

            last = props.last_update_date

            if last is None:

                return False

            # last_update_date may be tz-naive on some backends

            now = datetime.now(timezone.utc)

            if last.tzinfo is None:

                last = last.replace(tzinfo=timezone.utc)

            age_hours = (now - last).total_seconds() / 3600

            return age_hours > self.DRIFT_HOURS

        except (AttributeError, TypeError, Exception):

            return False


    def _t1_t2_for(self, qubits: list[int]) -> tuple[float, float]:

        """Mean T1 / T2 over the given physical qubits (seconds)."""

        qp = self.target.qubit_properties

        t1s, t2s = [], []

        for q in qubits:

            if q < len(qp) and qp[q] is not None:

                if qp[q].t1: t1s.append(qp[q].t1)

                if qp[q].t2: t2s.append(qp[q].t2)

        t1 = float(np.mean(t1s)) if t1s else 100e-6

        t2 = float(np.mean(t2s)) if t2s else 100e-6

        t2 = min(t2, 2 * t1)   # physical constraint T2 ≤ 2*T1

        return t1, t2


    def _circuit_duration_s(self, qc_t: QuantumCircuit) -> float:

        """

        Circuit duration in seconds.


        qc.duration is deprecated in Qiskit ≥ 1.3 and removed in 3.0.

        We compute it from op_start_times when available, otherwise fall

        back to the deprecated attribute with a suppressed warning.

        """

        # Preferred: use op_start_times (populated after ALAP scheduling)

        starts = getattr(qc_t, "op_start_times", None)

        if starts and len(starts):

            # Find the finish time of the last instruction

            max_end = 0

            for start, inst in zip(starts, qc_t.data):

                dur = getattr(inst.operation, "duration", 0) or 0

                max_end = max(max_end, start + dur)

            if max_end > 0:

                return max_end * self.dt


        # Fallback: deprecated .duration property

        try:

            with warnings.catch_warnings():

                warnings.simplefilter("ignore", DeprecationWarning)

                dur = qc_t.duration

            if dur:

                return float(dur) * self.dt

        except Exception:

            pass

        return 0.0


    @staticmethod

    def _max_idle_window_dt(qc_t: QuantumCircuit) -> int:

        """Largest idle gap (in dt units) across all qubits after ALAP scheduling."""

        starts = getattr(qc_t, "op_start_times", None)

        # Explicit None/empty check — avoids falsiness of [0, ...] lists

        if starts is None or len(starts) == 0:

            return 0


        busy: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for start, inst in zip(starts, qc_t.data):

            if inst.operation.name in ("barrier", "measure", "delay"):

                continue

            dur = getattr(inst.operation, "duration", 0) or 0

            for q in inst.qubits:

                busy[qc_t.find_bit(q).index].append((start, start + dur))


        max_gap = 0

        for ivs in busy.values():

            ivs.sort()

            for a, b in zip(ivs, ivs[1:]):

                gap = b[0] - a[1]

                if gap > max_gap:

                    max_gap = gap

        return max_gap


    @staticmethod

    def _used_physical_qubits(qc_t: QuantumCircuit) -> list[int]:

        """

        Physical qubits that carry real gates (barriers, delays, and measures

        are excluded — ALAP inserts delays on every qubit, so including them

        would return all 127 qubits of a large fake backend).

        """

        used = set()

        for inst in qc_t.data:

            if inst.operation.name in ("barrier", "delay", "measure"):

                continue

            for q in inst.qubits:

                used.add(qc_t.find_bit(q).index)

        return sorted(used)


    # ---- strategy selection -----------------------------------------------


    def select_strategy(self, qc_raw: QuantumCircuit) -> tuple[QuantumCircuit, Plan]:

        qc_t = transpile(

            qc_raw, self.backend,

            scheduling_method="alap",

            seed_transpiler=self.seed,

            optimization_level=1,

        )


        used   = self._used_physical_qubits(qc_t)

        t1, t2 = self._t1_t2_for(used)

        ratio  = t2 / (2 * t1)


        t_circ_s = self._circuit_duration_s(qc_t)

        r_coh    = (t2 / t_circ_s) if t_circ_s > 0 else 1e3


        idle_ns = self._max_idle_window_dt(qc_t) * self.dt * 1e9


        room_xy4  = idle_ns >= 4 * self.t_pi_ns + 20

        room_hahn = idle_ns >= self.t_pi_ns + 10


        need_dd  = (ratio < self.RATIO_DEPHASING) and room_hahn

        need_zne = (r_coh < self.COH_TIGHT) or self.drift_detected


        if   need_dd and need_zne: strat = Strategy.ZNE_DD

        elif need_dd:              strat = Strategy.DD

        elif need_zne:             strat = Strategy.ZNE

        else:                      strat = Strategy.BARE


        if need_dd and room_xy4:

            # XY-4: suppresses both dephasing and amplitude errors

            dd_seq, dd_name = [XGate(), YGate(), XGate(), YGate()], "XY-4"

        elif need_dd:

            # Hahn echo: simpler, fits in tighter idle windows

            dd_seq, dd_name = [XGate()], "Hahn"

        else:

            dd_seq, dd_name = None, "None"


        zne_scale = 3 if need_zne else 1


        plan = Plan(

            strategy=strat, dd_seq=dd_seq, dd_name=dd_name,

            zne_scale=zne_scale,

            diagnostics=dict(

                t1_us=t1 * 1e6, t2_us=t2 * 1e6,

                ratio=ratio, r_coh=r_coh, idle_ns=idle_ns,

                drift=self.drift_detected, used_qubits=used,

            ),

        )

        return qc_t, plan


    # ---- noise model for local simulation --------------------------------


    def build_lean_simulator(self, used: list[int]) -> AerSimulator:

        """

        Build a compact 3-qubit AerSimulator that mirrors the noise of the

        physical qubits in `used`.

        """

        qp = self.target.qubit_properties


        # 1Q gate duration

        x_dur = 60e-9

        try:

            x_dur = self.target["x"][(used[0],)].duration or x_dur

        except (KeyError, AttributeError):

            pass


        # 2Q gate: prefer ECR (IBM Heron/Falcon), then CZ, then CX

        twoq_name = next(

            (n for n in ("ecr", "cz", "cx") if n in self.target.operation_names),

            None,

        )

        twoq_dur, twoq_err = 600e-9, 1e-2

        if twoq_name:

            try:

                kk       = next(iter(self.target[twoq_name]))

                twoq_dur = self.target[twoq_name][kk].duration or twoq_dur

                twoq_err = self.target[twoq_name][kk].error   or twoq_err

            except (KeyError, StopIteration):

                pass


        nm = NoiseModel(basis_gates=self.BASIS)


        # 1Q noise: thermal relaxation during a π-pulse

        for log_q, phys_q in enumerate(used[:3]):

            if phys_q < len(qp) and qp[phys_q] is not None:

                t1 = qp[phys_q].t1 or 100e-6

                t2 = min(qp[phys_q].t2 or 100e-6, 2 * t1)

            else:

                t1, t2 = 100e-6, 100e-6

            err1 = thermal_relaxation_error(t1, t2, x_dur)

            for g in ("sx", "x", "y"):   # Y is also a π-pulse → same model

                nm.add_quantum_error(err1, g, [log_q])


        # 2Q noise: thermal relaxation + depolarizing on every ordered pair

        for i in range(min(3, len(used))):

            for j in range(min(3, len(used))):

                if i == j:

                    continue

                pi, pj = used[i], used[j]


                t1i = (qp[pi].t1 if pi < len(qp) and qp[pi] else 100e-6)

                t2i = min((qp[pi].t2 if pi < len(qp) and qp[pi] else 100e-6), 2 * t1i)

                t1j = (qp[pj].t1 if pj < len(qp) and qp[pj] else 100e-6)

                t2j = min((qp[pj].t2 if pj < len(qp) and qp[pj] else 100e-6), 2 * t1j)


                relax = thermal_relaxation_error(t1i, t2i, twoq_dur).expand(

                    thermal_relaxation_error(t1j, t2j, twoq_dur)

                )

                depol = depolarizing_error(min(twoq_err, 0.1), 2)

                nm.add_quantum_error(relax.compose(depol), "cx", [i, j])


        return AerSimulator(noise_model=nm, basis_gates=self.BASIS)


    # ---- execution -------------------------------------------------------


    def run_plan(

        self, qc_t: QuantumCircuit, plan: Plan,

        *, shots: int = 8192, target_state: str = "111",

    ) -> dict:

        """

        Execute the chosen mitigation plan on a local noise-model simulator.


        ZNE extrapolation: Richardson linear extrapolation at scale factors

        1 and 3.  If the extrapolated value is *worse* than the raw result,

        we fall back to the raw (scale=1) value — this prevents ZNE from

        hurting on short circuits where noise amplification dominates.

        """

        used = plan.diagnostics["used_qubits"]

        sim  = self.build_lean_simulator(used)


        # Strip measurements before we manipulate the circuit

        compact = self._extract_active(

            qc_t.remove_final_measurements(inplace=False), used[:3]

        )


        # Build the inverse *before* transpiling to basis gates so that

        # high-level gates (CCX, etc.) have reliable inverses.

        compact_inv = compact.inverse()


        def build(scale: int) -> QuantumCircuit:

            if scale % 2 == 0:

                raise ValueError(

                    f"ZNE scale factor must be odd (got {scale}). "

                    "Even values fold the circuit an odd number of times, "

                    "yielding scale-1 noise instead of scale×."

                )

            c = compact.copy()

            for _ in range((scale - 1) // 2):

                c = c.compose(compact_inv).compose(compact)


            # Transpile to basis *before* DD so the scheduler sees real gates

            c = transpile(

                c, basis_gates=self.BASIS,

                optimization_level=0,

                seed_transpiler=self.seed,

            )


            if plan.dd_seq is not None:

                dt_ns = self.dt * 1e9

                durs  = InstructionDurations(

                    [

                        ("x",  None, int(round(60  / dt_ns))),

                        ("y",  None, int(round(60  / dt_ns))),  # Y = π-pulse, same as X

                        ("sx", None, int(round(35  / dt_ns))),

                        ("rz", None, 0),

                        ("id", None, int(round(60  / dt_ns))),

                        ("cx", None, int(round(660 / dt_ns))),

                    ],

                    dt=self.dt,

                )

                pm = PassManager([

                    ALAPScheduleAnalysis(durs),

                    PadDynamicalDecoupling(durs, plan.dd_seq),

                ])

                c = pm.run(c)


            c.measure_all()

            return c


        if plan.zne_scale > 1:

            p1   = self._exec(sim, build(1),              shots, target_state)

            p3   = self._exec(sim, build(plan.zne_scale), shots, target_state)


            # Richardson linear extrapolation to zero noise

            p_zne_raw = (3 * p1 - p3) / 2


            # Safety clamp: never report worse than the raw (scale=1) run

            # and never exceed 1.0 or go below 0.0.

            p_zne = float(np.clip(p_zne_raw, p1, 1.0))


            return dict(probability=p_zne, p_s1=p1, p_s3=p3,

                        p_zne_raw=p_zne_raw)


        return dict(probability=self._exec(sim, build(1), shots, target_state))


    def baseline(

        self, qc_t: QuantumCircuit, plan: Plan,

        *, shots: int = 8192, target_state: str = "111",

    ) -> float:

        """Noisy execution without any mitigation (for comparison)."""

        used    = plan.diagnostics["used_qubits"]

        sim     = self.build_lean_simulator(used)

        compact = self._extract_active(

            qc_t.remove_final_measurements(inplace=False), used[:3]

        )

        c = transpile(compact, basis_gates=self.BASIS,

                      optimization_level=0, seed_transpiler=self.seed)

        c.measure_all()

        return self._exec(sim, c, shots, target_state)


    # ---- static helpers --------------------------------------------------


    @staticmethod

    def _extract_active(

        qc_full: QuantumCircuit, used: list[int]

    ) -> QuantumCircuit:

        """

        Re-index a full transpiled circuit to a compact N-qubit circuit

        containing only the gates on qubits in `used`.

        """

        idx_map = {phys: log for log, phys in enumerate(used)}

        out     = QuantumCircuit(len(used))

        for inst in qc_full.data:

            if inst.operation.name in ("barrier", "delay"):

                continue

            phys = [qc_full.find_bit(q).index for q in inst.qubits]

            if all(p in idx_map for p in phys):

                out.append(inst.operation, [idx_map[p] for p in phys])

            else:

                warnings.warn(

                    f"Gate '{inst.operation.name}' on qubits {phys} was dropped: "

                    f"not all qubits are in active set {list(idx_map)}. "

                    "This may indicate a qubit mapping mismatch.",

                    stacklevel=2,

                )

        return out


    def _exec(

        self, sim: AerSimulator, qc: QuantumCircuit,

        shots: int, target_state: str,

    ) -> float:

        res = sim.run(qc, shots=shots, seed_simulator=self.seed).result()

        return res.get_counts().get(target_state, 0) / shots



# ---------------------------------------------------------------------------

# Entry point — Fake backends demo + optional real IBM backend

# ---------------------------------------------------------------------------


def _run_demo_fake() -> None:

    """Run the controller on all available Fake backends."""

    from qiskit_ibm_runtime.fake_provider import (

        FakeBrisbane, FakeKyoto, FakeOsaka, FakeSherbrooke, FakeTorino,

    )


    fakes = [

        ("FakeOsaka",       FakeOsaka()),

        ("FakeBrisbane",    FakeBrisbane()),

        ("FakeKyoto",       FakeKyoto()),

        ("FakeSherbrooke",  FakeSherbrooke()),

        ("FakeTorino",      FakeTorino()),

    ]


    grover = build_grover_3q(iterations=1)

    p_th   = np.sin(3 * np.arcsin(1 / np.sqrt(8))) ** 2


    print(f"Теоретический максимум |111⟩: {p_th * 100:.2f}%\n")

    _print_header()


    for name, fake in fakes:

        _run_one(name, fake, grover)



def _run_real_backend(token: str, backend_name: str = "ibm_brisbane") -> None:

    """

    Connect to a real IBM Quantum backend via QiskitRuntimeService.


    Parameters

    ----------

    token        : Your IBM Quantum API token (find at quantum.ibm.com → Account).

    backend_name : Name of the backend, e.g. "ibm_brisbane", "ibm_kyoto",

                   "ibm_sherbrooke".  Run service.backends() to see available.

    """

    try:

        from qiskit_ibm_runtime import QiskitRuntimeService

    except ImportError:

        print(

            "qiskit-ibm-runtime не установлен.\n"

            "Установите: pip install qiskit-ibm-runtime"

        )

        return


    print(f"Подключаемся к IBM Quantum ({backend_name})…")

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    backend = service.backend(backend_name)

    print(f"Подключено: {backend.name}  —  {backend.num_qubits} кубитов\n")


    grover = build_grover_3q(iterations=1)

    p_th   = np.sin(3 * np.arcsin(1 / np.sqrt(8))) ** 2


    print(f"Теоретический максимум |111⟩: {p_th * 100:.2f}%\n")

    _print_header()

    _run_one(backend.name, backend, grover)



# ---- formatting helpers --------------------------------------------------


def _print_header() -> None:

    print(

        f"{'backend':<18}{'strategy':<10}{'DD':<10}{'ZNE':<5}"

        f"{'ratio':<7}{'r_coh':<7}{'idle,ns':<9}"

        f"{'base,%':<9}{'mit,%':<9}{'Δ pp':<7}"

    )

    print("-" * 95)



def _run_one(name: str, backend, grover: QuantumCircuit) -> None:

    ctrl    = AdaptiveMitigationController(backend, drift_mode=False)

    qc_t, plan = ctrl.select_strategy(grover)

    p_base  = ctrl.baseline(qc_t, plan)

    result  = ctrl.run_plan(qc_t, plan)

    p_mit   = result["probability"]

    d       = plan.diagnostics


    extras = ""

    if "p_zne_raw" in result and result["p_zne_raw"] != p_mit:

        extras = f"  [ZNE raw={result['p_zne_raw']*100:.2f}% → clamped]"


    print(

        f"{name:<18}{plan.strategy.value:<10}{plan.dd_name:<10}"

        f"{plan.zne_scale:<5}{d['ratio']:<7.2f}{d['r_coh']:<7.1f}"

        f"{d['idle_ns']:<9.0f}{p_base*100:<9.2f}{p_mit*100:<9.2f}"

        f"{(p_mit - p_base)*100:+.2f}{extras}"

    )



# ---------------------------------------------------------------------------


if __name__ == "__main__":

    import sys


    if len(sys.argv) == 3 and sys.argv[1] == "--real":

        # Usage: python grover_adaptive_controller.py --real <TOKEN> [backend_name]

        token        = sys.argv[2]

        backend_name = sys.argv[3] if len(sys.argv) > 3 else "ibm_brisbane"

        _run_real_backend(token, backend_name)

    elif len(sys.argv) == 4 and sys.argv[1] == "--real":

        token, backend_name = sys.argv[2], sys.argv[3]

        _run_real_backend(token, backend_name)

    else:

        # Default: demo with Fake backends

        _run_demo_fake()