"""

grover_mitigation_lab.py

========================


Измерительный стенд для исследования "когда митигация помогает алгоритму

Гровера, а когда вредит" — рамка физика-экспериментатора (Направление 2).


ОТЛИЧИЕ ОТ adaptive_controller

------------------------------

Старый код принимал решение (BARE/ZNE/DD/ZNE+DD) по калибровочным числам и

"проверял" его на локальном симуляторе, построенном из тех же чисел. Это

замкнутый круг: модель тестировалась против самой себя, реального исхода не

было вообще.


Здесь измерение и решение РАЗВЕДЕНЫ:


  1. Harness гоняет ВСЕ ЧЕТЫРЕ стратегии безусловно, на каждом наборе кубитов,

     на РЕАЛЬНОМ железе через SamplerV2 (или на честной шумовой симуляции).

  2. Для каждой точки пишется строка: признаки железа (T1, T2, ratio, ошибка

     2Q, R_coh, idle) -> вероятность успеха каждой стратегии + её ошибка (SEM).

  3. Решающее правило (features -> strategy) — ТОНКАЯ чистая функция, которую

     ты оцениваешь (accuracy, regret) и ПОДГОНЯЕШЬ по собранной таблице, а не

     зашиваешь пороги руками. Это и есть "модель" Направления 2.


ФИЗИЧЕСКАЯ ОГОВОРКА ПРО DD И СИМУЛЯЦИЮ

-------------------------------------

Aer thermal_relaxation — марковский (без памяти) шум. DD работает против

МЕДЛЕННОГО/коррелированного дефазирования, поэтому против марковского шума DD

выигрыша не даёт ПО ОПРЕДЕЛЕНИЮ. Вывод, зашитый в архитектуру:

  * ось DD валидируется ТОЛЬКО на реальном железе (там есть 1/f-шум);

  * симуляционная сетка честно используется для осей ZNE и R_coh;

  * для стилизованной проверки DD в симуляции есть опциональная инъекция

    КОГЕРЕНТНОГО дефазирования (coherent_dephasing) — но это модель, не железо.


Запуск

------

  # Реальное железо (токен из --token, QISKIT_IBM_TOKEN или сохранённого аккаунта)

  python main.py --list-backends --token $QISKIT_IBM_TOKEN

  python main.py --real --backend ibm_sherbrooke --triples "0,1,2; 3,4,5" \

      --out ibm_sherbrooke.csv --shots 4000

  python main.py --dry-run --backend ibm_kyoto --triples "0,1,2"


  # Симуляционная сетка (для осей ZNE/R_coh)

  python grover_mitigation_lab.py --sim-grid


  # Подгонка и оценка решающего правила по собранной таблице

  python grover_mitigation_lab.py --analyze results.csv

"""


from __future__ import annotations


import argparse

import csv

import itertools

import math

import os

import sys

import warnings

from dataclasses import dataclass, field, asdict

from datetime import datetime, timezone

from typing import Callable, Iterable, Optional, Sequence


import numpy as np


from qiskit import QuantumCircuit

from qiskit.circuit.library import XGate, YGate

from qiskit.transpiler import InstructionDurations, PassManager

from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling, PadDelay

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")

warnings.filterwarnings("ignore", category=UserWarning, module="qiskit")



# ===========================================================================

# 1. Алгоритм и стратегии

# ===========================================================================


TARGET_STATE = "111"          # ищем |111> в 3-кубитном Гровере

N_STATES = 8                  # 2**3



def build_grover_3q(iterations: int = 1) -> QuantumCircuit:

    """3-кубитный Гровер, цель |111>. Без измерений (их добавляем позже)."""

    qc = QuantumCircuit(3, name=f"grover_k{iterations}")

    qc.h([0, 1, 2])

    for _ in range(iterations):

        qc.barrier()

        qc.ccz(0, 1, 2)                 # оракул для |111>

        qc.barrier()

        qc.h([0, 1, 2])                 # диффузор

        qc.x([0, 1, 2])

        qc.ccz(0, 1, 2)

        qc.x([0, 1, 2])

        qc.h([0, 1, 2])

    return qc



def grover_theoretical_max(iterations: int = 1, n_states: int = N_STATES) -> float:

    """Идеальная вероятность успеха после k итераций амплитудной амплификации."""

    theta = math.asin(1.0 / math.sqrt(n_states))

    return math.sin((2 * iterations + 1) * theta) ** 2



# --- DD-последовательности -------------------------------------------------

# На IBM нативны только rz/sx/x/(ecr|cz). X-only последовательности безопасны.

# XY-4 включает Y (не нативен) — оставлен опцией с оговоркой в докстринге.

DD_SEQUENCES: dict[str, list] = {

    "Hahn":  [XGate()],                          # один pi-импульс

    "CPMG2": [XGate(), XGate()],                 # CPMG, только нативный X

    "XY4":   [XGate(), YGate(), XGate(), YGate()],  # требует калибровки Y

}



def fold_global(qc_no_meas: QuantumCircuit, scale: int) -> QuantumCircuit:

    """

    Глобальный unitary folding для ZNE: U -> U (U^dag U)^k, scale = 2k+1.

    Логически эквивалентно U, но время/шум растут в scale раз.

    """

    if scale % 2 == 0 or scale < 1:

        raise ValueError(f"scale must be odd >= 1, got {scale}")

    folded = qc_no_meas.copy()

    inv = qc_no_meas.inverse()

    for _ in range((scale - 1) // 2):

        folded = folded.compose(inv).compose(qc_no_meas)

    return folded



# ===========================================================================

# 2. Признаки железа (физические переменные исследования)

# ===========================================================================


@dataclass

class Features:

    backend: str

    triple: tuple[int, int, int]

    t1_us: float

    t2_us: float

    ratio: float          # T2 / (2 T1) — < 1 значит дефазировка доминирует

    twoq_err: float       # ошибка нативного 2Q гейта (ecr/cz/cx)

    r_coh: float          # T2 / t_circ — запас когерентности относительно схемы

    idle_ns: float        # макс. окно простоя после ALAP (есть ли куда ставить DD)

    cal_age_h: float      # возраст калибровки в часах (дрейф)



def _two_qubit_op_name(target) -> Optional[str]:

    for n in ("ecr", "cz", "cx"):

        if n in target.operation_names:

            return n

    return None



def extract_features(backend, triple: Sequence[int], t_circ_s: float,

                     idle_ns: float) -> Features:

    """Снять физические признаки для конкретной тройки кубитов конкретного чипа."""

    target = backend.target

    qp = target.qubit_properties


    t1s, t2s = [], []

    for q in triple:

        if q < len(qp) and qp[q] is not None:

            if qp[q].t1:

                t1s.append(qp[q].t1)

            if qp[q].t2:

                t2s.append(qp[q].t2)

    t1 = float(np.mean(t1s)) if t1s else 100e-6

    t2 = float(np.mean(t2s)) if t2s else 100e-6

    t2 = min(t2, 2 * t1)                      # физическое ограничение T2 <= 2 T1

    ratio = t2 / (2 * t1)


    # ошибка двухкубитного гейта на рёбрах внутри тройки (среднее по доступным)

    twoq_name = _two_qubit_op_name(target)

    errs = []

    if twoq_name:

        for a, b in itertools.permutations(triple, 2):

            try:

                props = target[twoq_name].get((a, b))

                if props is not None and props.error:

                    errs.append(props.error)

            except (KeyError, AttributeError):

                continue

    twoq_err = float(np.mean(errs)) if errs else float("nan")


    r_coh = (t2 / t_circ_s) if t_circ_s > 0 else float("inf")


    # возраст калибровки (совместимо с V2 Backend архитектурой)

    cal_age_h = float("nan")

    try:

        last = getattr(backend, "last_update_date", None)

        if last is None and hasattr(backend, "properties"):

            props = backend.properties()

            if props is not None:

                last = getattr(props, "last_update_date", None)

        if last is not None:

            if last.tzinfo is None:

                last = last.replace(tzinfo=timezone.utc)

            cal_age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600

    except Exception:

        pass


    return Features(

        backend=getattr(backend, "name", "sim"),

        triple=tuple(triple),

        t1_us=t1 * 1e6, t2_us=t2 * 1e6, ratio=ratio,

        twoq_err=twoq_err, r_coh=r_coh, idle_ns=idle_ns, cal_age_h=cal_age_h,

    )



# ===========================================================================

# 3. Сборка исполняемых схем под каждую стратегию

# ===========================================================================

#

# Минимальный набор из 4 схем покрывает все 4 стратегии:

#   c1     : scale=1, без DD          -> BARE,    и scale-1 для ZNE

#   c1_dd  : scale=1, с DD            -> DD,      и scale-1 для ZNE+DD

#   c3     : scale=3, без DD          -> scale-3 для ZNE

#   c3_dd  : scale=3, с DD            -> scale-3 для ZNE+DD

#

# Порядок операций строгий: fold -> transpile к ISA -> ALAP+DD.

# DD ставится ПОСЛЕ фолдинга, чтобы защищать реально исполняемые гейты.


@dataclass

class BuiltCircuits:

    c1: QuantumCircuit

    c1_dd: QuantumCircuit

    c3: QuantumCircuit

    c3_dd: QuantumCircuit

    t_circ_s: float

    idle_ns: float

    dd_name: str



def _durations_from(backend_or_durations) -> InstructionDurations:

    """InstructionDurations: с реального target берём как есть, для симуляции — модель."""

    if isinstance(backend_or_durations, InstructionDurations):

        return backend_or_durations

    try:

        durs = backend_or_durations.target.durations()

        if durs is not None:

            return durs

    except (AttributeError, TypeError):

        pass

    # запасной набор (нс приведены к dt=0.5ns условно) — только для симуляции

    return InstructionDurations(

        [("x", None, 60, "ns"), ("y", None, 60, "ns"), ("sx", None, 35, "ns"),

         ("rz", None, 0, "ns"), ("id", None, 60, "ns"),

         ("ecr", None, 660, "ns"), ("cz", None, 660, "ns"),

         ("cx", None, 660, "ns"), ("delay", None, 0, "ns")],

        dt=0.5e-9,

    )



def _measure_logical_qubits(isa: QuantumCircuit, n_logical: int = 3,

                            phys_indices: Optional[Sequence[int]] = None

                            ) -> QuantumCircuit:

    """Измерить только логические кубиты схемы, не весь регистр устройства."""

    from qiskit import ClassicalRegister

    layout = isa.layout

    if phys_indices is not None:

        phys = list(phys_indices[:n_logical])

    elif layout is not None and layout.final_index_layout() is not None:

        phys = layout.final_index_layout()[:n_logical]

    else:

        phys = list(range(n_logical))

    creg = ClassicalRegister(n_logical, "meas")

    isa.add_register(creg)

    for i, q in enumerate(phys):

        isa.measure(q, creg[i])

    return isa



def _schedule_and_pad_dd(isa: QuantumCircuit, durations: InstructionDurations,

                         dd_seq: list) -> QuantumCircuit:

    passes = [ALAPScheduleAnalysis(durations)]

    if dd_seq:

        passes.append(PadDynamicalDecoupling(durations, dd_seq))

    else:

        passes.append(PadDelay(durations=durations))   # BARE тоже получает delay → честный idle-шум

    return PassManager(passes).run(isa)



def _circuit_duration_s(scheduled: QuantumCircuit, dt: float) -> float:

    starts = getattr(scheduled, "op_start_times", None)

    if starts and len(starts):

        max_end = 0

        for start, inst in zip(starts, scheduled.data):

            dur = getattr(inst.operation, "duration", 0) or 0

            max_end = max(max_end, start + dur)

        return max_end * dt

    return 0.0



def _max_idle_dt(scheduled: QuantumCircuit) -> int:

    starts = getattr(scheduled, "op_start_times", None)

    if not starts:

        return 0

    busy: dict[int, list[tuple[int, int]]] = {}

    for start, inst in zip(starts, scheduled.data):

        if inst.operation.name in ("barrier", "measure", "delay"):

            continue

        dur = getattr(inst.operation, "duration", 0) or 0

        for q in inst.qubits:

            idx = scheduled.find_bit(q).index

            busy.setdefault(idx, []).append((start, start + dur))

    max_gap = 0

    for ivs in busy.values():

        ivs.sort()

        for a, b in zip(ivs, ivs[1:]):

            max_gap = max(max_gap, b[0] - a[1])

    return max_gap



def build_strategy_circuits(grover: QuantumCircuit, backend, *,

                            triple: Sequence[int], dd_name: str = "CPMG2",

                            opt_level: int = 1,

                            sim_durations: Optional[InstructionDurations] = None

                            ) -> BuiltCircuits:

    """

    Собрать 4 исполняемые ISA-схемы для тройки кубитов.

    `backend` — реальный бэкенд или Aer-бэкенд; `triple` фиксирует физический layout.

    """

    dd_seq = DD_SEQUENCES[dd_name]

    durations = sim_durations if sim_durations is not None else _durations_from(backend)

    dt = getattr(getattr(backend, "target", None), "dt", None) or 0.5e-9


    pm = generate_preset_pass_manager(

        backend=backend, optimization_level=opt_level,

        initial_layout=list(triple), seed_transpiler=42,

    )


    def make(scale: int, with_dd: bool) -> QuantumCircuit:

        folded = fold_global(grover, scale)

        isa = pm.run(folded)

        if isa.layout is not None and isa.layout.final_index_layout() is not None:

            meas_phys = isa.layout.final_index_layout()[:3]

        else:

            meas_phys = list(triple)

        if with_dd:

            isa = _schedule_and_pad_dd(isa, durations, dd_seq)

        else:

            isa = _schedule_and_pad_dd(isa, durations, [])  # только ALAP, без импульсов

        isa = _measure_logical_qubits(isa, phys_indices=meas_phys)

        return isa


    c1 = make(1, False)

    c1_dd = make(1, True)

    c3 = make(3, False)

    c3_dd = make(3, True)


    t_circ_s = _circuit_duration_s(c1, dt)

    idle_ns = _max_idle_dt(c1) * dt * 1e9


    return BuiltCircuits(c1, c1_dd, c3, c3_dd, t_circ_s, idle_ns, dd_name)



# ===========================================================================

# 4. Исполнители: реальное железо (SamplerV2) и честная симуляция

# ===========================================================================


def _counts_from_pub(pub_result) -> dict:

    """Достать counts из PubResult SamplerV2 без привязки к жестким именам."""

    data = pub_result.data

    if len(data) == 0:

        raise RuntimeError("В PubResult отсутствуют данные измерений.")

    # Просто берем первый доступный классический регистр из DataBin, как бы его ни назвал компилятор

    first_reg_name = list(data.keys())[0]

    return getattr(data, first_reg_name).get_counts()



def _prob_and_sem(counts: dict, shots: int, target: str = TARGET_STATE

                  ) -> tuple[float, float]:

    hit = counts.get(target, 0)

    p = hit / shots

    sem = math.sqrt(max(p * (1 - p), 0) / shots)   # шумовая ошибка доли

    return p, sem



class HardwareExecutor:

    """Исполнение на реальном бэкенде IBM через SamplerV2 (один Batch на тройку)."""


    def __init__(self, backend, shots: int = 10000):

        from qiskit_ibm_runtime import SamplerV2, Batch  # импорт по требованию

        self._SamplerV2 = SamplerV2

        self._Batch = Batch

        self.backend = backend

        self.shots = shots


    def run(self, circuits: list[QuantumCircuit]) -> list[tuple[float, float]]:

        with self._Batch(backend=self.backend) as batch:

            sampler = self._SamplerV2(mode=batch)

            # ВАЖНО: отключаем встроенный DD рантайма — измеряем НАШ явный DD

            sampler.options.dynamical_decoupling.enable = False

            sampler.options.twirling.enable_gates = False

            sampler.options.twirling.enable_measure = False

            job = sampler.run([(c,) for c in circuits], shots=self.shots)

            jid = job.job_id() if callable(getattr(job, "job_id", None)) else job.job_id

            print(f"    job_id={jid}  circuits={len(circuits)}  shots={self.shots}",

                  flush=True)

            res = job.result()

        out = []

        for pub in res:

            counts = _counts_from_pub(pub)

            out.append(_prob_and_sem(counts, self.shots))

        return out



@dataclass

class NoiseSpec:

    """Параметры стилизованной шумовой модели для симуляционной сетки."""

    t1_us: float = 100.0

    t2_us: float = 80.0

    twoq_err: float = 0.01

    readout_err: float = 0.02

    coherent_dephasing_rad_per_us: float = 0.0   # >0 включает медленный когерентный сдвиг



def _delay_duration_seconds(op, dt: float = 0.5e-9) -> float:

    """Длительность delay-инструкции в секундах."""

    dur = getattr(op, "duration", None)

    if dur is None:

        return 60e-9

    unit = getattr(op, "unit", "dt") or "dt"

    if unit == "dt":

        return float(dur) * dt

    if unit == "ns":

        return float(dur) * 1e-9

    if unit == "us":

        return float(dur) * 1e-6

    return float(dur) * dt



def _mean_coherent_theta_per_delay(spec: NoiseSpec, qc: QuantumCircuit,

                                   dt: float = 0.5e-9) -> Optional[float]:

    """Средний угол Rz(ω·τ) на один delay в конкретной схеме."""

    if spec.coherent_dephasing_rad_per_us <= 0:

        return None

    thetas = []

    for inst in qc.data:

        if inst.operation.name == "delay":

            dur_us = _delay_duration_seconds(inst.operation, dt) * 1e6

            thetas.append(spec.coherent_dephasing_rad_per_us * dur_us)

    return float(np.mean(thetas)) if thetas else None



def _build_sim_noise_model(spec: NoiseSpec, coherent_theta: Optional[float] = None) -> tuple:

    """Собрать NoiseModel: марковский шум + опц. когерентный Rz на delay."""

    from qiskit.quantum_info import Operator

    from qiskit_aer.noise import (NoiseModel, coherent_unitary_error,

                                   depolarizing_error, thermal_relaxation_error,

                                   ReadoutError)

    t1 = spec.t1_us * 1e-6

    t2 = min(spec.t2_us, 2 * spec.t1_us) * 1e-6

    x_dur, twoq_dur, idle_dur = 60e-9, 660e-9, 60e-9

    basis = ["rz", "sx", "x", "y", "id", "cx", "delay"]

    nm = NoiseModel(basis_gates=basis)

    relax1 = thermal_relaxation_error(t1, t2, x_dur)

    for g in ("sx", "x", "y", "id"):

        nm.add_all_qubit_quantum_error(relax1, g)

    relax_idle = thermal_relaxation_error(t1, t2, idle_dur)

    if coherent_theta is not None and coherent_theta > 0:

        rz = Operator([[np.exp(-1j * coherent_theta / 2), 0],

                       [0, np.exp(1j * coherent_theta / 2)]])

        relax_idle = relax_idle.compose(coherent_unitary_error(rz))

    nm.add_all_qubit_quantum_error(relax_idle, "delay")

    relax2 = thermal_relaxation_error(t1, t2, twoq_dur).expand(

        thermal_relaxation_error(t1, t2, twoq_dur))

    depol2 = depolarizing_error(min(spec.twoq_err, 0.5), 2)

    nm.add_all_qubit_quantum_error(relax2.compose(depol2), "cx")

    if spec.readout_err > 0:

        e = spec.readout_err

        nm.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))

    return nm, basis



class SimExecutor:

    """

    Честная локальная симуляция:

      * thermal_relaxation на 1Q-гейтах И на delay (idle декогерирует!),

      * depolarizing на 2Q,

      * readout error,

      * опционально когерентное дефазирование на idle (чтобы DD-путь был не мёртв).

    Без всякого клэмпа ZNE — модель должна иметь право быть хуже базы.

    """


    def __init__(self, spec: NoiseSpec, shots: int = 10000, seed: int = 42):

        from qiskit_aer import AerSimulator

        self._AerSimulator = AerSimulator

        self.spec = spec

        self.shots = shots

        self.seed = seed

        self.basis = ["rz", "sx", "x", "y", "id", "cx", "delay"]

        # InstructionDurations для ALAP+DD в симуляции

        ns = lambda x: int(round(x))

        self.sim_durations = InstructionDurations(

            [("x", None, 60, "ns"), ("y", None, 60, "ns"), ("sx", None, 35, "ns"),

             ("rz", None, 0, "ns"), ("id", None, 60, "ns"),

             ("cx", None, 660, "ns"), ("delay", None, 0, "ns")],

            dt=0.5e-9,

        )


    def _make_simulator(self, noise_model, basis_gates):

        from qiskit.transpiler import CouplingMap

        return self._AerSimulator(

            noise_model=noise_model,

            basis_gates=basis_gates,

            coupling_map=CouplingMap([(0, 1), (1, 2), (0, 2)]),

            method="automatic",

        )


    def backend(self):

        from qiskit.providers.fake_provider import GenericBackendV2

        return GenericBackendV2(

            num_qubits=3,

            coupling_map=[(0, 1), (1, 2), (0, 2)],

            basis_gates=self.basis,

            seed=42,

        )


    def run(self, circuits: list[QuantumCircuit]) -> list[tuple[float, float]]:

        out = []

        for qc in circuits:

            theta = _mean_coherent_theta_per_delay(self.spec, qc)

            nm, basis = _build_sim_noise_model(self.spec, coherent_theta=theta)

            sim = self._make_simulator(nm, basis)

            res = sim.run(qc, shots=self.shots, seed_simulator=self.seed).result()

            out.append(_prob_and_sem(res.get_counts(), self.shots))

        return out



# ===========================================================================

# 5. Richardson ZNE с честной ошибкой (без клэмпа)

# ===========================================================================


def richardson_zne(p1: float, sem1: float, p3: float, sem3: float

                   ) -> tuple[float, float]:

    """

    Линейная экстраполяция Ричардсона по scale 1 и 3:

        p0 = (3 p1 - p3) / 2

    Возвращает (p0, sem0). Значение НЕ обрезается — оно имеет право быть < p1

    или вне [0,1], потому что именно это и есть сигнал "ZNE здесь вредит".

    """

    p0 = (3 * p1 - p3) / 2

    sem0 = math.sqrt((9 * sem1 ** 2 + sem3 ** 2) / 4)

    return p0, sem0



# ===========================================================================

# 6. Harness: одна тройка -> строка таблицы со ВСЕМИ 4 стратегиями

# ===========================================================================


@dataclass

class Row:

    features: Features

    dd_name: str

    p_bare: float;  sem_bare: float

    p_dd: float;    sem_dd: float

    p_zne: float;   sem_zne: float

    p_zne_dd: float; sem_zne_dd: float

    p_theory: float


    def best_strategy(self) -> str:

        cands = {"BARE": self.p_bare, "DD": self.p_dd,

                 "ZNE": self.p_zne, "ZNE+DD": self.p_zne_dd}

        return max(cands, key=cands.get)


    def best_prob(self) -> float:

        return max(self.p_bare, self.p_dd, self.p_zne, self.p_zne_dd)


    def flat(self) -> dict:

        d = asdict(self.features)

        d["triple"] = "-".join(map(str, self.features.triple))

        d.update(dd_name=self.dd_name,

                 p_bare=self.p_bare, sem_bare=self.sem_bare,

                 p_dd=self.p_dd, sem_dd=self.sem_dd,

                 p_zne=self.p_zne, sem_zne=self.sem_zne,

                 p_zne_dd=self.p_zne_dd, sem_zne_dd=self.sem_zne_dd,

                 p_theory=self.p_theory, best=self.best_strategy())

        return d



def measure_triple(grover: QuantumCircuit, backend, executor, *,

                   triple: Sequence[int], dd_name: str = "CPMG2",

                   sim_durations: Optional[InstructionDurations] = None) -> Row:

    """Безусловно прогнать 4 схемы и собрать все 4 стратегии для одной тройки."""

    built = build_strategy_circuits(

        grover, backend, triple=triple, dd_name=dd_name, sim_durations=sim_durations)


    (p1, s1), (p1d, s1d), (p3, s3), (p3d, s3d) = executor.run(

        [built.c1, built.c1_dd, built.c3, built.c3_dd])


    p_zne, s_zne = richardson_zne(p1, s1, p3, s3)

    p_zne_dd, s_zne_dd = richardson_zne(p1d, s1d, p3d, s3d)


    feats = extract_features(backend, triple, built.t_circ_s, built.idle_ns)


    return Row(

        features=feats, dd_name=dd_name,

        p_bare=p1, sem_bare=s1,

        p_dd=p1d, sem_dd=s1d,

        p_zne=p_zne, sem_zne=s_zne,

        p_zne_dd=p_zne_dd, sem_zne_dd=s_zne_dd,

        p_theory=grover_theoretical_max(),

    )



def write_csv(rows: list[Row], path: str) -> None:

    if not rows:

        return

    fields = list(rows[0].flat().keys())

    with open(path, "w", newline="") as f:

        w = csv.DictWriter(f, fieldnames=fields)

        w.writeheader()

        for r in rows:

            w.writerow(r.flat())



# ===========================================================================

# 7. Решающее правило как ТОНКАЯ функция + подгонка/оценка по данным

# ===========================================================================


@dataclass

class PolicyThresholds:

    ratio_dephasing: float = 0.85   # ratio < это -> дефазировка -> есть смысл в DD

    coh_tight: float = 25.0         # r_coh < это -> схема длинная -> есть смысл в ZNE

    idle_min_ns: float = 0.0        # idle_ns > это -> окно достаточно для CPMG2

    drift_hours: float = 12.0       # калибровка старше -> принудительно ZNE



def decide(f: dict, thr: PolicyThresholds) -> str:

    """features(dict) -> стратегия. Это baseline-политика, НЕ главный результат."""

    need_dd = ((f["ratio"] < thr.ratio_dephasing)

               and (f["idle_ns"] > thr.idle_min_ns))

    drift = (not math.isnan(f.get("cal_age_h", float("nan")))

             and f["cal_age_h"] > thr.drift_hours)

    need_zne = (f["r_coh"] < thr.coh_tight) or drift

    if need_dd and need_zne:

        return "ZNE+DD"

    if need_dd:

        return "DD"

    if need_zne:

        return "ZNE"

    return "BARE"



_PROB_KEY = {"BARE": "p_bare", "DD": "p_dd", "ZNE": "p_zne", "ZNE+DD": "p_zne_dd"}



def evaluate_policy(table: list[dict], thr: PolicyThresholds) -> dict:

    """

    Метрики Направления 2:

      accuracy — доля точек, где политика выбрала эмпирически лучшую стратегию;

      mean_regret — средняя потеря вероятности успеха относительно лучшей;

      max_regret — худший случай.

    Regret честнее accuracy: промах мимо лучшего, но в пределах 0.5% — не провал.

    """

    hits, regrets = 0, []

    for row in table:

        chosen = decide(row, thr)

        best = max(_PROB_KEY, key=lambda s: row[_PROB_KEY[s]])

        hits += (chosen == best)

        regrets.append(row[_PROB_KEY[best]] - row[_PROB_KEY[chosen]])

    n = max(len(table), 1)

    return {"accuracy": hits / n,

            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,

            "max_regret": float(np.max(regrets)) if regrets else 0.0,

            "n": len(table)}



def fit_thresholds(table: list[dict],

                   ratio_grid: Iterable[float] = np.linspace(0.5, 1.0, 11),

                   coh_grid: Iterable[float] = np.linspace(5, 80, 16),

                   idle_grid: Iterable[float] = (0.0, 400.0, 600.0, 800.0, 1000.0, 1500.0)

                   ) -> tuple[PolicyThresholds, dict]:

    """

    Подгонка порогов по собранной таблице минимизацией среднего regret.

    Это и есть переход от 'пороги с потолка' к 'пороги, обоснованные данными'.

    Для чистоты на практике делай это с train/test split (см. split_eval ниже).

    """

    best_thr, best_metrics = None, {"mean_regret": float("inf")}

    for r in ratio_grid:

        for c in coh_grid:

            for idle in idle_grid:

                thr = PolicyThresholds(ratio_dephasing=float(r),

                                       coh_tight=float(c),

                                       idle_min_ns=float(idle))

                m = evaluate_policy(table, thr)

                if m["mean_regret"] < best_metrics["mean_regret"]:

                    best_thr, best_metrics = thr, m

    return best_thr, best_metrics



def evaluate_fixed_strategy(table: list[dict], strategy: str) -> dict:

    """Метрики для константной стратегии (бейзлайн «всегда ZNE+DD» и т.п.)."""

    hits, regrets = 0, []

    for row in table:

        best = max(_PROB_KEY, key=lambda s: row[_PROB_KEY[s]])

        hits += (strategy == best)

        regrets.append(row[_PROB_KEY[best]] - row[_PROB_KEY[strategy]])

    n = max(len(table), 1)

    return {"strategy": strategy,

            "accuracy": hits / n,

            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,

            "max_regret": float(np.max(regrets)) if regrets else 0.0,

            "n": len(table)}



def majority_baseline(table: list[dict]) -> dict:

    """Бейзлайн: всегда предсказывать самый частый класс oracle в выборке."""

    if not table:

        return {"strategy": "", "accuracy": 0.0, "mean_regret": 0.0, "max_regret": 0.0, "n": 0}

    counts: dict[str, int] = {}

    for row in table:

        counts[row["best"]] = counts.get(row["best"], 0) + 1

    strategy = max(counts, key=counts.get)

    return evaluate_fixed_strategy(table, strategy)



def kfold_eval(table: list[dict], k: int = 5, seed: int = 0) -> dict:

    """k-fold: на каждом фолде фитим пороги на train, меряем на test."""

    if len(table) < k:

        raise ValueError(f"Нужно >= {k} точек для {k}-fold, получено {len(table)}")

    rng = np.random.default_rng(seed)

    idx = rng.permutation(len(table))

    folds = np.array_split(idx, k)

    accs, regrets = [], []

    for i in range(k):

        test_idx = folds[i]

        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        train = [table[j] for j in train_idx]

        test = [table[j] for j in test_idx]

        thr, _ = fit_thresholds(train)

        m = evaluate_policy(test, thr)

        accs.append(m["accuracy"])

        regrets.append(m["mean_regret"])

    return {"k": k,

            "accuracy_mean": float(np.mean(accs)),

            "accuracy_std": float(np.std(accs)),

            "regret_mean": float(np.mean(regrets)),

            "regret_std": float(np.std(regrets)),

            "fold_accuracies": accs,

            "fold_regrets": regrets}



def split_eval(table: list[dict], seed: int = 0) -> dict:

    """Честная оценка: фитим пороги на train, меряем regret на отложенном test."""

    rng = np.random.default_rng(seed)

    idx = rng.permutation(len(table))

    cut = len(table) // 2

    train = [table[i] for i in idx[:cut]]

    test = [table[i] for i in idx[cut:]]

    thr, _ = fit_thresholds(train)

    return {"thresholds": asdict(thr),

            "train": evaluate_policy(train, thr),

            "test": evaluate_policy(test, thr)}



def load_csv(path: str) -> list[dict]:

    """Загрузить одну или несколько CSV (пути через запятую)."""

    rows = []

    for csv_path in (p.strip() for p in path.split(",") if p.strip()):

        with open(csv_path, newline="") as f:

            for d in csv.DictReader(f):

                for k, v in d.items():

                    if k not in ("backend", "triple", "dd_name", "best"):

                        try:

                            d[k] = float(v)

                        except (TypeError, ValueError):

                            d[k] = float("nan")

                rows.append(d)

    return rows



def combine_csv_files(paths: Sequence[str], out_path: str) -> int:

    """Объединить несколько CSV с одинаковой схемой в один файл."""

    table = load_csv(",".join(paths))

    if not table:

        return 0

    fields = list(table[0].keys())

    with open(out_path, "w", newline="") as f:

        w = csv.DictWriter(f, fieldnames=fields)

        w.writeheader()

        w.writerows(table)

    return len(table)



# ===========================================================================

# 8. (Опция) Прямое измерение механизма: даёт ли DD прирост T2?

# ===========================================================================

# Самый "физический" эксперимент Направления 2: померить ОТДЕЛЬНО, как DD

# меняет когерентность простаивающего кубита, а не только исход Гровера.

# Двухзвенная причинная цепочка: DD -> +dT2 -> +d(успех Гровера).


def coherence_probe_circuits(delays_ns: Sequence[float], dt: float,

                             with_dd: bool, dd_name: str = "CPMG2"

                             ) -> list[QuantumCircuit]:

    """

    Набор Ramsey-подобных схем: H - (idle tau, опц. DD) - H - measure.

    Сравнение затухания контраста с DD и без даёт T2_eff и dT2.

    """

    seq = DD_SEQUENCES[dd_name]

    out = []

    for tau in delays_ns:

        qc = QuantumCircuit(1, 1, name=f"probe_{int(tau)}ns_{'dd' if with_dd else 'bare'}")

        qc.h(0)

        n_dt = int(round(tau / (dt * 1e9)))

        if with_dd and n_dt > 0:

            # равномерно расставить импульсы внутри простоя

            k = len(seq)

            seg = max(n_dt // (k + 1), 0)

            for g in seq:

                if seg:

                    qc.delay(seg, 0, unit="dt")

                qc.append(g, [0])

            qc.delay(max(n_dt - seg * k, 0), 0, unit="dt")

        else:

            qc.delay(max(n_dt, 0), 0, unit="dt")

        qc.h(0)

        qc.measure(0, 0)

        out.append(qc)

    return out



def fit_t2_from_contrast(delays_ns: np.ndarray, p0: np.ndarray) -> float:

    """Грубая оценка T2_eff: контраст |p0-0.5| ~ 0.5 exp(-tau/T2). Возвращает T2 в нс."""

    contrast = np.abs(p0 - 0.5)

    contrast = np.clip(contrast, 1e-4, None)

    # линеаризация: ln(2*contrast) = -tau/T2

    y = np.log(2 * contrast)

    A = np.vstack([delays_ns, np.ones_like(delays_ns)]).T

    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(-1.0 / slope) if slope < 0 else float("inf")



# ===========================================================================

# 9. CLI

# ===========================================================================


def _parse_triples(s: str) -> list[tuple[int, int, int]]:

    triples = []

    for chunk in s.split(";"):

        chunk = chunk.strip()

        if not chunk:

            continue

        q = tuple(int(x) for x in chunk.split(","))

        if len(q) != 3:

            raise ValueError(f"Тройка должна содержать 3 кубита: {chunk!r}")

        triples.append(q)

    return triples



def resolve_ibm_token(cli_token: str = "") -> str:

    """Токен: --token → QISKIT_IBM_TOKEN → IBM_QUANTUM_TOKEN → сохранённый аккаунт."""

    for val in (cli_token, os.environ.get("QISKIT_IBM_TOKEN", ""),

                os.environ.get("IBM_QUANTUM_TOKEN", "")):

        if val and val.strip():

            return val.strip()

    return ""



def connect_ibm_service(token: str = "", channel: str = "ibm_quantum_platform"):

    from qiskit_ibm_runtime import QiskitRuntimeService

    if token:

        return QiskitRuntimeService(channel=channel, token=token)

    return QiskitRuntimeService(channel=channel)



def _triple_is_connected(backend, triple: Sequence[int]) -> bool:

    """Проверить, что тройка образует связный подграф coupling map."""

    cm = getattr(backend, "coupling_map", None)

    if cm is None:

        return True

    nodes = set(triple)

    edges: set[tuple[int, int]] = set()

    for a, b in cm.get_edges():

        edges.add((a, b))

        edges.add((b, a))

    seen = {triple[0]}

    frontier = [triple[0]]

    while frontier:

        q = frontier.pop()

        for n in nodes:

            if n not in seen and (q, n) in edges:

                seen.add(n)

                frontier.append(n)

    return seen == nodes



def validate_triples(backend, triples: Sequence[Sequence[int]]) -> None:

    """Проверить индексы кубитов и связность троек до отправки в очередь."""

    n = backend.num_qubits

    for triple in triples:

        if len(triple) != 3:

            raise ValueError(f"Тройка должна содержать 3 кубита: {triple!r}")

        for q in triple:

            if not 0 <= q < n:

                raise ValueError(

                    f"Кубит {q} вне диапазона [0, {n - 1}] для {backend.name}")

        if not _triple_is_connected(backend, triple):

            warnings.warn(

                f"Тройка {triple} может быть не связной на {backend.name} — "

                "транспилятор добавит SWAP, схема станет длиннее.",

                stacklevel=2,

            )



def list_ibm_backends(token: str = "", channel: str = "ibm_quantum_platform",

                      min_qubits: int = 3) -> None:

    service = connect_ibm_service(token, channel)

    backends = sorted(

        (b for b in service.backends() if b.num_qubits >= min_qubits),

        key=lambda b: (not b.status().operational, -b.num_qubits, b.name),

    )

    print(f"{'backend':<22} {'qubits':>6}  {'operational':<12}  status")

    print("-" * 60)

    for b in backends:

        st = b.status()

        flag = "yes" if st.operational else "no"

        msg = (st.status_msg or "")[:40]

        print(f"{b.name:<22} {b.num_qubits:>6}  {flag:<12}  {msg}")



def dry_run_real(backend_name: str, triples: list[tuple[int, int, int]],

                 dd_name: str, token: str = "", channel: str = "ibm_quantum_platform"

                 ) -> None:

    """Собрать и транспилировать схемы без отправки job — проверка перед очередью."""

    service = connect_ibm_service(token, channel)

    backend = service.backend(backend_name)

    validate_triples(backend, triples)

    grover = build_grover_3q(iterations=1)

    triple = triples[0]

    print(f"Бэкенд: {backend.name} ({backend.num_qubits} кубитов)")

    print(f"Dry-run для тройки {triple}, DD={dd_name}\n")

    built = build_strategy_circuits(grover, backend, triple=triple, dd_name=dd_name)

    dt = getattr(getattr(backend, "target", None), "dt", None) or 0.5e-9

    for label, qc in [("c1 BARE", built.c1), ("c1_dd DD", built.c1_dd),

                      ("c3 ZNE", built.c3), ("c3_dd ZNE+DD", built.c3_dd)]:

        ops = qc.count_ops()

        depth = qc.depth()

        dur_us = _circuit_duration_s(qc, dt) * 1e6

        delays = ops.get("delay", 0)

        print(f"  {label:<14} depth={depth:4d}  delays={delays:3d}  "

              f"T_circ≈{dur_us:6.1f}µs  ops={dict(ops)}")

    print(f"\nВсего троек для прогона: {len(triples)}  "

          f"(4 схемы × {len(triples)} = {4 * len(triples)} jobs в Batch на тройку)")



def run_real(token: str, backend_name: str, triples: list[tuple[int, int, int]],

             dd_name: str, shots: int, out_path: str,

             channel: str = "ibm_quantum_platform") -> None:

    service = connect_ibm_service(token, channel)

    backend = service.backend(backend_name)

    validate_triples(backend, triples)

    print(f"Бэкенд: {backend.name} ({backend.num_qubits} кубитов)")

    print(f"Теор. максимум |111>: {grover_theoretical_max()*100:.2f}%")

    print(f"Троек: {len(triples)}  shots={shots}  DD={dd_name}")

    print(f"Результат: {out_path}\n")


    grover = build_grover_3q(iterations=1)

    executor = HardwareExecutor(backend, shots=shots)

    rows = []

    for i, triple in enumerate(triples, 1):

        print(f"  [{i}/{len(triples)}] тройка {triple} ...", flush=True)

        try:

            row = measure_triple(grover, backend, executor, triple=triple, dd_name=dd_name)

            rows.append(row)

            _print_row(row)

            write_csv(rows, out_path)

        except Exception as exc:

            print(f"    ОШИБКА на тройке {triple}: {exc}", flush=True)

            if rows:

                write_csv(rows, out_path)

                print(f"    Частичный результат сохранён ({len(rows)} строк).")

            raise

    print(f"\nЗаписано {len(rows)} строк -> {out_path}")



def run_sim_grid(out_path: str, dd_name: str, shots: int) -> None:

    from qiskit_aer import AerSimulator  # noqa: F401  (проверка наличия Aer)

    grover = build_grover_3q(iterations=1)

    print(f"Теор. максимум |111>: {grover_theoretical_max()*100:.2f}%")

    print("Симуляционная сетка с когерентным дефазированием "

          "(coherent_dephasing_rad_per_us=0.3) для оси DD.\n")


    rows = []

    for t1 in (120.0, 80.0, 40.0):

        for t2_frac in (0.9, 0.6, 0.3):

            for err in (0.005, 0.012, 0.025):

                spec = NoiseSpec(t1_us=t1, t2_us=t1 * 2 * t2_frac,

                                 twoq_err=err, readout_err=0.02,

                                 coherent_dephasing_rad_per_us=0.3)

                ex = SimExecutor(spec, shots=shots)

                row = measure_triple(grover, ex.backend(), ex, triple=(0, 1, 2),

                                     dd_name=dd_name)

                rows.append(row)

                _print_row(row)

                write_csv(rows, out_path)   # инкрементальная запись на случай сбоя

    write_csv(rows, out_path)

    print(f"\nЗаписано {len(rows)} строк -> {out_path}")



class _AerExecutor:

    """Локальный исполнитель для Fake-бэкендов через AerSimulator.from_backend."""


    def __init__(self, backend, shots: int):

        self.backend = backend

        self.shots = shots


    def run(self, circuits: list[QuantumCircuit]) -> list[tuple[float, float]]:

        out = []

        for qc in circuits:

            res = self.backend.run(qc, shots=self.shots, seed_simulator=42).result()

            out.append(_prob_and_sem(res.get_counts(), self.shots))

        return out



def run_fake_grid(out_path: str, dd_name: str, shots: int) -> None:

    from qiskit_aer import AerSimulator

    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKyoto, FakeSherbrooke

    grover = build_grover_3q(iterations=1)

    print(f"Теор. максимум |111>: {grover_theoretical_max()*100:.2f}%")

    print("Fake-бэкенды: Brisbane, Kyoto, Sherbrooke\n")

    rows = []

    for name, Fake in [("FakeBrisbane", FakeBrisbane), ("FakeKyoto", FakeKyoto),

                       ("FakeSherbrooke", FakeSherbrooke)]:

        print(f"  [{name}] ...", flush=True)

        fb = Fake()

        sim = AerSimulator.from_backend(fb, method="matrix_product_state")

        ex = _AerExecutor(sim, shots)

        row = measure_triple(grover, fb, ex, triple=(0, 1, 2), dd_name=dd_name)

        row.features.backend = name

        rows.append(row)

        _print_row(row)

    write_csv(rows, out_path)

    print(f"\nЗаписано {len(rows)} строк -> {out_path}")



def run_analyze(csv_path: str) -> None:

    table = load_csv(csv_path)

    print(f"Загружено строк: {len(table)}")

    base = evaluate_policy(table, PolicyThresholds())

    print(f"\nИсходные пороги (0.85 / 25):  accuracy={base['accuracy']:.2f}  "

          f"mean_regret={base['mean_regret']*100:.2f}pp  "

          f"max_regret={base['max_regret']*100:.2f}pp")

    thr, m = fit_thresholds(table)

    print(f"Подогнанные пороги:           ratio<{thr.ratio_dephasing:.2f}  "

          f"r_coh<{thr.coh_tight:.1f}  idle>{thr.idle_min_ns:.0f}ns  ->  "

          f"accuracy={m['accuracy']:.2f}  mean_regret={m['mean_regret']*100:.2f}pp")

    maj = majority_baseline(table)

    print(f"\nБейзлайн majority-class ({maj['strategy']}):  "

          f"accuracy={maj['accuracy']:.2f}  "

          f"mean_regret={maj['mean_regret']*100:.2f}pp")

    for strat in ("BARE", "ZNE", "ZNE+DD"):

        b = evaluate_fixed_strategy(table, strat)

        print(f"  always {strat:<6}  accuracy={b['accuracy']:.2f}  "

              f"mean_regret={b['mean_regret']*100:.2f}pp")

    se = split_eval(table)

    print(f"\nЧестная оценка (train/test split 50/50):")

    print(f"  train: accuracy={se['train']['accuracy']:.2f} "

          f"regret={se['train']['mean_regret']*100:.2f}pp")

    print(f"  test:  accuracy={se['test']['accuracy']:.2f} "

          f"regret={se['test']['mean_regret']*100:.2f}pp  "

          f"(n={se['test']['n']})")

    kf = kfold_eval(table, k=5)

    print(f"\n5-fold CV (fit на k-1 фолдах, оценка на hold-out):")

    print(f"  accuracy={kf['accuracy_mean']:.2f} ± {kf['accuracy_std']:.2f}  "

          f"regret={kf['regret_mean']*100:.2f} ± {kf['regret_std']*100:.2f}pp")



def _print_row(r: Row) -> None:

    f = r.features

    print(f"    {f.backend:<14} {'-'.join(map(str,f.triple)):<10} "

          f"ratio={f.ratio:.2f} r_coh={f.r_coh:6.1f} idle={f.idle_ns:5.0f}ns | "

          f"BARE={r.p_bare*100:5.1f}  DD={r.p_dd*100:5.1f}  "

          f"ZNE={r.p_zne*100:5.1f}  ZNE+DD={r.p_zne_dd*100:5.1f}  "

          f"-> best={r.best_strategy()}")



def main(argv: Optional[list[str]] = None) -> None:

    ap = argparse.ArgumentParser(description="Grover mitigation lab (Direction 2)")

    ap.add_argument("--real", action="store_true", help="прогон на реальном железе")

    ap.add_argument("--dry-run", action="store_true",

                    help="транспиляция схем без отправки на IBM (preflight)")

    ap.add_argument("--list-backends", action="store_true",

                    help="список доступных IBM-бэкендов (>=3 кубитов)")

    ap.add_argument("--sim-grid", action="store_true", help="симуляционная сетка")

    ap.add_argument("--fake-grid", action="store_true",

                    help="прогон на FakeBrisbane/Kyoto/Sherbrooke")

    ap.add_argument("--combine", nargs="+", metavar="CSV",

                    help="объединить CSV-файлы и записать в --out")

    ap.add_argument("--analyze", metavar="CSV", help="подгонка/оценка политики по CSV")

    ap.add_argument("--token", default="",

                    help="IBM Quantum API token (или QISKIT_IBM_TOKEN)")

    ap.add_argument("--channel", default="ibm_quantum_platform")

    ap.add_argument("--backend", default="ibm_sherbrooke")

    ap.add_argument("--triples", default="0,1,2", help="'0,1,2; 3,4,5; ...'")

    ap.add_argument("--dd", default="CPMG2", choices=list(DD_SEQUENCES))

    ap.add_argument("--shots", type=int, default=10000)

    ap.add_argument("--out", default="results.csv")

    a = ap.parse_args(argv)

    token = resolve_ibm_token(a.token)


    if a.list_backends:

        list_ibm_backends(token, a.channel)

    elif a.dry_run:

        if not a.backend:

            sys.exit("Укажи --backend для --dry-run")

        dry_run_real(a.backend, _parse_triples(a.triples), a.dd, token, a.channel)

    elif a.analyze:

        run_analyze(a.analyze)

    elif a.combine:

        n = combine_csv_files(a.combine, a.out)

        print(f"Объединено {n} строк -> {a.out}")

    elif a.real:

        run_real(token, a.backend, _parse_triples(a.triples),

                 a.dd, a.shots, a.out, a.channel)

    elif a.sim_grid:

        run_sim_grid(a.out, a.dd, a.shots)

    elif a.fake_grid:

        run_fake_grid(a.out, a.dd, a.shots)

    else:

        ap.print_help()



if __name__ == "__main__":

    main() 