"""
Эксперимент на FakeKyoto: сравнение Raw / ZNE / DD / Hybrid (DD+ZNE)
=====================================================================
Ключевые исправления по сравнению с предыдущей версией:
  - DD реализован через PadDynamicalDecoupling (реальные idle-окна),
    а не через X-X перед измерением.
  - Поддерживаются три последовательности: Hahn Echo (X,X),
    XY4 (X,Y,X,Y) и XY8 (расширенный XY4).
  - ZNE использует три точки масштабирования (λ=1,3,5) с линейной
    экстраполяцией Ричардсона для повышения точности.
  - Добавлено повторное прогоняние (N_REPEATS) для оценки SEM.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeKyoto

from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler.passes import UnrollCustomDefinitions, BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling, BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel


# ── ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ──────────────────────────────────────────────────
SHOTS       = 2000
N_REPEATS   = 5          # сколько раз повторять каждый замер для SEM
TARGET_STATE = '111'  # Добавь это в блок параметров в начале
LAMBDAS     = [1, 3, 5]  # коэффициенты масштабирования шума для ZNE
RANDOM_THR  = 1 / 8      # 1/2^3 — порог случайного угадывания

DD_SEQUENCES = {
    'Hahn':  [XGate(), XGate()],
    'XY4':   [XGate(), YGate(), XGate(), YGate()],
    'XY8':   [XGate(), YGate(), XGate(), YGate(),
               YGate(), XGate(), YGate(), XGate()],
}
# Выберите нужную последовательность здесь:
DD_SEQ_NAME = 'XY4'

# ── ПУТИ ────────────────────────────────────────────────────────────────────
desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
img_path = os.path.join(desktop, "kyoto_dd_zne_v2.png")

# ── БЭКЕНД ──────────────────────────────────────────────────────────────────
backend  = FakeKyoto()
noise_sim = AerSimulator.from_backend(backend)
durations = InstructionDurations.from_backend(backend)


# ── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ──────────────────────────────────────────────────

def build_grover_3q() -> QuantumCircuit:
    """Базовая 3-кубитная схема алгоритма Гровера (k=1, цель |111>)."""
    qc = QuantumCircuit(3)
    # Инициализация суперпозиции
    qc.h(range(3))
    qc.barrier()
    # Фазовый оракул для |111>: CCZ через вспомогательную декомпозицию
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.barrier()
    # Оператор диффузии
    qc.h(range(3))
    qc.x(range(3))
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(range(3))
    qc.h(range(3))
    qc.measure_all()
    return qc


def fold_gates(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Локальное свёртывание гейтов для ZNE: U → U (U† U)^n, где scale = 2n+1.
    Логически эквивалентно исходной схеме, но шум увеличен в scale раз.
    """
    assert scale % 2 == 1, "scale должен быть нечётным (1, 3, 5, ...)"
    n_pairs = (scale - 1) // 2
    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in ('barrier', 'measure', 'reset'):
            for _ in range(n_pairs):
                folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                folded.append(inst.operation,           inst.qubits, inst.clbits)
    return folded

def apply_dd(qc_transpiled: QuantumCircuit, seq_name: str) -> QuantumCircuit:
    dd_seq = DD_SEQUENCES[seq_name]
    
    # 1. Снова durations (база)
    dur_obj = InstructionDurations.from_backend(backend)
    for q in range(backend.num_qubits):
        try:
            x_val = dur_obj.get('x', q)
            dur_obj.update([('y', [q], x_val)])
        except: continue

    # 2. Монолитный PassManager без лишних движений
    # Добавляем аргумент 'basis_gates', чтобы DD понимал, куда он вставляется
    pm = PassManager([
        ALAPScheduleAnalysis(dur_obj), 
        PadDynamicalDecoupling(dur_obj, dd_seq)
    ])
    
    qc_dd = pm.run(qc_transpiled)

    # 3. ВАЖНО: Мы не вызываем здесь transpile()!
    # Мы вернем схему как есть, а симулятору разрешим выполнить Y.
    return qc_dd

def run_shots(qc):
    # Только перевод в базис — без scheduling!
    pm = PassManager([
        BasisTranslator(sel, backend.target)
    ])
    
    qc_basis = pm.run(qc)

    job = noise_sim.run(qc_basis, shots=SHOTS)
    counts = job.result().get_counts()
    return counts.get(TARGET_STATE, 0) / SHOTS




def richardson_linear(ps: list[float], lambdas: list[int]) -> float:
    """
    Линейная экстраполяция Ричардсона по нескольким точкам (λ, P(λ)).
    Для двух точек: аналитическая формула.
    Для трёх и более: МНК-полином первой степени, экстраполяция в λ=0.
    """
    if len(ps) == 2:
        l1, l2 = lambdas
        p1, p2 = ps
        return p1 + (p1 - p2) * (l1 / (l2 - l1))
    # МНК для трёх точек
    coeffs = np.polyfit(lambdas, ps, deg=1)
    return float(np.polyval(coeffs, 0))


def repeated_measure(qc: QuantumCircuit, n: int = N_REPEATS) -> tuple[float, float]:
    """Возвращает (среднее, SEM) по n независимым запускам."""
    vals = [run_shots(qc) for _ in range(n)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(n))


# ── ОСНОВНАЯ ЛОГИКА ──────────────────────────────────────────────────────────

print(f">>> Бэкенд: {backend.name}")
print(f">>> DD-последовательность: {DD_SEQ_NAME}")
print(f">>> Масштабы ZNE: {LAMBDAS}\n")

grover = build_grover_3q()

# Базовая транспиляция с ALAP (нужна для корректного PadDD)
t_base = transpile(grover, backend, optimization_level=3, scheduling_method='alap')

# ── 1. RAW ───────────────────────────────────────────────────────────────────
p_raw, sem_raw = repeated_measure(t_base)
print(f"RAW:    P = {p_raw:.4f} ± {sem_raw:.4f}")

# ── 2. ZNE (без DD) ──────────────────────────────────────────────────────────
p_zne_points = []
for lam in LAMBDAS:
    if lam == 1:
        p_zne_points.append(p_raw)
    else:
        t_folded = transpile(
            fold_gates(t_base, lam), backend, optimization_level=0
        )
        p_l, _ = repeated_measure(t_folded)
        p_zne_points.append(p_l)

p_zne = richardson_linear(p_zne_points, LAMBDAS)
# SEM для ZNE — приближённо из дисперсии точек экстраполяции
sem_zne = float(np.std(p_zne_points) / np.sqrt(len(p_zne_points)))
print(f"ZNE:    P = {p_zne:.4f} ± {sem_zne:.4f}  (точки: {[f'{v:.4f}' for v in p_zne_points]})")

# ── 3. DD (без ZNE) ──────────────────────────────────────────────────────────
t_dd = apply_dd(t_base, DD_SEQ_NAME)
p_dd, sem_dd = repeated_measure(t_dd)
print(f"DD:     P = {p_dd:.4f} ± {sem_dd:.4f}")

# ── 4. HYBRID (DD + ZNE) ─────────────────────────────────────────────────────
# Порядок: сначала ALAP+DD (стабилизируем кубиты), затем фолдинг для ZNE
# ── 4. HYBRID (DD + ZNE) ──
# ── 4. HYBRID (DD + ZNE) ──
p_hyb_points = []

for lam in LAMBDAS:
    if lam == 1:
        p_hyb_points.append(p_dd)
    else:
        folded = fold_gates(t_base, lam)

        t_folded = transpile(
            folded,
            backend,
            optimization_level=0,
            scheduling_method='alap'
        )

        t_dd_folded = apply_dd(t_folded, DD_SEQ_NAME)

        p_l, _ = repeated_measure(t_dd_folded)
        p_hyb_points.append(p_l)

# 👉 ВАЖНО: эти строки ОБЯЗАТЕЛЬНЫ
p_hybrid = richardson_linear(p_hyb_points, LAMBDAS)
sem_hybrid = float(np.std(p_hyb_points) / np.sqrt(len(p_hyb_points)))

print(f"Hybrid: P = {p_hybrid:.4f} ± {sem_hybrid:.4f}  (точки: {[f'{v:.4f}' for v in p_hyb_points]})")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────────────────────
labels  = ['Raw', f'ZNE\n(λ={LAMBDAS})', f'DD\n({DD_SEQ_NAME})', f'Hybrid\n(DD+ZNE)']
values  = [p_raw, p_zne, p_dd, p_hybrid]
errors  = [sem_raw, sem_zne, sem_dd, sem_hybrid]
colors  = ['#bdc3c7', '#3498db', '#2ecc71', '#e67e22']

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(labels, values, color=colors, edgecolor='black',
              yerr=errors, capsize=6, error_kw={'linewidth': 1.5}, zorder=3)

ax.axhline(RANDOM_THR, color='red', linestyle='--', linewidth=1.5,
           label=f'Порог случайного угадывания (1/8 = {RANDOM_THR:.3f})', zorder=4)

ax.set_ylim(0, max(values) * 1.35)
ax.set_ylabel('Вероятность успеха P(|111⟩)', fontsize=12)
ax.set_title(f'Митигация ошибок на {backend.name} | DD: {DD_SEQ_NAME} | ZNE: λ={LAMBDAS}',
             fontsize=13, pad=16)
ax.grid(axis='y', alpha=0.3, linestyle=':', zorder=0)
ax.legend(loc='upper right', frameon=True)

for bar, val, err in zip(bars, values, errors):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + err + max(values) * 0.02,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(img_path, dpi=200)
print(f"\n>>> График сохранён: {img_path}")
