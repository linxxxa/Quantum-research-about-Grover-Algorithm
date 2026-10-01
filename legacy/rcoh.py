import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler import InstructionDurations, CouplingMap
from qiskit.transpiler import InstructionDurations


# --- ФИЗИЧЕСКИЕ ПАРАМЕТРЫ (IBM Eagle-like) ---
T1_FIXED = 200e3        # нс
GATE_TIME_1Q = 35       # нс
GATE_TIME_2Q = 320      # нс


# --- 1. СХЕМА ГРОВЕРА (3 кубита) ---
def build_grover_3q():
    qc = QuantumCircuit(3)

    # superposition
    qc.h([0, 1, 2])

    # oracle |111>
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)

    # diffuser
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])

    return qc


# --- 2. ТОПОЛОГИЯ + ДЛИТЕЛЬНОСТИ ---
durations = InstructionDurations([
    ('h', None, GATE_TIME_1Q),
    ('x', None, GATE_TIME_1Q),
    ('cx', None, GATE_TIME_2Q),
])

coupling = CouplingMap.from_line(3)
backend = AerSimulator()

# --- 3. ТРАНСПИЛЯЦИЯ + SCHEDULING ---
qc_base = build_grover_3q()


# ВАЖНО: передаём durations через backend properties (обходной путь)
qc_transpiled = transpile(
    qc_base,
    basis_gates=['cx', 'u1', 'u2', 'u3', 'h', 'x'],
    coupling_map=coupling,
    optimization_level=1
)

# --- РУЧНОЙ ПОДСЧЁТ ВРЕМЕНИ ---
gate_counts = qc_transpiled.count_ops()

n_1q = sum(gate_counts.get(g, 0) for g in ['h', 'x', 'u1', 'u2', 'u3'])
n_2q = gate_counts.get('cx', 0)
depth = qc_transpiled.depth()
t_circuit = depth * GATE_TIME_2Q

print(f"Estimated circuit time: {t_circuit:.1f} ns")


# --- 4. ИДЕАЛЬНОЕ СОСТОЯНИЕ (ВАЖНО: после transpile!) ---
ideal_state = Statevector.from_instruction(qc_transpiled)


# --- 5. МОДЕЛЬ ШУМА (ЧИСТАЯ ФИЗИКА T1/T2) ---
def build_noise_model(t1, t2):

    noise_model = NoiseModel()

    # 1Q ошибки
    err_1q = thermal_relaxation_error(t1, t2, GATE_TIME_1Q)

    # 2Q ошибки (тензор двух кубитов)
    err_2q = thermal_relaxation_error(t1, t2, GATE_TIME_2Q).expand(
             thermal_relaxation_error(t1, t2, GATE_TIME_2Q)
    )

    noise_model.add_all_qubit_quantum_error(
        err_1q, ['u1', 'u2', 'u3', 'h', 'x']
    )

    noise_model.add_all_qubit_quantum_error(
        err_2q, ['cx']
    )

    return noise_model


# --- 6. СКАН ПО T2 ---
t2_range = np.logspace(np.log10(1e3), np.log10(1e6), 50)

results_r_coh = []
results_fidelity = []


for t2 in t2_range:

    # физическое ограничение
    t2_actual = min(t2, 2 * T1_FIXED)

    r_coh = t2_actual / t_circuit

    noise_model = build_noise_model(T1_FIXED, t2_actual)

    sim = AerSimulator(
        noise_model=noise_model,
        method='density_matrix'
    )

    qc_run = qc_transpiled.copy()
    qc_run.save_density_matrix()

    result = sim.run(qc_run).result()
    rho = result.data()['density_matrix']

    fid = state_fidelity(ideal_state, rho)

    results_r_coh.append(r_coh)
    results_fidelity.append(fid)


# --- 7. ГРАФИК ---
plt.figure(figsize=(10, 6))

plt.semilogx(
    results_r_coh,
    results_fidelity,
    'o-',
    color='blue',
    label='Grover Fidelity (3 qubits)'
)

# эмпирические пороги
plt.axvline(x=20, color='red', linestyle='--', label='Hybrid threshold ~20')
plt.axvline(x=60, color='green', linestyle='--', label='Stable quantum ~60')

plt.xlabel('R_coh = T2 / t_circuit (log scale)')
plt.ylabel('State Fidelity')
plt.title('Quantum-to-Classical Transition via Coherence Ratio')

plt.grid(True, which='both', alpha=0.3)
plt.legend()

plt.show()