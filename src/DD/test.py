import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate, RXGate, RYGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule
from src.DD.dynamical_decoupling import DynamicalDecoupling
from qiskit.visualization import timeline_drawer

from qiskit import IBMQ, transpile, Aer
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-france', group='univ-montpellier', project='default')
backend = provider.get_backend('ibmq_guadalupe')

from src.tools.DD_insertion import construct_udd_sequence, \
                                 kdd_sequences, \
                                 kdd_spacing, \
                                 construct_bv_circuit, \
                                 construct_hs_circuit, \
                                 construct_graph_matrix, \
                                 convert_count_to_prob, \
                                 translate_circuit_to_basis

from qiskit.circuit.library import QFT, GraphState
bv_circuits = []
hs_circuits = []
qft_circuits = []
for i in range(3, 15):
    bv_circuits.append(construct_bv_circuit(i))

# for i in range(2, 15, 2):
#     hs_circuits.append(construct_hs_circuit(i))

for i in range(3, 15):
    qft_circuits.append(QFT(i))

for circuit in bv_circuits:
    circuit.measure_all()

# for circuit in hs_circuits:
#     circuit.measure_all()

for circuit in qft_circuits:
    circuit.measure_all()

durations = InstructionDurations.from_backend(backend)
## add duration of y gates which are used for DD sequences
bconf = backend.configuration()
for i in range(bconf.num_qubits):
    x_duration = durations.get('x', i)
    durations.update(InstructionDurations(
        [('y', i, x_duration)]
        ))

    durations.update(InstructionDurations(
        [('rx', i, x_duration)]
        ))

    durations.update(InstructionDurations(
        [('ry', i, x_duration)]
        ))

graph_state_circuits = []
coupling_map = backend.configuration().coupling_map

for i in range(3, 15):
    gs_circuit_matrix = construct_graph_matrix(i, coupling_map)
    graph_state_circuits.append(GraphState(gs_circuit_matrix))

for circuit in graph_state_circuits:
    circuit.measure_all()

from src.tools.DD_insertion import pm_DD_sequences
pms = pm_DD_sequences(durations)
bv_job_ids = []
bv_jobs = []

for circuit in bv_circuits:
    circuit_list = []
    transpiled_qc = transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=1)
    circuit_list.append(transpiled_qc)
    qc_transpile = pms[-1].run(transpiled_qc)
    for pm in pms:
        qc_transpile = pm.run(transpiled_qc)
        print(11)
        qc_transpile_base = translate_circuit_to_basis(qc_transpile, bconf)
        circuit_list.append(qc_transpile_base)
    job = backend.run(circuit_list, shots=8192)
    bv_jobs.append(job)
    job_id = job.job_id()
    print(job_id)
    bv_job_ids.append(job_id)