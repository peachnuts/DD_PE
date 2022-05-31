import numpy as np
from qiskit.circuit.library import RXGate, RZGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit import IBMQ, transpile
from qiskit.circuit.library import QFT

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-france', group='univ-montpellier', project='reservations')
backend = provider.get_backend('ibmq_jakarta')

durations = InstructionDurations.from_backend(backend)

def theta_phi(theta, phi):
    return [RZGate(phi), RXGate(-theta), RZGate(-phi)]

def one_sequence(phi):
    sequence = []
    sequence.extend(theta_phi(np.pi, np.pi/6 + phi))
    sequence.extend(theta_phi(np.pi, phi))
    sequence.extend(theta_phi(np.pi, np.pi/2 + phi))
    sequence.extend(theta_phi(np.pi, phi))
    sequence.extend(theta_phi(np.pi, np.pi/6 + phi))
    return sequence

seqences = []
seqences.extend(one_sequence(0))
seqences.extend(one_sequence(np.pi/2))
seqences.extend(one_sequence(0))
seqences.extend(one_sequence(np.pi/2))

def kdd_spacing(num_pulse=20):
    mid = 1 / num_pulse
    end = mid / 2
    spacing = []
    spacing.append(end)
    interval = [0] * 2
    for i in range(num_pulse):
        spacing.extend(interval)
        if i < num_pulse - 1:
            spacing.append(mid)
    spacing.append(end)
    return spacing


bconf = backend.configuration()
for i in range(bconf.num_qubits):
    x_duration = durations.get('x', i)
    durations.update(InstructionDurations(
        [('rx', i, x_duration)]
        ))


spacing = kdd_spacing()

pm = PassManager([ALAPSchedule(durations),
                 DynamicalDecoupling(durations, seqences, spacing=spacing, name='kdd')])


qc = QFT(4)
qc_t = transpile(qc, backend=backend, optimization_level=3)
qc_kdd = pm.run(qc_t)

