from qiskit import IBMQ, transpile
from qiskit.circuit import Delay
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import QFT, GraphState
from src.tools.DD_insertion import construct_bv_circuit, \
                                 construct_hs_circuit, \
                                 construct_graph_matrix
from networkx.generators.random_graphs import erdos_renyi_graph
from src.QAOA.QAOA_approximation import generate_three_regular_graph, create_qaoa_circ, get_ansatz_parm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-france', group='univ-montpellier', project='default')
backend = provider.get_backend('ibmq_montreal')

def idle_time(dag):
    ## calculate the total idle time of the circuit follwing the alap schedule
    t_idle = 0
    for nd in dag.topological_op_nodes():
        if not isinstance(nd.op, Delay):
            continue
        pred = next(dag.predecessors(nd))
        # succ = next(dag.successors(nd))
        if pred.cargs is None or pred.name == 'barrier':
            continue
        # if isinstance(succ.op, Measure):
        t_idle += nd.op.duration
    return t_idle

seed = 1
coupling_map = backend.configuration().coupling_map

bv_idles = []
hs_idles = []
qft_idles = []
gs_idles = []
qaoa_3reg_idles = []
qaoa_random_idles = []

for i in range(4,13,2):
    #BV
    bv_circ = transpile(construct_bv_circuit(i), backend=backend, optimization_level=3, scheduling_method='alap', seed_transpiler=seed)
    bv_idles.append(idle_time(circuit_to_dag(bv_circ)))
    #QFT
    qft_circ = transpile(QFT(i), backend=backend, optimization_level=3, scheduling_method='alap', seed_transpiler=seed)
    qft_idles.append(idle_time(circuit_to_dag(qft_circ)))
    #Graph state
    gs_circuit_matrix = construct_graph_matrix(i, coupling_map)
    gs_circ = transpile(GraphState(gs_circuit_matrix), backend=backend, optimization_level=3, scheduling_method='alap', seed_transpiler=seed)
    gs_idles.append(idle_time(circuit_to_dag(gs_circ)))

for i in range(4,13,2):
    # HS
    hs_circ = transpile(construct_hs_circuit(i), backend=backend, optimization_level=3, scheduling_method='alap',
                        seed_transpiler=seed)
    hs_idles.append(idle_time(circuit_to_dag(hs_circ)))

for i in range(4,13,2):
    # QAOA 3 regular graph
    G = generate_three_regular_graph(i)
    res = get_ansatz_parm(G)
    qc_3reg = transpile(create_qaoa_circ(G, res.x), backend=backend, optimization_level=3, scheduling_method='alap',
                        seed_transpiler=seed)
    qaoa_3reg_idles.append(idle_time(circuit_to_dag(qc_3reg)))
    # QAOA random graph
    G = erdos_renyi_graph(i, 0.5, 200)
    res = get_ansatz_parm(G)
    qc_random = transpile(create_qaoa_circ(G, res.x), backend=backend, optimization_level=3, scheduling_method='alap',
                          seed_transpiler=seed)
    qaoa_random_idles.append(idle_time(circuit_to_dag(qc_random)))



print('-------')
x_ticks = list(range(4,13,2))
X = np.arange(len(x_ticks))
plt.plot(X, bv_idles, linestyle='--', marker='x', color='y', label='bv')
plt.plot(X, qft_idles, linestyle='--', marker='^', color='r', label='qft')
plt.plot(X, gs_idles, linestyle='--', marker='*', color='g', label='gs')
plt.plot(X, qaoa_3reg_idles, linestyle='--', marker='o', color='c', label='3-reg')
plt.plot(X, qaoa_random_idles, linestyle='--', marker='+', color='m', label='rand')

plt.legend(loc='best')
plt.title('Idle times of different benchmarks on ibmq_montreal')
plt.xlabel('Benchmark size')
plt.ylabel('Idle time (dt)')
plt.xticks(X)

# ax.set_title('Idle times of different benchmarks on ibmq_montreal')
# ax.set_xticks(X)
# ax.set_xticklabels(x_ticks)
# ax.set_xlabel('Benchmark size')
# ax.set_ylabel('Idle time')
plt.savefig('idle_time.png')

plt.show()
print(bv_idles)
print(qft_idles)
print(gs_idles)
print(qaoa_3reg_idles)
print(qaoa_random_idles)