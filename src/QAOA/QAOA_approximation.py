from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit import execute, Aer, QuantumCircuit, IBMQ
import numpy as np
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler, Z, CircuitStateFn
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.quantumregister import Qubit
from scipy.optimize import minimize
from recirq.qaoa.problems import get_all_3_regular_problems
import networkx as nx
from qiskit.providers.aer.noise import NoiseModel



def generate_three_regular_graph(n_qubits: int):
    instance_i = 0
    np.random.seed(42)
    threereg_problems = get_all_3_regular_problems(max_n_qubits=22,
                                                   n_instances=10,
                                                   rs=np.random.RandomState(5))
    threereg_problem = threereg_problems[n_qubits, instance_i]

    working_graph = nx.Graph()
    for node in threereg_problem.graph.nodes():
        working_graph.add_node(node)

    for (q1, q2) in threereg_problem.graph.edges():
        working_graph.add_edge(q1, q2)

    return working_graph

def generate_graph_matrix(G):
    node_num = len(G.nodes)
    w = np.zeros((node_num, node_num))
    for i in G.nodes:
        for j in list(G.adj[i]):
            w[i][j] = 1
    return w


def create_qaoa_circ(G, theta):

    """
    Creates a parametrized qaoa circuit

    Args:
        G: networkx graph
        theta: list
               unitary parameters

    Returns:
        qc: qiskit circuit
    """

    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):

        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc

def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring

        G: networkx graph

    Returns:
        obj: float
             Objective
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1

    return obj


def compute_expectation(counts, G):

    """
    Computes expectation value based on measurement results

    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count

    return avg/sum_count


def get_expectation(G, noise_model, shots=8192,):

    """
    Runs parametrized circuit

    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        #noise_model1 = NoiseModel.from_backend(provider.get_backend('ibmq_jakarta'))
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=10, noise_model=noise_model,
                             nshots=8192).result().get_counts()

        return compute_expectation(counts, G)

    return execute_circ


def get_ansatz_parm(G, noise_model=None):
    expectation = get_expectation(G, noise_model)
    np.random.seed(42)
    beta = np.random.randint(100)
    gamma = np.random.randint(100)
    res = minimize(expectation,
                   [beta, gamma],
                   method='COBYLA')
    return res


def get_operator(weight_matrix):
    r"""Generate Hamiltonian for the graph partitioning
    Notes:
        Goals:
            1 separate the vertices into two set of the same size
            2 make sure the number of edges between the two set is minimized.
        Hamiltonian:
            H = H_A + H_B
            H_A = sum\_{(i,j)\in E}{(1-ZiZj)/2}
            H_B = (sum_{i}{Zi})^2 = sum_{i}{Zi^2}+sum_{i!=j}{ZiZj}
            H_A is for achieving goal 2 and H_B is for achieving goal 1.
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.
    Returns:
        PauliSumOp: operator for the Hamiltonian
        float: a constant shift for the obj function.
    """
    num_nodes = len(weight_matrix)
    pauli_list = []
    shift = 0

    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5, Pauli((z_p, x_p))])
                shift += 0.5

    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         if i != j:
    #             x_p = np.zeros(num_nodes, dtype=bool)
    #             z_p = np.zeros(num_nodes, dtype=bool)
    #             z_p[i] = True
    #             z_p[j] = True
    #             pauli_list.append([1, Pauli((z_p, x_p))])
    #         else:
    #             shift += 1

    pauli_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
    return PauliSumOp.from_list(pauli_list), shift, pauli_list


def circuit_without_classreg(circuit):
    """
    Return a new circuit which has the same operations as the original circuit without classical register and measurement
    """
    new_dag = DAGCircuit()
    original_dag = circuit_to_dag(circuit)
    for qreg in original_dag.qregs.values():
        new_dag.add_qreg(qreg)
    for node in original_dag.topological_op_nodes():
        if node.name != 'measure':
            new_dag.apply_operation_back(node.op,
                                         qargs=[Qubit(new_dag.qregs['q'], qarg.index ) for qarg in
                                                  node.qargs],
                                           )
    new_circuit = dag_to_circuit(new_dag)
    return new_circuit


def cut_counts(counts, bit_indexes):
    """
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param bit_indexes: a list of indexes
    :return: new_counts for the  specified bit_indexes
    """
    bit_indexes.sort(reverse=True)
    new_counts = {}
    for key in counts:
        new_key = ''
        for index in bit_indexes:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts


def expectation_term(counts, shots, z_index_list):
    """
    :param shots: shots of the experiment
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param z_index_list: a list of indexes
    :return: the expectation value of ZZ...Z operator for given z_index_list
    """

    expectation = 0
    z_counts = cut_counts(counts, z_index_list)
    for key in z_counts:
        sign = -1
        if key.count('1') % 2 == 0:
            sign = 1
        expectation += sign * z_counts[key] / shots

    return expectation

def expectation_result(counts, shots, pauli_list):
    """
    Calculate the expectation value based on the count rsult and hamiltonian
    :param counts: count result obtained by qiskit job
    :param shots: number of measurements
    :param pauli_list: a list of pauli strings that constitute a Hamiltonian
    :return: the expectation value
    """
    expectation = 0
    pauli_len = len(pauli_list[0][0])
    for pauli in pauli_list:
        z_index = []
        for index, p in enumerate(pauli[0]):
            if p == 'Z':
                z_index.append(pauli_len -1 - index)
        coeff = pauli[1]
        expectation += coeff * expectation_term(counts, shots, z_index)
    return expectation



