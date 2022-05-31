# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the EchoRZXWeylDecomposition pass"""
import unittest
from math import pi
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import (
    EchoRZXWeylDecomposition,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeParis

import qiskit.quantum_info as qi

from qiskit.quantum_info.synthesis.two_qubit_decompose import (
    TwoQubitWeylDecomposition,
)


class TestEchoRZXWeylDecomposition(QiskitTestCase):
    """Tests the EchoRZXWeylDecomposition pass."""

    def setUp(self):
        super().setUp()
        self.backend = FakeParis()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def test_rzx_number_native_weyl_decomposition(self):
        """Check the number of RZX gates for a hardware-native cx"""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])

        unitary_circuit = qi.Operator(circuit).data

        after = EchoRZXWeylDecomposition(self.inst_map)(circuit)

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

        # check whether after circuit has correct number of rzx gates
        self.assertRZXgates(unitary_circuit, after)

    def test_non_native_weyl_decomposition(self):
        """Check the number of RZX gates for a non-hardware-native rzz"""
        theta = pi / 9
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.rzz(theta, qr[1], qr[0])

        unitary_circuit = qi.Operator(circuit).data

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

        # check whether after circuit has correct number of rzx gates
        self.assertRZXgates(unitary_circuit, after)

    def assertRZXgates(self, unitary_circuit, after):
        """Check the number of rzx gates"""
        alpha = TwoQubitWeylDecomposition(unitary_circuit).a
        beta = TwoQubitWeylDecomposition(unitary_circuit).b
        gamma = TwoQubitWeylDecomposition(unitary_circuit).c

        # check whether after circuit has correct number of rzx gates
        expected_rzx_number = 0
        if not alpha == 0:
            expected_rzx_number += 2
        if not beta == 0:
            expected_rzx_number += 2
        if not gamma == 0:
            expected_rzx_number += 2

        circuit_rzx_number = QuantumCircuit.count_ops(after)["rzx"]

        self.assertEqual(expected_rzx_number, circuit_rzx_number)

    @staticmethod
    def count_gate_number(gate, circuit):
        """Count the number of a specific gate type in a circuit"""
        if gate not in QuantumCircuit.count_ops(circuit):
            gate_number = 0
        else:
            gate_number = QuantumCircuit.count_ops(circuit)[gate]
        return gate_number

    def test_h_number_non_native_weyl_decomposition_1(self):
        """Check the number of added Hadamard gates for an rzz gate"""
        theta = pi / 11
        qr = QuantumRegister(2, "qr")
        # rzz gate in native direction
        circuit = QuantumCircuit(qr)
        circuit.rzz(theta, qr[0], qr[1])

        # rzz gate in non-native direction
        circuit_non_native = QuantumCircuit(qr)
        circuit_non_native.rzz(theta, qr[1], qr[0])

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        dag_non_native = circuit_to_dag(circuit_non_native)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after_non_native = dag_to_circuit(pass_.run(dag_non_native))

        circuit_rzx_number = self.count_gate_number("rzx", after)

        circuit_h_number = self.count_gate_number("h", after)
        circuit_non_native_h_number = self.count_gate_number("h", after_non_native)

        # for each pair of rzx gates four hadamard gates have to be added in
        # the case of a non-hardware-native directed gate.
        self.assertEqual(
            (circuit_rzx_number / 2) * 4, circuit_non_native_h_number - circuit_h_number
        )

    def test_h_number_non_native_weyl_decomposition_2(self):
        """Check the number of added Hadamard gates for a swap gate"""
        qr = QuantumRegister(2, "qr")
        # swap gate in native direction
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])

        # swap gate in non-native direction
        circuit_non_native = QuantumCircuit(qr)
        circuit_non_native.swap(qr[1], qr[0])

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        dag_non_native = circuit_to_dag(circuit_non_native)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after_non_native = dag_to_circuit(pass_.run(dag_non_native))

        circuit_rzx_number = self.count_gate_number("rzx", after)

        circuit_h_number = self.count_gate_number("h", after)
        circuit_non_native_h_number = self.count_gate_number("h", after_non_native)

        # for each pair of rzx gates four hadamard gates have to be added in
        # the case of a non-hardware-native directed gate.
        self.assertEqual(
            (circuit_rzx_number / 2) * 4, circuit_non_native_h_number - circuit_h_number
        )

    def test_weyl_unitaries_random_circuit(self):
        """Weyl decomposition for random two-qubit circuit."""
        theta = pi / 9
        epsilon = 5
        delta = -1
        eta = 0.2
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)

        # random two-qubit circuit
        circuit.rzx(theta, 0, 1)
        circuit.rzz(epsilon, 0, 1)
        circuit.rz(eta, 0)
        circuit.swap(1, 0)
        circuit.h(0)
        circuit.rzz(delta, 1, 0)
        circuit.swap(0, 1)
        circuit.cx(1, 0)
        circuit.swap(0, 1)
        circuit.h(1)
        circuit.rxx(theta, 0, 1)
        circuit.ryy(theta, 1, 0)
        circuit.ecr(0, 1)

        unitary_circuit = qi.Operator(circuit).data

        dag = circuit_to_dag(circuit)
        pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))


if __name__ == "__main__":
    unittest.main()