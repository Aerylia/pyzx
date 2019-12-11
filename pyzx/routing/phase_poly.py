# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2019 - Aleks Kissinger, John van de Wetering,
#                      and Arianne Meijer-van de Griend

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pandas import DataFrame, concat
import numpy as np 

from ..circuit import Circuit, ZPhase, Fraction, HAD, XPhase, CNOT
from ..parity_maps import build_random_parity_map, CNOT_tracker
from ..graph.graph import  Graph
from ..linalg import Mat2
from ..routing.cnot_mapper import sequential_gauss, GAUSS_MODE, STEINER_MODE, TKET_COMPILER
from ..routing.tket_router import get_tk_architecture, pyzx_to_tk, graph_placement

TKET_STEINER_MODE = "tket-steiner"
        
def route_phase_poly(circuit, architecture, mode, **kwargs):
    phase_poly = PhasePoly.fromCircuit(circuit)
    new_circuit = phase_poly.synthesize(mode, architecture, **kwargs)[0]
    return new_circuit

def make_random_phase_poly(n_qubits, n_phase_layers, cnots_per_layer, return_circuit=False):
    c = CNOT_tracker(n_qubits)
    for _ in range(n_phase_layers):
        build_random_parity_map(n_qubits, cnots_per_layer, circuit=c)
        for i in range(n_qubits):
            phase = np.random.choice([1,-1])*Fraction(1, int(np.random.choice([1,2,4])))
            c.add_gate(ZPhase(target=i, phase=phase))
    if return_circuit:
        return c
    else:
        return PhasePoly.fromCircuit(c)

class PhasePoly():

    def __init__(self, zphase_dict, out_parities):
        self.zphases = zphase_dict
        self.out_par = out_parities
        self.n_qubits = len(out_parities[0])
        self.all_parities = list(zphase_dict.keys())

    @staticmethod
    def fromCircuit(circuit, initial_qubit_placement=None, final_qubit_placement=None):
        zphases = {}
        current_parities = mat22partition(Mat2.id(circuit.qubits))
        if initial_qubit_placement is not None:
            current_parities = ["".join([row[i] for i in initial_qubit_placement]) for row in current_parities]
        for gate in circuit.gates:
            parity = current_parities[gate.target]
            if gate.name in ["CNOT", "CX"]:
                # Update current_parities
                control = current_parities[gate.control]
                current_parities[gate.target] = "".join([str((int(i)+int(j))%2) for i,j in zip(control, parity)])
            elif isinstance(gate, ZPhase):
                # Add the T rotation to the phases
                if parity in zphases:
                    zphases[parity] += gate.phase
                else: 
                    zphases[parity] = gate.phase
            elif isinstance(gate, XPhase):
                print("X phases not yet supported!")
            else:
                print("Gate not supported!", gate.name)
        def clamp(phase):
            new_phase = phase%2
            if new_phase > 1:
                return new_phase -2
            return new_phase
        zphases = {par:clamp(r) for par, r in zphases.items() if clamp(r) != 0}
        if final_qubit_placement is not None:
            current_parities = [ current_parities[i] for i in final_qubit_placement]
        return PhasePoly(zphases, current_parities)

    def partition(self, skip_output_parities=True, optimize_parity_order=False):
        # Matroid partitioning based on wikipedia: https://en.wikipedia.org/wiki/Matroid_partitioning
        def add_edge(graph, vs_dict, node1, node2):
            v1 = [k for k,v in vs_dict.items() if v==node1][0]
            v2 = [k for k,v in vs_dict.items() if v==node2][0]
            graph.nedges += 1
            graph.graph[v1][v2] = 1

        partitions = []
        parities_to_partition = self.all_parities if not skip_output_parities else [p for p in self.all_parities if p not in self.out_par]
        for parity in parities_to_partition:
            graph = Graph()
            # Add current parity to the graph
            # Add each partition to the graph
            # Add each parity in the partition to the graph
            vs = graph.add_vertices(len(partitions)+sum([len(p) for p in partitions])+1)
            vs_dict = {i:node for i,node in zip(vs, partitions+[e for p in partitions for e in p]+[parity])}
            # Check which parities can be added to another partition
            for partition in partitions:
                # Check if the new parity can be added to the partition
                if self._independent(list(partition) + [parity]):
                    add_edge(graph, vs_dict, parity, partition)
                for p in partition:
                    # Check if p can be replaced with the new parity in the partition
                    for partition2 in partitions:
                        if partition2 != partition:
                            for p2 in partition2:
                                # Check if p2 can be replaced with p in their partitions
                                new_partition = [e for e in partition if e != p] + [p2]
                                if self._independent(new_partition):
                                    add_edge(graph, vs_dict, p2, p)
            # Find a path from the parity to a parition
            path = self._dfs([(parity, [parity])], graph, vs_dict, partitions)
            if path != []:
                # Apply those changes if such a path exists
                # Remember which partition to add the final element to
                p_idx = partitions.index(path[-1]) # Last element in the path is always a partition
                for p1, p2 in zip(path[:-1], path[1:]):
                    # Replace p2 with p1 in p1's partition
                    for partition in partitions:
                        if p1 in partition:
                            partition.pop(p1)
                            partition.append(p2)
                partitions[p_idx].append(path[-2])
            else:
                # Make a new partition if no such path exists.
                partitions.append([parity])
        ordered_partitions = []
        for partition in partitions:
            # Add identity parities to the partition if the set is not full yet.
            matrix = partition2mat2(partition)
            matrix, _ = inverse_hack(matrix)
            if optimize_parity_order:
                # Find a more optimal ordering of the parities
                parity_placement = self._place_parities(matrix)
            else:
                parity_placement = np.arange(self.n_qubits)
            partition = mat22partition(matrix)
            new_partition = [partition[i] for i in parity_placement]
            ordered_partitions.append(new_partition)
        return ordered_partitions

    def _order_partitions(self, partitions):
        n = len(partitions)
        numbered = {i:partition2mat2(partition) for i, partition in enumerate(partitions)}
        def cost_func(p1, p2):
            _, inv = inverse_hack(p1)
            #return sum(sum((p2*inv).data, []))
            c = CNOT_tracker(len(partitions[0]))
            (p2*inv).gauss(full_reduce=True, y=c)
            return len(c.gates)
        path_costs = {(i,j):cost_func(p1, p2) for i, p1 in numbered.items() for j, p2 in numbered.items() if i != j}
        # start at the back
        new_partitions = [partitions[-1]]
        visited = []
        current = n-1
        while len(visited) < n-1:
            # Pick the partition that was not yet visited whose 
            choice = min([i for i in range(n) if i not in visited + [current]], key=lambda i: path_costs[(i,current)])
            new_partitions = [mat22partition(numbered[choice])] + new_partitions
            visited += [current]
            current = choice
        return new_partitions

    def _place_parities(self, parities):
        if isinstance(parities, Mat2):
            parities = parities.data
        permutation = [None]*self.n_qubits
        skipped_parities = []
        # Greedily map identity rows
        for i, parity in enumerate(parities):
            if parity.count(1) == 1:
                permutation[parity.index(1)] = i
            else:
                skipped_parities.append((i, parity))
        skipped_idxs = []
        deliberating = []
        for i in range(self.n_qubits):
            if permutation[i] is None:
                # Find the best parity in skipped_parities to place there
                pivotted_parities = [(j, p) for j, p in skipped_parities if p[i] == 1]
                if pivotted_parities:
                    # The parity with the least 1s has the most priority because it can probably not be placed elsewhere
                    pivotted_parities = sorted(pivotted_parities, key=lambda parity: parity[1].count(1))
                    if len(pivotted_parities) == 1:
                        chosen = pivotted_parities[0]
                        permutation[i] = chosen[0]
                        skipped_parities.remove(chosen)
                    else:
                        # If there are multiple options, try them later
                        deliberating.append((i, pivotted_parities))
                else:
                    skipped_idxs.append(i)
        for i, pivotted_parities in deliberating:
            pivotted_parities = [p for p in pivotted_parities if p in skipped_parities]
            if pivotted_parities:
                chosen = pivotted_parities[0]
                permutation[i] = chosen[0]
                skipped_parities.remove(chosen)
            else:
                skipped_idxs.append(i)
        for i in skipped_idxs:
            permutation[i] = skipped_parities.pop()[0]
        return np.asarray(permutation)

    def _dfs(self, nodes, graph, inv_vs_dict, partitions):
        # recursive dfs to find the shortest path
        if nodes == []:
            return []
        new_nodes = []
        for (node, path) in nodes:
            # For each node2 connected to node
            v = [k for k,v in inv_vs_dict.items() if v==node][0]
            for node2 in graph.graph[v].keys():
                # If it's a partition, we found a path
                next_node = inv_vs_dict[node2]
                if next_node not in path: # Avoid loops!
                    new_path = path + [next_node]
                    if next_node in partitions:
                        return new_path
                    else:
                        new_nodes.append((next_node, new_path))
        return self._dfs(new_nodes, graph, inv_vs_dict, partitions)

    def _independent(self, partition):
        return inverse_hack(partition2mat2(partition)) is not None

    def synthesize(self, mode, architecture, optimize_parity_order=False, optimize_partition_order=True, iterative_placement=False, parity_permutation=True, iterative_initial_placement=False, **kwargs):
        kwargs["full_reduce"] = True
        n_qubits = architecture.n_qubits if architecture is not None else len(self.out_par)
        # Partition and order the parities
        partitions = self.partition(optimize_parity_order=optimize_parity_order)+[self.out_par]
        if optimize_partition_order:
            partitions = self._order_partitions(partitions)
        # Make the parity sets into matrices
        matrices = [partition2mat2(partition) for partition in partitions]
        # The matrices to be computed need to first undo the previous parities and then obtain the new parities
        other_matrices = []
        prev_matrix = Mat2.id(n_qubits)
        for m in matrices:
            new_matrix, inverse = inverse_hack(m)
            other_matrices.append(new_matrix*prev_matrix) 
            prev_matrix = inverse
        if mode == TKET_STEINER_MODE:
            #print(len(partitions))
            perms = []
            # TODO - this can be made more memory/time efficient by doing it in reverse.
            arch = get_tk_architecture(architecture)
            current_perm = np.arange(n_qubits)
            #suggestions = []
            for idx in range(len(other_matrices)):
                # Make the partial circuit
                circuit = Circuit(n_qubits)
                next_perm = current_perm
                if parity_permutation:
                    next_perm = self._place_parities([[row[j] for j in current_perm] for row in other_matrices[idx].data])
                possible_perms = []
                do_loop = True
                n_gates = None
                while do_loop and (next_perm.tolist() not in possible_perms) and (n_gates is None or len(circuit.gates) < n_gates):
                    possible_perms.append(next_perm.tolist())
                    n_gates = len(circuit.gates)
                    if iterative_initial_placement:
                        iterative_placement = False
                    else:
                        do_loop = False
                    permuted_matrices = [Mat2([[other_matrices[idx].data[k][l] for l in current_perm] for k in next_perm])]
                    if idx < len(other_matrices)-1:
                        permuted_matrices += [Mat2([[m.data[k][l] for l in next_perm] for k in next_perm]) for m in other_matrices[idx+1:]]
                    temp_CNOT_circuits, _, _ = sequential_gauss(permuted_matrices, mode=GAUSS_MODE, architecture=architecture, full_reduce=True)
                    circuit = Circuit(n_qubits)
                    for c in temp_CNOT_circuits:
                        for g in c.gates:
                            circuit.add_gate(g)
                    if iterative_placement or idx == 0:
                        # Get a better qubit placement from tket
                        tk_circuit = pyzx_to_tk(circuit)
                        new_perm = graph_placement(tk_circuit, arch)
                        # Parse the placement into a np.array permutation
                        perm_dict = {p[1].index[0]: p[0].index[0] for p in new_perm.items()} 
                        #suggestions.append((perm_dict, current_perm, next_perm, circuit.gates[:len(temp_CNOT_circuits[0].gates)]))
                        new_perm = [perm_dict[i] if i in perm_dict else None for i in range(n_qubits)]
                        perm_idx  = [i for i in range(n_qubits) if new_perm[i] is None]
                        perm_val = [i for i in range(n_qubits) if i not in new_perm]
                        for j, v in zip(perm_idx, perm_val):
                            new_perm[j] = v
                        new_perm = np.asarray(new_perm)
                    if idx == 0:
                        current_perm = current_perm[new_perm]
                    next_perm = next_perm[new_perm]
                if iterative_initial_placement:
                    next_perm = np.asarray(possible_perms[-1])
                perms.append(current_perm)
                current_perm = next_perm
            # Add the intitial permutation of the last CNOT block as final permutation of the full circuit
            perms.append(next_perm) 
            # Rotate all matrices accordingly
            new_matrices = []
            for i, input_perm, output_perm in zip(range(len(other_matrices)), perms, perms[1:]):
                new_matrices.append(Mat2([[other_matrices[i].data[k][l] for l in input_perm] for k in output_perm]))
            CNOT_circuits, _, _ = sequential_gauss([m.copy() for m in new_matrices], mode=STEINER_MODE, architecture=architecture, **kwargs)
            #print(*["\n".join([str(x) for x in [m1,s, p1, p2, m2, c.gates]]) for p1, p2, m1, m2,c,s in zip(perms, perms[1:], other_matrices, new_matrices, CNOT_circuits, suggestions)], sep="\n\n")
        else:
            CNOT_circuits, perms, _ = sequential_gauss([m.copy() for m in other_matrices], mode=mode, architecture=architecture, **kwargs)
        zphases = list(self.zphases.keys())
        circuit = Circuit(n_qubits)
        # Keep track of the parities
        current_parities = Mat2([[Mat2.id(n_qubits).data[i][perms[0].tolist().index(j)] for j in range(n_qubits)] for i in range(n_qubits)])
        for c in CNOT_circuits:
            # Obtain the specified parity
            for gate in c.gates:
                # CNOTs have been mapped already, do not need to be adjusted!
                circuit.add_gate(gate)
                current_parities.row_add(gate.control, gate.target)
            # Place the rotations at each parity
            for target, p in enumerate(current_parities.data):
                parity = "".join([str(v) for v in p])
                # Apply the phases at current parity if needed.
                if parity in zphases: 
                    phase = self.zphases[parity]
                    gate = ZPhase(target=target, phase=phase)
                    zphases.remove(parity)
                    circuit.add_gate(gate)
        return circuit, perms[0], perms[-1]

def tpar(circuit, mode, architecture, input_perm=True, output_perm=True, **kwargs):
    """
    Broken!
    """
    n_qubits = architecture.n_qubits
    current_perm = [i for i in range(n_qubits)]
    initial_perm = current_perm
    c = Circuit(n_qubits)
    final_circuit = Circuit(n_qubits)
    for gate in circuit.gates:
        if gate.name == "HAD":
            # Deal with the phase polynomial that came before
            phase_poly = PhasePoly.fromCircuit(c, initial_qubit_placement=current_perm)
            new_circ, in_perm, out_perm = phase_poly.synthesize(mode, architecture, input_perm=input_perm, output_perm=output_perm, **kwargs)
            # Store the initial permutation
            if input_perm:
                initial_perm = in_perm
                input_perm = False
            # Copy the synthesized circuit
            for g in new_circ.gates:
                final_circuit.add_gate(g)

            # Update new permutation
            current_perm = out_perm
            # Add the hadamard gate
            g = HAD(current_perm[gate.target])
            final_circuit.add_gate(g)
            # Start a new circuit
            c = Circuit(n_qubits)
        else:
            c.add_gate(gate)
    phase_poly = PhasePoly.fromCircuit(c)
    new_circ, in_perm, out_perm = phase_poly.synthesize(mode, architecture, input_perm=input_perm, output_perm=output_perm, **kwargs)
    # Store the initial permutation
    if input_perm:
        initial_perm = in_perm
    # Copy the synthesized circuit
    for g in new_circ.gates:
        final_circuit.add_gate(g)
    return final_circuit, initial_perm, out_perm

def inverse_hack(matrix):
    m = matrix.copy()
    if matrix.rows() != matrix.cols(): 
        # Compute which columns are independent from the matrix
        rank = m.gauss(full_reduce=False)
        cols = []
        row = 0
        for col in range(m.cols()):
            if row >= m.rows() or m.data[row][col] == 0 :
                cols.append(col)
            else:
                row += 1
        # Add those columns as rows to the matrix
        matrix = Mat2(matrix.data + [[1 if c==i else 0 for i in range(m.cols())] for c in cols])
        m = matrix.copy()
    # Compute the inverse
    inv = Mat2.id(matrix.rows())
    rank = m.gauss(x=inv, full_reduce=True)
    if rank < matrix.rows(): return None
    else: return matrix, inv

def partition2mat2(partition):
    return Mat2([[int(i) for i in parity] for parity in partition])

def mat22partition( m):
    return ["".join(str(i) for i in parity) for parity in m.data]