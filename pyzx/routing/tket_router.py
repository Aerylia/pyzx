from pytket.pyzx import pyzx_to_tk, tk_to_pyzx
from pytket._routing import route, Architecture, graph_placement, QubitMap
from pytket._transform import Transform
from pytket import OpType, UnitID

from ..circuit import Circuit, ZPhase, CNOT

def route_tket(circuit, architecture, initial_mapping=None):
    tk_circuit = pyzx_to_tk(circuit)
    arch = get_tk_architecture(architecture)
    if initial_mapping is None:
        outcirc = route(tk_circuit, arch)
    else:
        qmap = QubitMap()
        for i, j in enumerate(initial_mapping):
            qmap[UnitID('q', i)] = UnitID('flatnode', j) # TODO make flatnode into node once pytket is updated to 0.4
        outcirc = route(tk_circuit, arch, initial_map=qmap)
    Transform.DecomposeSWAPtoCX().apply(outcirc)
    #outcirc.decompose_SWAP_to_CX()
    return outcirc

def tket_to_pyzx(circuit):
    Transform.RebaseToPyZX().apply(circuit)
    c = Circuit(len(circuit.qubits))
    for gate in circuit.get_commands():
        if gate.op.get_type() == OpType.Rz:
            c.add_gate(ZPhase(gate.qubits[0].index[0], gate.op.get_params()[0]))
        elif gate.op.get_type() == OpType.CX:
            c.add_gate(CNOT(gate.qubits[0].index[0], gate.qubits[1].index[0]))
        else:
            print("Warning: gate not supported, skipping", gate)
    return c

def get_tk_architecture(architecture):
    coupling_graph = [e for e in architecture.graph.edges()]
    return Architecture(coupling_graph)
    