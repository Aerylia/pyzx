import sys, os, re
import datetime
import numpy as np
from pandas import DataFrame, concat

if __name__ == '__main__':
    print("Please call this as python -m pyzx phasepoly ...")
    exit()

from ..routing.architecture import architectures, SQUARE, create_architecture, dynamic_size_architectures, FULLY_CONNNECTED
from ..routing.cnot_mapper import STEINER_MODE, TKET_COMPILER, sequential_map_cnot_circuits, elim_modes, compiler_modes, GENETIC_STEINER_MODE, PSO_STEINER_MODE, GAUSS_MODE, sequential_gauss
from ..routing.phase_poly import route_phase_poly, TKET_STEINER_MODE
from ..utils import make_into_list, restricted_float
from ..circuit import Circuit, ZPhase, CNOT
from ..routing.tket_router import pyzx_to_tk, OpType, route_tket, tket_to_pyzx
from ..parity_maps import CNOT_tracker

TKET_THEN_STEINER_MODE = "tket->steiner"
PHASEPOLY_TKET_MODE = "phase_poly->tket"


description = "Compiles given qasm files or those in the given folder to a given architecture."

import argparse
parser = argparse.ArgumentParser(prog="pyzx phase poly", description=description)
parser.add_argument("QASM_source", nargs='+', help="The QASM file or folder with QASM files to be routed.")
parser.add_argument("-m", "--mode", nargs='+', dest="mode", default=STEINER_MODE, help="The mode specifying how to route. choose 'all' for using all modes.", choices=[TKET_COMPILER, STEINER_MODE, TKET_STEINER_MODE, TKET_THEN_STEINER_MODE, PHASEPOLY_TKET_MODE, GAUSS_MODE])
parser.add_argument("-a", "--architecture", nargs='+', dest="architecture", default=SQUARE, choices=architectures, help="Which architecture it should run compile to.")
parser.add_argument("-q", "--qubits", nargs='+', default=None, type=int, help="The number of qubits for the fully connected architecture.")
#parser.add_argument("--population", nargs='+', default=10, type=int, help="The population size for the genetic algorithm.")
#parser.add_argument("--iterations", nargs='+', default=5, type=int, help="The number of iterations for the genetic algorithm.")
#parser.add_argument("--crossover_prob", nargs='+', default=0.8, type=restricted_float, help="The crossover probability for the genetic algorithm. Must be between 0.0 and 1.0.")
#parser.add_argument("--mutation_prob", nargs='+', default=0.2, type=restricted_float, help="The mutation probability for the genetic algorithm. Must be between 0.0 and 1.0.")
#parser.add_argument("--perm", default="both", choices=["row", "col", "both"], help="Whether to find a single optimal permutation that permutes the rows, columns or both with the genetic algorithm.")
#parser.add_argument("--destination", help="Destination file or folder where the compiled circuit should be stored. Otherwise the source folder is used.")
parser.add_argument("--metrics_csv", default=None, help="The location to store compiling metrics as csv, if not given, the metrics are printed.")
#parser.add_argument("--n_compile", default=1, type=int, help="How often to run the Quilc compiler, since it is not deterministic.")
parser.add_argument("--subfolder", default=None, type=str, nargs="+", help="Possible subfolders from the main QASM source to compile from. Less typing when source folders are in the same folder. Can also be used for subfiles.")
parser.add_argument("--raw", default=False, type=bool, help="Whether the results should be raw or aggregated with mean/median/min/max")
parser.add_argument("--notes", default="", type=str, help="Extra notes that can be added to the csv")
parser.add_argument("--placement", default=True, type=bool, help="Whether tket should optimize placement")
#parser.add_argument("-n", "--n_circuits", nargs='+', dest="n", default=20, type=int, help="The number of circuits to generate.")
#parser.add_argument("-p", "--n_phase_layers", nargs='+', dest="phase_layers", default=1, type=int, help="Number of layers with phases in the circuits to be generated.")
#parser.add_argument("-c", "--cnots_between_layers", nargs='+', dest="cnots", default=5, type=int, help="Number of CNOT gates between each phase layer in the circuits to be generated.")

#TODO add PSO arguments

def main(args):
    args = parser.parse_args(args)
    if args.metrics_csv is not None and os.path.exists(args.metrics_csv):
        delete_csv = None
        text = input("The given metrics file [%s] already exists. Do you want to overwrite it? (Otherwise it is appended) [y|n]" % args.metrics_csv)
        if text.lower() in ['y', "yes"]:
            delete_csv = True
        elif text.lower() in ['n', 'no']:
            delete_csv = False
        while delete_csv is None:
            text = input("Please answer yes or no.")
            if text.lower() in ['y', "yes"]:
                delete_csv = True
            elif text.lower() in ['n', 'no']:
                delete_csv = False
        if delete_csv:
            os.remove(args.metrics_csv)

    sources = make_into_list(args.QASM_source)
    if args.subfolder is not None:
        print(args.subfolder)
        if args.subfolder == ["."]:
            sources = [os.path.join(d, o) for d in sources for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
        else:
            sources = [os.path.join(source, subfolder) for source in sources for subfolder in args.subfolder if os.path.isdir(source)]
        # Remove non existing paths

    sources = [source for source in sources if os.path.exists(source) or print("Warning, skipping non-existing source:", source)]
    sources = [os.path.join(source, f) for source in sources for f in os.listdir(source)]
    sources = [source for source in sources if os.path.isfile(source)]

    if "all" in args.mode:
        mode = elim_modes + [TKET_COMPILER]
    else:
        mode = args.mode

    #all_circuits = [] 
    #for source in sources:
    #    print("Mapping qasm files in path:", source)
    all_results = []
    for a in args.architecture:
        if a in dynamic_size_architectures:
            archs = [create_architecture(a, n_qubits=q) for q in args.qubits]
        else:
            archs = [create_architecture(a)]
        for architecture in archs:
            results_df = map_phase_poly_circuits(sources, architecture, mode)
            if not args.raw:
                kwargs = {"level":["mode", "#cnots_per_layer", "#phase_layers"]}
                results_df = concat([results_df.mean(**kwargs).add_suffix("_mean"), 
                                    results_df.median(**kwargs).add_suffix("_median"), 
                                    results_df.min(**kwargs).add_suffix("_min"), 
                                    results_df.max(**kwargs).add_suffix("_max")], axis=1)
                #results_df["# phase layers"] = n_phase_layers
                #results_df["# cnots per layer"] = n_cnots_per_layer
                #results_df["architecture"] = architecture.name
                #results_df.set_index(["# phase layers","# cnots per layer", "architecture"], inplace=True, append=True)
            results_df["notes"] = args.notes 
            all_results.append(results_df)
        if args.metrics_csv and all_results != []:
            final_df = concat(all_results)
            if os.path.exists(args.metrics_csv):
                with open(args.metrics_csv, 'a') as f:
                    final_df.to_csv(f, header=False)
            else:
                final_df.to_csv(args.metrics_csv)
            all_results = []
    if not args.metrics_csv and all_results != []:
        print(concat(all_results))
         

def map_phase_poly_circuits(sources, architecture, modes, placement=True, **kwargs):
    modes = make_into_list(modes)
    circuits = [Circuit.from_qasm_file(f) for f in sources]
    all_results = []
    full_connected = create_architecture(FULLY_CONNNECTED, n_qubits=architecture.n_qubits)
    tket_initial_mapping = None if placement else [i for i in range(architecture.n_qubits)]
    for mode in modes:
        for i, circuit in enumerate(circuits):
            t = datetime.datetime.now()
            if mode == TKET_COMPILER or mode == TKET_THEN_STEINER_MODE:
                a = architecture if mode == TKET_COMPILER else full_connected
                c = route_tket(circuit.copy(), a, initial_mapping=tket_initial_mapping)
            elif mode == PHASEPOLY_TKET_MODE or mode == GAUSS_MODE:
                c = route_phase_poly(circuit.copy(), architecture, GAUSS_MODE)
            else:
                c = route_phase_poly(circuit.copy(), architecture, mode, n_steps=5, population=5)
            if mode == PHASEPOLY_TKET_MODE:
                c = route_tket(c.copy(), architecture, initial_mapping=tket_initial_mapping)
            elif mode == TKET_THEN_STEINER_MODE:
                # TODO split the circuit c into CNOT sections and route that sequential steiner gauss
                matrices, phases = zip(*tket_to_cnots(c))
                circuits, _, _ = sequential_gauss([m.copy() for m in matrices], mode=STEINER_MODE, architecture=architecture, full_reduce=True)
                c = Circuit(architecture.n_qubits)
                for circuit, p in zip(circuits, phases):
                    for gate in circuit.gates:
                        c.add_gate(gate)
                    for phase in p:
                        c.add_gate(phase)
            t = datetime.datetime.now() - t
            original_CNOTs = get_metrics(circuit).add_prefix("Original ")
            results = get_metrics(c)
            results["time"] = t
            results["idx"] = i
            results["mode"] = mode 
            file = sources[i]
            match = re.search('/(\d+)layers(\d+)cnots', file)
            results["file"] = file
            if match is not None:
                results["#phase_layers"] = int(match.group(1))
                results["#cnots_per_layer"] = int(match.group(2))
            else:
                results["#phase_layers"] = None
                results["#cnots_per_layer"] = None
            results = results.join(original_CNOTs)
            results.set_index(["#phase_layers", "#cnots_per_layer"], inplace=True)
            all_results.append(results)
            #print("done", i)
    results_df = concat(all_results)
    results_df.set_index(["idx", "mode", "file"], inplace=True, append=True)
    return results_df

def get_metrics(circuit):
    if isinstance(circuit, Circuit):
        tk_circuit = pyzx_to_tk(circuit)
    else:
        tk_circuit = circuit
    metrics = {}
    metrics["CX depth"] = tk_circuit.depth_by_type(OpType.CX)
    metrics["# CX"] = tk_circuit.n_gates_of_type(OpType.CX)
    metrics["Rz depth"] = tk_circuit.depth_by_type(OpType.Rz)
    metrics["# Rz"] = tk_circuit.n_gates_of_type(OpType.Rz)
    return DataFrame([metrics])

def tket_to_cnots(circuit):
    cnots = CNOT_tracker(circuit.n_qubits)
    phases = []
    split_gates = []
    for gate in circuit.get_commands():
        if gate.op.get_type() == OpType.Rz:
            phases.append(ZPhase(gate.qubits[0].index[0], gate.op.get_params()[0]))
        elif gate.op.get_type() == OpType.CX:
            if phases != []:   
                split_gates.append((cnots.matrix, phases))
                cnots = CNOT_tracker(circuit.n_qubits)
                phases = []             
            cnots.add_gate(CNOT(gate.qubits[0].index[0], gate.qubits[1].index[0]))
        else:
            print("Warning: gate not supported, skipping", gate)
    split_gates.append((cnots.matrix, phases))
    return split_gates