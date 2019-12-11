import sys, os
if __name__ == '__main__':
    sys.path.append('..')
import numpy as np
try:
    from pandas import DataFrame
except:
    DataFrame = None
    if __name__ == '__main__':
        print("Warning: could not import pandas. No performance data will be exported.")
import time

from ..linalg import Mat2
from .architecture import create_fully_connected_architecture, create_architecture, dynamic_size_architectures
from ..parity_maps import CNOT_tracker
from ..machine_learning import GeneticAlgorithm, ParticleSwarmOptimization
from ..utils import make_into_list
#from .steiner import steiner_gauss
from .steiner import rec_steiner_gauss as steiner_gauss

debug = False

# ELIMINATION MODES:
GAUSS_MODE = "gauss"
STEINER_MODE = "steiner"
GENETIC_STEINER_MODE = "genetic_steiner"
GENETIC_GAUSS_MODE = "genetic_gauss"
PSO_GAUSS_MODE = "pso_gauss"
PSO_STEINER_MODE = "pso_steiner"

elim_modes = [STEINER_MODE, GAUSS_MODE, GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE]
genetic_elim_modes = [GENETIC_STEINER_MODE, GENETIC_GAUSS_MODE]
pso_elim_modes = [PSO_GAUSS_MODE, PSO_STEINER_MODE]
basic_elim_modes = [STEINER_MODE, GAUSS_MODE]
elim_modes = genetic_elim_modes + pso_elim_modes + basic_elim_modes

# COMPILE MODES
QUIL_COMPILER = "quilc"
NO_COMPILER = "not_compiled"
TKET_COMPILER = "tket"

compiler_modes = [QUIL_COMPILER, NO_COMPILER]

COMBINED_METRIC = "combined"
DEPTH_METRIC = "depth"
COUNT_METRIC = "count"
metrics = [COMBINED_METRIC, DEPTH_METRIC, COUNT_METRIC]

def depth_fitness_func(mode, matrix, architecture, row=True, col=True, full_reduce=True, **kwargs):
    metric_func = lambda c: c.cnot_depth()
    return basic_fitness_func(DEPTH_METRIC, mode, matrix, architecture, row, col, full_reduce, **kwargs)

def cnot_fitness_func(mode, matrix, architecture, row=True, col=True, full_reduce=True, **kwargs):
    metric_func = lambda c: c.count_cnots()
    return basic_fitness_func(COUNT_METRIC, mode, matrix, architecture, row, col, full_reduce, **kwargs)

def combined_fitness_func(mode, matrix, architecture, row=True, col=True, full_reduce=True, **kwargs):
    metric_func = lambda c: c.cnot_depth()*10000 + c.count_cnots()
    return basic_fitness_func(COMBINED_METRIC, mode, matrix, architecture, row, col, full_reduce, **kwargs)

def basic_fitness_func(metric_func, mode, matrix, architecture, row=True, col=True, full_reduce=True, **kwargs):
    """
    Creates and returns a fitness function to be used for the genetic algorithm that uses CNOT gate count as fitness.

    :param mode: The type of Gaussian elimination to be used
    :param matrix: A Mat2 parity map to route.
    :param architecture: The architecture to take into account when routing
    :param row: Whether to find a row permutation
    :param col: Whether to find a column permutation
    :param full_reduce: Whether to fully reduce the matrix, thus rebuild the full circuit.
    :return: A fitness function that calculates the number of gates needed for a given permutation.
    """
    #n_qubits = len(matrix.data)

    #def fitness_func(permutation):
    #    row_perm = permutation if row else np.arange(len(matrix.data))
    #    col_perm = permutation if col else np.arange(len(matrix.data[0]))
    #    circuit = CNOT_tracker(n_qubits)
    #    mat = Mat2([[matrix.data[r][c] for c in col_perm] for r in row_perm])
    #    gauss(mode, mat, architecture=architecture, y=circuit, full_reduce=full_reduce, **kwargs)
    #    return metric_func(circuit)
    fitness_func = FitnessFunction(metric_func, matrix, mode, architecture, row=row, col=col, full_reduce=full_reduce, **kwargs)
    return fitness_func

class FitnessFunction(object):

    def __init__(self, metric, matrix, mode, architecture, row=True, col=True, full_reduce=True, **kwargs):
        self.metric = metric
        self.matrix = matrix
        self.mode = mode
        self.architecture = architecture
        self.row = row
        self.col = col
        self.full_reduce = full_reduce
        self.n_qubits = architecture.n_qubits
        self.kwargs = kwargs
    
    def _make_function(self):
        if self.metric == COMBINED_METRIC:
            f = lambda c: c.cnot_depth()*10000 + c.count_cnots()
        elif self.metric == COUNT_METRIC:
            f = lambda c: c.count_cnots()
        elif self.metric == DEPTH_METRIC:
            f = lambda c: c.cnot_depth()
        
        def fitness_func(permutation):
            row_perm = permutation if self.row else np.arange(len(self.matrix.data))
            col_perm = permutation if self.col else np.arange(len(self.matrix.data[0]))
            circuit = CNOT_tracker(self.n_qubits)
            mat = Mat2([[self.matrix.data[r][c] for c in col_perm] for r in row_perm])
            gauss(self.mode, mat, architecture=self.architecture, y=circuit, full_reduce=self.full_reduce, **self.kwargs)
            return f(circuit)
        return fitness_func

    def __call__(self, permutation):
        f = self._make_function()
        return f(permutation)
        
class StepFunction():

    def __init__(self, matrices, mode, architecture, fitness_func, **kwargs):
        self.matrices=matrices
        self.mode = mode
        self.architecture = architecture
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.rev_matrices = [Mat2(np.asarray(m.data).T.tolist()) for m in reversed(matrices)] # Reverse and transpose the parity matrices to create the reversed equivalent sequence 
        

    def __call__(self, initial_perm):
        matrices=self.matrices
        new_mode = self.mode
        architecture = self.architecture
        fitness_func = self.fitness_func
        rev_matrices = self.rev_matrices
        kwargs = self.kwargs
        # Apply the original qubit placement 
        ms = [Mat2([[row[i] for i in initial_perm] for row in m.data]) if j == 0 else Mat2([r for r in m.data]) for j, m in enumerate(matrices)]
        # Optimize the sequence
        circs, perms, score = sequential_gauss(ms, new_mode, architecture=architecture, fitness_func=fitness_func, input_perm=False, output_perm=True, n_threads=1, **kwargs)
        # Resulting permutation is the initial permutation of the reverse pass
        perms[0] = initial_perm
        ms = [Mat2([[row[i] for i in perms[-1]] for row in m.data]) if j == 0 else Mat2([r for r in m.data]) for j,m in enumerate(rev_matrices)] 
        # Optimize the reverse sequences.
        _, new_perms, _ = sequential_gauss(ms, new_mode, architecture=architecture, fitness_func=fitness_func, input_perm=False, output_perm=True, n_threads=1, **kwargs)
        # New initial placement is the final placement of the reverse pass.
        #new_perms[0] = perms[-1] 
        return new_perms[-1], (circs, perms), score


def gauss(mode, matrix, architecture=None, permutation=None, try_transpose=False, **kwargs):
    """
    Performs gaussian elimination of type mode on Mat2 matrix on the given architecture, if needed.

    :param mode: Type of Gaussian elimination to be used
    :param matrix: Mat2 matrix to run the algorithm on
    :param architecture: Device architecture to take into account [optional]
    :param kwargs: Other arguments that can be given to the Mat2.gauss() function or parameters for the genetic algorithm.
    :return: The rank of the matrix. Mat2 matrix is transformed.
    """
    if try_transpose:
        matrix = matrix.transpose()
        architecture = architecture.transpose()

    if mode == GAUSS_MODE:
        # TODO - adjust to get the right gate locations for the given permutation.
        
        if permutation is not None:
            #print("\033[91m Warning: Permutation parameter with Gauss-Jordan elimination is not yet supported, it can be optimized with permutated_gauss(). \033[0m ")
            #return matrix.gauss(**kwargs)
            # Broken code that tries to implement this.
            matrix = Mat2([[row[i] for i in permutation] for row in matrix.data])
            old_x, old_y = None, None
            if "x" in kwargs:
                old_x = kwargs["x"]
            if "y" in kwargs:
                old_y = kwargs["y"]
            n_qubits = len(matrix.data)
            x = CNOT_tracker(n_qubits)
            kwargs["x"] = x
            kwargs["y"] = None
            rank = matrix.gauss(**kwargs)
            #for gate in x.gates:
            #    #c = permutation[gate.control]
            #    #t = permutation[gate.target]
            #    if old_x != None: old_x.row_add(c, t)
            #    if old_y != None: old_y.col_add(t, c)
            #return rank
        else:
            rank = matrix.gauss(**kwargs)
    elif mode == STEINER_MODE:
        if architecture is None:
            print(
                "\033[91m Warning: Architecture is not given, assuming fully connected architecture of size matrix.shape[0]. \033[0m ")
            architecture = create_fully_connected_architecture(len(matrix.data))
        rank =  steiner_gauss(matrix, architecture, permutation=permutation, **kwargs)
    elif mode == GENETIC_STEINER_MODE:
        perm, circuit, rank = permutated_gauss(matrix, STEINER_MODE, architecture=architecture, permutation=permutation, **kwargs)
        #return rank
    elif mode == GENETIC_GAUSS_MODE:
        perm, circuit, rank = permutated_gauss(matrix, GAUSS_MODE, architecture=architecture, permutation=permutation, **kwargs)
    if try_transpose:
        # TODO - fix x and y circuits... - Needed? 
        # TODO pick which gauss version was chosen
        pass
    return rank

def permutated_gauss(matrix, mode=None, architecture=None, population_size=30, crossover_prob=0.8, mutate_prob=0.2, n_iterations=5,
                     row=True, col=True, full_reduce=True, fitness_func=None, x=None, y=None, n_threads=None, **kwargs):
    """
    Finds an optimal permutation of the matrix to reduce the number of CNOT gates.
    
    :param matrix: Mat2 matrix to do gaussian elimination over
    :param population_size: For the genetic algorithm
    :param crossover_prob: For the genetic algorithm
    :param mutate_prob: For the genetic algorithm
    :param n_iterations: For the genetic algorithm
    :param row: If the rows should be permutatedA
    :param col: If the columns should be permutated
    :param full_reduce: Whether to do full gaussian reduction
    :return: Best permutation found, list of CNOTS corresponding to the elimination.
    """
    if row or col:
        if fitness_func is None:
            fitness_func =  combined_fitness_func(mode, matrix, architecture, row=row, col=col, full_reduce=full_reduce, **kwargs)
        optimizer = GeneticAlgorithm(population_size, crossover_prob, mutate_prob, fitness_func, quiet=True, n_threads=n_threads)
        permsize = len(matrix.data) if row else len(matrix.data[0])
        best_permutation = optimizer.find_optimimum(permsize, n_iterations, continued=True)
    else:
        best_permutation = np.arange(len(matrix.data))

    n_qubits=len(matrix.data)
    row_perm = best_permutation if row else np.arange(len(matrix.data))
    col_perm = best_permutation if col else np.arange(len(matrix.data[0]))
    if y is None:
        circuit = CNOT_tracker(n_qubits)
    else:
        circuit = y
    mat = Mat2([[matrix.data[r][c] for c in col_perm] for r in row_perm])
    circuit.row_perm = row_perm
    circuit.col_perm = col_perm
    rank = gauss(mode, mat, architecture, x=x, y=circuit, full_reduce=full_reduce, **kwargs)
    return best_permutation, circuit, rank

def sequential_gauss(matrices, mode=None, architecture=None, fitness_func=None, input_perm=True, output_perm=True, 
                        swarm_size=15, n_steps=5, s_crossover=0.4, p_crossover=0.3, pso_mutation=0.2, n_threads=None, full_reduce=True, **kwargs):
    n_qubits = len(matrices[0].data)
    kwargs["full_reduce"] = full_reduce
    #print(mode)
    #print(*matrices, sep="\n\n")
    if mode in basic_elim_modes or mode is None:
        circuits = [CNOT_tracker(n_qubits) for _ in matrices]
        permutations = [np.arange(n_qubits) for _ in range(len(matrices)+1)]
        for i, m in enumerate(matrices):
            gauss(mode, m, architecture=architecture, y=circuits[i], **kwargs)
    elif mode in genetic_elim_modes:
        col = input_perm
        if mode == GENETIC_GAUSS_MODE:
            new_mode = GAUSS_MODE
        else:
            new_mode = STEINER_MODE
        row = True
        circuits = []
        permutations = []
        current_perm = np.arange(n_qubits)#[i for i in range(n_qubits)]
        if not col:
            permutations.append(current_perm) # Add initial permutation if it is not optimized
        for i, m in enumerate(matrices):
            # Adjust matrix according to current input perm.
            m = Mat2([[row[r] for r in current_perm] for row in m.data])
            if i == len(matrices) - 1:
                row = output_perm # Last permutation is only optimized if the output qubit locations are flexible.
            perm, circuit, _ = permutated_gauss(m, new_mode, architecture=architecture, fitness_func=fitness_func, row=row, col=col, n_threads=n_threads, **kwargs)
            #if not col and not row:
            #    perm = current_perm
            circuits.append(circuit) # Store the extracted circuit
            # Update the new permutation
            current_perm = perm
            if col:
                permutations.append(current_perm) # Add optimized inital permutation
                if not row:
                    current_perm = np.arange(n_qubits)
            permutations.append(current_perm) # Store the obtained permutation
            col = False # Subsequent initial permutations are determined by the previous output permutation.
        #input("current perm - should be [0..] ")
    else: # pso modes
        if mode == PSO_STEINER_MODE:
            new_mode = GENETIC_STEINER_MODE
        else:
            new_mode = GENETIC_GAUSS_MODE
        if not input_perm or not output_perm:
            # You cannot do subsequent passes to optimize the permutation, so pso is useless.
            return sequential_gauss(matrices, new_mode, architecture=architecture, fitness_func=fitness_func, input_perm=input_perm, output_perm=output_perm, n_threads=n_threads, **kwargs)
        
        step_func = StepFunction(matrices, new_mode, architecture, fitness_func, **kwargs)
        optimizer = ParticleSwarmOptimization(swarm_size=swarm_size, fitness_func=fitness_func, step_func=step_func, s_best_crossover=s_crossover, p_best_crossover=p_crossover, mutation=pso_mutation)
        best_solution = optimizer.find_optimimum(architecture.n_qubits, n_steps, False)
        circuits, permutations = best_solution
    return circuits, permutations, sum([c.cnot_depth()*10000 + c.count_cnots() for c in circuits])

def count_cnots_mat2(mode, matrix, compile_mode=None, architecture=None, n_compile=1, store_circuit_as=None, **kwargs):
    if compile_mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        circuit = PyQuilCircuit(architecture)
    else:
        circuit = CNOT_tracker(matrix.data.shape[0])
    mat = Mat2(np.copy(matrix.data))
    gauss(mode, mat, architecture=architecture, y=circuit, **kwargs)
    return count_cnots_circuit(compile_mode, circuit, n_compile, store_circuit_as)

def count_cnots_circuit(mode, circuit, n_compile=1, store_circuit_as=None):
    count = -1
    if mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        if isinstance(circuit, PyQuilCircuit):
            count = sum([circuit.compiled_cnot_count() for i in range(n_compile)])/n_compile
    elif mode == NO_COMPILER:
        count = circuit.count_cnots()
    if store_circuit_as is not None:
        with open(store_circuit_as, 'w') as f:
            f.write(circuit.to_qasm())
    return count

def create_dest_filename(original_file, population=None, iteration=None, crossover_prob=None, mutation_prob=None, index=None):
    pop_ext = "" if population is None else "pop" + str(population)
    iter_ext = "" if iteration is None else "iter" + str(iteration)
    crosover_ext = "" if crossover_prob is None else "crossover" + str(crossover_prob)
    mutation_ext = "" if mutation_prob is None else "mutate" + str(mutation_prob)
    index_ext = "" if index is None else "(" + str(index) + ")"
    filename = os.path.basename(original_file)
    base_file, extension = os.path.splitext(filename)
    new_filename = '_'.join([part for part in [base_file, pop_ext, iter_ext, crosover_ext, mutation_ext, index_ext] if part != ""]) + extension
    return new_filename

def get_metric_header():
    metrics = CNOT_tracker.get_metric_names()
    return ["id", "architecture", "mode", "index", "population", "n_iterations", "crossover", "mutation"] + metrics + ["time", "destination_file"]

def make_metrics(circuit, id, architecture_name, mode, dest_file=None, population=None, iteration=None, crossover_prob=None, mutation_prob=None, passed_time=None, index=None):
    result = circuit.gather_metrics()
    result["id"] = id
    result["mode"] = mode
    result["architecture"] = architecture_name
    result["population"] = population
    result["n_iterations"] = iteration
    result["crossover"] = crossover_prob
    result["mutation"] = mutation_prob
    result["time"] = passed_time
    result["index"] = index
    result["destination_file"] = dest_file
    return result


def batch_map_cnot_circuits(source, modes, architectures, n_qubits=None, populations=30, iterations=15, crossover_probs=0.8,
                            mutation_probs=0.5, dest_folder=None, metrics_file=None, n_compile=1):
    modes = make_into_list(modes)
    architectures = make_into_list(architectures)
    populations = make_into_list(populations)
    iterations = make_into_list(iterations)
    crossover_probs = make_into_list(crossover_probs)
    mutation_probs = make_into_list(mutation_probs)

    if os.path.isfile(source):
        source, file = os.path.split(source)
        files = [file]
    else:
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    if not os.path.exists(source):
        raise IOError("Folder does not exist: " + source)
    if dest_folder is None:
        dest_folder = source
    else:
        os.makedirs(dest_folder, exist_ok=True)

    arch_iter = []
    circuits = {}
    metrics = []
    for architecture in architectures:
        if architecture in dynamic_size_architectures:
            if n_qubits is None:
                raise KeyError("Number of qubits not specified for architecture" + architecture)
            else:
                n_qubits = make_into_list(n_qubits)
                arch_iter.extend([create_architecture(architecture, n_qubits=q) for q in n_qubits])
        else:
            arch_iter.append(create_architecture(architecture))
    for architecture in arch_iter:
        circuits[architecture.name] = {}
        for mode in modes:
            if mode == QUIL_COMPILER:
                n_compile_list = range(n_compile)
            else:
                n_compile_list = [None]
            new_dest_folder = os.path.join(dest_folder, architecture.name, mode)
            os.makedirs(new_dest_folder, exist_ok=True)
            if mode in genetic_elim_modes:
                pop_iter = populations
                iter_iter = iterations
                crossover_iter = crossover_probs
                mutation_iter = mutation_probs
                circuits[architecture.name][mode] = {}
            else:
                if mode == QUIL_COMPILER:
                    circuits[architecture.name][mode] = []
                pop_iter = [None]
                iter_iter = [None]
                crossover_iter = [None]
                mutation_iter = [None]

            for population in pop_iter:
                for iteration in iter_iter:
                    for crossover_prob in crossover_iter:
                        for mutation_prob in mutation_iter:
                            for file in files:
                                if os.path.splitext(file)[1].lower() == ".qasm":
                                    origin_file = os.path.join(source, file)
                                    for i in n_compile_list:
                                        dest_filename = create_dest_filename(origin_file, population, iteration, crossover_prob, mutation_prob, i)
                                        dest_file = os.path.join(dest_folder, architecture.name, mode, dest_filename)
                                        try:
                                            start_time = time.time()
                                            circuit = map_cnot_circuit(origin_file, architecture, mode=mode, dest_file=dest_file,
                                                                       population=population, iterations=iteration,
                                                                       crossover_prob=crossover_prob, mutation_prob=mutation_prob)
                                            end_time = time.time()
                                            if metrics_file is not None:
                                                metrics.append(make_metrics(circuit, origin_file, architecture.name, mode, dest_file, population, iteration, crossover_prob, mutation_prob, end_time-start_time, i))
                                            if mode in genetic_elim_modes:
                                                circuits[architecture.name][mode][(population, iteration, crossover_prob, mutation_prob)] = circuit
                                            elif mode == QUIL_COMPILER:
                                                circuits[architecture.name][mode].append(circuit)
                                            else:
                                                circuits[architecture.name][mode] = circuit
                                        except KeyError as e: # Should only happen with quilc
                                            if mode == QUIL_COMPILER:
                                                print("\033[31mCould not compile", origin_file, "into", dest_file, end="\033[0m\n")
                                            else:
                                                raise e

    if len(metrics) > 0 and DataFrame is not None:
        df = DataFrame(metrics)
        print("Average gate count:", df["n_cnots"].mean())
        print("Average gate depth:", df["depth"].mean())
        if os.path.exists(metrics_file): # append to the file - do not overwrite!
            df.to_csv(metrics_file, columns=get_metric_header(), header=False, index=False, mode='a')
        else:
            df.to_csv(metrics_file, columns=get_metric_header(), index=False)
    return circuits

def map_cnot_circuit(file, architecture, mode=GENETIC_STEINER_MODE, dest_file=None, population=30, iterations=15, crossover_prob=0.8, mutation_prob=0.2, **kwargs):
    if type(architecture) == type(""):
        architecture = create_architecture(architecture)
    circuit = CNOT_tracker.from_qasm_file(file)
    matrix = circuit.matrix
    compiled_circuit = CNOT_tracker(circuit.n_qubits)
    if mode in basic_elim_modes:
        rank = gauss(mode, matrix, architecture, full_reduce=True, y=compiled_circuit, **kwargs)
    elif mode in genetic_elim_modes:
        rank = gauss(mode, matrix, architecture, full_reduce=True, y=compiled_circuit,
                     population_size=population, crossover_prob=crossover_prob, mutate_prob=mutation_prob,
                     n_iterations=iterations, **kwargs)
    elif mode == QUIL_COMPILER:
        from pyzx.pyquil_circuit import PyQuilCircuit
        compiled_circuit = PyQuilCircuit.from_CNOT_tracker(circuit, architecture)
        compiled_circuit.compile()

    if dest_file is not None:
        compiled_qasm = compiled_circuit.to_qasm()
        with open(dest_file, "w") as f:
            f.write(compiled_qasm)
    return compiled_circuit

def sequential_map_cnot_circuits(source, modes, architectures, n_qubits=None, populations=30, iterations=15, crossover_probs=0.8,
                            mutation_probs=0.5, dest_folder=None, metrics_file=None, n_compile=1):
    modes = make_into_list(modes)
    architectures = make_into_list(architectures)
    populations = make_into_list(populations)
    iterations = make_into_list(iterations)
    crossover_probs = make_into_list(crossover_probs)
    mutation_probs = make_into_list(mutation_probs)

    if os.path.isfile(source):
        source, file = os.path.split(source)
        files = [file]
    else:
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    if not os.path.exists(source):
        raise IOError("Folder does not exist: " + source)
    if dest_folder is None:
        dest_folder = source
    else:
        os.makedirs(dest_folder, exist_ok=True)

    arch_iter = []
    circuits = {}
    metrics = []
    for architecture in architectures:
        if architecture in dynamic_size_architectures:
            if n_qubits is None:
                raise KeyError("Number of qubits not specified for architecture" + architecture)
            else:
                n_qubits = make_into_list(n_qubits)
                arch_iter.extend([create_architecture(architecture, n_qubits=q) for q in n_qubits])
        else:
            arch_iter.append(create_architecture(architecture))
    for architecture in arch_iter:
        circuits[architecture.name] = {}
        for mode in modes:
            if mode == QUIL_COMPILER:
                n_compile_list = range(n_compile)
            else:
                n_compile_list = [None]
            new_dest_folder = os.path.join(dest_folder, architecture.name, mode)
            os.makedirs(new_dest_folder, exist_ok=True)
            if mode in genetic_elim_modes or mode in pso_elim_modes:
                pop_iter = populations
                iter_iter = iterations
                crossover_iter = crossover_probs
                mutation_iter = mutation_probs
                circuits[architecture.name][mode] = {}
            else:
                if mode == QUIL_COMPILER:
                    circuits[architecture.name][mode] = []
                pop_iter = [None]
                iter_iter = [None]
                crossover_iter = [None]
                mutation_iter = [None]

            for population in pop_iter:
                for iteration in iter_iter:
                    for crossover_prob in crossover_iter:
                        for mutation_prob in mutation_iter:
                            matrices = []
                            for file in files:
                                if os.path.splitext(file)[1].lower() == ".qasm":
                                    origin_file = os.path.join(source, file)
                                    matrices.append(CNOT_tracker.from_qasm_file(origin_file).matrix)        
                            for i in n_compile_list:
                                #dest_filename = create_dest_filename(origin_file, population, iteration, crossover_prob, mutation_prob, i)
                                #dest_file = os.path.join(dest_folder, architecture.name, mode, dest_filename)
                                try:
                                    print("Sequence length", len(matrices))
                                    start_time = time.time()
                                    circuits, permutations, score = sequential_gauss(matrices, architecture=architecture, mode=mode,
                                                                population_size=population, n_iterations=iteration,
                                                                crossover_prob=crossover_prob, mutate_prob=mutation_prob)
                                    end_time = time.time()
                                    print("score:", score)
                                    print("time (s):", end_time-start_time)
                                    # TODO store metrics
                                    """
                                    if metrics_file is not None:
                                        metrics.append(make_metrics(circuit, origin_file, architecture.name, mode, dest_file, population, iteration, crossover_prob, mutation_prob, end_time-start_time, i))
                                    if mode in genetic_elim_modes:
                                        circuits[architecture.name][mode][(population, iteration, crossover_prob, mutation_prob)] = circuit
                                    elif mode == QUIL_COMPILER:
                                        circuits[architecture.name][mode].append(circuit)
                                    else:
                                        circuits[architecture.name][mode] = circuit
                                    """
                                except KeyError as e: # Should only happen with quilc
                                    if mode == QUIL_COMPILER:
                                        print("\033[31mCould not compile", origin_file, end="\033[0m\n")
                                    else:
                                        raise e

    if len(metrics) > 0 and DataFrame is not None:
        df = DataFrame(metrics)
        print("Average gate count:", df["n_cnots"].mean())
        print("Average gate depth:", df["depth"].mean())
        if os.path.exists(metrics_file): # append to the file - do not overwrite!
            df.to_csv(metrics_file, columns=get_metric_header(), header=False, index=False, mode='a')
        else:
            df.to_csv(metrics_file, columns=get_metric_header(), index=False)
    return circuits