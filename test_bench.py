import pyzx as zx
import os
import time

directory = "../Pytket-dev_attempt/tkcOutputs2/tests/"

with open("cowtan_bench.txt",'w') as df:
    for fname in os.listdir(directory):
        if not fname.endswith('.qasm'): continue
        print(fname)
        fname = directory+fname
        with open(fname, 'r') as rf:
            lines = rf.read().splitlines()
            if len(lines) > 5000: continue
            print(len(lines))
            circ = zx.Circuit.load(fname)
            g = circ.to_graph()
            start = time.process_time()
            zx.simplify.clifford_simp(g)
            # assert(zx.tensor.compare_tensors(circ,g))
            circ2 = zx.extract.streaming_extract(g)
            end = time.process_time()
            print("time = " + str(end-start))
            df.write(str(end-start) + "," + str(len(circ2.gates)) + "," + str(circ2.twoqubitcount()) + "\n")