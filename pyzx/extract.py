# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

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

from __future__ import print_function

__all__ = ['clifford_extract', 'streaming_extract', 'modified_extract']

from fractions import Fraction
import itertools

from .linalg import Mat2, greedy_reduction, permutation_as_swaps, column_optimal_swap
from .graph import Graph
from .simplify import id_simp, tcount
from .rules import apply_rule, pivot, match_spider_parallel, spider
from .circuit import Circuit
from .circuit.gates import ParityPhase, CNOT, HAD, ZPhase, CZ, InitAncilla
from .optimize import basic_optimization


def bi_adj(g, vs, ws):
    return Mat2([[1 if g.connected(v,w) else 0 for v in vs] for w in ws])

def cut_rank(g, left, right):
    return bi_adj(g, left, right).rank()

def cut_edges(g, left, right, available=None):
    m = bi_adj(g, left, right)
    max_r = max(g.row(v) for v in left)
    for v in g.vertices():
        r = g.row(v)
        if (r > max_r):
            g.set_row(v, r+2)
    x,y = m.factor()

    for v1 in left:
        for v2 in right:
            if (g.connected(v1,v2)):
                g.remove_edge(g.edge(v1,v2))
    
    cut_rank = y.rows()

    #g.add_vertices(2*cut_rank)
    left_verts = []
    right_verts = []
    
    if available == None:
        qs = range(cut_rank)
    else:
        qs = available

    for i in qs:
        v1 = g.add_vertex(1,i,max_r+1)
        v2 = g.add_vertex(1,i,max_r+2)
        #v = vi+cut_rank+i
        #g.add_edge((vi+i,v))
        g.add_edge((v1,v2),2)
        left_verts.append(v1)
        right_verts.append(v2)
        #g.set_edge_type(g.edge(vi+i,v), 2)

    for i in range(y.rows()):
        for j in range(y.cols()):
            if (y.data[i][j]):
                g.add_edge((left[j],left_verts[i]),2)
                #g.add_edge((left[j], vi + i))
                #g.set_edge_type(g.edge(left[j], vi + i), 2)
    for i in range(x.rows()):
        for j in range(x.cols()):
            if (x.data[i][j]):
                g.add_edge((right_verts[j],right[i]),2)
                #g.add_edge((vi + cut_rank + j, right[i]))
                #g.set_edge_type(g.edge(vi + cut_rank + j, right[i]), 2)
    return left_verts


def unspider_by_row(g, v):
    r = g.row(v)
    w = g.add_vertex(1,g.qubit(v),r-1)
    for n in list(g.neighbours(v)):
        if g.row(n) < r:
            e = g.edge(n,v)
            g.add_edge((n,w), edgetype=g.edge_type(e))
            g.remove_edge(e)
    g.add_edge((w, v))
    return w

def connectivity_from_biadj(g, m, left, right, edgetype=2):
    for i in range(len(right)):
        for j in range(len(left)):
            if m.data[i][j] and not g.connected(right[i],left[j]):
                g.add_edge((right[i],left[j]),edgetype)
            elif not m.data[i][j] and g.connected(right[i],left[j]):
                g.remove_edge((right[i],left[j]))

def streaming_extract(g, allow_ancillae=False, quiet=True, stopcount=-1):
    """Given a graph put into semi-normal form by :func:`simplify.full_reduce`, 
    it extracts its equivalent set of gates into an instance of :class:`circuit.Circuit`.
    """
    g.normalise()
    qs = g.qubits() # We are assuming that these are objects that update...
    rs = g.rows()   # ...to reflect changes to the graph, so that when...
    ty = g.types()  # ... g.set_row/g.set_qubit is called, these things update directly to reflect that
    phases = g.phases()
    c = Circuit(g.qubit_count())
    leftrow = 1
    maxq = max(qs.values()) + 1

    nodestotal = tcount(g)
    nodesparsed = 0
    nodesmarker = 10

    # special_nodes contains the ParityPhase like nodes
    special_nodes = {}
    for v in g.vertices():
        if len(list(g.neighbours(v))) == 1 and v not in g.inputs and v not in g.outputs:
            n = list(g.neighbours(v))[0]
            special_nodes[n] = v
        if rs[v] > 1:
            g.set_row(v, rs[v]+20)
    
    tried_id_simp = False
    while True:
        left = [v for v in g.vertices() if rs[v] == leftrow]
        boundary_verts = []
        right = set()
        good_verts = []
        good_neighs = []
        postselects = []
        for v in left:
            # First we add the gates to the circuit that can be processed now,
            # and we simplify the graph to represent this.
            q = qs[v]
            phase = phases[v]
            t = ty[v]
            neigh = [w for w in g.neighbours(v) if rs[w]<leftrow]
            if len(neigh) != 1:
                raise TypeError("Graph doesn't seem circuit like: multiple parents")
            n = neigh[0]
            if qs[n] != q:
                raise TypeError("Graph doesn't seem circuit like: cross qubit connections")
            if g.edge_type(g.edge(n,v)) == 2:
                c.add_gate("HAD", q)
                g.set_edge_type(g.edge(n,v),1)
            if t == 0: continue # it is an output
            if phase != 0:
                if phase.denominator > 2: nodesparsed += 1
                if t == 1: c.add_gate("ZPhase", q, phase=phase)
                else: c.add_gate("XPhase", q, phase=phase)
                g.set_phase(v, 0)
        for v in left:
            q = qs[v]
            t = ty[v]
            neigh = [w for w in g.neighbours(v) if rs[w]==leftrow and w<v]
            for n in neigh:
                t2 = ty[n]
                q2 = qs[n]
                if t == t2:
                    if g.edge_type(g.edge(v,n)) != 2:
                        raise TypeError("Invalid vertical connection between vertices of the same type")
                    if t == 1: c.add_gate("CZ", q2, q)
                    else: c.add_gate("CX", q2, q)
                else:
                    if g.edge_type(g.edge(v,n)) != 1:
                        raise TypeError("Invalid vertical connection between vertices of different type")
                    if t == 1: c.add_gate("CNOT", q, q2)
                    else: c.add_gate("CNOT", q2, q)
                g.remove_edge(g.edge(v,n))
            
            # Done processing gates, now we look to see if we can shift the frontier
            d = [w for w in g.neighbours(v) if rs[w]>leftrow]
            right.update(d)
            if len(d) == 0: 
                if not allow_ancillae: raise TypeError("Not circuit like")
                else:
                    postselects.append(v)
            if len(d) == 1: # Only connected to one node in its future
                if ty[d[0]] != 0: # which is not an output
                    good_verts.append(v) # So we can make progress
                    good_neighs.append(d[0])
                else:  # This node is done processing, since it is directly (and only) connected to an output
                    boundary_verts.append(v)
                    right.remove(d[0])
        for v in postselects:
            if not quiet: print("postselect", v, qs[v])
            c.add_gate("PostSelect", qs[v])
            left.remove(v)
            g.set_row(v, leftrow-0.5)
            if qs[v] == maxq - 1:
                maxq = maxq -1
        if not good_verts:  # There are no 'easy' nodes we can use to progress
            if all(ty[v] == 0 for v in right): break # Actually we are done, since only outputs are left
            for v in boundary_verts: left.remove(v) # We don't care about the nodes only connected to outputs
            have_removed_gadgets = False
            for n in right.intersection(special_nodes): # Neighbours that are phase gadgets
                targets = set(g.neighbours(n))
                targets.remove(special_nodes[n])
                if targets.issubset(left): # Only connectivity on the lefthandside, so we can extract it
                    nphase = phases[n]
                    if nphase not in (0,1):
                        raise Exception("Can't parse ParityPhase with non-Pauli Phase")
                    phase = phases[special_nodes[n]]
                    c.add_gate("ParityPhase", phase*(-1 if nphase else 1), *[qs[t] for t in targets])
                    g.remove_vertices([special_nodes[n],n])
                    nodesparsed += 1
                    right.remove(n)
                    del special_nodes[n]
                    have_removed_gadgets = True
            if stopcount != -1 and len(c.gates) > stopcount: return c
            if have_removed_gadgets: continue
            right = list(right)
            m = bi_adj(g,right,left)
            m2 = m.copy()
            m2.gauss(full_reduce=True)
            if not any(sum(l)==1 for l in m2.data):
                if not tried_id_simp:
                    tried_id_simp = True
                    i = id_simp(g, matchf=lambda v: rs[v]>leftrow, quiet=True)
                    if i: 
                        if not quiet: print("id_simp found some matches")
                        m = match_spider_parallel(g, matchf=lambda e: rs[g.edge_s(e)]>=leftrow and rs[g.edge_t(e)]>=leftrow)
                        m = [(v1,v2) if v1 in left else (v2,v1) for v1,v2 in m]
                        if not quiet and m: print("spider fusion found some matches")
                        etab, rem_verts, not_needed1, not_needed2 = spider(g, m)
                        g.add_edge_table(etab)
                        g.remove_vertices(rem_verts)
                        continue
                try:
                    gates, lr = handle_phase_gadget(g, left, set(right), special_nodes, quiet=quiet)
                except ValueError:
                    if not allow_ancillae:
                        raise
                    raise Exception
                    gates, maxq = find_ancilla_qubits(g, left, set(right), special_nodes, maxq, quiet=quiet)
                    c.gates.extend(gates)
                    continue
                c.gates.extend(gates)
                nodesparsed += 1
                tried_id_simp = False
                if lr > leftrow:
                    for v in boundary_verts:
                        g.set_row(v, lr)
                    leftrow = lr
                continue
            sequence = greedy_reduction(m) # Find the optimal set of CNOTs we can apply to get a frontier we can work with
            if not isinstance(sequence, list): # Couldn't find any reduction, hopefully we can fix this
                right = set(right)
                gates, success = try_greedy_cut(g, left, right, right.difference(special_nodes), quiet=quiet)
                if success:
                    c.gates.extend(gates)
                    continue
                raise Exception("We should never get here")
                
            if not quiet: print("Greedy reduction with {:d} CNOTs".format(len(sequence)))
            for control, target in sequence:
                c.add_gate("CNOT", qs[left[target]], qs[left[control]])
                # If a control is connected to an output, we need to add a new node.
                for v in g.neighbours(left[control]):
                    if v in g.outputs:
                        #print("Adding node before output")
                        q = qs[v]
                        r = rs[v]
                        w = g.add_vertex(1,q,r-1)
                        e = g.edge(left[control],v)
                        et = g.edge_type(e)
                        g.remove_edge(e)
                        g.add_edge((left[control],w),2)
                        g.add_edge((w,v),3-et)
                        k = right.index(v)
                        right[k] = w
                        break
                for k in range(len(m.data[control])): # We update the graph to represent the extraction of a CNOT
                    if not m.data[control][k]: continue
                    if m.data[target][k]: g.remove_edge((left[target],right[k]))
                    else: g.add_edge((left[target],right[k]), 2)
                m.row_add(control, target)
            for v in left:
                d = [w for w in g.neighbours(v) if rs[w]>leftrow]
                if len(d) == 1 and ty[d[0]] != 0:
                    good_verts.append(v)
                    good_neighs.append(d[0])
            if not good_verts: continue
        
        for v in g.vertices():
            if rs[v] < leftrow: continue
            if v in good_verts: continue
            g.set_row(v,rs[v]+1) # Push the frontier one layer up
        for i,v in enumerate(good_neighs): 
            g.set_row(v,leftrow+1) # Bring the new nodes of the frontier to the correct position
            g.set_qubit(v,qs[good_verts[i]])

        tried_id_simp = False

        if not quiet and nodesparsed > nodesmarker:
            print("{:d}/{:d}".format(nodesparsed, nodestotal))
            nodesmarker = int(round(nodesparsed-5,-1))
            nodesmarker += 10
        leftrow += 1
        if stopcount != -1 and len(c.gates) > stopcount: return c
            
    swap_map = {}
    leftover_swaps = False
    for v in left: # Finally, check for the last layer of Hadamards, and see if swap gates need to be applied.
        q = qs[v]
        neigh = [w for w in g.neighbours(v) if rs[w]>leftrow]
        if len(neigh) != 1: 
            raise TypeError("Algorithm failed: Not fully reducable")
            return c
        n = neigh[0]
        if ty[n] != 0: 
            raise TypeError("Algorithm failed: Not fully reducable")
            return c
        if g.edge_type(g.edge(n,v)) == 2:
            c.add_gate("HAD", q)
            g.set_edge_type(g.edge(n,v),1)
        if qs[n] != q: leftover_swaps = True
        swap_map[q] = qs[n]
    if leftover_swaps: 
        for t1, t2 in permutation_as_swaps(swap_map):
            c.add_gate("SWAP", t1, t2)
    return c


def try_greedy_cut(g, left, right, candidates, quiet=True):
    q = len(left)
    left = list(left)
    # Take care nothing is connected directly to an output
    for w in right.copy():
        if w in g.outputs:
            w2 = g.add_vertex(1, g.qubit(w), g.row(w)-1)
            n = list(g.neighbours(w))[0] # Outputs should have unique neighbours
            e = g.edge(n,w)
            et = g.edge_type(e)
            g.remove_edge(e)
            g.add_edge((n,w2),2)
            g.add_edge((w2,w),3-et)
            right.remove(w)
            right.add(w2)
            if w in candidates:
                candidates.remove(w)
                candidates.add(w2)

    right = list(right)
    # We want to figure out which vertices in candidates are 'pivotable'
    # That is, that removing them will decrease the cut rank of the remainder
    m = bi_adj(g, right, left)
    m.gauss(full_reduce=True) # Gaussian elimination doesn't change this property
    good_nodes = []
    for r in m.data:
        if sum(r) == 1: # Exactly one nonzero value, so removing the column with the nonzero value...
            i = next(i for i in range(len(r)) if r[i]) # ...decreases the rank of the matrix
            w = right[i]
            if w in candidates:
                good_nodes.append(w)
    if not good_nodes:
        return [], False
    right = [w for w in right if w not in good_nodes]

    new_right = cut_edges(g, left, right)
    leftrow = g.row(left[0])
    for w in good_nodes: 
        g.set_row(w, leftrow+2)
        new_right.append(unspider_by_row(g, w))

    left.sort(key=g.qubit)
    qs = [g.qubit(v) for v in left]
    m = bi_adj(g, new_right, left)
    target = column_optimal_swap(m)
    for i, j in target.items():
        g.set_qubit(new_right[i],qs[j])
    new_right.sort(key=g.qubit)
    m = bi_adj(g, new_right, left)
    gates = m.to_cnots(optimize=True)
    for cnot in gates:
        cnot.target = qs[cnot.target]
        cnot.control = qs[cnot.control]
    for i in range(q):
        for j in range(q):
            if g.connected(left[i],new_right[j]):
                if i != j:
                    g.remove_edge(g.edge(left[i],new_right[j]))
            elif i == j:
                g.add_edge((left[i],new_right[j]), 2)
    if not quiet: print("Greedy extract with {:d} nodes and {:d} CNOTs".format(len(good_nodes),len(gates)))
    return gates, True



def handle_phase_gadget(g, left, neigh, special_nodes, quiet=True):
    """Tries to find a cut of the graph at the given leftrow so that a single phase-gadget can be extracted.
    Returns a list of extracted gates and modifies the graph g in place. Used by :func:`streaming_extract`"""
    q = len(left)
    qs = g.qubits() # We are assuming this thing automatically updates
    rs = g.rows()
    leftrow = rs[left[0]]
    gadgets = neigh.intersection(special_nodes) # These are the phase gadgets that are attached to the left row
    if len(gadgets) == 0: raise ValueError("No phase gadget connected to this row")
    all_verts = neigh.union(left).union(special_nodes.values())
    right = list(neigh)
    options = []
    for gadget in gadgets:
        if all(w in all_verts for w in g.neighbours(gadget)):
            options.append(gadget)
    #print(options)
    for o in options: # We move the candidates gadgets to the end of the list
        right.remove(o)
        right.append(o)
    #print(right)
    m = bi_adj(g, right, left+options)
    r = reduce_bottom_rows(m, q)
    gadget = options[r-len(left)] # This is a gadget that works
    right.remove(gadget)

    g.set_row(gadget,leftrow+1)
    g.set_row(special_nodes[gadget],leftrow+1)

    # Take care nothing is connected directly to an output
    for i in range(len(right)):
        w = right[i]
        if w in g.outputs:
            w2 = g.add_vertex(1, qs[w], rs[w]-1)
            n = list(g.neighbours(w))[0] # Outputs should have unique neighbours
            e = g.edge(n,w)
            et = g.edge_type(e)
            g.remove_edge(e)
            g.add_edge((n,w2),2)
            g.add_edge((w2,w),3-et)
            right[i] = w2

    if len(right) == q:
        if not quiet: print("No cutting necessary")
        for w in right:
            g.set_row(w, leftrow+2)
    else:
        right = cut_edges(g, left+[gadget], right)
    # We have now prepared the stage to do the extraction of the phase gadget
    
    phase = g.phase(special_nodes[gadget])
    phase = -1*phase if g.phase(gadget) != 0 else phase
    left.sort(key=g.qubit)
    qv = [qs[v] for v in left]
    m = bi_adj(g, right, left)
    target = column_optimal_swap(m)
    for i, j in target.items():
        g.set_qubit(right[i],qv[j])
    right.sort(key=g.qubit)

    m = bi_adj(g, right, left)
    if m.rank() != q:
        raise Exception("Rank in phase gadget reduction too low.")
    operations = Circuit(q)
    operations.row_add = lambda r1,r2: operations.gates.append((r1,r2))
    m.gauss(full_reduce=True,x=operations)
    gates = [CNOT(qv[r2],qv[r1]) for r1,r2 in operations.gates]
    m = bi_adj(g, right+[gadget], left)
    for r1,r2 in operations.gates:
        m.row_add(r1,r2)
    connectivity_from_biadj(g, m, right+[gadget], left)

    # Now the connections from the left to the right are like the identity
    # with some wires coming to the gadget from the left and from the right
    gadget_left = [v for v in left if g.connected(gadget, v)]
    gadget_right = [w for w in right if g.connected(gadget, w)]
    targets = [qs[v] for v in gadget_left]
    # We bring as many connections on the right to the left
    for i in reversed(range(len(gadget_right))): # The following checks if every phase connected node is on the right
        w = gadget_right[i]
        v = next(v for v in left if g.connected(w,v))
        g.set_edge_type((v,w),1)
        g.set_qubit(w, qs[v])
        if qs[w] not in targets:
            gates.append(HAD(qs[w]))
            gadget_right.pop(i)
            targets.append(qs[w])
            gadget_left.append(v)
        else:
            g.set_row(w, leftrow+1)

    if not gadget_right: #Only connected on leftside so we are done
        if not quiet: print("Simple phase gadget")
        gate = ParityPhase(phase, *targets)
        g.remove_vertices([special_nodes[gadget],gadget])
        gates.append(gate)
        return gates, leftrow
    
    if not quiet: print("Complicated phase gadget") # targets on left and right, so need to do more
    if len(gadget_right) % 2 != 0 or len(gadget_left) == 1:
        raise Exception("Gadget seems non-unitary")
    
    #Now we can finally extract the phase gadget
    rtargets = []
    for w in gadget_right: 
        t = qs[w]
        rtargets.append(t)
        gates.extend([HAD(t),ZPhase(t,Fraction(-1,2)),HAD(t)])
    if len(gadget_right)%4 != 0: # This is either 2 or 0
        phase = (-phase)%2
    gates.append(ParityPhase(phase, *targets))
    for t in rtargets:
        gates.extend([HAD(t),ZPhase(t, Fraction(1,2))])
    for v in left:
        if qs[v] not in rtargets:
            g.set_row(v, leftrow+1)

    g.remove_vertices([special_nodes[gadget],gadget])
    return gates, leftrow+1

def reduce_bottom_rows(m, qubits):
    """Using just row_add's from the first qubit rows in m, tries to find a row that can be 
    completely zero'd out. Returns the rownumber of this row when successful."""
    cols = m.cols()
    leading_one = {}
    adds = []
    for r in range(qubits):
        while True:
            i = next(i for i in range(cols) if m.data[r][i])
            if i in leading_one:
                m.row_add(leading_one[i],r)
                adds.append((leading_one[i],r))
            else:
                leading_one[i] = r
                break
    for r in range(qubits, m.rows()):
        while True:
            if not any(m.data[r]): 
                return r
            i = next(i for i in range(cols) if m.data[r][i])
            if i not in leading_one: break
            m.row_add(leading_one[i], r)
            adds.append((leading_one[i],r))
    raise ValueError("Did not find any completely reducable row")

def find_ancilla_qubits(g, left, right, gadgets, maxq, quiet=True):
    leftrow = g.row(left[0])
    nodes = list(right.difference(gadgets))
    right = list(right)
    for w in nodes:
        right.remove(w)
        right.append(w)
    m = bi_adj(g, right, left)
    m.gauss(full_reduce=True)
    candidates = []
    ancilla_count = 100000
    for row in m.data:
        if not any(row[:-len(nodes)]):
            verts = [right[i] for i,a in enumerate(row) if a]
            if len(verts) < ancilla_count:
                candidates = [verts]
                ancilla_count = len(verts)
            elif len(verts) == ancilla_count:
                candidates.append(verts)
    if not candidates:
        raise ValueError("No valid ancilla vertices found")
    if not quiet: print("Adding {:d} ancillas".format(ancilla_count-1))
    if len(candidates) == 1:
        ancillas = candidates[0][:-1]
    else:
        all_candidates = set()
        for cand in candidates: all_candidates.update(cand)
        best_set = None
        best_count = 100000
        for poss in itertools.combinations(all_candidates, ancilla_count-1):
            s = sum(1 for cand in candidates if all(v in cand for v in poss))
            if s < best_count:
                best_count = s
                best_set = poss
        ancillas = best_set

    gates = []
    for i, v in enumerate(ancillas):
        g.set_row(v, leftrow)
        g.set_qubit(v, maxq+i)
        w = g.add_vertex(1, maxq+i, leftrow-1)
        g.add_edge((v,w),1)
        gates.append(InitAncilla(maxq+i))
    #raise Exception
    return gates, maxq+len(ancillas)



class CNOTMaker(object):
    def __init__(self, qubits, cnot_swaps=False):
        self.qubits = qubits
        self.cnot_swaps = cnot_swaps
        self.g = Graph()
        self.qs = list(range(qubits))  # tracks qubit indices of vertices
        self.v = 0                     # next vertex to add
        self.r = 0                     # current row
        
        for i in range(qubits):
            self.add_node(i, 0, False)
            self.g.inputs.append(self.v)
            self.v += 1
        self.r += 1
    
    def finish(self):
        for i in range(self.qubits):
            self.add_node(i, 0)
            self.g.outputs.append(self.v-1)
        self.r += 1
    
    def add_node(self, q, t, update_index=True):
        self.g.add_vertex(t,q,self.r)
        if update_index:
            self.g.add_edge((self.qs[q],self.v))
            self.qs[q] = self.v
            self.v += 1
    
    def row_swap(self, r1, r2):
        #print("row_swap", r1,r2)
        if self.cnot_swaps:
            self.row_add(r1, r2)
            self.row_add(r2, r1)
            self.row_add(r1, r2)
        else:
            self.add_node(r1, 1)
            self.add_node(r2, 1)
            self.r += 1
            self.add_node(r1, 1, False)
            self.g.add_edge((self.qs[r2],self.v))
            self.v += 1
            self.add_node(r2, 1, False)
            self.g.add_edge((self.qs[r1],self.v))
            self.qs[r1] = self.v - 1
            self.qs[r2] = self.v
            self.v += 1
            self.r += 1
    
    def row_add(self, r1, r2):
        #print("row_add", r1,r2)
        self.add_node(r1, 1)
        self.add_node(r2, 2)
        self.g.add_edge((self.qs[r1],self.qs[r2]))
        self.r += 1


def clifford_extract(g, left_row, right_row, cnot_blocksize=2):
    """When ``left_row`` and ``right_row`` are adjacent rows of green nodes
    that are interconnected with Hadamard edges, that section of the graph
    is equal to some permutation matrix. This permutation matrix can be 
    decomposed as a series of CNOT gates. That is what this function does. """
    qubits = g.qubit_count()
    qleft = [v for v in g.vertices() if g.row(v)==left_row]
    qright= [v for v in g.vertices() if g.row(v)==right_row]
    qleft.sort(key=g.qubit)
    qright.sort(key=g.qubit)
    for q in range(qubits):
        no_left = False
        if len(qleft) <= q or g.qubit(qleft[q]) != q: #missing vertex
            vert = max((v for v in g.vertices() if g.qubit(v)==q and g.row(v)<left_row), key=g.row)
            neigh = [n for n in g.neighbours(vert) if g.qubit(n)==q and g.row(n)>=right_row]
            if neigh:
                conn = min(neigh,key=g.row)
            else:
                neigh = [n for n in g.neighbours(vert) if g.row(n)>=right_row]
                if len(neigh) > 1: raise TypeError("Too many neighbours")
                conn = neigh[0]
            e = g.edge(vert, conn)
            t = g.edge_type(e)
            g.remove_edge(e)
            v1 = g.add_vertex(1,q,left_row)
            g.add_edge((vert,v1),3-t)
            g.add_edge((v1,conn), 2)
            qleft.insert(q,v1)
            no_left = True
        else:
            v1 = qleft[q]
        if len(qright) <= q or g.qubit(qright[q]) != q: #missing vertex
            if no_left: vert = conn
            else: vert = min((v for v in g.vertices() if g.qubit(v)==q and g.row(v)>right_row), key=g.row)
            neigh = [n for n in g.neighbours(vert) if g.qubit(n)==q and g.row(n)<=left_row]
            if neigh:
                conn2 = max(neigh,key=g.row)
                if v1 != conn2: raise TypeError("vertices mismatching")
            else:
                neigh = [n for n in g.neighbours(vert) if g.row(n)==left_row]
                if len(neigh) > 1: raise TypeError("Too many neighbours")
                conn2 = neigh[0]
            e = g.edge(conn2,vert)
            t = g.edge_type(e)
            g.remove_edge(e)
            v2 = g.add_vertex(1,q,right_row)
            g.add_edge((conn2,v2),2)
            g.add_edge((v2,vert),3-t)
            qright.insert(q,v2)

    if len(qleft) != len(qright):
        raise ValueError("Amount of qubits should match on left and right side")
    m = bi_adj(g,qleft,qright)
    if m.rank() != qubits:
        raise ValueError("Adjency matrix rank does not match amount of qubits")
    for v in qright:
       g.set_type(v,2)
       for e in g.incident_edges(v):
           if (g.row(g.edge_s(e)) <= right_row
               and g.row(g.edge_t(e)) <= left_row): continue
           g.set_edge_type(e,3-g.edge_type(e)) # 2 -> 1, 1 -> 2
    c = CNOTMaker(qubits, cnot_swaps=True)
    m.gauss(full_reduce=True,x=c,blocksize=cnot_blocksize)
    c.finish()

    g.replace_subgraph(left_row, right_row, c.g.adjoint())



def simple_extract(g, quiet=True):
    g.normalise()
    qs = g.qubits() # We are assuming that these are objects that update...
    rs = g.rows()   # ...to reflect changes to the graph, so that when...
    ty = g.types()  # ... g.set_row/g.set_qubit is called, these things update directly to reflect that
    phases = g.phases()
    
    h = Graph()
    
    qindex = {}
    depth = 0
    for i in range(len(g.inputs)):
        v = h.add_vertex(0,i,depth)
        h.inputs.append(v)
        qindex[i] = v
    depth = 1
    
    def add_phase_gate(q, phase):
        nonlocal depth
        v = h.add_vertex(1, q, depth, phase)
        h.add_edge((qindex[q],v),1)
        qindex[q] = v
        depth += 1
        return v
    def add_hadamard(q):
        nonlocal depth
        v = h.add_vertex(1, q, depth)
        h.add_edge((qindex[q],v),2)
        qindex[q] = v
        depth += 1
        return v
    def add_cnot(ctrl, tgt):
        nonlocal depth
        v1 = h.add_vertex(1, ctrl, depth)
        v2 = h.add_vertex(2, tgt, depth)
        h.add_edges([(qindex[ctrl],v1),(qindex[tgt],v2),(v1,v2)],1)
        qindex[ctrl] = v1
        qindex[tgt] = v2
        depth += 1
        return v1,v2
    def add_cz(ctrl, tgt):
        nonlocal depth
        v1 = h.add_vertex(1, ctrl, depth)
        v2 = h.add_vertex(1, tgt, depth)
        h.add_edges([(qindex[ctrl],v1),(qindex[tgt],v2)],1)
        h.add_edge((v1,v2),2)
        qindex[ctrl] = v1
        qindex[tgt] = v2
        depth += 1
        return v1,v2
    
    def add_gadget(targets, phase):
        nonlocal depth
        verts = {q:h.add_vertex(1,q,depth) for q in targets}
        axel = h.add_vertex(2,-1,depth+0.5)
        leaf = h.add_vertex(1,-2,depth+0.5,phase)
        h.add_edges([(qindex[q],verts[q]) for q in targets] + [(verts[q],axel) for q in targets] + [(axel,leaf)], 1)
        for q in targets: qindex[q] = verts[q]
        depth += 1
        return targets, axel, leaf
    
    def add_nonlocal_gadget(qubits, vertices, phase):
        nonlocal depth
        new_verts = {q:h.add_vertex(1,q,depth) for q in qubits}
        axel = h.add_vertex(2,-1,depth+0.5)
        leaf = h.add_vertex(1,-2,depth+0.5,phase)
        h.add_edges([(qindex[q],new_verts[q]) for q in qubits] + [(new_verts[q],axel) for q in qubits] + 
                    [(v,axel) for v in vertices] + [(axel,leaf)], 1)
        for q in qubits: qindex[q] = new_verts[q]
        depth += 1
        return new_verts, axel, leaf
    
    leftrow = 1
    #maxq = max(qs.values()) + 1
    
    gadgets = {}
    nodes = [] # Non phase-gadgets
    for v in g.vertices(): # Find which vertices are gadgets
        if rs[v] > 1: g.set_row(v, rs[v]+20)
        if v in g.inputs or v in g.outputs: continue
        if len(list(g.neighbours(v))) == 1: #phase gadget
            n = list(g.neighbours(v))[0]
            gadgets[n] = v
        elif all(w in g.inputs or w in g.outputs or len(list(g.neighbours(w)))!=1 for w in g.neighbours(v)): # regular vertex
            nodes.append(v)
    
    nodestotal = len(nodes)
    nodesparsed = 0
    nodestotal = 19
    
    processed_targets = {}
    while True:
        left = [v for v in g.vertices() if rs[v] == leftrow]
        for v in left:
            # First we add the gates to the circuit that can be processed now,
            # and we simplify the graph to represent this.
            q = qs[v]
            phase = phases[v]
            t = ty[v]
            if t != 1: raise TypeError("Only supports zx-diagrams in graph-like state")
            neigh = [w for w in g.neighbours(v) if rs[w]<leftrow]
            if len(neigh) != 1:
                raise TypeError("Graph doesn't seem circuit like: multiple parents")
            n = neigh[0]
            if qs[n] != q:
                raise TypeError("Graph doesn't seem circuit like: cross qubit connections")
            if g.edge_type(g.edge(n,v)) == 2:
                add_hadamard(q)
                g.set_edge_type(g.edge(n,v),1)
            #if t == 0: continue # it is an output
            if phase != 0:
                add_phase_gate(q, phase)
                g.set_phase(v, 0)
        
        boundary_verts = []
        neighbours = set()
        for v in left: # Parse CZ gates between frontier
            q = qs[v]
            neigh = [w for w in g.neighbours(v) if rs[w]==leftrow and w<v]
            for n in neigh:
                q2 = qs[n]
                if g.edge_type(g.edge(v,n)) != 2:
                    raise TypeError("Invalid vertical connection between vertices of the same type")
                add_cz(q2, q)
                g.remove_edge(g.edge(v,n))
            d = [w for w in g.neighbours(v) if rs[w]>leftrow]
            neighbours.update(d)
        
        for w in neighbours: # Phase gadget stuff
            if w in gadgets:
                tgts = set(g.neighbours(w))
                tgts.remove(gadgets[w])
                if tgts.issubset(left):
                    add_gadget([qs[v] for v in tgts], phases[gadgets[w]])
                    g.remove_vertex(gadgets[w])
                    g.remove_vertex(w)
                elif tgts.issubset(left+list(processed_targets.keys())):
                    qubits = [qs[v] for v in left if v in tgts]
                    verts = [processed_targets[v] for v in tgts if v in processed_targets]
                    add_nonlocal_gadget(qubits,verts, phases[gadgets[w]])
                    g.remove_vertex(gadgets[w])
                    g.remove_vertex(w)
        neighbours = set()
        for v in left.copy(): # Deal with frontier connected to outputs
            d = [w for w in g.neighbours(v) if rs[w]>leftrow]
            if any(w in g.outputs for w in d):
                if len(d) == 1:
                    left.remove(v)
                    continue
                b = [w for w in d if w in g.outputs][0]
                if all(w in gadgets or w==b for w in d):
                    processed_targets[v] = add_phase_gate(qs[v],0)
                    left.remove(v)
                    continue
                else:
                    q = qs[b]
                    r = rs[b]
                    w = g.add_vertex(1,q,r-1)
                    nodes.append(w)
                    e = g.edge(v,b)
                    et = g.edge_type(e)
                    g.remove_edge(e)
                    g.add_edge((v,w),2)
                    g.add_edge((w,b),3-et)
                    d.remove(b)
                    d.append(w)
            neighbours.update(d)
                
        if not left: break # We are done
        right = [w for w in neighbours if w in nodes] # Only get non-phase-gadget neighbours
        m = bi_adj(g,right,left)
        #print(m)
#         target = column_optimal_swap(m)
#         right = [right[j] for (i,j) in sorted(target,key=lambda x:x[0])]
#         m = bi_adj(g,right,left)
#         print()
#         print(m)
        neighbours.difference_update(right)
        neighbours = right + list(neighbours)
        cnots = m.to_cnots()
        m2 = bi_adj(g, neighbours, left)
        for cnot in cnots:
            m.row_add(cnot.target,cnot.control)
            m2.row_add(cnot.target, cnot.control)
            add_cnot(qs[left[cnot.control]],qs[left[cnot.target]])
        connectivity_from_biadj(g,m2,neighbours,left)
        good_verts = {}
        for i, row in enumerate(m.data):
            if sum(row) == 1:
                v = left[i]
                w = right[[j for j in range(len(m.data[i])) if m.data[i][j]][0]]
                good_verts[v] = w
        if not good_verts:
            print(m)
            print(left)
            print(right)
            print(nodes)
            raise Exception("No good match found")
        for v in left:
            if v not in good_verts:
                g.set_row(v,leftrow+1)
            else:
                g.set_row(good_verts[v],leftrow+1)
                g.set_qubit(good_verts[v],qs[v])
                if len(list(g.neighbours(v))) > 2: # Gadgets are still connected to it
                    w = add_phase_gate(qs[v],0)
                    processed_targets[v] = w
        leftrow += 1
        if leftrow >= nodestotal:
            nodestotal += 20
            for v in g.vertices():
                if rs[v] > leftrow: g.set_row(v,rs[v]+20)
    # We are done processing now. Time to deal with swaps.
    swap_map = {}
    for w in g.outputs:
        v = list(g.neighbours(w))[0]
        if g.edge_type(g.edge(v,w)) == 2:
            add_hadamard(qs[v])
            g.set_edge_type(g.edge(v,w),1)
        swap_map[qs[v]] = qs[w]
    for t1, t2 in permutation_as_swaps(swap_map):
        add_cnot(t1,t2)
        add_cnot(t2,t1)
        add_cnot(t1,t2)
    
    for i in range(len(g.outputs)):
        v = h.add_vertex(0,i,depth)
        h.outputs.append(v)
        h.add_edge((qindex[i],v),1)
        qindex[i] = v
    
    return h

# O(N^3)
def max_overlap(cz_matrix):
    """Given an adjacency matrix of qubit connectivity of a CZ circuit, returns:
    a) the rows which have the maximum inner product
    b) the list of common qubits between these rows
    """
    N = len(cz_matrix.data[0])

    max_inner_product = 0
    final_common_qbs = list()
    overlapping_rows = tuple()
    for i in range(N):
        for j in range(i+1,N):
            inner_product = 0
            i_czs = 0
            j_czs = 0
            common_qbs = list()
            for k in range(N):
                i_czs += cz_matrix.data[i][k]
                j_czs += cz_matrix.data[j][k]

                if cz_matrix.data[i][k]!=0 and cz_matrix.data[j][k]!=0:
                    inner_product+=1
                    common_qbs.append(k)

            if inner_product > max_inner_product:
                max_inner_product = inner_product
                if i_czs < j_czs:
                    overlapping_rows = [j,i]
                else:
                    overlapping_rows = [i,j]
                final_common_qbs = common_qbs
    return [overlapping_rows,final_common_qbs]


def filter_duplicate_cnots(cnots):
    qubits = max([max(cnot.control,cnot.target) for cnot in cnots]) + 1
    c = Circuit(qubits)
    c.gates = cnots.copy()
    c = basic_optimization(c,do_swaps=False)
    return c.gates


def modified_extract(g, optimize_czs=True, optimize_cnots=2, quiet=True):
    """Given a graph put into semi-normal form by :func:`simplify.full_reduce`, 
    it extracts its equivalent set of gates into an instance of :class:`circuit.Circuit`.

    :param g: The ZX-diagram graph to be extracted into a Circuit.
    :param optimize_czs: Whether to try to optimize the CZ-subcircuits by exploiting overlap between the CZ gates
    :param optimize_cnots: (0,1,2,3) Level of CNOT optimization to apply.
    :param quiet: Whether to print detailed output of the extraction process.
    """
    qs = g.qubits() # We are assuming that these are objects that update...
    rs = g.rows()   # ...to reflect changes to the graph, so that when...
    ty = g.types()  # ... g.set_row/g.set_qubit is called, these things update directly to reflect that
    phases = g.phases()
    c = Circuit(g.qubit_count())

    gadgets = {}
    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in g.inputs and v not in g.outputs:
            n = list(g.neighbours(v))[0]
            gadgets[n] = v
    
    qubit_map = dict()
    frontier = []
    for o in g.outputs:
        v = list(g.neighbours(o))[0]
        if v in g.inputs: continue
        frontier.append(v)
        qubit_map[v] = qs[o]
        
    czs_saved = 0
    
    while True:
        # preprocessing
        for v in frontier: # First removing single qubit gates
            q = qubit_map[v]
            b = [w for w in g.neighbours(v) if w in g.outputs][0]
            e = g.edge(v,b)
            if g.edge_type(e) == 2: # Hadamard edge
                c.add_gate("HAD",q)
                g.set_edge_type(e,1)
            if phases[v]: 
                c.add_gate("ZPhase", q, phases[v])
                g.set_phase(v,0)
        # And now on to CZ gates
        cz_mat = Mat2([[0 for i in range(g.qubit_count())] for j in range(g.qubit_count())])
        for v in frontier:
            for w in list(g.neighbours(v)):
                if w in frontier:
                    cz_mat.data[qubit_map[v]][qubit_map[w]] = 1
                    cz_mat.data[qubit_map[w]][qubit_map[v]] = 1
                    g.remove_edge(g.edge(v,w))
        
        if optimize_czs:
            overlap_data = max_overlap(cz_mat)
            while len(overlap_data[1]) > 2: #there are enough common qubits to be worth optimising
                i,j = overlap_data[0][0], overlap_data[0][1]
                czs_saved += len(overlap_data[1])-2
                c.add_gate("CNOT",i,j)
                for qb in overlap_data[1]:
                    c.add_gate("CZ",j,qb)
                    cz_mat.data[i][qb]=0
                    cz_mat.data[j][qb]=0
                    cz_mat.data[qb][i]=0
                    cz_mat.data[qb][j]=0
                c.add_gate("CNOT",i,j)
                overlap_data = max_overlap(cz_mat)

        for i in range(g.qubit_count()):
            for j in range(i+1,g.qubit_count()):
                if cz_mat.data[i][j]==1:
                    c.add_gate("CZ",i,j)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbours = set()
        for v in frontier.copy():
            d = [w for w in g.neighbours(v) if w not in g.outputs]
            if any(w in g.inputs for w in d): #frontier vertex v is connected to an input
                if len(d) == 1: # Only connected to input, remove from frontier
                    frontier.remove(v)
                    continue
                # We disconnect v from the input b via a new spider
                b = [w for w in d if w in g.inputs][0]
                q = qs[b]
                r = rs[b]
                w = g.add_vertex(1,q,r+1)
                e = g.edge(v,b)
                et = g.edge_type(e)
                g.remove_edge(e)
                g.add_edge((v,w),2)
                g.add_edge((w,b),3-et)
                d.remove(b)
                d.append(w)
            neighbours.update(d)
        
        if not frontier: break # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        removed_gadget = False
        for w in neighbours:
            if w not in gadgets: continue
            for v in g.neighbours(w):
                if v in frontier:
                    apply_rule(g,pivot,[(w,v,[],[o for o in g.neighbours(v) if o in g.outputs])])
                    frontier.remove(v)
                    del gadgets[w]
                    frontier.append(w)
                    qubit_map[w] = qubit_map[v]
                    removed_gadget = True
                    break
        if removed_gadget: # There was indeed a gadget in the way. Go back to the top
            continue
            
        neighbours = list(neighbours)
        m = bi_adj(g,neighbours,frontier)
        if all(sum(row)!=1 for row in m.data): # No easy vertex
            if optimize_cnots>1:
                 greedy = greedy_reduction(m)
            else: greedy = None
            if greedy:
                greedy = [CNOT(target,control) for control,target in greedy]
                if (len(greedy)==1 or optimize_cnots<3) and not quiet: print("Found greedy reduction with", len(greedy), "CNOT")
                cnots = greedy
            if not greedy or (optimize_cnots == 3 and len(greedy)>1):
                perm = column_optimal_swap(m)
                perm = {v:k for k,v in perm.items()}
                neighbours2 = [neighbours[perm[i]] for i in range(len(neighbours))]
                m2 = bi_adj(g, neighbours2, frontier)
                if optimize_cnots > 0:
                    cnots = m2.to_cnots(optimize=True)
                else:
                    cnots = m2.to_cnots(optimize=False)
                cnots = filter_duplicate_cnots(cnots) # Since the matrix is not square, the algorithm sometimes introduces duplicates
                if greedy:
                    m3 = m2.copy()
                    for cnot in cnots:
                        m3.row_add(cnot.target,cnot.control)
                    reductions = sum(1 for row in m3.data if sum(row)==1)
                    if greedy and (len(cnots)/reductions > len(greedy)-0.1):
                        if not quiet: print("Found greedy reduction with", len(greedy), "CNOTs")
                        cnots = greedy
                    else:
                        neighbours = neighbours2
                        m = m2
                        if not quiet: print("Gaussian elimination with", len(cnots), "CNOTs")
            # We now have a set of CNOTs that suffice to extract at least one vertex.
            m2 = m.copy()
            for cnot in cnots:
                m2.row_add(cnot.target,cnot.control)
            extractable = set()
            for i, row in enumerate(m2.data):
                if sum(row) == 1:
                    extractable.add(i)
            # We now know which vertices are extractable, and hence the CNOTs on qubits that do not involved
            # these vertices aren't necessary.
            # So first, we get rid of all the CNOTs that happen in the Gaussian elimination after 
            # all the extractable vertices have become extractable
            m2 = m.copy()
            for count, cnot in enumerate(cnots):
                if sum(1 for row in m2.data if sum(row)==1) == len(extractable): #extractable rows equal to maximum
                    cnots = cnots[:count] # So we do not need the remainder of the CNOTs
                    break
                m2.row_add(cnot.target, cnot.control)
            # We now recalculate which vertices were extractable, because the deleted cnots
            # might have acted to swap this vertex around some.
            extractable = set()
            for i, row in enumerate(m2.data):
                if sum(row) == 1:
                    extractable.add(i)
            # And now we try to get rid of some more CNOTs, that can be commuted to the end of the CNOT circuit
            # without changing extractability.
            necessary_cnots = []
            blocked = {i:'A' for i in extractable} # 'A' stands for "blocked for All". 'R' for
            for cnot in reversed(cnots):
                if cnot.target not in blocked and cnot.control not in blocked: continue #CNOT not needed
                should_add = False
                if cnot.target in blocked and blocked[cnot.target] != 'R': 
                    should_add = True
                    blocked[cnot.target] = 'A'
                if cnot.control in blocked and blocked[cnot.control] != 'G':
                    should_add = True
                    blocked[cnot.control] = 'A'
                if cnot.control in extractable: should_add = True
                if cnot.target in extractable: should_add = True
                if not should_add: continue
                necessary_cnots.append(cnot)
                if cnot.control not in blocked: blocked[cnot.control] = 'G' # 'G' stands for Green
                if cnot.target not in blocked: blocked[cnot.target] = 'R' # 'R' stands for Red
            if not quiet: print("Actual realization required", len(necessary_cnots), "CNOTs")
            cnots = []
            for cnot in reversed(necessary_cnots):
                m.row_add(cnot.target,cnot.control)
                cnots.append(CNOT(qubit_map[frontier[cnot.control]],qubit_map[frontier[cnot.target]]))
            #for cnot in cnots:
            #    m.row_add(cnot.target,cnot.control)
            #    c.add_gate("CNOT",qubit_map[frontier[cnot.control]],qubit_map[frontier[cnot.target]])
            connectivity_from_biadj(g,m,neighbours,frontier)
        else:
            if not quiet: print("Simple vertex")
            cnots = []
        good_verts = dict()
        for i, row in enumerate(m.data):
            if sum(row) == 1:
                v = frontier[i]
                w = neighbours[[j for j in range(len(row)) if row[j]][0]]
                good_verts[v] = w
        if not good_verts: raise Exception("No extractable vertex found. Something went wrong")
        hads = []
        for v,w in good_verts.items(): # Update frontier vertices
            hads.append(qubit_map[v])
            #c.add_gate("HAD",qubit_map[v])
            qubit_map[w] = qubit_map[v]
            b = [o for o in g.neighbours(v) if o in g.outputs][0]
            g.remove_vertex(v)
            g.add_edge((w,b))
            frontier.remove(v)
            frontier.append(w)
        if not quiet: print("Vertices extracted:", len(good_verts))
        for cnot in cnots: c.add_gate(cnot)
        for h in hads: c.add_gate("HAD",h)
            
    if optimize_czs:
        if not quiet: print("CZ gates saved:", czs_saved)
    # Outside of loop. Finish up the permutation
    id_simp(g,quiet=True) # Now the graph should only contain inputs and outputs
    swap_map = {}
    leftover_swaps = False
    for v in g.outputs: # Finally, check for the last layer of Hadamards, and see if swap gates need to be applied.
        q = qs[v]
        i = list(g.neighbours(v))[0]
        if i not in g.inputs: 
            raise TypeError("Algorithm failed: Not fully reducable")
            return c
        if g.edge_type(g.edge(v,i)) == 2:
            c.add_gate("HAD", q)
            g.set_edge_type(g.edge(v,i),1)
        if qs[i] != q: leftover_swaps = True
        swap_map[q] = qs[i]
    if leftover_swaps: 
        for t1, t2 in permutation_as_swaps(swap_map):
            c.add_gate("SWAP", t1, t2)
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))
    return c