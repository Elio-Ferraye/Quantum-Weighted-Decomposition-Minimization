"""
QWD-M IMPLEMENTATION AND COMPARATIVE BENCHMARKING

The following code implements QWD-M and compares it with conventional methods by running a MaxCut benchmark.

This code can be opened via an IDE such as PyCharm and executed on the D-Wave platform after connecting the Virtual Environment to the API token.

"""

import math
import random
from typing import List, Tuple, Dict

# D-Wave / Ocean imports
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import (
    BinaryQuadraticModel,
    ExactSolver,
    SimulatedAnnealingSampler,
    SampleSet
)
import networkx as nx

##############################################################################
#                          GLOBAL PARAMETERS                               #
##############################################################################

SEED = 123
random.seed(SEED)

NUM_READS = 100  # Number of readings for quantum annealer
USE_SIMULATED_ANNEALING_BASELINE = True  # Classic baseline method
VERBOSE = True

##############################################################################
#       1) BENCHMARK GRAPH LOADING/CREATION (MAXCUT TYPE)           #
##############################################################################

def load_benchmark_graph(num_nodes: int = 15) -> nx.Graph:
    """
    Create or load an Erdos-Renyi (or other) graph.
    For demonstration purposes, we'll generate an ER random graph.
    """
    p = 0.5
    G = nx.erdos_renyi_graph(num_nodes, p, seed=SEED)
    return G

##############################################################################
#       2) QUBO GENERATION FOR MAXCUT                                   #
##############################################################################

def maxcut_bqm_from_graph(G: nx.Graph) -> BinaryQuadraticModel:
    """
    Construct a QUBO for MaxCut of a graph G.
    Formula: Minimize the opposite of the number of “cut” edges.
    MaxCut => - sum_{(u,v)}( x_u XOR x_v ), rewritten in binary.
    => E_{uv} ~ - ( x_u + x_v - 2 x_u x_v ).
    """
    bqm = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

    for (u,v) in G.edges():
        # We want E_{uv} = - x_u - x_v + 2 x_u x_v
        # -> linear[u] -= 1, linear[v] -= 1, quadratic[u,v] += 2

        # linear[u] -= 1
        curr_lu = bqm.linear.get(u, 0.0)
        bqm.linear[u] = curr_lu - 1.0

        # linear[v] -= 1
        curr_lv = bqm.linear.get(v, 0.0)
        bqm.linear[v] = curr_lv - 1.0

        # quadratic[u,v] += 2
        existing_q = bqm.quadratic.get((u,v), bqm.quadratic.get((v,u), 0.0))
        bqm.quadratic[(u,v)] = existing_q + 2.0

    return bqm

##############################################################################
#     3) CLASSIC METHOD (BASELINE)                                      #
##############################################################################

def solve_classically(bqm: BinaryQuadraticModel):
    """
    Baseline: simulated annealing or exact solver.
    """
    if USE_SIMULATED_ANNEALING_BASELINE:
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=NUM_READS)
        best = sampleset.first
        return best, best.energy
    else:
        solver = ExactSolver()
        solutions = solver.sample(bqm)
        best = next(iter(solutions))
        return best, best.energy

##############################################################################
#     4) ANNEALING STANDARD SANS QWD-M                                      #
##############################################################################

def solve_qanneal_standard(bqm: BinaryQuadraticModel,
                           num_reads: int = NUM_READS
                           ) -> Tuple[Dict[int,int], float]:
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm,
                               num_reads=num_reads,
                               label="StandardAnnealMaxCut")
    best = sampleset.first
    return best, best.energy

##############################################################################
#  5) QWD-M: BUILDING A RICH GROUND SPACE (H'_i) + FUSION           #
##############################################################################

def build_degenerate_groundspace(num_qubits: int,
                                 patterns: List[str],
                                 weights: List[float]) -> BinaryQuadraticModel:
    """
    Builds a BQM defining a degenerate ground space including
    several patterns bitstrings,synergy etc.
    """
    linear = {i: 0.0 for i in range(num_qubits)}
    quadratic = {}
    offset = 0.0

    big_reward_factor = 4.0
    synergy_factor = 2.0
    mismatch_penalty = 2.5

    synergy_matrix = {}

    # For each pattern => we reinforce the presence of '1' bits, penalize '0' bits
    # + synergy
    for pat, w in zip(patterns, weights):
        w_eff = w * big_reward_factor
        for i, bit in enumerate(pat):
            if bit == '1':
                # linear[i] -= w_eff/(1+mismatch_penalty)
                curr_li = linear.get(i, 0.0)
                linear[i] = curr_li - w_eff/(1.0 + mismatch_penalty)
            else:
                # linear[i] += w_eff*mismatch_penalty/(2*n)
                curr_li = linear.get(i, 0.0)
                linear[i] = curr_li + w_eff*mismatch_penalty/(2.0*num_qubits)

        # synergy => (i,j)
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if pat[i] == '1' and pat[j] == '1':
                    key = (min(i,j), max(i,j))
                    synergy_matrix[key] = synergy_matrix.get(key, 0.0) - synergy_factor * w

    # Incorporate synergy_matrix into quadratic
    for (i, j), val in synergy_matrix.items():
        existing_q = quadratic.get((i,j), quadratic.get((j,i), 0.0))
        quadratic[(i,j)] = existing_q + val

    bqm_init = BinaryQuadraticModel(linear, quadratic, offset, vartype='BINARY')
    return bqm_init

def fuse_bqm_initial_problem(bqm_init: BinaryQuadraticModel,
                             bqm_problem: BinaryQuadraticModel,
                             alpha: float = 1.0,
                             beta: float = 1.0
                             ) -> BinaryQuadraticModel:
    """
    Fusion H'_i + H_p => BQM combines
    """
    bqm_combined = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

    # BQM init
    for i in bqm_init.linear.keys():
        existing = bqm_combined.linear.get(i, 0.0)
        bqm_combined.linear[i] = existing + alpha*bqm_init.linear[i]
    for (i, j), val in bqm_init.quadratic.items():
        existing_q = bqm_combined.quadratic.get((i,j), bqm_combined.quadratic.get((j,i), 0.0))
        bqm_combined.quadratic[(i,j)] = (existing_q if existing_q else 0.0) + alpha*val
    bqm_combined.offset += alpha*bqm_init.offset

    # BQM problem
    for i in bqm_problem.linear.keys():
        existing = bqm_combined.linear.get(i, 0.0)
        bqm_combined.linear[i] = existing + beta*bqm_problem.linear[i]
    for (i, j), val in bqm_problem.quadratic.items():
        existing_q = bqm_combined.quadratic.get((i,j), bqm_combined.quadratic.get((j,i), 0.0))
        bqm_combined.quadratic[(i,j)] = (existing_q if existing_q else 0.0) + beta*val
    bqm_combined.offset += beta*bqm_problem.offset

    return bqm_combined

def solve_qanneal_qwdm(bqm_problem: BinaryQuadraticModel,
                       patterns: List[str],
                       weights: List[float],
                       alpha: float = 1.0,
                       beta: float = 1.0,
                       num_reads: int = 100
                       ):
    num_qubits = len(bqm_problem.variables)  # we recover the dimension
    bqm_init = build_degenerate_groundspace(num_qubits, patterns, weights)
    bqm_combined = fuse_bqm_initial_problem(bqm_init, bqm_problem, alpha, beta)

    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm_combined, num_reads=num_reads, label="QWD-M-MaxCut")
    best = sampleset.first
    return best, best.energy

###############################################################################
#            6) MAIN CODE: TEST ON MAXCUT         #
###############################################################################
def main():
    print("==== QWD-M IMPLEMENTATION TEST ON MAXCUT ====")

    # 1) Graphe
    num_nodes = 15
    G = load_benchmark_graph(num_nodes)
    print(f"[INFO] Loaded random Erdos-Renyi graph with {num_nodes} nodes.\n")

    # 2) QUBO MaxCut
    bqm_maxcut = maxcut_bqm_from_graph(G)
    print("[INFO] MaxCut BQM built =>", len(bqm_maxcut.variables), " variables.")

    # 3) Baseline
    best_classic, e_classic = solve_classically(bqm_maxcut)
    print("\n--- Baseline (Classical) ---")
    print(f"Solution sample = {best_classic}")
    print(f"Energy          = {e_classic:.4f}")

    # 4) QA standard
    best_std, e_std = solve_qanneal_standard(bqm_maxcut, NUM_READS)
    print("\n--- Standard QA ---")
    print(f"Solution sample = {best_std}")
    print(f"Energy          = {e_std:.4f}")

    # 5) QWD-M
    # Generate random patterns => non-biased use
    patterns = []
    weights = []
    for _ in range(5):
        pat_str = "".join(str(random.randint(0,1)) for __ in range(num_nodes))
        wval = random.uniform(0.5,1.5)
        patterns.append(pat_str)
        weights.append(wval)

    best_qwdm, e_qwdm = solve_qanneal_qwdm(
        bqm_maxcut,
        patterns,
        weights,
        alpha=1.0,
        beta=1.0,
        num_reads=NUM_READS
    )
    print("\n--- QWD-M ---")
    print(f"Patterns used   : {patterns}")
    print(f"Weights used    : {weights}")
    print(f"Solution sample = {best_qwdm}")
    print(f"Energy          = {e_qwdm:.4f}")

    # 6) Recap
    print("\n===== Performance Summary =====")
    print(f"Classical energy   = {e_classic:.4f}")
    print(f"Standard QA energy = {e_std:.4f}")
    print(f"QWD-M QA energy    = {e_qwdm:.4f}")

    if e_qwdm < e_std:
        print("QWD-M obtains a better or equal solution => Potential strong improvement!")
    else:
        print("QWD-M did not surpass standard QA => Possibly refine synergy or patterns further.")

    print("\n==== End of QWD-M Implementation Test on MaxCut ====")

if __name__ == "__main__":
    main()
