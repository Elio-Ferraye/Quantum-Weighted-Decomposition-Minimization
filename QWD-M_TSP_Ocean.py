"""
QWD-M IMPLEMENTATION AND COMPARATIVE BENCHMARKING (TSP EXAMPLE)

The following code implements QWD-M (Quantum Weighted Decomposition Minimization)
and compares it with conventional methods on a QUBO formulation of the Traveling
Salesman Problem (TSP).

You can open this code in an IDE (e.g., PyCharm) and execute it on the D-Wave
platform after connecting your Virtual Environment with the D-Wave API token.
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

##############################################################################
#                       GLOBAL PARAMETERS & SETTINGS                         #
##############################################################################

SEED = 123
random.seed(SEED)

NUM_READS = 100    # Number of reads for quantum annealer
USE_SIMULATED_ANNEALING_BASELINE = True  # Classic baseline method
VERBOSE = True

##############################################################################
#      1) TSP QUBO GENERATION (TRAVELING SALESMAN PROBLEM) EXAMPLE           #
##############################################################################

def generate_random_tsp_matrix(num_cities: int = 5) -> List[List[int]]:
    """
    Generate a random distance matrix for a TSP instance with 'num_cities' cities.
    The matrix is symmetric with zero diagonals.
    Distances are random integers in [1..10], except 0 for diagonal.
    """
    matrix = [[0]*num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            dist = random.randint(1, 10)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix

def tsp_bqm_from_distance_matrix(distance_matrix: List[List[int]]) -> BinaryQuadraticModel:
    """
    Construct a QUBO for the TSP from a given distance matrix using
    a standard TSP -> QUBO approach.

    We define a binary variable x_{i,t} meaning "city i is visited in step t".
    The objective: minimize total distance + enforce each city is visited exactly once.
    For a simpler illustration here, we create a partial or approximate TSP QUBO
    to keep the code short.
    """
    n = len(distance_matrix)  # num of cities
    # We'll interpret variables as x_{(i,t)} with i in [0..n-1], t in [0..n-1].
    # Flatten (i,t) into single index: var_index = i*n + t.

    def var_index(i, t):
        return i*n + t

    # We'll create a BQM
    bqm = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

    # 1) Distance cost: sum of dist(i,j) if x_{(i,t)} & x_{(j,t+1)}
    #    We'll do a ring t -> t+1 mod n to ensure a cycle.
    big_dist_scale = 2.0  # scale up distance cost
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = distance_matrix[i][j]
                for t in range(n):
                    t_next = (t + 1) % n
                    # quadratic var: x_{(i,t)} & x_{(j,t_next)}
                    idxA = var_index(i, t)
                    idxB = var_index(j, t_next)
                    existing = bqm.quadratic.get((idxA, idxB), bqm.quadratic.get((idxB, idxA), 0.0))
                    bqm.quadratic[(idxA, idxB)] = (existing if existing else 0.0) + big_dist_scale*dist

    # 2) Constraint: each city visited exactly once
    #    sum_{t} x_{(i,t)} = 1 for each i => We add penalty if sum != 1
    #    We'll do a simple pairwise penalty approach for sum_{t}(x_{(i,t)})=1.
    big_constraint = 5.0
    for i in range(n):
        # sum_{t} x_{(i,t)} = 1 => penalize x_{(i,t)} + x_{(i,u)} for t!=u
        for t in range(n):
            idxT = var_index(i, t)
            # linear term to slightly push usage
            curr_lin = bqm.linear.get(idxT, 0.0)
            bqm.linear[idxT] = curr_lin - 1.0  # partial push

            for u in range(t+1, n):
                idxU = var_index(i, u)
                # penalty if both x_{(i,t)} & x_{(i,u)}=1
                existing = bqm.quadratic.get((idxT, idxU), bqm.quadratic.get((idxU, idxT), 0.0))
                bqm.quadratic[(idxT, idxU)] = (existing if existing else 0.0) + big_constraint

    return bqm

##############################################################################
#       2) CLASSICAL METHOD (BASELINE)                                      #
##############################################################################

def solve_classically(bqm: BinaryQuadraticModel):
    """
    Baseline: either simulated annealing or exact solver.
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
#     3) ANNEALING STANDARD (NO QWD-M)                                      #
##############################################################################

def solve_qanneal_standard(bqm: BinaryQuadraticModel,
                           num_reads: int = NUM_READS
                           ) -> Tuple[Dict[int,int], float]:
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm,
                               num_reads=num_reads,
                               label="StandardAnnealTSP")
    best = sampleset.first
    return best, best.energy

##############################################################################
#  4) QWD-M: BUILDING A RICH GROUND SPACE (H'_i) + FUSION FOR TSP           #
##############################################################################

def build_degenerate_groundspace(num_vars: int,
                                 patterns: List[str],
                                 weights: List[float]) -> BinaryQuadraticModel:
    """
    Builds a BQM for an "initial Hamiltonian" H'_i that has a degenerate ground
    space covering the given patterns. The synergy approach is similar to the
    MaxCut example, but now we treat them as bit-vectors in the TSP var space.

    patterns: list of bitstrings, length = num_vars
    weights : list of floats of same length
    """
    linear = {i: 0.0 for i in range(num_vars)}
    quadratic = {}
    offset = 0.0

    big_reward_factor = 4.0
    synergy_factor = 2.0
    mismatch_penalty = 2.5

    synergy_matrix = {}

    for pat, w in zip(patterns, weights):
        w_eff = w * big_reward_factor
        for i, bit in enumerate(pat):
            if bit == '1':
                curr_li = linear.get(i, 0.0)
                linear[i] = curr_li - w_eff/(1.0 + mismatch_penalty)
            else:
                curr_li = linear.get(i, 0.0)
                linear[i] = curr_li + w_eff*mismatch_penalty/(2.0* num_vars)

        # synergy: if pat[i]=='1' and pat[j]=='1' => add synergy
        for i_idx in range(num_vars):
            for j_idx in range(i_idx+1, num_vars):
                if pat[i_idx] == '1' and pat[j_idx] == '1':
                    key = (min(i_idx,j_idx), max(i_idx,j_idx))
                    synergy_matrix[key] = synergy_matrix.get(key, 0.0) - synergy_factor * w

    for (i, j), val in synergy_matrix.items():
        existing_q = quadratic.get((i,j), quadratic.get((j,i), 0.0))
        quadratic[(i,j)] = (existing_q if existing_q else 0.0) + val

    bqm_init = BinaryQuadraticModel(linear, quadratic, offset, vartype='BINARY')
    return bqm_init

def fuse_bqm_initial_problem(bqm_init: BinaryQuadraticModel,
                             bqm_problem: BinaryQuadraticModel,
                             alpha: float = 1.0,
                             beta: float = 1.0
                             ) -> BinaryQuadraticModel:
    """
    Merge H'_i + H_p => combined BQM.
    """
    bqm_combined = BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

    # Add bqm_init
    for i in bqm_init.linear.keys():
        existing = bqm_combined.linear.get(i, 0.0)
        bqm_combined.linear[i] = existing + alpha*bqm_init.linear[i]
    for (i, j), val in bqm_init.quadratic.items():
        existing_q = bqm_combined.quadratic.get((i,j), bqm_combined.quadratic.get((j,i), 0.0))
        bqm_combined.quadratic[(i,j)] = (existing_q if existing_q else 0.0) + alpha*val
    bqm_combined.offset += alpha*bqm_init.offset

    # Add bqm_problem
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
    """
    QWD-M step:
      1) Build a degenerate groundspace BQM (H'_i).
      2) Fuse with bqm_problem => single BQM
      3) Run on D-Wave
    """
    num_vars = len(bqm_problem.variables)
    bqm_init = build_degenerate_groundspace(num_vars, patterns, weights)
    bqm_combined = fuse_bqm_initial_problem(bqm_init, bqm_problem, alpha, beta)

    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm_combined, num_reads=num_reads, label="QWD-M-TSP")
    best = sampleset.first
    return best, best.energy

##############################################################################
#                 5) MAIN: TEST ON TSP QUBO                                 #
##############################################################################

def main():
    print("==== QWD-M IMPLEMENTATION TEST ON TSP QUBO ====")

    # 1) Generate random TSP distance matrix
    num_cities = 6
    dist_matrix = generate_random_tsp_matrix(num_cities)
    print(f"[INFO] TSP distance matrix generated for {num_cities} cities.\n")

    # 2) Build TSP BQM
    bqm_tsp = tsp_bqm_from_distance_matrix(dist_matrix)
    num_vars = len(bqm_tsp.variables)
    print("[INFO] TSP BQM built =>", num_vars, " variables.\n")

    # 3) Baseline (classic)
    best_classic, e_classic = solve_classically(bqm_tsp)
    print("--- Baseline (Classical) ---")
    print(f"Solution sample = {best_classic}")
    print(f"Energy          = {e_classic:.4f}\n")

    # 4) QA standard
    best_std, e_std = solve_qanneal_standard(bqm_tsp, NUM_READS)
    print("--- Standard QA ---")
    print(f"Solution sample = {best_std}")
    print(f"Energy          = {e_std:.4f}\n")

    # 5) QWD-M
    # We'll randomly generate some patterns for the 'num_vars' bits
    # so as not to bias QWD-M to the known solution
    patterns = []
    weights = []
    for _ in range(5):  # e.g. 5 patterns
        pat_str = "".join(str(random.randint(0,1)) for __ in range(num_vars))
        wval = random.uniform(0.5, 1.5)
        patterns.append(pat_str)
        weights.append(wval)

    best_qwdm, e_qwdm = solve_qanneal_qwdm(
        bqm_tsp,
        patterns,
        weights,
        alpha=1.0,
        beta=1.0,
        num_reads=NUM_READS
    )
    print("--- QWD-M ---")
    print(f"Patterns used   : {patterns}")
    print(f"Weights used    : {weights}")
    print(f"Solution sample = {best_qwdm}")
    print(f"Energy          = {e_qwdm:.4f}\n")

    # 6) Recap
    print("===== Performance Summary =====")
    print(f"Classical energy   = {e_classic:.4f}")
    print(f"Standard QA energy = {e_std:.4f}")
    print(f"QWD-M QA energy    = {e_qwdm:.4f}")

    if e_qwdm < e_std:
        print("QWD-M obtains a better or equal solution => Potential strong improvement!")
    else:
        print("QWD-M did not surpass standard QA => Possibly refine synergy or patterns further.")

    print("\n==== End of QWD-M Implementation Test on TSP QUBO ====")

if __name__ == "__main__":
    main()
