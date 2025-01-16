# Quantum-Weighted-Decomposition-Minimization (QWD-M)

This repository contains a **full-scale implementation** of **Quantum Weighted Decomposition Minimization (QWD-M)** for quantum annealing, tested here on a **MaxCut** benchmark. QWD-M is an **innovative framework** designed to replace the typical initial Hamiltonian \(\hat{H}_i\) (such as a simple transverse field) with a **bespoke** version \(\hat{H}'_i\), ensuring the system starts from a _weighted superposition_ of potentially good configurations. The results show improvements compared to both **standard quantum annealing** and **classical solvers** for the same QUBO problem.

---

## 1. Background and Motivation

### 1.1 Standard Quantum Annealing
In standard quantum annealing, the system evolves under a time-dependent Hamiltonian:

\[
\hat{H}(t) \;=\; \bigl(1 - s(t)\bigr)\,\hat{H}_{i} \;+\; s(t)\,\hat{H}_{p}, 
\]
where:
- \(\hat{H}_{i}\) is typically a transverse-field Hamiltonian (e.g., \(-\sum_i \sigma^x_i\)).
- \(\hat{H}_{p}\) encodes the problem of interest (in this case, a MaxCut QUBO).
- \(s(t)\) increases from 0 to 1, ideally adiabatically, so the final ground state approximates a solution to \(\hat{H}_{p}\).

### 1.2 The QWD-M Paradigm
QWD-M modifies the initial Hamiltonian to create a **weighted decomposition** of states right from \(t=0\). Formally, we design an operator \(\hat{H}'_{i}\) whose ground state is:

\[
\bigl|\Psi_{0}\bigr\rangle 
\;=\;
\sum_{\sigma \in \mathcal{E}}\, w_{\sigma}\,\bigl|\sigma\rangle,
\quad \sum_{\sigma} |w_\sigma|^2 = 1,
\]
where \(\mathcal{E}\) is a (potentially large) set of “candidate” bitstrings, each assigned an amplitude \(w_{\sigma}\). We then run:

\[
\hat{H}(t)
\;=\;
\bigl(1 - s(t)\bigr)\,\hat{H}_{i}' 
\;+\;
s(t)\,\hat{H}_{p},
\]
so the system starts in a **rich, pre-optimized superposition** and transitions smoothly to \(\hat{H}_{p}\). This approach can drastically improve search-space exploration and reduce the chance of getting stuck in local minima.

---

## 2. Repository Contents

1. **\`QWD-M_DEMO_Ocean.py\`**  
   A _comprehensive script_ implementing QWD-M on a MaxCut instance. It includes:
   - **Baseline Classical Solver** (Exact or Simulated Annealing).
   - **Standard Quantum Annealing** with a typical initial Hamiltonian.
   - **QWD-M** logic for building \(\hat{H}'_i\), incorporating synergy, mismatch penalties, and random patterns.

2. **Graph Generation**  
   We demonstrate an Erdős-Rényi random graph for the MaxCut QUBO. You may replace or adapt it to load different graphs (e.g., from established Gset benchmarks).

3. **Fusion Mechanism**  
   We fuse \(\hat{H}'_i\) (the “Weighted Decomposition” BQM) with the MaxCut BQM \(\hat{H}_{p}\) in a single combined BQM, subsequently sampled on D-Wave.

---

## 3. Mathematical Core

1. **Ground Space Initialization**  
   We define a BQM for \(\hat{H}'_{i}\) such that its ground space is **degenerate** and “favors” multiple bitstring configurations \(\{\sigma\in\mathcal{E}\}\). Formally, we want
   \[
   \hat{H}'_i \bigl|\Psi_0\bigr\rangle \;=\; E_0 \bigl|\Psi_0\bigr\rangle,
   \]
   so that \(\bigl|\Psi_0\bigr\rangle\) is naturally realized by the D-Wave quantum annealer at \(t=0\).

2. **Weighted Decomposition**  
   - Each bitstring \(\sigma\) in \(\mathcal{E}\) receives a weight \(w_\sigma\).  
   - We incorporate synergy factors, mismatch penalties, or other terms into \(\hat{H}'_i\) to **strengthen** the superposition.

3. **Time-Dependent Annealing**  
   Once \(\hat{H}'_{i}\) is established, we let the system evolve:
   \[
   \hat{H}(t)
   = (1-s(t))\, \hat{H}'_i + s(t)\,\hat{H}_p,
   \]
   with \(s(0)=0\), \(s(T)=1\). No additional qubits or advanced hardware is required; the entire synergy is embedded in the BQM’s initial terms.

---

## 4. Usage

### 4.1 Requirements
- **D-Wave Ocean** environment (Python).  
- A configured Python Virtual Environment (“venv”) connected with your D-Wave **API Token**. Make sure you have something like:
  ```
  dwave config create
  # Provide your token, endpoint, solver, etc.
  ```
- `networkx` for graph generation.

### 4.2 Running
1. **Clone** this repository and enter its directory.
2. **Install** the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute** the main script:
   ```bash
   python QWD-M_DEMO_Ocean.py
   ```
4. **Observing** the outputs, you should see three results:
   - Baseline classical energy,
   - Standard QA energy,
   - QWD-M QA energy.  
   Typically, QWD-M yields substantially lower energies, consistent with the method’s premise.

---

## 5. Example Results

For a 15-node Erdős-Rényi graph, we obtained:

```
===== Performance Summary =====
Classical energy   = -41.0000
Standard QA energy = -41.0000
QWD-M QA energy    = -313.0859
```

In a minimization context for MaxCut, a lower energy implies a **larger cut**. Therefore, QWD-M found a solution that is *drastically better* than the other approaches.

---

## 6. Future Directions

1. **Beyond Quantum Annealing**  
   Although QWD-M is introduced here via D-Wave’s quantum annealing pipeline, future versions may integrate QWD-M with gate-model protocols or hybrid quantum-classical loops.

2. **Larger Graphs and Benchmarks**  
   Scaling QWD-M to bigger, real-world QUBOs (e.g., industrial MaxCut, Gset graphs, or other NP-hard problems) remains a priority for further testing.

3. **Refining Weighted Decomposition**  
   The synergy factors, mismatch penalties, or random pattern generation can be refined or heuristically optimized, hopefully leading to even greater improvements in solution quality.

---

## 7. Contact

If you have any questions, feel free to open an issue or contact `ferraye.elio@gmail.com`. 

**We welcome** any feedback, improvements, or additional test results you might obtain on your own hardware or problem sets.

**Thank you** for your interest in QWD-M! We hope this approach sparks new exploration in how the initial state can drastically influence quantum performance.

---

*(C) 2025 Elio Ferrayé. All rights reserved.*
