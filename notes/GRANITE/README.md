# Graph Neural Ising Transformer for Efficient Quantum Optimization

## **1. Background**  

### **1.1. Related Work**  
- **Ising Models & NP-Hard Problems**  
  - Ising Hamiltonian:  
    \[
    H = -\sum_{i,j} J_{ij} s_i s_j - \sum_i h_i s_i, \quad s_i \in \{-1, +1\}
    \]
  - Equivalent to Quadratic Unconstrained Binary Optimization (QUBO):  
    \[
    \min_{x} \sum_{i,j} Q_{ij} x_i x_j + \sum_i c_i x_i, \quad x_i \in \{0,1\}
    \]
    where \( s_i = 2x_i - 1 \).
  - Finding ground states is **NP-hard**.

- **Quantum Annealing (QA)**  
  - Adiabatic evolution:  
    \[
    H_{\text{system}}(s) = - \frac{A(s)}{2} \sum_i^n \sigma^x_i + \frac{B(s)}{2} H_{\text{problem}}
    \]
  - Limited qubit counts hinder large-scale problem-solving.

- **Graph-Based Compression & Reduction**  
  - Roof Duality \cite{hammer1984roof}, Minor Embedding \cite{choi2008minor}.  
  - Existing methods reduce **<20% of instances** \cite{thai2022fasthare}.

### **1.2. Motivation**  
- **Qubit limitation in quantum hardware (D-Wave ≤ 5,640 qubits)**.  
- **Existing reduction methods are rigid, non-tunable**.  
- **Need a flexible, learning-based approach** for progressive reduction.  

---

## **2. Method**  

### **2.1. Dataset Generation**  
- **Graph representation of Ising models**:  
  - Nodes \( V = \{0, 1, \dots, n\} \), Edges \( E = \{(i, j) \mid J_{ij} \neq 0\} \).  
  - Auxiliary vertex 0 for biases \( h_i \).  
  - **Edge weights**:  
    \[
    w(i, j) = \begin{cases}
        J_{ij}, & i \neq j \\
        h_j, & i = 0
    \end{cases}
    \]

- **Ground state determination**:  
  - Solve Ising via **Gurobi MIQP solver**:  
    \[
    \min_{x} \sum_{i,j} Q_{ij} x_i x_j + \sum_i c_i x_i
    \]
  - **Label edges**:  
    - **Alignment**: \( s_i = s_j \) → **Merge**.  
    - **Anti-alignment**: \( s_i = -s_j \) → **Flip-merge**.  
    - **Neutral**: No consistent alignment.  

### **2.2. Qubit Alignment & Compression**  
- **Merge operation \( M(i, j) \)**:  
  \[
  V' = V \setminus \{j\}, \quad E' = \{(i, k) \mid (j, k) \in E\}
  \]
  \[
  w(i, k) = w(i,k) + w(j,k)
  \]

- **Flip-Merge operation \( FM(i, j) \)**:  
  1. Flip edge weights: \( w(j, k) = -w(j, k) \)  
  2. Merge: Apply \( M(i, j) \).  

- **Edge classification is Co-NP-hard**:  
  - Reduction from **All-SAT-EQUAL (Co-NP-complete)**.  
  - Proof uses **Boolean-to-Ising encoding**:  
    \[
    Q(C) = x_c(2 - (x_1 + x_2 + x_3)) + (x_1x_2 + x_2x_3 + x_3x_1) - (x_1 + x_2 + x_3) + 1
    \]

### **2.3. GRANITE: Graph Neural Ising Transformer**  
- **GNN architecture**:  
  - **Node features**: \( h_v^{(0)} \) = (degree, weighted degree, absolute degree).  
  - **Edge features**: \( e_{uv}^{(0)} \) = (edge weights, absolute weights).  
  - **Message Passing**:  
    \[
    h_v^{(\ell)} = \text{MLP}_1 \left(  h_v^{(\ell-1)} \oplus \sum_{u \in \mathcal{N}(v)}  h_{u}^{(\ell-1)} \oplus\sum_{u\in \mathcal{N}(v)} e_{v, u}^{(\ell-1)} \right)
    \]
    \[
    e_{uv}^{(\ell)} =  \text{MLP}_2\left(h_u^{(\ell-1)} \oplus h_v^{(\ell-1)} \oplus e_{uv}^{(\ell-1)}\right)
    \]

- **Prediction function**:  
  \[
  z_{uv} = h_u^{(L)} \oplus h_v^{(L)} \oplus e_{uv}^{(L)}
  \]
  \[
  \hat{y}_{uv} = \sigma(\langle w, z_{uv} \rangle)
  \]

- **Hybrid Loss Function** (confidence-weighted BCE):  
  \[
  c_i = \frac{\exp\left(|\hat{y}_{i} - 0.5| / T\right)}{\sum_{j} \exp\left(|\hat{y}_{j} - 0.5| / T\right)}
  \]
  \[
  w_i = \lambda \cdot c_i + (1 - \lambda)
  \]
  \[
  \mathcal{L} = \sum_{i} w_i \left( -y_i \log(\hat{y}_i) - (1 - y_i) \log(1 - \hat{y}_i) \right)
  \]

- **Iterative Reduction Algorithm**:  
  1. Compute **node & edge embeddings** via GNN.  
  2. Predict **merge/flip-merge candidates**.  
  3. Apply **highest-confidence contraction**.  
  4. Repeat until desired reduction is achieved.  

---

## **3. Experiments**  

### **3.1. Experimental Setup**  
- **Datasets**:  
  - 97,500 graphs across **Erdős-Rényi, Barabási-Albert, Watts-Strogatz** models.  
  - Edge weights \( J_{ij} \sim \text{Uniform}(-5, 5) \).  
- **Evaluation**:  
  - **Optimality**:  
    \[
    \text{Optimality} = 1 - \frac{|E_{\text{best}} - E_{\text{min}}|}{|E_{\text{min}}|}
    \]
  - **Qubit Reduction**:  
    \[
    \text{Reduction} = 1 - \frac{q_{\text{compressed}}}{q_{\text{original}}}
    \]

### **3.2. Results**  
- **High optimality across all graph types**:  
  - **75% reduction still maintains 91% optimality**.  
- **Significant qubit savings**:  
  - **Erdős-Rényi: 5.3% qubits left at 75% reduction**.  
  - **Watts-Strogatz: 16.5% qubits left at 75% reduction**.  
- **Outperforms random compression strategies**.  

### **3.3. Ablation Study**  
- **Hybrid loss outperforms BCE/MSE**.  
- **3-layer GNN achieves best results**.  

---

## **4. Conclusion & Future Work**  
- **GRANITE enables tunable compression of Ising models**.  
- **Bridges gap between large optimization problems & quantum annealers**.  
- **Future work**: Real-world applications, noise-aware models for quantum devices.  