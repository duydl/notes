# Graph Neural Ising Transformer for Efficient Quantum Optimization

## **1. Background**  

### **1.1. Related Work**  
- **Ising Models & NP-Hard Problems**  
  - **Ising Hamiltonian**:  
    \[
    H = -\sum_{i,j} J_{ij} s_i s_j - \sum_i h_i s_i, \quad s_i \in \{-1, +1\}
    \]
    The energy function defines interactions between spins. The goal is to minimize \( H \), which leads to finding the ground state configuration.

  - **QUBO Formulation**:  
    \[
    \min_{x} \sum_{i,j} Q_{ij} x_i x_j + \sum_i c_i x_i, \quad x_i \in \{0,1\}
    \]
    QUBO is equivalent to Ising models in its ground state, where the optimal solutions correspond to the lowest-energy configurations of the Ising system, with \( s_i = 2x_i - 1 \), mapping combinatorial problems to quantum hardware.


- **Quantum Annealing (QA)**  
  - Adiabatic Evolution:  



- **Graph Format**  
  - Edge Feature: Both spin-spin interactions and external fields are treated as edge weights.
    \[
    w(i, j) = \begin{cases}
        J_{ij}, & i \neq j \\
        h_j, & i = 0
    \end{cases}
    \]


### **1.2. Motivation**  
- Qubit limitation in quantum hardware (D-Wave ≤ 5,640 qubits).  
- Existing reduction methods not tunable.   

## **2. Method**  

<!-- The GNN is not directly trained to compress graphs—it is trained to classify edges. Compression is a separate iterative process that applies the trained GNN model. -->

### **2.1. Dataset Generation**  
- Graph Construction:  
  - The Ising model can be represented as a weighted graph where nodes are spins and edges represent interactions:  
    \[
    V = \{0, 1, \dots, n\}, \quad E = \{(i, j) \mid J_{ij} \neq 0\}
    \]
- Ground state determination:  
  - Solve Ising via **Gurobi MIQP solver** (Mixed-Integer Quadratic Programming)

  - Label edges \((i,j)\):  
    - **Alignment** (\(s_i = s_j\) in all ground states) → **Label: Merge**
    - **Anti-alignment** (\(s_i = -s_j\) in all ground states) → **Label: Flip-merge**
    - **Neutral** (Different states appear across ground states) → **Label: No merge**  
 
- **Edge classification is Co-NP-hard**
### 2.2. GNN Model
The GNN Model lean to classify edge...

- GNN Node & Edge Updates: Each node (spin) and each edge updates its feature based on neighboring spins and the connecting edges as follow: 
  \[
  h_v^{(\ell)} = \text{MLP}_1 \left(  h_v^{(\ell-1)} \oplus \sum_{u\in \mathcal{N}(v)}  h_{u}^{(\ell-1)} \oplus\sum_{u\in \mathcal{N}(v)} e_{v, u}^{(\ell-1)} \right)
  \]

  \[
  e_{uv}^{(\ell)} =  \text{MLP}_2\left(h_u^{(\ell-1)} \oplus h_v^{(\ell-1)} \oplus e_{uv}^{(\ell-1)}\right)
  \]


- Prediction of Merge Candidates: The neural network predicts whether two spins should be merged based on their learned features.  
  \[
  z_{uv} = h_u^{(L)} \oplus h_v^{(L)} \oplus e_{uv}^{(L)}
  \]

  \[
  \hat{y}_{uv} = \sigma(\langle w, z_{uv} \rangle)
  \]

- Loss Function : The softmax function prioritizes high-confidence predictions, ensuring the most certain merges are performed first.    
  \[
  c_i = \frac{\exp\left(|\hat{y}_{i} - 0.5| / T\right)}{\sum_{j} \exp\left(|\hat{y}_{j} - 0.5| / T\right)}
  \]

  \[
  w_i = \lambda \cdot c_i + (1 - \lambda)
  \]


  \[
  \mathcal{L} = \sum_{i} w_i \left( -y_i \log(\hat{y}_i) - (1 - y_i) \log(1 - \hat{y}_i) \right)
  \]


### 2.3 Compression Phase :

Operation
  - Merge Operation: When two spins are always aligned, merging them reduces problem size while preserving the original optimization structure.
    \[
    w(i, k) = w(i,k) + w(j,k)
    \]

  - Flip-Merge Operation: If two spins are always opposite, flipping before merging. 


Evaluation Metrics


  - **Optimality**: Measures how close the reduced problem's solution is to the original optimal solution
    \[
    \text{Optimality} = 1 - \frac{|E_{\text{best}} - E_{\text{min}}|}{|E_{\text{min}}|}
    \] 

  - **Qubit Reduction**: Measures how many qubits are saved after compression, needed for quantum annealer feasibility.
    \[
    \text{Reduction} = 1 - \frac{q_{\text{compressed}}}{q_{\text{original}}}
    \]

## 3. Results & Ablation
-
  - 75% reduction still maintains 91% optimality.  
  - Significant qubit savings across all graph types.  
  - Outperforms random baseline compression strategies.  

-
    - Hybrid loss achieves best performance compared to BCE/MSE.  
    - 3-layer GNN is optimal for accuracy and generalization.  

## 4. Conclusion & Future Work

- Apply GRANITE to real-world datasets, extend to noisy quantum hardware.

## 5. Personal Ideas

- New optimization i.e Reinforcement Learning
