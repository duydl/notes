---
# theme: academic
# layout: cover
# coverAuthor: [ DO LE DUY ]
# coverAuthorUrl: [ https://github.com/duydl ]
# coverBackgroundUrl: https://www.bch.cuhk.edu.hk/kbwong/pymol/movie.gif
# themeConfig:
#   paginationX: r
#   paginationY: t
#   paginationPagesDisabled: [ 1 ]

theme: ../theme_academic

title: HySonLab Protein - Meeting Report
transition: slide-left
background: https://www.bch.cuhk.edu.hk/kbwong/pymol/movie.gif
info: |
  ### Progress Report on Protein Pipeline
  Overview of algorithms, databases, and tools for protein/bioinformatics study.
---

<!-- <style>
body {
  background-color: black; 
  color: white;
}
</style> -->

# Meeting Report
<!-- ### *HySonLab Material* -->
## *A Universal Graph Deep Learning Interatomic Potential for the Periodic Table*

---

## Abstract

- **Problem**: Current interatomic potentials (IAPs) are either narrowly fitted or inaccurate.
- **Solution**: Introduce M3GNet, a graph neural network-based IAP with three-body interactions.
- **Data**: Trained on Materials Project's 10 years of data.
- **Applications**: Structural relaxation, dynamic simulations, and property prediction.
- **Results**: Identified 1.8 million potentially stable materials; 1578 verified stable using DFT.

---

## Introduction

- **Atomistic Simulations**: Essential for materials design.
- **Equilibrium Structures**: First step in computational studies.
- **Electronic Structure Methods**: Accurate but computationally expensive.
- **Interatomic Potentials (IAPs)**: Needed for large-scale studies.
- **Machine Learning IAPs**: Promise better accuracy but lack general applicability.
- **M3GNet**: Combines many-body interactions with graph neural networks.

**Important Terms**:
- **Potential Energy Surface (PES)**
- **Structural Relaxation**
- **Dynamic Simulations**


---

## Materials Graphs with Many-Body Interactions

- **Graphs**: Nodes (atoms) and edges (bonds).
- **Many-Body Interactions**: New graph architecture with three-body interactions.
- **Graph Representation**: 
  $$
  \mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{X}, [\bm{M}, \bm{u}])
  $$
  - $\mathcal{V}$: Atom information.
  - $\mathcal{E}$: Bond information.
  - $\mathcal{X}$: Atomic coordinates.
  - $\bm{M}$: Lattice matrix.
  - $\bm{u}$: Global state information.
- **Bond Interaction**: Considering neighbors:
  $$
  \tilde{\bm{e}}_{ij} = \sum_{\substack{k_1, k_2, \ldots, k_{n-2} \in \mathcal{N}_i/j \\ k_1 \neq k_2 \neq \ldots \neq k_{n-2}}} \phi_n(\bm{e}_{ij}, \bm{r}_{ij}, \bm{v}_{j}, \bm{r}_{ik_1}, \bm{r}_{ik_2}, \ldots, \bm{r}_{ik_{n-2}}, \bm{v}_{k_1}, \bm{v}_{k_2}, \ldots, \bm{v}_{k_{n-2}})
  $$

---

### Three-Body Interaction

- **Angular Interactions**:
  $$
  \tilde{\bm{e}}_{ij} = \sum_k j_l(z_{ln}\frac{r_{ik}}{r_c})Y_l^0(\theta_{jik}) \odot\sigma(\bm{W}_v\bm{v}_k + \bm{b}_v)f_c(r_{ij})f_c(r_{ik})
  $$
  - $j_l$: Spherical Bessel function.
  - $z_{ln}$: Roots.
  - $r_c$: Cutoff radius.
  - $Y_l^0$: Spherical harmonics.
  - $f_c(r)$: Cutoff function.
  - $\sigma$: Sigmoid activation.

---

### Graph Convolution Updates

- **Bond Update**:
  $$
  \bm{e}_{ij}^\prime = \bm{e}_{ij} + \phi_e(\bm{v}_i \oplus \bm{v}_j \oplus \bm{e}_{ij} \oplus \bm{u}) \bm{W}_e^0 \bm{e}_{ij}^0
  $$
- **Atom Update**:
  $$
  \bm{v}_i^\prime = \bm{v}_i + \sum_j \phi_e^\prime(\bm{v}_i \oplus \bm{v}_j \oplus \bm{e}_{ij}^\prime \oplus \bm{u}) \bm{W}_e^{0\prime} \bm{e}_{ij}^0
  $$
- **State Update**:
  $$
  \bm{u}^\prime = g(\bm{W}^u_2 g(\bm{W}^u_1 (\frac{1}{N_v} \sum_{i=1}^{N_v} \bm{v}_i \oplus \bm{u}) + \bm{b}^u_1) + \bm{b}^u_2)
  $$

---

## M3GNet Interatomic Potential

### Model Training

- **Dataset**: Materials Project, 187,687 ionic steps.
- **Training Data**: Energies ($E$), Forces ($\bm{f}$), Stresses ($\bm{\sigma}$).
  $$
  \bm{f} = -\frac{\partial E}{\partial \bm{x}}, \quad \bm{\sigma} = \frac{1}{V} \frac{\partial E}{\partial \bm{\epsilon}}
  $$
  - $\bm{x}$: Atomic coordinates.
  - $V$: Volume.
  - $\bm{\epsilon}$: Strain.

---

**Benchmark on IAP Datasets**
- **Comparison**: Classical IAPs (EAM, MEAM), ML-IAPs (NNP, MTP).
- **M3GNet Performance**: Outperforms classical potentials, comparable to ML-IAPs.

**Universal IAP for the Periodic Table**
- **Data Coverage**: 89 elements.
- **Model Accuracy**: Low errors for energy, force, and stress predictions.
  - Energy error: 0.035 eV/atom.
  - Force error: 0.072 eV/Å.
  - Stress error: 0.41 GPa.

---

**Dataset and Methods**
- **Materials Project Data**: 187,687 ionic steps of 62,783 compounds.
- **Training Data**: 
  $$
  E, \quad \bm{f} = -\frac{\partial E}{\partial \bm{x}}, \quad \bm{\sigma} = \frac{1}{V} \frac{\partial E}{\partial \bm{\epsilon}}
  $$
- **Dataset Distribution**:
  - Energy Range: [-28.731, 49.575] eV/atom.
  - Force Range: [-2570.567, 2552.991] eV/Å.
  - Stress Range: [-5474.488, 1397.567] GPa.

- **Model Performance**
  - **Training and Validation**: 90% training, 5% validation, 5% test split.
  - **Errors**: Energy: 0.035 eV/atom, Force: 0.072 eV/Å, Stress: 0.41 GPa.

---

## New Materials Discovery

- **Hypothetical Materials Generation**:
  - Generated 31.7 million hypothetical materials via substitutions.
- **Relaxation and Validation**:
  - Relaxed structures using M3GNet.
  - Identified 1.8 million potentially stable materials.
  - Verified stability of top candidates using DFT.
- **Results**:
  - High agreement between M3GNet and DFT energies.
  - Phonon calculations for dynamic stability.

---

## Appendix: Methods

### Neural Network Definition

- One layer of the perceptron model:
  $$
  \mathcal{L}_g^k : x \mapsto g(\bm{W}_k x + \bm{b}_k)
  $$

- $K$-layer multi-layer perceptron (MLP):
  $$
  \xi_K(x) = (\mathcal{L}_g^{K} \circ \mathcal{L}_g^{K-1} \circ \ldots \circ \mathcal{L}_g^1)(x)
  $$

- $K$-layer gated MLP:
  $$
  \phi_K(x) = \left( (\mathcal{L}_g^{K} \circ \mathcal{L}_g^{K-1} \circ \ldots \circ \mathcal{L}_g^1)(x) \right) \odot \left( (\mathcal{L}_\sigma^{K} \circ \mathcal{L}_g^{K-1} \circ \ldots \circ \mathcal{L}_g^1)(x) \right)
  $$
  - $\mathcal{L}_\sigma^K(x)$ uses sigmoid function $\sigma(x)$.
  - $\odot$ denotes element-wise product.

---

### Model Architecture and Hyperparams

- **Graph Construction**:
  - Radial cutoff: 5 Å.
  - Three-body interactions cutoff: 4 Å.

- **Graph Featurizer**:
  - Atomic number embeddings: dimension 64.
  - Bond distances expanded using smooth basis functions:
    $$
    h_m(r) = \frac{1}{\sqrt{d_m}}\left[f_m(r) + \sqrt{\frac{e_m}{d_{m-1}}}h_{m-1}(r)\right]
    $$
    - Parameters:
      $$
      d_m = 1 - \frac{e_m}{d_{m-1}}, \quad e_m = \frac{m^2(m+2)^2}{4(m+1)^4 + 1}
      $$
      $$
      f_m(r) = (-1)^m \frac{\sqrt{2}\pi}{r_c^{3/2}} \frac{(m+1)(m+2)}{\sqrt{(m+1)^2 + (m+2)^2}} \left( \text{sinc}\left(r \frac{(m+1)\pi}{r_c}\right) + \text{sinc} \left(r \frac{(m+2)\pi}{r_c}\right) \right)
      $$
      $$
      \text{sinc}(x) = \frac{\sin{x}}{x}
      $$

---

### Model Architecture and Hyperparams

- **Main Blocks**:
  - Three three-body information exchange and graph convolutions.

- **Prediction of Extensive Properties**:
  $$
  p_{\rm{ext}} = \sum_i \phi_3(\bm{v}_i)
  $$
  - Gated MLP: [64, 64, 1] neurons.

- **Prediction of Intensive Properties**:
  $$
  p_{\rm{int}} = \xi_3\left( \sum_i w_i \xi_2(\bm{v}_i) \oplus \bm{u} \right)
  $$
  - Weights:
    $$
    w_i = \frac{\xi_3^\prime(\bm{v}_i)}{\sum_i \xi_3^\prime(\bm{v}_i)}
    $$

