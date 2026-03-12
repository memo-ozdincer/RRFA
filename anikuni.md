# Crash Course: The Physics & ML Behind Your Matter Lab Project
## Everything Eddie Kim might ask you about — with the math

---

# 1. THE POTENTIAL ENERGY SURFACE (PES) — THE LANDSCAPE YOU'RE NAVIGATING

A molecule with N atoms lives in 3N-dimensional configuration space (the positions of all atoms). The potential energy E(R) defines a hypersurface over this space. The key features of this surface are:

**Minima** — stable configurations. All eigenvalues of the Hessian are positive. These are your reactants and products.

**First-order saddle points (transition states)** — exactly ONE negative Hessian eigenvalue. The transition state is the highest point along the minimum energy path (MEP) connecting two minima. It represents the energy barrier a reaction must overcome.

**The Hessian** is the 3N × 3N matrix of second derivatives:

    H_{IJ}^{αβ} = ∂²E / (∂R_I^α ∂R_J^β)

where I,J index atoms and α,β ∈ {x,y,z}. This matrix tells you the curvature of the PES at any point. At a minimum, all curvatures are positive (bowl-shaped in every direction). At a transition state, one curvature is negative (saddle-shaped along the reaction coordinate) and all others are positive.

**Why transition states matter:** The Eyring equation gives the reaction rate k ∝ exp(−ΔG‡/k_BT), where ΔG‡ is the free energy barrier at the transition state. Finding transition states tells you whether a reaction is feasible and how fast it goes. This has direct industrial relevance for drug design, catalyst optimization, and chemical process engineering.

---

# 2. FINDING TRANSITION STATES — THE METHODS

## 2.1 Double-Ended Methods (Know Both Endpoints)

These methods know the reactant and product, and try to find the path between them.

**Nudged Elastic Band (NEB):** Creates a chain of "images" (molecular configurations) connected by spring forces between reactant and product. Each image feels: the real force perpendicular to the path (pushing it toward the MEP) and spring forces along the path (keeping images evenly spaced). A "climbing image" variant inverts the force along the band for the highest-energy image, driving it toward the true saddle point.

**Growing String Method (GSM):** Instead of starting with a full chain, GSM grows two path fragments from each endpoint that meet in the middle. At each step, a new node is added along the path tangent and relaxed perpendicular to it:

    F_⊥ = F − (t̂ · F)t̂

where F = −∇E and t̂ is the local tangent. Once fragments merge, the highest-energy node is a TS estimate. This is what your ReactBench workflow uses for initial guesses.

## 2.2 Single-Ended Methods (Climb From a Starting Point)

**Eigenvector-following / Mode-following:** Given a Hessian at your current point, you identify the eigenvector with the smallest (most negative) eigenvalue — the "reaction coordinate direction." You step uphill along this eigenvector while minimizing in all other directions. This is why the Hessian's smallest eigenvalue/eigenvector pair is so critical.

**Gentlest Ascent Dynamics (GAD):** A continuous version of eigenvector-following. You simultaneously evolve the position x and a direction vector v:

    dx/dt = −(I − 2vvᵀ)∇E(x)
    dv/dt = −(I − vvᵀ)(∇²E(x))v

The first equation moves uphill along v and downhill perpendicular to v. The second equation tracks the lowest eigenvector of the Hessian. This is what your saddle-point search algorithm builds on.

**Rational Function Optimization (RFO):** A second-order optimizer that uses a Padé approximation to the energy:

    E(R + Δx) − E(R) ≈ (gᵀΔx + ½ΔxᵀHΔx) / (1 + |Δx|²)

The key property: choosing different eigenvectors of the augmented eigenvalue problem gives either minimum-seeking or saddle-point-seeking updates. This is what the HIP paper uses for TS refinement after GSM.

## 2.3 Your Contribution — Why It Matters

Your saddle-point search algorithm achieves 100% convergence from molecules displaced by 2Å of noise, where existing methods (Sella, iHiSD, GAD) fail below 65%. The key was likely leveraging analytical second derivatives (Hessians) rather than approximations. Your benchmarking of 8 algorithms across 120K+ simulations showed that analytical second derivatives outperform NN-predicted ones (94% vs 75%, 33× faster).

---

# 3. THE HESSIAN — WHY IT'S EXPENSIVE AND HOW HIP FIXES IT

## 3.1 Computing Hessians from DFT

In Kohn-Sham DFT, computing the Hessian requires solving the Coupled Perturbed Kohn-Sham (CPKS) equations. The key difficulty: for forces, the density response ∂C/∂x cancels out due to stationarity (the Hellmann-Feynman theorem), so forces are cheap. For Hessians, you MUST compute the density response, which requires solving a large linear system scaling as O(N⁵) in compute and O(N⁴) in memory.

Finite differences avoid the CPKS complexity but need 6N gradient evaluations (central differences), each scaling O(N⁴), giving the same O(N⁵) overall but with more noise and a larger prefactor.

## 3.2 Hessians from Neural Networks (Auto-Differentiation)

If you have a trained MLIP (Machine Learning Interatomic Potential) that predicts energy E(R), you can get the Hessian by differentiating twice via automatic differentiation (AD). But: you can only compute one Hessian-vector product HvB at a time. To get the full 3N × 3N Hessian, you need 3N such products (one per column), giving O(N²) cost from an O(N) model. Memory also scales O(N²), and batching makes it even worse because AD sees a batch of molecules as one bigger system.

## 3.3 HIP: Direct Hessian Prediction (The Paper From Your Lab)

The HIP paper (Burger, Thiede, Aspuru-Guzik et al.) is from YOUR lab at Matter. The key insight: you can construct SE(3)-equivariant, symmetric Hessians directly from the irreducible representation features already computed during message passing in a GNN. No auto-differentiation needed.

**How it works:**
1. Run the EquiformerV2 backbone (message passing GNN) to get node features
2. Add a Hessian-specific readout layer that computes atom-pair features (messages without aggregation)
3. Project these features down to irreps up to l=2 (scalar + vector + rank-2 tensor)
4. Use the Clebsch-Gordan tensor product expansion to convert from the coupled basis back to Cartesian 3×3 blocks:

    H'_{IJ,m₁,m₂} = Σ_{l,m} C^{l,m}_{1,m₁,1,m₂} h̃_{IJ,l,m}

5. Symmetrize: H = H' + H'ᵀ

**Why l=2 is enough:** A 3×3 matrix (two l=1 vectors coupled) decomposes into irreps l=0 (trace/scalar), l=1 (antisymmetric part), and l=2 (symmetric traceless part). So l=2 is exactly what you need — nothing higher contributes.

**Results:** 10-70× faster than AD, 2× lower Hessian MAE, 92% accuracy at classifying extrema (vs 75% for AD EquiformerV2). Training is also orders of magnitude cheaper.

**Why direct prediction is OK (non-conservative Hessians):** For MD, non-conservative forces cause energy drift that blows up over time. But for transition state search, geometry optimization, and ZPE — the errors are bounded because you're not integrating over long time horizons. BFGS updates (the standard practice) aren't conservative either.

---

# 4. ECKART PROJECTION — REMOVING RIGID-BODY MODES

A molecule with N atoms has 3N degrees of freedom. But 3 correspond to translation and 3 (or 2 for linear molecules) correspond to rotation. These "rigid body modes" have zero frequency — the energy doesn't change if you translate or rotate the whole molecule.

The Eckart projection removes these 5-6 redundant modes from the Hessian so you only analyze true vibrational modes. Steps:

**1. Mass-weight the Hessian:**

    H̃ = M^{-1/2} H M^{-1/2}

where M is the diagonal mass matrix (each atom's mass repeated 3 times).

**2. Build the translational vectors** (3 vectors, each 3N-dimensional):

    t_α^{(i)} = √m_i · ê_α,    α ∈ {x,y,z}

These represent uniform translation of all atoms.

**3. Build the rotational vectors** (3 vectors) using the inertia tensor:

Compute the inertia tensor I = Σᵢ mᵢ(rᵢ·rᵢ I₃ − rᵢrᵢᵀ), diagonalize it to get principal axes {î₁, î₂, î₃}, then:

    r_k^{(i)} = √m_i · (î_k × rᵢ),    k = 1,2,3

**4. Build the projector:** Collect these 5-6 vectors, orthonormalize them (QR or SVD), and project them out:

    P = I_{3N} − Σₐ uₐuₐᵀ

Then the Eckart-projected Hessian is:

    H̃_Eckart = P H̃ Pᵀ

**Why Eddie might ask about this:** Eckart projection is essential for frequency analysis (determining if you're at a TS or minimum), ZPE calculation, and vibrational spectroscopy. If you skip it, the 5-6 near-zero eigenvalues from rigid-body modes contaminate your spectrum and can lead to misclassification of stationary points.

---

# 5. GRAPH NEURAL NETWORKS FOR MOLECULES — THE BACKBONE

## 5.1 The Idea

Represent a molecule as a graph: atoms are nodes, edges connect atoms within a cutoff radius. Each node carries features that get updated through "message passing" — information flows between connected atoms over multiple layers, gradually building up each atom's awareness of its chemical environment.

## 5.2 Invariance vs Equivariance

**Invariant features** (l=0, scalars): don't change under rotation. Examples: energy, interatomic distances, bond angles.

**Equivariant features** (l≥1): transform predictably under rotation. l=1 features are vectors (forces, dipoles). l=2 features are rank-2 tensors (Hessian blocks, quadrupoles).

Under a rotation Q, features of degree l transform via Wigner D-matrices:

    h_{i,l,m}(Q{rₖ}) = Σ_{m'} D^{(l)}_{mm'}(Q) h_{i,l,m'}({rₖ})

This is the generalization of "vectors rotate with the rotation matrix" to arbitrary angular momentum.

## 5.3 Message Passing with Tensor Products

Messages between atoms couple node features with edge features (spherical harmonics of the displacement direction) via Clebsch-Gordan tensor products:

    v_{IJ,l₃} = Σ_{l₁,l₂} w_{l₁,l₂,l₃}(||r_{IJ}||) (f_{l₁}(h_I, h_J) ⊗ Y_{l₂}(r̂_{IJ}))_{l₃}

where Y_{l₂} are spherical harmonics encoding the direction between atoms, w is a learned radial function encoding the distance, and ⊗ is the CG tensor product.

**Spherical harmonics** Y_l^m(θ,φ) form a complete orthonormal basis for functions on the sphere. They're the angular part of atomic orbital wavefunctions — the s, p, d, f orbitals correspond to l=0,1,2,3.

**Clebsch-Gordan coefficients** C^{l₃,m₃}_{l₁,m₁,l₂,m₂} tell you how to combine two angular momenta. Coupling l₁ and l₂ produces outputs with l₃ ∈ {|l₁−l₂|, ..., l₁+l₂}. This is why l=2 features require l_max ≥ 2 in your backbone.

## 5.4 EquiformerV2 (The Specific Architecture Used)

EquiformerV2 is the backbone for both the HIP paper and your project. Key features:
- Transformer-style architecture with graph attention (not just sum aggregation)
- Uses SO(2) convolutions instead of full SO(3) tensor products for efficiency (reduces cost from O(L⁶) to O(L³))
- Attention re-normalization, separable S² activation, and separable layer normalization to leverage higher-degree representations
- State-of-the-art on OC20 (catalysis dataset) and QM9 (small molecules)

## 5.5 E(3)-Equivariance vs SE(3)

**SE(3)** = special Euclidean group = rotations + translations (no reflections)
**E(3)** = full Euclidean group = rotations + translations + reflections (parity)

For molecules, parity matters (chiral molecules). E(3)-equivariant networks assign parity labels (even/odd) to each feature, which determines how they transform under coordinate inversion. In e3nn notation: "1x0e" means one scalar with even parity, "1x1o" means one pseudovector with odd parity, etc.

---

# 6. ADJOINT SAMPLING — DIFFUSION FOR GENERATIVE CHEMISTRY

This is the foundation for the second part of your Matter Lab project: using diffusion-based methods with adjoint sampling to generate molecular configurations (conformers, transition states) from Boltzmann distributions.

## 6.1 The Problem

You want to sample from the Boltzmann distribution:

    μ(x) = exp(−E(x)/τ) / Z

where E(x) is an energy function (from DFT or a neural network potential) and Z is the unknown normalization constant. You don't have training data — you only have the energy function. Standard diffusion models need data; this doesn't.

## 6.2 The Stochastic Optimal Control Framing

Instead of training on data, you learn a controlled SDE:

    dX_t = σ(t) u(X_t, t) dt + σ(t) dB_t,    X₀ = 0

where u is a learnable control (neural network) and B_t is Brownian motion. You want to find u such that the distribution at time t=1 matches the target Boltzmann distribution: p^u₁ = μ.

The optimal control is the one that deviates least from the uncontrolled process (Schrödinger bridge):

    p*(X) = p^base(X|X₁) μ(X₁)

The training objective is a KL divergence between your controlled process and this optimal one:

    L_SOC(u) = E_{p^u}[∫₀¹ ½||u(X_t,t)||² dt + log(p^base₁(X₁)/μ(X₁))]

This decomposes into: a "control cost" (penalizing large drifts) plus a "terminal cost" (how far your endpoint distribution is from the target).

## 6.3 Why Adjoint Sampling Is Special

Previous methods (Path Integral Sampler, DIS, etc.) needed to simulate the full SDE for EVERY gradient update — extremely expensive when energy evaluations are costly (like DFT).

**Adjoint Sampling's key trick:** Use the "Reciprocal Projection" — given a sample X₁ from your current policy, you can reconstruct the entire trajectory distribution without re-simulating, by using the base process posterior conditioned on X₁. This means you can take MANY gradient updates per energy evaluation, dramatically improving efficiency.

The mathematical foundation is the Reciprocal Adjoint Matching (RAM) objective, which decomposes the stochastic control problem into local-in-time matching conditions that can be solved without simulating the SDE.

## 6.4 Incorporating Symmetries

For molecules, you need:
- **E(3) equivariance:** The learned control u must be equivariant to rotations, translations, and reflections
- **Permutation invariance:** Swapping identical atoms shouldn't change the energy
- **Periodic boundary conditions:** For torsion angle representations, angles wrap around

Adjoint Sampling handles all of these by building the control network from equivariant GNN architectures (like the ones described in Section 5).

## 6.5 Your Application: Generating Transition States and Reaction Pathways

The vision for your project: instead of running expensive saddle-point searches from scratch every time, train a diffusion-based generative model that can directly sample transition state geometries and reaction pathways from the Boltzmann distribution, conditioned on reactant/product information. This uses:

1. **Transition state data** (from your saddle-point search algorithm + DFT) to define training energies
2. **E(3)-equivariant GNNs** as the score/control network
3. **Adjoint sampling** to efficiently train without needing enormous amounts of TS data — you just need the energy function
4. **Stochastic optimal control** to learn to generate samples from the Boltzmann distribution over molecular configurations, with learned symmetry constraints

This is targeting NeurIPS 2026 — it would let you generate chemically valid reaction pathways without explicit simulation, potentially orders of magnitude faster than current TS search workflows.

---

# 7. WHAT EDDIE WILL LIKELY ASK & HOW TO ANSWER

**"Walk me through your saddle-point search. What was the key insight?"**
→ Existing vector-following methods (GAD, Sella's eigenvector-following) struggle when initialized far from the true TS because the Hessian eigenvectors at the starting point don't align well with the reaction coordinate. My algorithm achieves 100% convergence even from 2Å displacement because [your specific insight — likely related to how you handle the Hessian information during the walk, or a better step-control mechanism]. I benchmarked 8 algorithms across 120K+ simulations and found that analytical second derivatives dramatically outperform NN-predicted ones (94% vs 75%).

**"What's an Eckart projection and why do you need it?"**
→ The Hessian of an N-atom molecule is 3N×3N, but 6 eigenvalues correspond to rigid-body translation and rotation (they're theoretically zero). The Eckart projection constructs mass-weighted translational and rotational basis vectors from the center of mass and inertia tensor principal axes, then projects them out. Without this, near-zero rigid-body eigenvalues contaminate the vibrational spectrum and you can't reliably classify whether a stationary point is a TS (one imaginary frequency) or a minimum (no imaginary frequencies).

**"How does the GNN predict the Hessian directly?"**
→ The HIP approach from our lab constructs each 3×3 Hessian sub-block H_{IJ} from the irreducible representation features computed during message passing. A rank-2 Cartesian tensor decomposes into irreps l=0,1,2, so you only need features up to l=2. The CG tensor product expansion reassembles the 3×3 block from these irreps while guaranteeing SE(3) equivariance by construction. The result is 10-70× faster than auto-differentiation and more accurate.

**"How would you generate good training data for this from scratch?"**
→ For the saddle-point search: I used the Transition1x dataset (reactions computed at the ωB97X/6-31G* level of DFT), which covers diverse regions of the PES including non-equilibrium geometries. Good data means covering the PES broadly — not just equilibria but displaced configurations, transition regions, and reaction pathways. For quality assurance, I validate against DFT Hessians and use frequency analysis to confirm that predicted TSs have exactly one imaginary frequency.

**"How do you know your diffusion model is generating chemically valid structures?"**
→ Three levels of validation: (1) The energy of generated configurations under the target energy function should match the Boltzmann distribution; (2) Generated transition states should have exactly one negative Hessian eigenvalue when verified by DFT; (3) IRC (Intrinsic Reaction Coordinate) calculations from the generated TS should connect back to the correct reactant and product minima. The E(3)-equivariance built into the network ensures rotational/translational invariance, and the stochastic optimal control formulation provides theoretical guarantees on convergence to the target distribution.