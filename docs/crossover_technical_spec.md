# Neural Network Crossover with CCA Neuron Alignment — Technical Specification

**Version:** 1.0.0  
**Date:** 2026-03-31  
**Author:** Julian Bolivar  
**Status:** Specification  
**Scope:** Generic neural network crossover for GA-evolved populations  

---

## 1. Overview

This document specifies a crossover operator for neural networks within a Genetic Algorithm (GA). The operator combines weights from two parent networks to produce a child network, preserving functional specialization of neurons through CCA (Canonical Correlation Analysis) alignment.

The crossover operates **layer by layer**, aligning neurons functionally before blending weights. This resolves the *permutation problem* — the fact that two networks can compute the same function with neurons in different order.

### 1.1 Scope

This specification applies to any feedforward or iterative network with:

- One or more hidden layers of configurable size
- An input layer with fixed semantic positions (no permutation)
- An output layer with fixed semantic positions (no permutation)
- Optionally: residual skip connections with ReZero scaling
- Optionally: skip projections for heterogeneous layer sizes

The operator is generic over the linear algebra backend (`LinAlg` trait) and works with any activation function.

### 1.2 Design Principles

- **Encapsulation**: crossover is a method on the network struct, not an external function
- **Zero overhead during inference**: activation capture is the caller's responsibility
- **CCA always on CPU**: SVD is negligible (<1s per generation) and avoids polluting the LinAlg trait
- **Deterministic**: given the same seed, crossover produces identical results

---

## 2. The Permutation Problem

Two neural networks can compute exactly the same function with their hidden neurons in different order. A naive crossover that averages weights by position destroys specialization:

```
Parent A neurons: [crash_detector, ramp_detector, flat_detector]
Parent B neurons: [flat_detector, crash_detector, ramp_detector]

Naive average at position 0: (crash + flat) / 2 = meaningless hybrid
Naive average at position 1: (ramp + crash) / 2 = meaningless hybrid
Naive average at position 2: (flat + ramp) / 2 = meaningless hybrid
```

All three child neurons become identical blends that detect nothing well. The specialized detectors from both parents are destroyed.

### 2.1 Why This Matters More With Larger Layers

For a layer of N neurons, there are N! possible orderings. The probability that two independently trained networks have the same neuron order is 1/N!:

| Layer size | Permutations | Probability of same order |
|-----------|-------------|--------------------------|
| 4 | 24 | 4.2% |
| 8 | 40,320 | 0.0025% |
| 16 | 2.09 × 10^13 | ~0% |
| 32 | 2.63 × 10^35 | ~0% |

For any practical layer size, naive crossover is guaranteed to fail.

---

## 3. Solution: CCA Neuron Alignment

### 3.1 Concept

CCA is a "functional MRI" of the network. We show both parents the same set of inputs (reference batch) and observe how each neuron responds — its "functional fingerprint". Two neurons that detect the same signal (e.g., crashes) produce similar activation patterns regardless of their position in the layer. CCA finds these correspondences automatically.

### 3.2 Activation Capture

Activations are captured **during fitness evaluation** — not as a separate step. When the GA evaluates an individual's fitness, it processes inputs through the network normally. The hidden layer activations that are already computed are simply copied to a cache. Additional compute cost: **zero**.

```
For each individual in population:
  During normal fitness evaluation:
    for each input in reference_batch:
      result = network.infer(input)
      cache.record(result.hidden_states)  // copy already-computed values
    // continue fitness evaluation normally

  Cache lifetime: one generation — discarded at start of next
```

#### 3.2.1 ActivationCache Structure

```rust
pub struct ActivationCache<L: LinAlg = CpuLinAlg> {
    /// activations[layer_idx][batch_sample_idx] = neuron activations
    layers: Vec<Vec<L::Vector>>,
}
```

- `layers.len()` = number of hidden layers
- `layers[L].len()` = batch size (number of reference inputs)
- Each `L::Vector` contains the neuron activations for that layer and sample

The cache uses the same `LinAlg` backend as the network. All CCA operations (covariance, SVD) go through `L::` trait methods.

#### 3.2.2 Reference Batch Selection

The reference batch should contain diverse inputs that exercise different neuron specializations. Recommended composition:

- ~10% inputs that produce high surprise (extreme events)
- ~10% inputs with unusual characteristics (outliers)
- ~80% uniformly sampled from validation set

The **same batch** is used for **all individuals** in a generation. This ensures CCA comparisons are consistent across the population.

Batch size recommendation: 500 samples is sufficient for layers up to ~256 neurons.

### 3.3 CCA Algorithm

For each hidden layer L, given activation matrices from parent A and parent B:

```
Input:
  A_activations: [batch_size × n_neurons_a]  — rows are samples, cols are neurons
  B_activations: [batch_size × n_neurons_b]

PHASE 1 — Standardize columns
  For each column (neuron): subtract mean, divide by std dev.
  This ensures CCA compares functional patterns, not magnitude.

PHASE 2 — Compute cross-covariance
  C_a  = A^T × A / (batch_size - 1)     [n_a × n_a] auto-covariance of A
  C_b  = B^T × B / (batch_size - 1)     [n_b × n_b] auto-covariance of B
  C_ab = A^T × B / (batch_size - 1)     [n_a × n_b] cross-covariance

PHASE 3 — SVD on normalized cross-covariance
  M = C_a^(-1/2) × C_ab × C_b^(-1/2)
  U, S, V^T = SVD(M)

  U columns: canonical directions in A's neuron space
  V columns: canonical directions in B's neuron space
  S diagonal: correlation strength of each pair (sigma_1 >= sigma_2 >= ...)

PHASE 4 — Extract neuron-to-neuron permutation (greedy matching)
  For each canonical pair (u_k, v_k) ordered by sigma_k descending:
    neuron_a = argmax(|u_k|)  — neuron of A with largest coefficient
    neuron_b = argmax(|v_k|)  — neuron of B with largest coefficient
    If both neuron_a and neuron_b are unmatched:
      match[neuron_b] = neuron_a
      mark both as matched
    Skip if either already matched (enforce unique pairing)

  Assign remaining unmatched neurons of B to remaining positions of A
  (arbitrary order — these are weakly correlated neurons)

Output:
  permutation: Vec<usize>  — permutation[i] = position in A that neuron i of B maps to
```

### 3.4 SVD Implementation

SVD is a method on the `LinAlg` trait:

```rust
pub trait LinAlg {
    // ... existing methods ...

    /// Singular Value Decomposition: M = U × S × V^T
    fn svd(m: &Self::Matrix) -> (Self::Matrix, Self::Vector, Self::Matrix);
}
```

The `CpuLinAlg` implementation uses Jacobi eigenvalue rotations — no external dependencies. A GPU backend would use cuBLAS, wgpu compute shaders, or equivalent. The crossover calls `L::svd()` and is agnostic to the execution backend.

All CCA operations (covariance computation, matrix inverse square root, SVD, permutation extraction) use `LinAlg` trait methods, maintaining full backend genericity.

### 3.5 Inverse Square Root of Covariance Matrix

Computing `C^(-1/2)` for the normalized cross-covariance:

```
C = V × D × V^T          (eigendecomposition)
C^(-1/2) = V × D^(-1/2) × V^T

where D^(-1/2)_ii = 1 / sqrt(D_ii) for D_ii > epsilon
                   = 0               for D_ii <= epsilon  (regularization)
```

The epsilon threshold (e.g., 1e-10) prevents division by zero for degenerate dimensions.

---

## 4. Crossover Operator

### 4.1 Inputs

```
parent_a:       network with weights W_a, the reference (typically higher fitness)
parent_b:       network with weights W_b, to be aligned to A
cache_a:        ActivationCache for parent A on the reference batch
cache_b:        ActivationCache for parent B on the reference batch
alpha:          blending weight = fitness_a / (fitness_a + fitness_b)
child_config:   topology of the child (may differ from both parents due to mutation)
rng:            random number generator for Xavier initialization
```

### 4.2 Layer-by-Layer Crossover

The crossover processes each layer independently, applying the appropriate strategy based on layer type and dimension compatibility.

```
For each layer index i in child:

  IF input layer (i == 0):
    → Positional crossover (Section 4.3)

  ELSE IF hidden layer (0 < i < num_layers - 1):
    → CCA-aligned crossover (Section 4.4)

  ELSE IF output layer (i == num_layers - 1):
    → Positional crossover or Xavier (Section 4.5)
```

### 4.3 Input Layer Crossover (No CCA)

The input layer has fixed semantic positions (e.g., board positions, latent blocks). Neurons are not permutable. Direct positional crossover:

```
For each weight position (row, col):
  child_weight = alpha * parent_a_weight + (1 - alpha) * parent_b_weight
```

If the child's input layer has different dimensions than the parents (e.g., different input_size), Xavier initialization is used for the entire layer.

### 4.4 Hidden Layer Crossover (CCA-Aligned)

This is the core of the crossover. For each hidden layer:

#### Step 1: CCA Alignment

```
permutation = cca_neuron_alignment(cache_a[layer], cache_b[layer])
B_aligned = apply_permutation(parent_b.weights[layer], permutation)
B_bias_aligned = apply_permutation(parent_b.bias[layer], permutation)
```

After alignment, position i in both parents corresponds to the same functional role.

#### Step 2: Dimensional Crossover

Four cases based on the neuron counts:

```
n_a = parent A neuron count for this layer
n_b = parent B neuron count for this layer
n_h = child neuron count for this layer (from child_config)
n_min = min(n_a, n_b)
n_max = max(n_a, n_b)
```

**Case 1 — All equal (n_a == n_b == n_h):**
Most frequent case. Pure CCA-aligned blending.

```
For i in [0..n_h):
  child.weights[i] = alpha * A.weights[i] + (1 - alpha) * B_aligned.weights[i]
  child.bias[i]    = alpha * A.bias[i]    + (1 - alpha) * B_aligned.bias[i]
```

**Case 2 — Child smaller (n_h < n_min):**
Child is smaller than both parents. Only the blending zone exists.

```
For i in [0..n_h):
  child.weights[i] = alpha * A.weights[i] + (1 - alpha) * B_aligned.weights[i]
  child.bias[i]    = alpha * A.bias[i]    + (1 - alpha) * B_aligned.bias[i]

Excess neurons from both parents are discarded.
```

**Case 3 — Parents differ (n_a != n_b, n_h between n_min and n_max):**
Blending zone + copy zone from the larger parent.

```
For i in [0..n_min):                          // Blending zone
  child.weights[i] = alpha * A.weights[i] + (1 - alpha) * B_aligned.weights[i]
  child.bias[i]    = alpha * A.bias[i]    + (1 - alpha) * B_aligned.bias[i]

larger = parent with more neurons in this layer
For i in [n_min..min(n_max, n_h)):            // Copy zone
  child.weights[i] = larger.weights[i]
  child.bias[i]    = larger.bias[i]
```

**Case 4 — Child larger (n_h > n_max):**
Blending + copy + Xavier zone for new neurons.

```
For i in [0..n_min):                          // Blending zone
  child.weights[i] = alpha * A.weights[i] + (1 - alpha) * B_aligned.weights[i]
  child.bias[i]    = alpha * A.bias[i]    + (1 - alpha) * B_aligned.bias[i]

larger = parent with more neurons in this layer
For i in [n_min..n_max):                      // Copy zone
  child.weights[i] = larger.weights[i]
  child.bias[i]    = larger.bias[i]

For i in [n_max..n_h):                        // Xavier zone
  child.weights[i] = xavier_init(fan_in, fan_out, rng)
  child.bias[i]    = 0.0
```

### 4.5 Output Layer Crossover

The output layer has fixed semantic positions (action logits, value estimate). No CCA alignment needed.

```
IF child output input_size == parents output input_size:
  // Dimensions compatible — positional crossover
  For each weight position:
    child_weight = alpha * A_weight + (1 - alpha) * B_weight

ELSE:
  // input_size changed (hidden topology mutation added/removed layer or changed size)
  // Parent output weights are incompatible — reinitialize
  child.output_layer = xavier_init(new_input_size, output_size, rng)
```

### 4.6 Layer Count Mismatch

When the child has a different number of hidden layers than the parents (due to topology mutation):

```
For each hidden layer index i in child:
  IF both parents have layer i:
    → CCA-aligned crossover (Section 4.4)
  ELSE IF only one parent has layer i:
    → Copy weights from that parent (adjusted for child dimensions)
  ELSE (neither parent has layer i):
    → Xavier initialization for entire layer

Output layer:
  IF input_size changed (due to different last hidden layer size):
    → Xavier initialization
  ELSE:
    → Positional crossover
```

This case should be **rare** — well-configured GAs apply aggressive topology mutations with low probability.

---

## 5. Auxiliary Components Crossover

### 5.1 ReZero Scaling Factors

ReZero alphas are scalar values — no permutation problem. Simple weighted average:

```
For each skip layer i:
  IF both parents have rezero_alpha[i]:
    child.rezero_alpha[i] = alpha * A.rezero_alpha[i] + (1 - alpha) * B.rezero_alpha[i]
  ELSE IF only one parent has it:
    child.rezero_alpha[i] = that parent's value
  ELSE:
    child.rezero_alpha[i] = child_config.rezero_init  // default
```

### 5.2 Skip Projections

Skip projections are matrices that transform between layers of different sizes. They follow the same CCA pattern but are **not aligned by CCA** — they are positional matrices tied to specific layer pairs.

```
For each skip projection index i:
  IF both parents have projection[i] with same dimensions:
    child.projection[i] = alpha * A.projection[i] + (1 - alpha) * B.projection[i]
  ELSE IF dimensions differ or only one has it:
    child.projection[i] = xavier_init(target_size, source_size, rng)
```

---

## 6. Integration with Existing Architecture

### 6.1 Method Signatures

```rust
// In pc_actor_critic.rs
pub struct ActivationCache<L: LinAlg = CpuLinAlg> {
    layers: Vec<Vec<L::Vector>>,
}

impl<L: LinAlg> ActivationCache<L> {
    /// Create cache for a network with given number of hidden layers.
    pub fn new(num_hidden_layers: usize) -> Self;

    /// Record hidden states from an inference result.
    pub fn record(&mut self, infer_result: &InferResult<L>);

    /// Number of samples recorded.
    pub fn batch_size(&self) -> usize;
}

// In matrix.rs
/// Compute CCA neuron alignment between two sets of activations.
/// Returns permutation: permutation[i] = position in A that neuron i of B maps to.
/// All operations (covariance, SVD) use L:: trait methods.
pub fn cca_neuron_alignment<L: LinAlg>(
    activations_a: &[L::Vector],
    activations_b: &[L::Vector],
) -> Vec<usize>;

// In pc_actor.rs
impl<L: LinAlg> PcActor<L> {
    /// Produce a child by crossing self (parent A) with other (parent B).
    ///
    /// Parent A is the reference for CCA alignment. Alpha controls the
    /// blending weight of parent A (typically fitness_a / total_fitness).
    pub fn crossover(
        &self,
        other: &Self,
        cache_self: &ActivationCache<L>,
        cache_other: &ActivationCache<L>,
        alpha: f64,
        child_config: PcActorConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, PcError>;
}

// In mlp_critic.rs
impl<L: LinAlg> MlpCritic<L> {
    /// Produce a child by crossing self (parent A) with other (parent B).
    pub fn crossover(
        &self,
        other: &Self,
        cache_self: &ActivationCache<L>,
        cache_other: &ActivationCache<L>,
        alpha: f64,
        child_config: MlpCriticConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, PcError>;
}

// In pc_actor_critic.rs
impl<L: LinAlg> PcActorCritic<L> {
    /// Produce a child agent by crossing self with other.
    /// Delegates to PcActor::crossover and MlpCritic::crossover.
    pub fn crossover(
        &self,
        other: &Self,
        cache_self: &ActivationCache<L>,
        cache_other: &ActivationCache<L>,
        alpha: f64,
        child_config: PcActorCriticConfig,
        seed: u64,
    ) -> Result<Self, PcError>;
}
```

### 6.2 Usage Example

```rust
// --- Phase 1: Fitness evaluation with activation capture ---
let reference_batch: Vec<Vec<f64>> = select_reference_batch(&validation_set, 500);
let mut population_caches: HashMap<usize, ActivationCache> = HashMap::new();

for (idx, individual) in population.iter().enumerate() {
    let mut cache = ActivationCache::new(individual.agent.actor.config.hidden_layers.len());

    for input in &reference_batch {
        let (action, infer_result) = individual.agent.act(input, &valid_actions, Training);
        cache.record(&infer_result);
        // ... accumulate fitness metrics normally
    }

    population_caches.insert(idx, cache);
}

// --- Phase 2: Selection + Crossover ---
let (parent_a, parent_b) = tournament_select(&population);
let alpha = parent_a.fitness / (parent_a.fitness + parent_b.fitness);
let child_config = mutate_genome(&parent_a.config, &parent_b.config, &mut rng);

let child = parent_a.agent.crossover(
    &parent_b.agent,
    &population_caches[&parent_a.idx],
    &population_caches[&parent_b.idx],
    alpha,
    child_config,
    42,  // seed for Xavier init
)?;

// --- Phase 3: Train child with PPO/backprop to refine ---
train(&mut child, &training_data);
```

### 6.3 What Does NOT Change

| Component | Change | Reason |
|-----------|--------|--------|
| `LinAlg` trait | Add `svd()` method | SVD is a linear algebra operation — belongs in the trait |
| `InferResult` | None | Already returns `hidden_states` |
| `infer()` | None | Activation capture is caller's responsibility |
| `act()`, `learn()` | None | Training loop unchanged |
| Normal inference | Zero overhead | No flags, no switches |

---

## 7. Computational Cost

```
Activation capture (Phase 1):
  Additional compute: ZERO — activations are already computed during inference.
  Only cost: clone hidden_states vectors (~256 floats × 500 samples = 128KB per layer).
  Memory: ~1-5 MB per individual for typical architectures.

CCA alignment (Phase 2) per layer:
  One SVD of matrix [min(n_a, n_b) × min(n_a, n_b)]
  Complexity: O(n^3) where n = min(n_a, n_b)
  Wall time: microseconds for n <= 256

Total CCA per generation (population 20, ~10 crossovers):
  ~60-80 SVDs of 32-256 dim matrices = <1 second total

Comparison with training (minutes to hours per generation): negligible
```

---

## 8. Edge Cases and Error Handling

### 8.1 Degenerate CCA (Zero Variance Neurons)

If a neuron has constant activation across the batch (zero variance), standardization produces NaN. Detection and handling:

```
During standardization:
  IF std_dev(neuron_i) < epsilon:
    Mark neuron_i as "dead" — exclude from CCA
    In the permutation: assign dead neurons to remaining unmatched positions
```

### 8.2 Identical Parents

If both parents have identical weights (e.g., from cloning + different mutations):

```
CCA produces identity permutation (all correlations ≈ 1.0)
Crossover reduces to: child = alpha * A + (1-alpha) * A = A
Result: child ≈ parent A (as expected)
```

### 8.3 Incompatible Batch Sizes

Both parents must have been evaluated on the same reference batch.

```
IF cache_a.batch_size() != cache_b.batch_size():
  Return PcError::DimensionMismatch
```

### 8.4 Empty Hidden Layers

If the network has zero hidden layers (input → output only):

```
No CCA to perform — only output layer crossover (positional)
ActivationCache has zero layers — record() is a no-op
crossover() only crosses the output layer
```

---

## 9. Reference

Method based on: Uriot & Izzo (2020). "Safe Crossover of Neural Networks Through Neuron Alignment." GECCO '20, European Space Agency.

Adapted with:
- Weighted average by fitness (instead of simple average from the original paper)
- Four dimension cases for evolvable topologies (original assumes fixed topology)
- Layer count mismatch handling for topology-evolving GAs
- Extension to residual networks (ReZero scaling, skip projections)
- Separation of CCA computation (CPU) from network backend (generic LinAlg)
