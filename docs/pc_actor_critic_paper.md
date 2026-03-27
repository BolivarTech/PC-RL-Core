# Predictive Coding Actor-Critic: Architecture, Experiments, and Lessons Learned

## Abstract

This document presents a novel reinforcement learning architecture that integrates predictive coding (PC) inference into an actor-critic framework. The actor replaces the standard feedforward pass with an iterative top-down/bottom-up inference loop that minimizes prediction error across layers before selecting actions. A hybrid learning rule blending backpropagation with local PC prediction errors is introduced as a regularization mechanism. The architecture is evaluated on Tic-Tac-Toe against minimax opponents with curriculum learning. Key results: (1) PC inference adds +1 minimax depth level over equivalent MLP, (2) a 99/1 backprop/PC-error blend (lambda=0.99) is statistically significant (p=0.034, N=35) for breaking depth ceilings, (3) bounded activations are mandatory for PC loop stability. Implementation is in pure Rust with ~1,900 parameters total.

---

## 1. Architecture

### 1.1 Overview

The system consists of two networks:

- **PC Actor**: Selects actions using predictive coding inference
- **MLP Critic**: Estimates state value V(s) for advantage computation

```
Input (board state, 9 dims)
    |
    v
[Hidden Layer, 27 neurons, Tanh]  <-- PC inference loop (top-down / bottom-up)
    |                                  iterates until convergence or max_steps
    v
[Output Layer, 9 neurons, Linear]  --> Softmax --> Action
    |
    +-- Latent Concat (27 dims) --+
                                  |
         [Board (9)] ++ [Latent (27)] = 36 dims
                                  |
                                  v
                   [Critic Hidden, 36 neurons, Tanh] --> V(s)
```

**Total parameters**: ~1,900 (549 actor + 1,369 critic)

### 1.2 PC Inference Loop

Instead of a single feedforward pass, the actor performs iterative inference:

1. **Forward pass**: Standard bottom-up computation initializes hidden states
2. **Top-down prediction**: Each layer generates predictions of the layer below using transposed weights: `prediction = W^T * state_above`
3. **Prediction error**: `error = prediction - hidden_state`
4. **State update**: `hidden_state += alpha * error` (gradient descent on free energy)
5. **Convergence check**: Repeat steps 2-4 until RMS error < tolerance or max_steps reached
6. **Output recompute**: Final output from converged hidden states

The inference loop runs identically during both training and play. The surprise score (RMS prediction error) is computed but does not affect inference -- it is used for learning rate modulation.

### 1.3 Latent Concatenation

All hidden layer activations are concatenated into a latent vector and fed to the critic alongside the raw board state. This gives the critic access to the actor's internal representations, not just the raw observations. The critic input is `[board_state(9) ++ latent(27)] = 36 dimensions`.

### 1.4 Hybrid Learning Rule (local_lambda)

Hidden layer weight updates blend two gradient signals:

```
delta = lambda * backprop_gradient + (1 - lambda) * pc_prediction_error
```

- **Backprop gradient** (lambda component): Carries reward information from the policy gradient through the network. Subject to vanishing gradients through saturating activations.
- **PC prediction error** (1-lambda component): Local signal measuring representational consistency between adjacent layers. Not subject to vanishing gradients but carries no reward information.
- **Output layer**: Always uses pure backprop (no PC error available for the output).

Default: `lambda = 0.99` (99% backprop, 1% PC error as regularizer).

### 1.5 Training Pipeline

- **Algorithm**: REINFORCE with baseline (advantage actor-critic)
- **Opponent**: Minimax with configurable search depth
- **Curriculum**: Starts at depth 1, advances when non-loss rate (win + draw) > 95% over 1000-game window. Metrics reset on advancement.
- **Action selection**: Softmax sampling during training, argmax during play

---

## 2. Experiments

### 2.1 PC Inference vs MLP Baseline

The first question: does the PC inference loop add value over a standard feedforward pass?

| Architecture | Alpha | Max Depth | Notes |
|-------------|-------|-----------|-------|
| MLP (alpha=0) | 0 | 6 | Pure feedforward, no PC |
| PC (alpha=0.03) | 0.03 | 8 | +2 depth levels |
| PC (alpha=0.01) | 0.01 | 8 | Same ceiling, different alpha |

**Finding**: PC inference consistently adds +1 to +2 minimax depth levels over equivalent MLP architecture. The iterative convergence produces richer internal representations that improve policy quality. The value comes from inference, not from using PC errors for learning.

### 2.2 Network Topology

| Topology | Max Depth | Notes |
|----------|-----------|-------|
| 18 hidden | 7 | Underfitting |
| **27 hidden** | **8** | **Sweet spot** |
| 32 hidden | 8 | No improvement, same ceiling |
| 27 -> 18 hidden | 7 | Worse: vanishing gradients through double tanh |
| 2 x 36 hidden | 7 | Worse: same vanishing gradient problem |

**Finding**: Single hidden layer of 27 neurons is optimal. Adding layers degrades performance due to vanishing gradients through cascaded tanh. The gradient must pass through `1 - fx^2` at each tanh layer; with two layers, the signal reaching the first layer is attenuated by the product of both derivatives.

### 2.3 Activation Functions

The PC inference loop imposes unique constraints on activation functions:

| Activation | Max Depth | Failure Mode |
|------------|-----------|-------------|
| **Tanh** | **8** | None -- bounded output stabilizes PC loop |
| ReLU | 4 | **Dying neurons**: PC update `h += alpha * error` pushes activations negative; ReLU derivative = 0 permanently |
| ELU | 6 | **Unbounded explosion**: Positive activations grow without limit; PC loop creates positive feedback; policy collapses |

**Finding**: PC inference loops require bounded activations. The iterative update `h[i] += alpha * error` accumulates across iterations. Unbounded activations (ReLU, ELU) allow hidden states to grow without limit, creating unstable feedback loops. Tanh's natural bound to [-1, 1] provides implicit regularization that is essential for PC convergence.

**Implication for other PC projects**: Always use bounded activations (tanh, sigmoid) in layers that participate in PC inference loops. If unbounded activations are desired, consider clamping hidden states after each PC iteration.

### 2.4 Pure Local PC Learning

Testing the hypothesis from Millidge et al. 2022 that PC prediction errors can replace backpropagation:

| Lambda | Max Depth | Notes |
|--------|-----------|-------|
| 1.0 (backprop) | 8 | Baseline |
| 0.0 (local PC) | 7 | Worse: no reward signal in hidden layers |
| 0.0, 2 layers | 6 | Much worse: double tanh + no reward |

**Finding**: Pure local PC learning is inferior to backpropagation for RL. The prediction errors measure representational consistency (how well layer A predicts layer B), not policy quality (which actions maximize reward). Hidden layers learn coherent internal representations but not ones oriented toward the task objective.

**Implication for other PC projects**: The Millidge et al. 2022 approximation may work well for supervised learning (where the output error directly reflects the objective), but in RL the reward signal must propagate through the network. Local errors alone are insufficient.

### 2.5 Hybrid Lambda Sweep

Testing whether a small amount of PC error can regularize backprop:

#### Fixed seeds (seed=42, seed=123)

| Lambda | seed=42 | seed=123 |
|--------|---------|----------|
| 1.0 | depth 8 | depth 7 |
| 0.99 | **depth 9** | depth 7 |
| 0.975 | depth 7 | -- |
| 0.75 | depth 7 | -- |
| 0.50 | depth 7 | -- |
| 0.25 | depth 7 | -- |
| 0.0 | depth 7 | -- |

#### Statistical validation (N=35 random seeds)

| Lambda | Mean Depth | StdDev | D>=8 | D=9 | p-value |
|--------|-----------|--------|------|-----|---------|
| 0.95 | 7.00 | 1.19 | 23% | 11% | 0.564 |
| 0.96 | 6.66 | 1.33 | 14% | 9% | 0.072 |
| 0.97 | 6.94 | 1.16 | 14% | 11% | 0.413 |
| 0.98 | 7.00 | 0.87 | 11% | 9% | 0.490 |
| **0.99** | **7.57** | **0.81** | **37%** | **20%** | **0.034\*** |
| 1.00 | 7.14 | 0.85 | 26% | 9% | baseline |

**Finding**: Lambda=0.99 is the only statistically significant improvement (p < 0.05). It increases mean depth by +0.43, doubles the rate of reaching depth 9 (20% vs 9%), and never drops below depth 7 (min=7 vs min=6 for baseline). All other lambda values perform equal or worse.

### 2.6 Critical Hyperparameters

| Parameter | Optimal | Effect of Wrong Value |
|-----------|---------|----------------------|
| output_activation | linear | Tanh bounds logits to [-1,1], softmax becomes uniform, no learning |
| entropy_coeff | 0.0 | Any entropy destabilizes learned defensive play |
| alpha | 0.03 | 0 = no PC benefit; too high = inference instability |
| lr_weights | 0.005 | Higher (0.01) = faster but lower ceiling; lower = too slow |
| local_lambda | 0.99 | < 0.975 = PC error overwhelms reward signal |

---

## 3. Theoretical Analysis

### 3.1 Why PC Inference Helps

The PC inference loop performs variational free energy minimization on the hidden states before action selection. This has several effects:

1. **Richer representations**: The converged hidden states encode not just the bottom-up input but also top-down contextual expectations. This is analogous to running a recurrent network for multiple steps on the same input.

2. **Implicit ensemble**: Each inference iteration produces a slightly different hidden state. The converged state represents a consensus that is more robust than a single feedforward pass.

3. **Denoising**: The top-down predictions act as a prior that regularizes the hidden states, suppressing noise from the bottom-up pathway.

4. **Compute-parameter tradeoff**: The PC loop multiplies inference compute by ~5x (max_steps=5) but achieves performance that would require more parameters in a feedforward network. This is efficient for small networks.

### 3.2 Why Lambda=0.99 Works

The 1% PC prediction error component acts as a structured micro-regularizer:

1. **Backprop gradient (99%)**: Carries the reward signal from the policy gradient. Tells hidden layers which representations lead to good actions. Subject to vanishing gradients through tanh.

2. **PC prediction error (1%)**: Nudges weights toward representational consistency between layers. Acts as a subtle perturbation of the optimization surface with meaningful direction (not random noise).

3. **Escape mechanism**: The perturbation helps the optimizer escape saddle points and shallow local minima that pure backprop cannot cross. The prediction errors point in a direction that improves internal consistency, which correlates with (but is not identical to) improved policy quality.

4. **Non-monotonic effect**: Too much PC error (lambda < 0.975) overwhelms the reward signal. The prediction errors point in a fundamentally different direction than the policy gradient; blending too much of the wrong direction is worse than none. The transition between "helpful" and "harmful" is extremely narrow (1.5% difference in lambda = 2 depth levels).

### 3.3 Seed Dependency and Loss Landscape

Different random seeds create different initial weight configurations, placing the optimizer in different basins of attraction. Each basin has its own local topology:

- Some basins contain deep minima (depth 9) accessible only with the right perturbation
- Other basins have shallower optima (depth 7) regardless of lambda
- Lambda=0.99 increases the probability of escaping to deeper basins but cannot guarantee it

This is consistent with the statistical results: lambda=0.99 doesn't always reach depth 9 (only 20% of seeds), but it reaches depth 9 more than twice as often as the baseline (9%).

---

## 4. Lessons for Other PC Projects

### 4.1 Activation Functions

**Use bounded activations in PC layers.** The iterative update `h += alpha * error` accumulates across inference steps. Unbounded activations (ReLU, ELU, GELU) can cause hidden states to diverge. Tanh and sigmoid are safe. If unbounded activations are required, clamp hidden states after each PC iteration:

```
h[i] = clamp(h[i] + alpha * error, -bound, bound)
```

### 4.2 PC Inference vs PC Learning

**PC inference is valuable; PC learning (alone) is not -- at least for RL.** The inference loop produces better representations (free energy minimization), but using prediction errors as the sole learning signal loses the task objective. The prediction errors optimize for internal consistency, not for external reward.

For supervised learning, where the output error directly reflects the objective, pure PC learning may be more competitive (as Millidge et al. 2022 suggest). For RL, the reward signal must reach the hidden layers through backpropagation.

### 4.3 The Hybrid Sweet Spot

**A tiny fraction of PC error as regularizer can be beneficial, but the optimal amount is extremely narrow.** In this architecture, only lambda=0.99 (1% PC error) improved over pure backprop. Lambda=0.975 (2.5% PC error) already degraded performance. This suggests:

- Grid search over lambda with coarse steps will likely miss the sweet spot
- The optimal lambda probably depends on the specific architecture, task, and other hyperparameters
- Evolutionary optimization (genetic algorithms) co-evolving lambda with other hyperparameters is the recommended approach for finding the optimal blend

### 4.4 Network Depth

**Prefer single hidden layers in PC architectures with tanh.** Multi-layer networks suffer compounded vanishing gradients: each tanh layer multiplies the gradient by `1 - fx^2`, which is < 1 for any non-zero activation. Two layers of tanh with activations at 0.7 reduce the gradient to `(1-0.49) * (1-0.49) = 0.26`. The PC inference loop does not help with this during learning because the backprop path goes through the standard network structure, not through the inference iterations.

If deeper networks are needed, consider:
- Skip connections (residual)
- Layer normalization
- Separate learning rates per layer
- Or increase the hybrid lambda closer to 0 for deeper layers (gradient-free local learning)

### 4.5 Latent Concatenation for Critics

**Feeding converged PC hidden states to the critic provides a richer state representation.** The latent concatenation (`board_state ++ hidden_activations`) gives the critic access to the actor's internal model of the situation, not just raw observations. This is particularly valuable when the PC inference loop has converged to a stable representation that captures relational structure in the input.

### 4.6 Surprise Score as Diagnostic

**The RMS prediction error (surprise) is a useful diagnostic even if not used for learning.** High surprise indicates the network's internal model is inconsistent with the input, which correlates with novel or difficult states. This can be used for:

- Prioritized experience replay (focus on high-surprise transitions)
- Curriculum scheduling (advance when surprise is consistently low)
- Anomaly detection (flag states with unexpectedly high surprise)

### 4.7 Compute-Parameter Tradeoff

**PC networks trade compute for parameters.** A 27-neuron PC network with 5 inference iterations achieves performance comparable to larger feedforward networks. Each inference step costs roughly one forward pass, so the compute is ~5x higher, but the parameter count stays at ~500 for the actor. This makes PC architectures attractive for:

- Embedded systems with limited memory but available compute cycles
- Problems where overfitting is a concern (small parameter count acts as regularizer)
- Situations where inference latency is acceptable but model size must be minimized

---

## 5. Implementation Notes

### 5.1 Technology

- **Language**: Rust (pure, no ML framework dependencies)
- **Library crate**: `pc_core` v0.2.0 (published on crates.io)
- **Dependencies**: serde, serde_json, rand, chrono (core); toml, clap, ctrlc (binary)
- **Tests**: 268 unit tests, TDD methodology throughout
- **Repository**: https://github.com/BolivarTech/PC-TicTacToe

### 5.2 Key Design Decisions

1. **Synchronous vs in-place PC updates**: Synchronous mode (snapshot all states, then update) is the default. In-place mode (updates immediately visible to subsequent layers) converges differently and produces slightly different results.

2. **Weight and gradient clipping**: Both set to 5.0. Prevents exploding weights from large policy gradient updates while allowing sufficient dynamic range.

3. **Deterministic training**: With a fixed seed, training is fully deterministic (identical results across runs). This enables reproducible experimentation and precise ablation studies.

4. **Curriculum reset on advancement**: Metrics (win/loss/draw window) reset when minimax depth increases. Without reset, high win rates from easier opponents carry over and cause premature advancement at higher depths.

---

## 6. References

1. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.

2. Millidge, B., Seth, A., & Buckley, C. L. (2022). Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs. *Neural Computation*, 34(6), 1329-1368.

3. Whittington, J. C., & Bogacz, R. (2017). An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. *Neural Computation*, 29(5), 1229-1262.

4. Salvatori, T., Song, Y., Hong, Y., Sha, L., Frieder, S., Xu, Z., Bogacz, R., & Lukasiewicz, T. (2022). Associative Memories via Predictive Coding. *NeurIPS 2022*.

5. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

*Author: Julian Bolivar -- BolivarTech*
*Date: March 2026*
*Repository: https://github.com/BolivarTech/PC-TicTacToe*
*Crate: https://crates.io/crates/pc_core*
