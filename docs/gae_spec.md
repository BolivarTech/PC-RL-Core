# GAE(λ) — Generalized Advantage Estimation via Output-Level Eligibility Traces

**Version:** 1.0
**Date:** 2026-04-12
**Author:** Julian Bolivar
**Crate:** `pc-rl-core`
**Status:** Implemented on branch `continuos_learning`, merged to `main`

---

## 1. Overview

GAE(λ) adds output-level eligibility traces to the actor's policy gradient,
smoothly interpolating between TD(0) (λ=0) and Monte Carlo (λ=1) return
estimation. Each step computes a TD(0) error and applies it scaled by the
accumulated trace:

```
trace = γλ × trace + ∇log π(a|s)
delta = td_error × trace
```

This provides temporal credit assignment across steps without buffering
full transitions (unlike TD(n)).

### 1.1 Comparison with TD(n)

| Aspect | TD(n) | GAE(λ) |
|---|---|---|
| Memory | O(n × network_size) per agent | O(output_size) per agent |
| Stale inference | Uses InferResult from n steps ago | Always uses current InferResult |
| Credit assignment | Fixed n-step horizon | Weighted average of all horizons |
| Backprop fidelity | Exact (cached per-layer activations) | Approximate (output-level trace) |
| Best for | Short episodes with known optimal n | Long/variable episodes, unknown optimal n |

### 1.2 When to Use Which

| Scenario | Recommendation |
|---|---|
| Short episodes, fixed length (TicTacToe 5-9) | TD(n) where n ≈ episode_length |
| Long episodes, variable length (Go, Chess) | GAE(λ=0.95) |
| Unknown environment, exploration | GAE(λ=0.95) |
| Deep networks (5+ layers) with short episodes | TD(n) — output-level trace approximation degrades with depth |
| 1-layer network | GAE(λ) — output-level trace is exact for single layer |
| Memory-constrained | GAE(λ) — O(output_size) vs O(n × network_size) |
| Reproducibility in ablations | TD(n) — explicit, simple horizon |

---

## 2. Configuration

### 2.1 Config Field

```rust
/// GAE lambda for eligibility traces. Mutually exclusive with `td_steps > 0`.
/// - `Some(λ)`: GAE(λ) with output-level eligibility traces.
/// - `None`: Disabled — standard TD(0) or TD(n) via `td_steps`.
///
/// Default: `None` (backward compatible). Recommended: `Some(0.95)` for short episodes.
#[serde(default = "default_gae_lambda")]
pub gae_lambda: Option<f64>,
```

### 2.2 Mutual Exclusion

GAE and TD(n) are mutually exclusive — rejected at construction:

```rust
if config.gae_lambda.is_some() && config.td_steps > 0 {
    return Err(PcError::ConfigValidation(...));
}
```

### 2.3 Valid Combinations

| `gae_lambda` | `td_steps` | Mode |
|---|---|---|
| `None` | `0` | TD(0) — default, backward compatible |
| `None` | `N >= 2` | TD(N) — n-step buffer |
| `Some(λ)` | `0` | GAE(λ) — eligibility traces |
| `Some(λ)` | `N >= 2` | **Rejected** — mutually exclusive |

### 2.4 TOML Usage

```toml
[agent]
gae_lambda = 0.95    # GAE enabled (serde: Some(0.95))
# OR
td_steps = 5         # TD(5) enabled (gae_lambda absent = None)
```

---

## 3. Architecture

### 3.1 Data Structures

```rust
/// Output-level eligibility trace (action-space sized).
/// Not serialized — transient mid-episode state.
actor_trace: Vec<f64>,  // len = output_size when enabled, 0 when disabled
```

Initialized via `gae_trace_len()` helper in all 3 constructors
(`new`, `crossover`, `from_parts`).

### 3.2 Trace Lifecycle

```
Episode start:  trace = [0, 0, ..., 0]
Each step:      trace = γλ × trace + grad_direction
                trace = clamp(trace, -GRAD_CLIP, GRAD_CLIP)
                delta = td_error × trace
                → backprop delta through all layers
Terminal:       trace = [0, 0, ..., 0]
reset_step():   trace = [0, 0, ..., 0]
crossover():    child trace = [0, 0, ..., 0]
Save/Load:      trace NOT serialized → fresh zeros on load
```

### 3.3 GAE Learning Path

Inside `learn_continuous_inner`, when `gae_lambda` is set:

1. Compute TD(0) error: `td_error = target - V(s)` (same as standard)
2. NaN guard: if `!td_error.is_finite()` → return 0.0 (before trace is touched)
3. Compute gradient direction: `grad = π - one_hot(action)` (without td_error)
4. Decay trace: `trace *= γλ`
5. Accumulate: `trace += grad_direction`
6. Clip: `trace = clamp(trace, -GRAD_CLIP, GRAD_CLIP)`
7. Scale: `delta = td_error × trace`
8. Apply entropy per-step (not accumulated in trace)
9. Call shared `apply_actor_update_and_bookkeeping(delta, ...)`
10. Early return (standard path skipped)

When `gae_lambda` is `None`, the standard TD(0)/TD(n) path runs unchanged.

### 3.4 DRY Helper

`apply_actor_update_and_bookkeeping()` is shared between GAE and standard paths.
It handles:
- Effective surprise scale + actor decay
- EWC/Fisher accumulation + correction (if ewc_lambda > 0)
- Actor weight update via `update_weights`
- M3b layer error EMA update
- Surprise buffer push + TD error buffer push

---

## 4. Interaction with PC Inference

PC inference and GAE operate in **sequential phases** — no interference:

```
Phase 1: PC inference loop (free energy minimization)
    → Converged y_conv, hidden_states, prediction_errors, surprise_score

Phase 2: GAE trace update + weight update
    → Uses converged activations from Phase 1
```

### 4.1 Complementary Mechanisms

| Mechanism | Modulates | Source | Phase |
|---|---|---|---|
| PC inference | Internal representation quality | Free energy minimization | 1 |
| GAE trace | Gradient direction (temporal credit) | History of policy gradients | 2 |
| PC surprise | Learning rate magnitude | Residual prediction error | 2 |
| local_lambda | Backprop vs PC error blend | Config (per-layer) | 2 |

### 4.2 local_lambda as Per-Layer Correction

With `local_lambda=0.99`, each hidden layer receives:

```
effective_delta = 0.99 × (trace-propagated backprop signal)
                + 0.01 × (local PC prediction error)
```

The 1% PC error component acts as a **per-layer anchor** that partially
compensates for the output-level trace approximation error. Where the
propagated trace signal is inaccurate (historical gradient through current
activations), the local PC error injects correct layer-specific information.

### 4.3 Refined Gradients

The PC loop refines hidden states before gradient computation. The
`grad_direction` accumulated in the trace is therefore computed from
**optimized representations** (post-convergence), not raw feedforward
activations. This improves the quality of the gradient directions
accumulated in the trace.

---

## 5. Output-Level Trace Approximation

### 5.1 What It Is

Full per-weight eligibility traces maintain one trace value per parameter:
`z_ij = γλ × z_ij + ∂log(π)/∂w_ij`. This requires separating gradient
computation from application in `Layer::backward()`.

Output-level traces maintain one trace value per output neuron:
`z_j = γλ × z_j + ∂log(π)/∂y_j`. The trace-scaled delta is then
backpropagated through the network using `update_weights`.

### 5.2 Exactness by Depth

| Network Depth | Output-Level Trace | Reason |
|---|---|---|
| 1 hidden layer | **Exact** | dW = delta ⊗ input — trace scaling on delta is equivalent |
| 2-3 layers | **Good approximation** | Backprop propagation error is bounded |
| 5+ layers | **Degrades** | Each layer multiplication introduces activation mismatch |

### 5.3 Why TD(n) Can Be Better for Deep Networks

TD(n) stores the complete `InferResult` (hidden_states, prediction_errors)
from the original step. When learning n steps later, backprop uses the
**exact activations** from when the action was taken.

GAE backprops the trace-scaled delta through **current activations**, but
the trace contains gradient components from previous steps. The outer
product `dW = grad ⊗ input` in each layer pairs current inputs with
historical gradient directions — a mismatch that compounds with depth.

### 5.4 Experimental Evidence (PC-TicTacToe)

N=35 seeds, 200k episodes, 3-layer [12,12,8] network, alternating P1/P2:

| Method | Mean Depth | D=9 | Notes |
|---|---|---|---|
| TD(5) | 6.71 | 8.6% | Best step_masked method |
| GAE(1.0) | 6.23 | 2.9% | MC-equivalent but output-trace approximation |
| TD(0) | 6.43 | 0% | Insufficient signal quality |
| GAE(0.95) | 6.00 | 0% | Trace decay adds noise for short episodes |

**Key finding:** GAE(1.0) does NOT match TD(5) despite both being
theoretically Monte Carlo for 5-9 step episodes. The 0.48 depth gap
is attributable to the output-level trace approximation error in the
3-layer network.

**Offensive bias:** GAE(1.0) produced 25.7% high-win + 37.1% no-draw
agents — the trace biases learning toward early moves (offense) because
backprop through current activations distorts the historical gradient
signal for earlier steps.

---

## 6. Trace Clipping

Traces are clipped to `[-GRAD_CLIP, GRAD_CLIP]` (GRAD_CLIP=5.0) after
each accumulation step:

```rust
for v in &mut self.actor_trace {
    *v = v.clamp(-GRAD_CLIP, GRAD_CLIP);
}
```

### 6.1 Why Clipping Is Needed

The geometric series bound for the trace is `1/(1 - γλ)`:
- γ=0.99, λ=0.95 → max ≈ 16.8
- γ=0.99, λ=1.0 → max ≈ 100
- γ=1.0, λ=1.0 → unbounded

Clipping at 5.0 is conservative but consistent with the gradient clipping
already applied inside `Layer::backward()`.

---

## 7. NaN Safety

The NaN guard at the top of `learn_continuous_inner` fires BEFORE any
trace modification:

```rust
if !td_error.is_finite() {
    return 0.0;  // trace unchanged, weights unchanged
}
```

A NaN reward → NaN td_error → guard fires → trace and weights untouched.
Verified by test with trace snapshot before/after.

---

## 8. GAE(0) = TD(0) Equivalence

When λ=0:
- `γλ = 0` → trace decays to zero each step
- `trace = 0 × trace + grad = grad` (just current gradient)
- `delta = td_error × grad` (identical to standard TD(0))

This equivalence holds **only when `entropy_coeff = 0.0`**. With
entropy > 0, the GAE path applies entropy to trace-scaled delta while
the standard path applies entropy after per-element td_error scaling —
producing slightly different results.

The default `entropy_coeff = 0.0` (recommended, as entropy destabilizes
learned defensive play) ensures equivalence.

---

## 9. Serialization

| Component | Serialized | Notes |
|---|---|---|
| `gae_lambda` config field | Yes | `#[serde(default)]` → None for legacy |
| `actor_trace` | **No** | Transient — reconstructed as zeros on load |

The trace is episode-specific state. Loading a saved agent starts a fresh
episode, so the trace must be zero.

---

## 10. Testing

### 10.1 Tests (14 total)

| Test | Purpose |
|---|---|
| `test_gae_lambda_default_is_none` | Default None (backward compat) |
| `test_gae_lambda_and_td_steps_mutually_exclusive` | Validation |
| `test_gae_lambda_none_allows_td_steps` | Mutual exclusion one-way |
| `test_gae_lambda_out_of_range_rejected` | Range [0.0, 1.0] |
| `test_gae_trace_field_exists_and_correct_size` | output_size, starts zero |
| `test_gae_trace_empty_when_disabled` | Empty when None |
| `test_gae_trace_accumulates_across_steps` | Non-zero after learning |
| `test_gae_trace_resets_on_terminal` | Zero after terminal |
| `test_gae_trace_resets_on_reset_step` | Zero after reset |
| `test_gae_produces_different_weights_than_td0` | GAE(0.95) ≠ TD(0) |
| `test_gae_lambda_zero_matches_td0` | GAE(0) = TD(0) when entropy=0 |
| `test_gae_nan_reward_safe` | Trace+weights unchanged after NaN |
| `test_gae_serialization_config` | Config round-trip |
| `test_gae_trace_not_serialized` | Trace transient on load |

---

## 11. Design Decisions

### 11.1 Why Output-Level Traces (Not Per-Weight)

Full per-weight traces require separating gradient computation from
application in `Layer::backward()`. This is a significant refactor
(new `compute_gradient` method + accumulate + apply separately).
Output-level traces capture the main GAE benefit (temporal credit
assignment) by accumulating the gradient DIRECTION at the output level
and letting backprop propagate it. For 1-3 layer networks, this is a
close approximation.

### 11.2 Why No Critic Traces

The critic uses TD(0) with `update_with_decay`. Adding per-weight critic
traces would require separating gradient computation in
`MlpCritic::update()`. The marginal benefit for short episodes (5-30
steps) doesn't justify the complexity. The actor is the bottleneck.

### 11.3 Why Default None (Not Some(0.95))

Backward compatibility. Existing configs and saved agents load as TD(0)
without behavioral changes. Users must explicitly opt into GAE. The
`Some(0.95)` recommendation is documented in the field docstring.

### 11.4 Why Entropy Outside Trace

Entropy regularization is a constant exploration force, not subject to
temporal credit assignment. Accumulating entropy in the trace would
amplify its effect proportionally to episode length, which is undesirable.
The per-step application ensures consistent regularization regardless of
trace magnitude.

