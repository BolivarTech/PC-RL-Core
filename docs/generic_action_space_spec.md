# Generic Action Space: Raw Output API

## Status

**Draft** -- Future work. Not part of continuous learning (v2.1.0). To be
implemented after continuous learning is validated.

## Motivation

The current architecture assumes a **discrete action space**: the actor outputs
logits, the library applies softmax, samples or argmax, and returns a single
`usize` action index. This assumption is embedded throughout the learning pipeline.

This works for Tic-Tac-Toe and similar discrete environments, but limits the
library as a general-purpose RL framework. Real-world environments require richer
action representations:

| Environment | Output interpretation | Current support |
|-------------|---------------------|-----------------|
| Board game (TTT) | 9 neurons = 9 board positions, pick one | Yes (discrete) |
| Robot locomotion | 4 neurons = [forward, back, right, left] torques | No |
| Continuous control | N neurons = N-dimensional continuous action | No |
| Multi-discrete | Groups of neurons, one selection per group | No |

### The Diagonal Problem

Consider a robot with 4 output neurons: `[forward, back, right, left]`. If the
actor produces `[0.8, 0.1, 0.7, 0.05]`, the robot should move **diagonally
forward-right** -- both `forward` and `right` are active simultaneously. The
current architecture forces a single choice (argmax = `forward`, index 0),
discarding the multi-dimensional nature of the output.

The consumer who designed the network topology knows what the outputs mean. The
library should expose the raw output and let the consumer interpret it.

## Current Architecture (v2.0.0)

### Action Selection Pipeline

```
actor.infer(state) -> InferResult { y_conv, latent_concat, surprise, ... }
                          |
                          v
              softmax_masked(y_conv, valid_actions)
                          |
                          v
              sample/argmax -> action: usize
                          |
                          v
              learn_continuous(state, infer, action, ...)
                          |
                          v
              delta[action] -= 1.0   <-- discrete gradient
              delta *= td_error
                          |
                          v
              update_weights_hybrid(delta, infer, surprise_scale)
```

### Where `usize` is Embedded

| Location | Usage | File |
|----------|-------|------|
| `act()` return | `(usize, InferResult<L>)` | `pc_actor_critic.rs:420` |
| `select_action()` return | `usize` | `pc_actor.rs:777` |
| `learn_continuous()` param | `action: usize` | `pc_actor_critic.rs:545` |
| Policy gradient | `delta[action] -= 1.0` | `pc_actor_critic.rs:597` |
| `softmax_masked()` | Normalizes over discrete indices | `matrix.rs:234` |
| `argmax_masked()` | Returns single index | `matrix.rs:274` |
| `sample_from_probs()` | Samples single index | `matrix.rs:331` |
| step() (CL spec) | Returns `usize` | `continuous_learning_spec.md` |

### Policy Gradient Formula (Discrete)

The current REINFORCE-style gradient for discrete actions:

```
pi = softmax(y_conv)                    // probability distribution
delta = pi                              // for all actions
delta[action] -= 1.0                    // one-hot subtraction
delta *= td_error                       // scale by advantage
// Equivalent to: -nabla log(pi[action]) * advantage
```

This formula is **fundamentally discrete**: it requires knowing which single action
was selected (`action: usize`) to construct the one-hot gradient.

## Proposed Architecture

### Core Principle

**`step()` returns the raw output activations. The consumer decides what they
mean.** The library does not apply softmax, does not sample, does not argmax. The
output is the direct result of the actor's forward pass (post `output_activation`,
which is already configurable -- linear, tanh, sigmoid, etc.).

```rust
let output: Vec<f64> = agent.step(&state, reward, terminal);
// Consumer interprets output based on their network design:
//   Discrete: apply own softmax + sampling
//   Continuous: use directly as motor commands
//   Multi-discrete: split into groups, argmax per group
```

### Training Diagnostics

Diagnostic information (td_error, loss, plasticity state) is **not bundled** in
the return value. The consumer queries it explicitly via dedicated methods when
needed:

```rust
// Only call these if you need the data -- no overhead per step otherwise
let td = agent.last_td_error();                // Option<f64>
let loss = agent.last_loss();                   // Option<f64>
let actor_state = agent.actor_plasticity();     // PlasticityState
let critic_state = agent.critic_plasticity();   // PlasticityState
```

### Policy Gradient: Discrete vs Continuous

The architectural change centers on how the policy gradient is computed. Two modes
are needed:

#### Mode 1: Discrete (current behavior, adapted)

For environments where the output represents a probability distribution over
discrete choices. The consumer must inform the library which action was selected,
because the gradient requires it:

```rust
// Consumer applies softmax + sampling externally
let output = agent.step(&state, reward, terminal);
let probs = softmax(&output);
let action = sample(probs);
agent.set_action_taken(action);  // informs library for gradient computation
execute(action);
```

Internally, the gradient remains REINFORCE:
```
delta = pi
delta[action] -= 1.0
delta *= advantage
```

#### Mode 2: Continuous (new)

For environments where the output IS the action. No selection step -- the entire
output vector was "chosen". The gradient is computed differently:

```rust
// Consumer uses output directly
let output = agent.step(&state, reward, terminal);
// No set_action_taken() needed -- the output IS the action
apply_torques(&output);
```

The policy gradient for continuous output is the **deterministic policy gradient**:

```
delta = advantage * d_output / d_weights
```

More precisely, for each output neuron `j`:
```
delta[j] = advantage * activation_derivative(pre_activation[j])
```

This pushes all output values in the direction that increases expected return,
proportional to how much each output contributed. No softmax, no one-hot, no
action index.

### Action Space Configuration

A new config parameter determines the gradient mode:

```rust
pub enum ActionSpace {
    /// Discrete: softmax + sampling + REINFORCE gradient.
    /// Consumer must call set_action_taken(usize) after each step.
    Discrete,
    /// Continuous: raw output = action, deterministic policy gradient.
    /// No action feedback needed from consumer.
    Continuous,
}

pub action_space: ActionSpace,  // default Discrete (backward compatible)
```

### Impact on Existing Components

| Component | Change required | Scope |
|-----------|----------------|-------|
| `step()` return type | `usize` -> `Vec<f64>` (or `L::Vector`) | API breaking |
| `act()` return type | Same change, deprecation path | API breaking |
| `select_action()` | Removed from library, consumer responsibility | Internal |
| `softmax_masked()` | No longer called by library (consumer may use it) | Utility stays |
| `learn_continuous()` | Split gradient computation by ActionSpace mode | Internal |
| `update_weights_hybrid()` | Receives delta from mode-specific gradient | Unchanged signature |
| `InferResult.y_conv` | Becomes the return value of step() | Semantic change |
| Critic input | `latent_concat` unchanged -- critic sees hidden states | No change |
| PC inference loop | Unchanged -- still produces y_conv via top-down/bottom-up | No change |
| Serialization | ActionSpace added to config with serde default | Backward compat |

### What Does NOT Change

- **PC inference loop**: Top-down/bottom-up convergence is independent of action
  space. The loop produces `y_conv` regardless.
- **Critic architecture**: Still receives `state + latent_concat`, outputs value
  estimate. The TD error computation is action-space agnostic.
- **Surprise scoring**: RMS prediction error across layers. Independent of output
  interpretation.
- **Continuous learning (M1-M4)**: Scale range, hysteresis, layer decay, EWC --
  all operate on learning rates and weight updates, not on action selection.
- **CCA crossover**: Operates on weight matrices, not on action semantics.

### Migration Path

#### Phase 1: Return raw output (non-breaking)

Add a new method alongside existing API:

```rust
// New: returns raw output
pub fn step_raw(&mut self, state: &[f64], reward: f64, terminal: bool) -> Vec<f64>

// Existing: unchanged, still returns usize
pub fn step(&mut self, state: &[f64], reward: f64, terminal: bool) -> usize
```

Both share the same internal learning logic. `step()` calls `step_raw()` internally
and applies softmax + sampling before returning.

#### Phase 2: ActionSpace enum

Add `ActionSpace::Continuous` with deterministic policy gradient. `step_raw()`
uses the configured mode for gradient computation.

#### Phase 3: Deprecate discrete wrappers

Once consumers have migrated to `step_raw()`, deprecate `step()` (usize return)
and `act()`. The raw output API becomes the canonical interface.

## Interaction with Continuous Learning

The generic action space spec is **independent** from continuous learning. The
mechanisms interact as follows:

| CL Mechanism | Interaction | Notes |
|--------------|-------------|-------|
| M1 (scale range) | None -- scales effective_lr, not action | Compatible |
| M2 (hysteresis) | Actor signal = PC surprise (not action-dependent) | Compatible |
| M3 (layer decay) | Scales per-layer LR, independent of gradient mode | Compatible |
| M4 (EWC) | Fisher computed from raw gradients -- mode determines which gradients | Gradient mode matters |

**M4 note**: Fisher diagonal estimation uses `g_raw^2`. In discrete mode, `g_raw`
comes from the REINFORCE gradient. In continuous mode, `g_raw` comes from the
deterministic policy gradient. The Fisher lifecycle (accumulate, decay, snapshot)
is identical -- only the source of `g_raw` changes.

## Implementation Order

1. **First**: Complete and validate continuous learning (v2.1.0) with the current
   discrete architecture. This is the priority.
2. **Then**: Implement generic action space as a separate version (v3.0.0 -- API
   breaking change) using this spec.
3. **Validate**: Re-run Tic-Tac-Toe benchmarks with `ActionSpace::Discrete` to
   confirm regression-free. Then test with a continuous environment (e.g., CartPole,
   simple robot).

## Open Questions

1. **L::Vector vs Vec\<f64\> return type**: Should `step_raw()` return `L::Vector`
   (generic, zero-copy for GPU backend) or `Vec<f64>` (always on host)? For GPU
   backends, the consumer likely needs host data. A `vec_to_vec()` call may be
   needed regardless.

2. **Temperature in continuous mode**: Temperature scaling (`y_conv / temperature`)
   currently affects softmax sharpness. In continuous mode, it could scale output
   magnitude -- or be irrelevant. Needs experimental validation.

3. **Entropy regularization**: Currently adds `entropy_coeff * (log(pi) + 1)` to
   the gradient. In continuous mode with no probability distribution, entropy is
   undefined. Alternative: output variance regularization?

4. **Hybrid action spaces**: Some environments combine discrete and continuous
   outputs (e.g., "which tool to use" + "how hard to press"). This may require
   splitting the output layer into groups with different gradient modes.

5. **`valid_actions` in continuous mode**: The mask concept is discrete. For
   continuous mode, constraints are typically enforced by activation functions
   (sigmoid for [0,1] bounds, tanh for [-1,1]) or by the environment rejecting
   invalid outputs.
