# Continuous Learning via Intrinsic Surprise Modulation

## Vision

Eliminate the binary training/inference distinction. The PC actor-critic becomes a
**perpetually adaptive** system where learning intensity is governed entirely by the
network's own surprise signal — the prediction error already computed by the PC
inference loop. When the environment is predictable, weights freeze naturally. When
novelty arrives, plasticity reactivates automatically. No external scheduler, no
mode switch, no replay buffer. The mechanism is fully internal to the predictive
coding architecture.

### Two Problems, One Signal

The surprise-driven plasticity solves two problems simultaneously:

1. **Overtraining collapse**: An agent that has mastered its environment (e.g.,
   Tic-Tac-Toe at depth 9) continues to receive weight updates under the current
   system (scale floor = 0.1). Over thousands of episodes with no new information,
   these updates cause weight drift and performance degradation. With surprise-driven
   plasticity, the network detects that it has nothing left to learn (low surprise)
   and freezes — preserving peak performance indefinitely without external
   intervention.

2. **Environment transfer**: When the same agent is placed in a new environment
   (e.g., from Tic-Tac-Toe to Connect Four, or from one opponent to a radically
   different one), the PC loop produces high prediction errors because its internal
   model no longer matches observations. This sustained surprise automatically
   reactivates plasticity and the agent begins learning the new environment — while
   EWC protects the consolidated knowledge from the previous one.

The lifecycle of an agent becomes:

```
Environment A                          Environment B
────────────────────────               ────────────────────────
1. High surprise → PLASTIC             4. High surprise → PLASTIC
   (learning new environment)             (adapting to new environment)
2. Surprise decreases → convergence       EWC anchors protect A knowledge
3. Low surprise → FROZEN               5. Surprise decreases → convergence
   (peak performance, no drift)        6. Low surprise → FROZEN
   Agent can stay here indefinitely       (new stable state)
```

No mode switch, no retraining command, no curriculum scheduler. The agent's own
prediction errors are the sole signal that drives the entire lifecycle.

## Current State (v2.0.0)

The infrastructure is 90% in place:

| Component | Status |
|-----------|--------|
| Surprise score (RMS prediction error) | Implemented — `PcActor::infer` |
| Surprise-based LR scaling | Implemented — `PcActorCritic::surprise_scale` |
| Adaptive thresholds (mean ± std) | Implemented — `surprise_buffer` with circular buffer |
| Scale range | Hardcoded `[0.1, 2.0]` — floor never reaches zero |
| `SelectionMode` | Controls action selection (stochastic vs argmax) AND implies train/no-train |

### Gap

The current scale floor of 0.1 means the network always learns, even when surprise
is minimal. There is no mechanism to truly freeze weights when the environment is
stable, nor to protect consolidated knowledge when surprise spikes.

### SelectionMode Redefinition

Currently `SelectionMode` conflates two concerns:

1. **Action selection policy**: `Training` → stochastic sampling, `Play` → argmax.
2. **Weight updates**: In practice, `Play` mode implies no learning because the
   consuming application (e.g., PC-TicTacToe) simply does not call `learn()`.

With continuous learning, concern (2) disappears. **Surprise alone governs
plasticity.** `SelectionMode` becomes purely an exploration/exploitation switch:

- `Training` → explores (stochastic action) + learns according to surprise
- `Play` → exploits (argmax action) + **also learns according to surprise**

In `Play` mode against a known opponent, surprise stays low → scale ≈ 0 → weights
are effectively frozen. If the opponent changes strategy, surprise rises → the
network adapts **without any external mode change**.

This means the consuming application always calls `learn()` regardless of mode. The
network self-regulates via its internal surprise signal.

### Unified API: `step()`

The current API requires the consumer to orchestrate learning explicitly:

```rust
// Current API (v2.0.0) — consumer manages the learning lifecycle:
let (action, infer) = agent.act(&state, &valid_actions, mode);
// ... execute action in environment, get reward, next_state ...
let loss = agent.learn_continuous(&state, &infer, action, &valid, reward, ...);
```

With continuous learning, the agent should manage its own learning internally.
The consumer should only provide what the agent cannot know: the current observation
and the reward from its last action. The proposed API:

```rust
let action = agent.step(&state, reward, terminal);
```

| Parameter  | Type     | Description |
|------------|----------|-------------|
| `state`    | `&[f64]` | Current environment observation |
| `reward`   | `f64`    | Reward received for the **previous** action |
| `terminal` | `bool`   | Whether the previous state was terminal (episode end) |

#### Internal state machine

The agent stores `(state_prev, action_prev, infer_prev)` from the previous step.
On each call to `step()`:

```
step(state, reward, terminal):
    1. IF has_previous_step:
           learn from (state_prev, action_prev, reward) → TD update
           (plasticity governed by surprise — may be zero if FROZEN)
    2. infer = pc_inference(state)
    3. action = select_action(infer)      // stochastic or argmax per config
    4. store (state, action, infer) as previous step
    5. IF terminal: clear previous step   // next call starts fresh episode
    6. RETURN action
```

First call of an episode: no previous step exists, so `reward` and `terminal` are
ignored. The consumer passes `0.0` and `false` as convention.

#### Manual reset

```rust
agent.reset_step();
```

Clears the internal `(state_prev, action_prev, infer_prev)` state, returning the
agent to the same condition as before its first `step()` call. The next `step()`
behaves as the first call of a fresh episode — `reward` and `terminal` are ignored.

This does **not** affect weights, surprise buffer, hysteresis state, or EWC anchors.
It only resets the temporal link between consecutive steps.

Use cases:
- Placing the agent in a completely new environment mid-episode
- Recovering from an external error where the previous step is no longer meaningful
- Testing or benchmarking with a clean step state without reconstructing the agent

#### Removal of `valid_actions`

The v2.0.0 API accepts a `valid_actions` parameter that externally masks the softmax
output, preventing the agent from selecting invalid actions. This conflicts with the
continuous learning philosophy: **the agent should learn what is valid by observing
its environment**, not by receiving external constraints.

Without the mask, an agent that selects an invalid action (e.g., walking into a wall)
receives feedback from the environment:

- The state does not change (or changes unfavorably)
- A negative (or zero) reward is received
- The agent's PC loop predicts a different outcome → surprise increases
- The agent learns to avoid that action in that context

This is how biological agents learn environmental constraints — through interaction,
not instruction. The `state` vector must encode enough information for the agent to
eventually infer action validity (e.g., adjacent cell occupancy in a grid world).

**Backward compatibility**: A `step_masked(&state, &valid_actions, reward, terminal)`
variant can coexist for environments where external masking is desired (e.g.,
Tic-Tac-Toe with strict rule enforcement). But the canonical API for continuous
learning is `step()` without the mask.

#### What the consumer no longer manages

| v2.0.0 (current)                     | Continuous learning (proposed)       |
|---------------------------------------|--------------------------------------|
| Call `act()` + `learn()` separately   | Single `step()` call                 |
| Handle `InferResult` externally       | Internal to agent                    |
| Choose `SelectionMode` per call       | Configured once at construction      |
| Provide `valid_actions` per step      | Agent learns from environment        |
| Decide when to call `learn()`         | Surprise governs plasticity          |

## Design Goals

1. **Zero-surprise convergence**: When surprise is consistently low, effective
   learning rate approaches zero — weights stabilize without explicit mode switch.
2. **Automatic reactivation**: A sustained increase in surprise (novel data,
   curriculum advancement, environment shift) restores full plasticity without
   external signal.
3. **Forgetting protection**: Transient noise (single outlier observations) must
   not trigger catastrophic weight updates. Only sustained novelty reactivates
   learning.
4. **Layer-wise consolidation**: Early layers (generic features) consolidate
   faster and are harder to perturb than later layers (task-specific features).
5. **Internal-only mechanisms**: All plasticity modulation uses signals already
   present in the PC loop (prediction errors, surprise scores). No replay buffer,
   no external task boundaries, no stored datasets.
6. **Backward compatible**: Existing behavior reproducible via configuration
   (e.g., `scale_floor = 0.1` recovers v2.0.0 behavior).

## Proposed Mechanisms

### M1 — Configurable Scale Range

Replace the hardcoded `[0.1, 2.0]` with configurable bounds:

```rust
pub scale_floor: f64,   // default 1e-6
pub scale_ceil: f64,    // default 2.0
```

The `scale_floor` is configurable via TOML config or CLI override. The default of
`1e-6` provides a micro-plasticity residual — weights change infinitesimally, which
allows the network to adapt to slow environmental drift without requiring the
hysteresis gate to wake. A value of `0.0` gives total freeze (maximum protection
against drift); `0.1` recovers v2.0.0 legacy behavior. The optimal value is
determined experimentally.

```toml
# In config TOML:
scale_floor = 1e-6
scale_ceil = 2.0
```

```bash
# CLI override:
--scale-floor 0.0      # total freeze
--scale-floor 0.1      # legacy behavior
--scale-ceil 3.0       # more aggressive learning on high surprise
```

### M2 — Schmitt Trigger Hysteresis

Replace the linear surprise-to-scale mapping with a hysteresis gate that requires
**sustained** surprise to change state:

```
State: FROZEN  →  surprise > wake_threshold for N consecutive steps  →  PLASTIC
State: PLASTIC →  surprise < sleep_threshold for M consecutive steps →  FROZEN
```

Where `wake_threshold > sleep_threshold` (asymmetric thresholds). This prevents
single noisy observations from triggering plasticity.

The hysteresis parameters:

```rust
pub wake_threshold: f64,    // surprise level to reactivate (relative to adaptive stats)
pub sleep_threshold: f64,   // surprise level to freeze (relative to adaptive stats)
pub wake_window: usize,     // consecutive high-surprise steps required to wake
pub sleep_window: usize,    // consecutive low-surprise steps required to sleep
```

When adaptive surprise is enabled, thresholds are expressed relative to the buffer
statistics (e.g., `wake = mean + 1.0 * std`, `sleep = mean - 0.5 * std`).

Within the PLASTIC state, the existing linear interpolation (M1 range) still applies.
Within the FROZEN state, `scale = scale_floor` (typically 0.0).

### M3 — Layer-Wise Consolidation Decay

Deeper layers (closer to input) learn slower and consolidate faster. Two modes are
available: fixed decay (position-based) and adaptive decay (surprise-based).

#### M3a — Fixed Decay (default)

A single global parameter at the actor level. The per-layer factor is computed
automatically via an exponential formula based on layer position:

```
effective_lr[i] = lr * surprise_scale * consolidation_decay^(n_hidden - 1 - i)
```

Where `i = 0` is the first hidden layer (closest to input), `n_hidden` is the number
of hidden layers **of the network being updated** (actor or critic — each has its own
layer count), and `consolidation_decay ∈ (0, 1]` controls how much protection early
layers get. A value of `1.0` means all layers have full plasticity (legacy behavior).

Example with `n_hidden = 3` and `consolidation_decay = 0.5`:

| Layer | Index (i) | Formula (0.5^(3-1-i)) | Factor | Effect |
|-------|-----------|----------------------|--------|--------|
| Hidden 0 (input-side) | 0 | 0.5^2 | 0.25 | Most protected |
| Hidden 1 (middle) | 1 | 0.5^1 | 0.50 | Moderate protection |
| Hidden 2 (output-side) | 2 | 0.5^0 | 1.00 | Full plasticity |

This reflects the intuition that early layers capture general features shared across
all regimes, while later layers encode task-specific mappings that should adapt freely.
One parameter controls the entire gradient of protection across the network.

```toml
[agent.actor]
adaptive_consolidation = false    # use exponential formula (default)
consolidation_decay = 0.5
```

```bash
--consolidation-decay 0.5
```

#### M3b — Adaptive Decay (surprise-per-layer)

Instead of a fixed positional decay, each layer's protection is derived from its own
prediction error stability. The PC loop already computes per-layer prediction errors
— layers with consistently low error have converged and should be protected more;
layers with volatile error are still learning and need full plasticity.

Each layer maintains an exponential moving average (EMA) of its prediction error
magnitude:

```
ema_error[i] = β * ema_error[i] + (1 - β) * |error[i]|
```

The per-layer decay factor is then:

```
adaptive_decay[i] = sigmoid(-k * (ema_error[i] - error_threshold))
```

Where:
- `β` = EMA smoothing factor (e.g., 0.99 — slow adaptation)
- `k` = sigmoid steepness (controls how sharply the transition occurs)
- `error_threshold` = midpoint where decay = 0.5

Behavior:
- `ema_error[i] << threshold` → `adaptive_decay[i] ≈ 1.0` → layer fully protected
- `ema_error[i] >> threshold` → `adaptive_decay[i] ≈ 0.0` → layer learns freely
- Transition is smooth (sigmoid), not abrupt

The effective learning rate becomes:

```
effective_lr[i] = lr * surprise_scale * (1.0 - adaptive_decay[i])
```

Note the inversion: high decay (converged layer) → low effective LR.

Example with 3 hidden layers after training:

| Layer | ema_error | adaptive_decay | effective LR factor | Interpretation |
|-------|-----------|----------------|---------------------|----------------|
| Hidden 0 | 0.002 (stable) | 0.95 | 0.05 | Nearly frozen — converged early |
| Hidden 1 | 0.08 (moderate) | 0.50 | 0.50 | Moderate protection |
| Hidden 2 | 0.25 (volatile) | 0.05 | 0.95 | Learning freely |

The key advantage over M3a: protection is based on **what each layer has actually
learned**, not its position. A deep layer that converges early gets protected; a
shallow layer that remains unstable keeps learning. This is more faithful to the
principle that surprise governs everything internally.

Actor-level flag and parameters:

```rust
// In PcActorConfig (or PcActorCriticConfig)
pub adaptive_consolidation: bool,           // default false (use M3a exponential)
pub consolidation_ema_beta: f64,            // default 0.99
pub consolidation_sigmoid_k: f64,           // default 10.0
pub consolidation_error_threshold: f64,     // default 0.05 (tuned experimentally)
```

```toml
[agent.actor]
adaptive_consolidation = true
consolidation_ema_beta = 0.99
consolidation_sigmoid_k = 10.0
consolidation_error_threshold = 0.05

# consolidation_decay is IGNORED when adaptive_consolidation = true
```

#### M3a vs M3b — When to use which

| Criterion | M3a (fixed) | M3b (adaptive) |
|-----------|-------------|----------------|
| Simplicity | One parameter | Four parameters |
| Debuggability | Deterministic per topology | Depends on training history |
| Assumption | Early layers always generalize | Let data decide |
| Best for | Initial experiments, small nets | Heterogeneous topologies, environment transfer |
| Risk | Over/under-protects if assumption is wrong | Sigmoid tuning may be sensitive |

Recommended approach: start with M3a fixed decay for initial validation. Switch to
M3b if experiments show that optimal decay varies significantly across layers or
training regimes. Both modes are mutually exclusive — when `adaptive_consolidation`
is true, the global `consolidation_decay` parameter is ignored.

### M4 — Elastic Weight Consolidation (EWC) with Surprise-Triggered Snapshots

When the network transitions from PLASTIC to FROZEN (M2), it takes a snapshot of
current weights and estimates parameter importance using the Fisher Information
diagonal (approximated from recent gradients).

During subsequent PLASTIC phases, weight updates are regularized:

```
total_gradient[i] = task_gradient[i] + λ_ewc * F[i] * (θ[i] - θ_snapshot[i])
```

Where:
- `F[i]` = diagonal Fisher estimate for parameter `i` (importance)
- `θ_snapshot[i]` = weight value at consolidation time
- `λ_ewc` = consolidation strength

The surprise signal eliminates EWC's biggest weakness (requiring explicit task
boundaries): the FROZEN→PLASTIC transition IS the task boundary, detected
automatically.

```rust
pub ewc_lambda: f64,          // default 0.0 (disabled, legacy behavior)
pub ewc_fisher_samples: usize // episodes to estimate Fisher at consolidation
```

#### EWC Computation Details

##### Step 1 — Weight snapshot (`θ_snapshot`)

When the hysteresis gate transitions PLASTIC → FROZEN, the agent clones all
trainable parameters:

```
θ_snapshot = clone(θ)
```

This includes per-layer weights (`W`, `b`), `rezero_alpha` scalars, and skip
projection matrices (`W_proj`). The snapshot is the anchor point — EWC will
penalize deviations from these values during subsequent PLASTIC phases.

##### Step 2 — Fisher diagonal estimation (`F[i]`)

The full Fisher Information Matrix is N×N (intractable). The diagonal approximation
computes one importance score per parameter by averaging squared gradients over
recent experience:

```
F[i] = (1/K) * Σ_k (∂ log π(a_k | s_k) / ∂θ[i])²
```

Where K = `ewc_fisher_samples` (last K steps before consolidation).

In the context of the existing architecture, the per-parameter gradients are already
computed by `Layer::backward()` — specifically `dW` (weight gradients) and the bias
gradient. Instead of applying the update, the gradients are squared and accumulated:

```
fn estimate_fisher(recent_steps) -> Vec<f64>:
    fisher = vec![0.0; num_parameters]

    for step in recent_steps:
        // Forward pass: already available from step.infer_result
        // Backward pass: compute policy gradient (delta) as in learn()
        // Extract per-parameter gradients from each layer

        grads = compute_parameter_gradients(step)
        for i in 0..num_parameters:
            fisher[i] += grads[i] * grads[i]

    fisher /= len(recent_steps)
    return fisher
```

The Fisher estimation reuses the same backward pass logic as `update_weights()` but
accumulates `g²` instead of applying `g * lr`. No new mathematical machinery is
needed — only a mode where backward produces gradients without updating.

##### Step 3 — Regularized update (during PLASTIC phase)

In `update_weights()`, after computing the normal task gradient for each parameter,
the EWC penalty is added before the weight update:

```
For each parameter i:
    task_grad   = normal backprop gradient (already computed)
    ewc_penalty = λ_ewc * F[i] * (θ[i] - θ_snapshot[i])
    total_grad  = task_grad + ewc_penalty
    θ[i] -= effective_lr * total_grad
```

The penalty acts as an **elastic spring** pulling each weight back toward its
consolidated value. The spring stiffness is proportional to the parameter's
importance (`F[i]`):

- **High F[i]** (critical for previous environment): strong resistance to change.
  The network can only move this weight if the task gradient is large enough to
  overcome the spring — meaning the new environment genuinely requires it.
- **Low F[i]** (unimportant for previous environment): the weight moves freely
  to accommodate the new task.
- **λ_ewc** controls global stiffness. Higher values prioritize preserving old
  knowledge; lower values prioritize adapting to the new environment.

##### Numerical example

```
Weight #42 (important for previous environment):
    θ_snapshot[42] = 0.5     (consolidated value)
    θ[42]          = 0.8     (current, drifted +0.3)
    F[42]          = 10.0    (high importance)
    λ_ewc          = 0.1
    task_grad[42]  = 0.05    (new environment wants to push further)

    ewc_penalty    = 0.1 * 10.0 * (0.8 - 0.5) = 0.30
    total_grad     = 0.05 + 0.30 = 0.35
    → Net effect: weight pulled BACK toward snapshot (penalty dominates)

Weight #99 (unimportant for previous environment):
    θ_snapshot[99] = 0.5
    θ[99]          = 0.8     (same drift)
    F[99]          = 0.01    (low importance)
    λ_ewc          = 0.1
    task_grad[99]  = 0.05

    ewc_penalty    = 0.1 * 0.01 * (0.8 - 0.5) = 0.0003
    total_grad     = 0.05 + 0.0003 = 0.0503
    → Net effect: weight moves freely (task gradient dominates)
```

##### Storage requirements

| Data              | Size                        | When computed            |
|-------------------|-----------------------------|--------------------------|
| `θ_snapshot`      | 1× all trainable parameters | PLASTIC → FROZEN transition |
| `F` (Fisher diag) | 1× f64 per parameter        | PLASTIC → FROZEN transition |

For current network sizes (layers of 18-27 neurons), this roughly doubles the weight
memory — negligible. For larger networks on a future GPU backend, storage should be
profiled.

##### Integration with M2 (hysteresis) and M3 (layer decay)

The full update formula combining all mechanisms:

```
For layer i, parameter j:
    IF adaptive_consolidation:
        layer_decay = 1.0 - adaptive_decay[i]     // from M3b per-layer surprise EMA
    ELSE:
        layer_decay = consolidation_decay^(n_hidden - 1 - i)  // from M3a exponential

    effective_lr   = lr * surprise_scale * layer_decay
    ewc_penalty    = λ_ewc * F[i,j] * (θ[i,j] - θ_snapshot[i,j])
    total_grad     = task_grad[i,j] + ewc_penalty
    θ[i,j]        -= effective_lr * total_grad
```

When the hysteresis gate is FROZEN: `surprise_scale ≈ scale_floor (1e-6)`, so
`effective_lr ≈ 0` regardless of EWC — the spring is irrelevant because nothing
moves. EWC only matters during PLASTIC phases when the network is actively learning
a new environment and needs to protect previously consolidated weights.

**Note**: EWC adds per-parameter storage (Fisher diagonal + snapshot), roughly
doubling memory for weights. For the current network sizes (layers of 18-27 neurons)
this is negligible. For larger networks with a future GPU backend, this should be
considered.

## Mechanism Interaction

The four mechanisms form a layered system:

```
Surprise Score (from PC loop)
    │
    ▼
┌─────────────────────────┐
│  M2: Hysteresis Gate     │ ── Determines FROZEN / PLASTIC state
│  (Schmitt trigger)       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  M1: Scale Mapping       │ ── Maps surprise → scale factor [floor, ceil]
│  (linear interpolation)  │    (floor = 0.0 when FROZEN)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  M3: Layer Decay         │ ── Per-layer attenuation of effective LR
│  (consolidation_decay)   │    (early layers ≈ frozen, late layers ≈ full)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  M4: EWC Regularization  │ ── Penalizes drift from consolidated weights
│  (Fisher-weighted anchor)│    (snapshot taken at PLASTIC→FROZEN transition)
└──────────┬──────────────┘
           │
           ▼
     effective_lr per layer per parameter
```

## Configuration

All new parameters have defaults that reproduce v2.0.0 behavior:

```rust
pub struct PcActorCriticConfig {
    // ... existing fields ...

    // M1 — Scale range
    pub scale_floor: f64,            // default 1e-6
    pub scale_ceil: f64,             // default 2.0

    // M2 — Hysteresis
    pub hysteresis: bool,            // default false (legacy linear mapping)
    pub wake_window: usize,          // default 5
    pub sleep_window: usize,         // default 10

    // M3 — Layer consolidation
    pub consolidation_decay: f64,              // default 1.0 (no decay, legacy)
    pub adaptive_consolidation: bool,          // default false (use M3a exponential)
    pub consolidation_ema_beta: f64,           // default 0.99 (only when adaptive)
    pub consolidation_sigmoid_k: f64,          // default 10.0 (only when adaptive)
    pub consolidation_error_threshold: f64,    // default 0.05 (only when adaptive)

    // M4 — EWC
    pub ewc_lambda: f64,             // default 0.0 (disabled)
    pub ewc_fisher_samples: usize,   // default 100
}
```

Legacy equivalent (v2.0.0 behavior):

```toml
[agent.actor]
scale_floor = 0.1
scale_ceil = 2.0
hysteresis = false
consolidation_decay = 1.0
ewc_lambda = 0.0
```

Continuous learning mode with fixed decay (M3a):

```toml
[agent.actor]
scale_floor = 1e-6
scale_ceil = 2.0
hysteresis = true
wake_window = 5
sleep_window = 10
consolidation_decay = 0.5
ewc_lambda = 0.1
ewc_fisher_samples = 100

hidden_layers = [
    { size = 27, activation = "softsign" },
    { size = 27, activation = "softsign" },
    { size = 18, activation = "softsign" },
]
```

Continuous learning mode with adaptive decay (M3b):

```toml
[agent.actor]
scale_floor = 1e-6
scale_ceil = 2.0
hysteresis = true
wake_window = 5
sleep_window = 10
adaptive_consolidation = true
consolidation_ema_beta = 0.99
consolidation_sigmoid_k = 10.0
consolidation_error_threshold = 0.05
ewc_lambda = 0.1
ewc_fisher_samples = 100

# consolidation_decay is IGNORED when adaptive_consolidation = true
hidden_layers = [
    { size = 27, activation = "softsign" },
    { size = 27, activation = "softsign" },
    { size = 18, activation = "softsign" },
]
```

All parameters are at the actor level and accept CLI override (e.g., `--scale-floor 0.0`,
`--consolidation-decay 0.5`, `--adaptive-consolidation`) for experimentation without
modifying config files.

## Implementation Order

Suggested phased approach, each phase independently testable:

| Phase | Mechanism | Scope | Risk |
|-------|-----------|-------|------|
| 1 | M1 — Configurable scale range | `surprise_scale()` + config | Minimal — parameter change only |
| 2 | M2 — Hysteresis gate | `surprise_scale()` + state field | Low — additive, behind flag |
| 3 | M3 — Layer-wise decay | `update_weights()` | Low — multiplicative factor |
| 4 | M4 — EWC regularization | `update_weights()` + snapshot storage | Medium — new storage + consolidation trigger |

Each phase should be validated independently with the Tic-Tac-Toe benchmark before
proceeding to the next.

## Biological Analogy

This design mirrors known neuromodulatory mechanisms:

| Mechanism | Biological Analogue |
|-----------|-------------------|
| Surprise score | Prediction error signal (dopamine, norepinephrine) |
| Hysteresis gate | Tonic vs phasic neuromodulator modes |
| Layer-wise decay | Cortical hierarchy — V1 consolidates before PFC |
| EWC consolidation | Sleep-dependent synaptic consolidation |
| FROZEN→PLASTIC transition | Arousal / orienting response to novelty |

The key insight is that biological systems never truly stop learning — they modulate
**plasticity** continuously based on internal signals. This spec brings the same
principle to the PC architecture using signals the network already computes.

## Critic Under Continuous Learning

The mechanisms described above (M1–M4) focus on the actor (PC network). The critic
(standard MLP) also has weights that must be managed under continuous learning. This
section describes how each mechanism applies to the critic and where the critic
diverges from the actor.

### Current State

In v2.0.0, the critic always updates at full learning rate regardless of surprise:

```rust
let loss = self.critic.update(&critic_input, target);  // no surprise modulation
```

The surprise scale is only applied to the actor's `update_weights()`. This means
that when the actor freezes (low surprise), the critic continues learning and can
drift — producing value estimates that diverge from the actor's stable policy.

### Surprise Signal

The critic does not have a PC loop and therefore does not produce its own surprise
score. The **actor's surprise** governs both networks. This is architecturally
correct: the surprise measures how well the agent's internal model matches the
environment. When the environment is predictable (low actor surprise), both the
policy (actor) and the value estimates (critic) should stabilize.

A single FROZEN/PLASTIC state machine (M2) controls both actor and critic. There
are no separate hysteresis gates.

### Mechanism Applicability

#### M1 — Scale Range: applies

The critic's learning rate is modulated by the same `surprise_scale` as the actor:

```
critic_effective_lr = critic_lr * surprise_scale
```

When `surprise_scale ≈ scale_floor`, the critic freezes alongside the actor. When
surprise rises, both learn together.

#### M2 — Hysteresis: applies (shared state)

The same FROZEN/PLASTIC state governs both networks. No separate thresholds or
windows for the critic.

#### M3a — Fixed Exponential Decay: applies

The critic has its own `consolidation_decay` parameter (independent from the actor's),
applied with the same exponential formula:

```
critic_effective_lr[i] = critic_lr * surprise_scale * critic_consolidation_decay^(n_critic_hidden - 1 - i)
```

```toml
[agent.critic]
consolidation_decay = 0.5    # default 1.0 (no decay, legacy)
```

#### M3b — Adaptive Decay: future exploration only

M3b does **not** apply to the critic. The critic is a standard MLP without a PC
inference loop, so it has no per-layer prediction errors to drive adaptive
consolidation.

The closest proxy would be per-layer gradient magnitude from the TD error:

```
ema_grad[i] = β * ema_grad[i] + (1 - β) * ||∂loss/∂W[i]||
```

However, this proxy has significant weaknesses:

1. **Numerical, not semantic**: gradient magnitude measures how much a layer's
   weights need to change, but says nothing about what the layer has "learned" or
   how stable its representation is. Small gradients can mean convergence or a
   flat loss landscape (saddle point).
2. **Confounded by learning rate**: the gradient magnitude depends on the scale of
   the TD error and the critic LR, not just layer convergence. A well-converged
   layer can still show large gradients if the TD error spikes due to environment
   change — which is exactly when you want it to adapt, not protect.
3. **No hierarchical structure**: unlike the actor where prediction errors flow
   between layers (top-down vs bottom-up), critic gradients backpropagate uniformly.
   There is no principled reason why early critic layers should behave differently
   from later ones beyond the generic "early = general" assumption — which M3a
   already captures with the exponential formula.

**Status**: deferred as optional future exploration. If a future critic architecture
incorporates predictive coding elements (e.g., a PC-based critic), M3b would become
directly applicable. Until then, the critic uses M3a exclusively.

#### M4 — EWC: applies (independent snapshots)

The critic maintains its own EWC state, separate from the actor:

- **Snapshot**: `θ_critic_snapshot` taken at PLASTIC → FROZEN transition (same
  trigger as the actor)
- **Fisher diagonal**: `F_critic[i]` estimated from the critic's own gradients
  (∂loss/∂θ_critic) over the last `ewc_fisher_samples` steps
- **Penalty**: applied during PLASTIC phases, same formula as the actor:

```
critic_total_grad[i] = critic_task_grad[i] + ewc_lambda * F_critic[i] * (θ_critic[i] - θ_critic_snapshot[i])
```

The actor's `ewc_lambda` is shared — one consolidation strength for both networks.
If experiments show the critic needs different stiffness, a separate
`critic_ewc_lambda` can be introduced.

### Critic Configuration

```toml
[agent.critic]
consolidation_decay = 0.5    # default 1.0 (no decay, legacy)
# All other parameters (scale_floor, scale_ceil, hysteresis, ewc_lambda, etc.)
# are shared at the agent level — the critic inherits them from [agent.actor]
# or from a shared [agent] section.
```

### Summary

| Mechanism | Actor | Critic |
|-----------|-------|--------|
| M1 (scale range) | Actor surprise → scale | Same surprise → same scale |
| M2 (hysteresis) | FROZEN/PLASTIC state | Shared state (no separate gate) |
| M3a (fixed decay) | `consolidation_decay` (actor) | `consolidation_decay` (critic, independent) |
| M3b (adaptive decay) | Per-layer prediction error EMA | Not applicable (future exploration) |
| M4 (EWC) | Actor snapshots + Fisher | Critic snapshots + Fisher (independent) |
| Surprise source | PC loop prediction errors | Inherited from actor |
