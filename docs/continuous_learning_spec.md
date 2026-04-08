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

**Policy gradient**: `step()` uses unmasked softmax over all output neurons. All
actions receive gradient signal — the agent learns to avoid invalid actions through
negative reward, not through masking. `step_masked()` uses `softmax_masked` as in
v2.0.0, restricting gradient to valid actions only.

**Backward compatibility**: `step_masked(&state, &valid_actions, reward, terminal)`
coexists for environments where external masking is desired (e.g., Tic-Tac-Toe with
strict rule enforcement). The canonical API for continuous learning is `step()`
without the mask.

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
pub scale_floor: f64,   // default 0.0
pub scale_ceil: f64,    // default 2.0
```

The `scale_floor` is configurable via TOML config or CLI override. The default of
`0.0` means FROZEN state is a true freeze — no weight updates, no drift. This is
safe because the EWMA-smoothed hysteresis (M2) detects gradual environmental
changes and wakes the network when needed, eliminating the need for micro-plasticity
during FROZEN. A value of `1e-6` provides optional micro-plasticity for environments
where the hysteresis may not detect very slow drift; `0.1` recovers v2.0.0 legacy
behavior.

```toml
# In config TOML:
scale_floor = 0.0       # true freeze (default)
scale_ceil = 2.0
```

```bash
# CLI override:
--scale-floor 1e-6     # micro-plasticity for slow drift
--scale-floor 0.1      # legacy behavior
--scale-ceil 3.0       # more aggressive learning on high surprise
```

### M2 — Dual-EWMA Hysteresis

Replace the linear surprise-to-scale mapping with a hysteresis gate driven by two
EWMAs with different time constants: a **fast** EWMA that tracks recent signal
changes and a **slow** EWMA that represents the long-term baseline. State
transitions occur when the fast signal deviates significantly from the baseline.

#### EWMA with warmup

Both EWMAs use the same warmup formula:

```
A(k) = A(k-1) + (Val(k) - A(k-1)) / min(N, k)
```

Where:
- `Val(k)` = raw plasticity signal at step k (surprise for actor, |TD error| for critic)
- `N` = EWMA window size (different for fast and slow)
- `k` = step counter (`u64`, increments monotonically, never resets on episode boundaries;
  u64 prevents overflow — at 1M steps/sec, overflow would take ~600,000 years)

Behavior:
- **Warmup (k ≤ N)**: computes the exact arithmetic mean of the first k values.
  No cold-start bias — A(1) = Val(1), A(2) = (Val(1)+Val(2))/2, etc.
- **Steady state (k > N)**: equivalent to EMA with α = 1/N, β = (N-1)/N.
  A(k) = A(k-1) × (1 - 1/N) + Val(k) × (1/N)

Properties:
- **Outlier robustness**: a single anomalous step moves the average by only 1/N
- **Magnitude sensitivity**: surprise of 5.0 contributes more than 0.16 — raw
  per-step thresholding discards this information
- **Cross-episode continuity**: k increments across episodes, the EWMA naturally
  persists without needing to decide whether to reset at episode boundaries
- **O(1) memory**: only stores A(k) and k per EWMA instance

#### Dual-EWMA state transitions

Two EWMAs with different windows track the same raw signal:

- **Fast EWMA** (`fast_window`, default 10): responds quickly to changes (~10 steps)
- **Slow EWMA** (`slow_window`, default 50): represents the long-term baseline

State transitions use **relative thresholds** — the fast EWMA is compared against
the slow EWMA scaled by a fraction:

```
State: FROZEN  →  fast_ewma > slow_ewma × (1 + wake_fraction)  →  PLASTIC
State: PLASTIC →  fast_ewma < slow_ewma × (1 - sleep_fraction) →  FROZEN
```

Where `wake_fraction > 0` and `sleep_fraction > 0` (both configurable).

This design is **self-calibrating**: thresholds are relative to the current baseline
(slow_ewma), not absolute values. They automatically adapt to different signal
magnitudes across environments without requiring a separate buffer, mean/std
computation, or adaptive threshold logic. The slow EWMA IS the adaptive threshold.

Example with slow_ewma = 0.05, wake_fraction = 0.5:
- Wake threshold = 0.05 × 1.5 = 0.075
- Single noise spike (surprise 0.15 for 1 step): fast_ewma rises to ~0.06 < 0.075 → **no wake**
- Sustained environment change (surprise 0.5): fast_ewma rises to ~0.3 in ~10 steps > 0.075 → **wake**

Example with slow_ewma ≈ 0.0 (fully converged agent):
- Wake threshold = 0.0 × 1.5 = 0.0
- Any nonzero surprise triggers wake → **correct**: if surprise was zero and now
  it's nonzero, the environment changed

Sleep example with slow_ewma = 0.3 (learning), sleep_fraction = 0.3:
- Sleep threshold = 0.3 × 0.7 = 0.21
- As agent converges, fast_ewma drops to 0.1 < 0.21 → **freeze**

Initial state: both EWMAs initialized to 0, network starts in PLASTIC (learning
from the beginning). As the network converges, the fast EWMA drops below the slow
EWMA's sleep band and the network transitions to FROZEN for the first time.

#### Initial warmup guard

During initial training, surprise naturally decreases as the network learns. The
fast EWMA (N=10) tracks this decrease quickly while the slow EWMA (N=50) retains
the higher early values — creating a false signal that can trigger premature sleep
within 20-30 steps. This is problematic because (a) the slow EWMA is not in steady
state during warmup, making the fast vs slow comparison unfair, and (b) a PLASTIC
phase shorter than `min_fisher_phase` produces unreliable Fisher data that gets
discarded by the short-phase guard.

To prevent premature sleep, the first FROZEN transition is suppressed until the
network has accumulated enough steps for both the dual EWMA and the Fisher EMA
to be in steady state:

```
min_initial_plastic = max(slow_window, min_fisher_phase)
                    = max(slow_window, ceil(1 / (1 - fisher_ema_beta)))
```

With default parameters: `max(50, 100) = 100` steps (~15 TTT games).

This guard is **derived from existing parameters** — no new configurable knob. It
ensures:
- The slow EWMA has completed warmup → fair fast vs slow comparison
- The Fisher EMA has accumulated ~63% of steady-state value → reliable data for
  the first EWC anchor

The guard uses the per-network step counter `k`. Since `k` is monotonic (`u64`,
never resets on episode boundaries), the guard is active only during the first
`min_initial_plastic` steps of the network's lifetime, then becomes a permanent
no-op. After GA crossover, `k` resets to 0 — the child automatically receives
warmup protection.

When the guard lifts at step `min_initial_plastic`:
- If the network has already converged: fast is well below slow → immediate FROZEN
  on the first evaluation, with reliable Fisher for EWC protection
- If the network is still learning: fast tracks slow closely → stays PLASTIC
  naturally until genuine convergence

#### Why dual EWMA instead of single EWMA + adaptive thresholds

A single EWMA with adaptive thresholds computed from a buffer has a **variance
mismatch**: the EWMA (window N=50) has ~7× less variance than the raw signal. If
thresholds are calibrated from raw signal statistics (mean ± std), the smoothed
EWMA almost never reaches them — the agent gets stuck FROZEN permanently.

Computing thresholds from the EWMA's own history fixes the mismatch but produces
very tight thresholds that can cause brief PLASTIC flickers on single noise spikes.

The dual EWMA eliminates both problems:
- **No variance mismatch**: both fast and slow are EWMAs of the same signal
- **No separate buffer or stats**: the slow EWMA IS the adaptive baseline
- **No tight-threshold flickers**: the fast EWMA (N=10) needs ~10 steps of
  sustained change to deviate from the slow baseline by wake_fraction
- **Self-calibrating across environments**: thresholds are relative (fractions),
  not absolute — a surprise magnitude of 0.05 or 5.0 works equally well

#### Independent state machines for actor and critic

The actor and critic have **separate** FROZEN/PLASTIC state machines, each with
its own pair of fast/slow EWMAs:

| Network | Plasticity signal | Source | EWMAs |
|---------|------------------|--------|-------|
| Actor | PC surprise (RMS prediction error) | `PcActor::infer` (existing) | Own fast + slow pair |
| Critic | \|TD error\| | `learn_continuous` line 578 (existing) | Own fast + slow pair |

This independence is essential because the two signals measure different things:

- **Actor surprise** = how well the internal model predicts across layers
- **\|TD error\|** = how well the value function predicts expected returns

Key scenarios that require independence:
- Actor converged (low surprise) but critic still inaccurate (high TD error) →
  critic must keep learning while actor stays frozen.
- Reward function changes without changing state dynamics → actor surprise stays
  low (PC errors unchanged) but TD error spikes → critic detects and adapts.

Caveat: TD error is a lagging indicator of actor policy changes. When the actor
enters PLASTIC and starts changing its policy, the critic's TD error may not spike
immediately. This lag is self-correcting (TD error rises as the policy diverges)
and is inherent to TD learning, not introduced by this design.

#### Hysteresis parameters

```rust
// Per-network hysteresis (actor and critic each have their own set)
pub hysteresis: bool,          // default false (legacy linear mapping)
pub fast_window: usize,        // default 10 (fast EWMA window)
pub slow_window: usize,        // default 50 (slow EWMA baseline window)
pub wake_fraction: f64,        // default 0.5 (fast > slow × 1.5 → PLASTIC)
pub sleep_fraction: f64,       // default 0.3 (fast < slow × 0.7 → FROZEN)
```

Within the PLASTIC state, the existing linear interpolation (M1 range) still applies.
Within the FROZEN state, `scale = scale_floor`.

#### Scale discontinuity at FROZEN→PLASTIC transition

When transitioning FROZEN→PLASTIC, the scale jumps from `scale_floor` (0.0) to
whatever M1 computes from the current raw surprise — potentially 1.0-2.0. This is
a discontinuous step, not a ramp. This is **intentional**: the dual EWMA required
~10 steps of sustained elevated surprise to trigger the wake, confirming a genuine
environment change. Aggressive learning at wake time is the correct response —
delaying with a soft-start ramp would slow adaptation precisely when speed matters
most. The first PLASTIC step's update magnitude is bounded by GRAD_CLIP (5.0) and
WEIGHT_CLIP (5.0), and EWC protection (active from the first cycle onward, thanks
to the warmup guard) resists large changes to important parameters.

### M3 — Layer-Wise Consolidation Decay

Deeper layers (closer to input) learn slower and consolidate faster. Two modes are
available for the **actor**: fixed decay (M3a, position-based) and adaptive decay
(M3b, surprise-based). Both are implemented; `adaptive_consolidation` flag selects
which is active. The **critic** uses M3a exclusively — M3b is not applicable (see
Critic section).

#### M3a — Fixed Decay (default)

A single global parameter at the actor level. The per-layer factor is computed
automatically via an exponential formula based on layer position:

```
effective_lr[i] = lr * surprise_scale * consolidation_decay^(n_hidden - 1 - i)
```

Where `i = 0` is the first hidden layer (closest to input), `n_hidden` is the number
of hidden layers **of the network being updated** (actor or critic — each has its own
layer count), and `consolidation_decay ∈ [0.0, 1.0]` controls how much protection
early layers get. A value of `1.0` means all layers have full plasticity (legacy
behavior). A value of `0.0` freezes all layers except the last hidden layer.

Values outside `[0.0, 1.0]` are rejected at construction with a `ConfigValidation`
error. A value > 1.0 would amplify early layers exponentially (opposite of intended
consolidation); negative values produce undefined behavior (alternating signs in
powers).

Example with `n_hidden = 3` and `consolidation_decay = 0.5`:

| Layer | Index (i) | Formula (0.5^(3-1-i)) | Factor | Effect |
|-------|-----------|----------------------|--------|--------|
| Hidden 0 (input-side) | 0 | 0.5^2 | 0.25 | Most protected |
| Hidden 1 (middle) | 1 | 0.5^1 | 0.50 | Moderate protection |
| Hidden 2 (output-side) | 2 | 0.5^0 | 1.00 | Full plasticity |

This reflects the intuition that early layers capture general features shared across
all regimes, while later layers encode task-specific mappings that should adapt freely.
One parameter controls the entire gradient of protection across the network.

#### Interaction with local_lambda

M3a and `local_lambda` operate on different aspects of the weight update:

- `local_lambda` controls **what** the layer learns: the blend ratio between
  backprop gradient and PC prediction error (`delta = λ*bp + (1-λ)*pc`).
- M3a controls **how much** each layer learns: the effective learning rate.

M3a scales the result of the lambda blend, not the blend itself. The ratio between
backprop and PC error within the delta is unchanged by M3a. Therefore, M3a does
not invalidate the local_lambda sweet spot directly.

However, there is a second-order interaction: if early layers learn less (due to
M3a), their weight updates alter the gradient landscape for subsequent steps,
which could shift the optimal lambda value. Additionally, M3a provides partially
redundant protection with lambda — both reduce the magnitude of updates to early
layers. With M3a active, lambda may tolerate a wider range.

**Recommendation**: the depth-lambda scaling law (`λ ≈ 1 - 10^(-(L+1))`) was
calibrated without M3a (consolidation_decay=1.0). When M3a is active, re-validate
lambda with a sweep to confirm the sweet spot has not shifted significantly. Do not
make M3a and local_lambda mutually exclusive — they are complementary.

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

**Sigmoid calibration required before M3b activation**: with the default `k=10` and
`error_threshold=0.05`, the sigmoid saturates for real prediction error ranges
[0.02, 0.15] — the actual dynamic range is only [0.38, 0.73], providing minimal
per-layer differentiation. The sigmoid parameters must be calibrated during M3
implementation using one of:
- Increase `k` significantly (50+) for sharper transitions
- Rescale sigmoid output to [0, 1] range: `(sigmoid(x) - min) / (max - min)`
- Replace sigmoid with a piecewise linear function
The chosen approach must be validated against experimental prediction error
distributions from the Tic-Tac-Toe benchmark. M3b is implemented in the actor
with `adaptive_consolidation = true`; the calibration is a prerequisite for
meaningful results, not a reason to defer implementation.

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

Both modes are implemented for the actor and selectable via `adaptive_consolidation`.
They are mutually exclusive — when `adaptive_consolidation` is true, the global
`consolidation_decay` parameter is ignored. M3a is the recommended starting point
for initial experiments; M3b should be activated once the sigmoid is calibrated
against experimental prediction error distributions. The critic uses M3a exclusively.

### M4 — Elastic Weight Consolidation (EWC) with Surprise-Triggered Snapshots

EWC protects consolidated knowledge during new PLASTIC phases by penalizing weight
deviations from a snapshot taken at convergence. The hysteresis state machine (M2)
eliminates EWC's biggest weakness (requiring explicit task boundaries): the
FROZEN→PLASTIC transition IS the task boundary, detected automatically.

During PLASTIC phases, weight updates are regularized:

```
total_gradient[i] = task_gradient[i] + λ_ewc * F[i] * (θ[i] - θ_snapshot[i])
```

Where:
- `F[i]` = diagonal Fisher estimate for parameter `i` (importance)
- `θ_snapshot[i]` = weight value at consolidation time
- `λ_ewc` = consolidation strength

```rust
pub ewc_lambda: f64,          // default 0.0 (disabled, legacy behavior)
pub fisher_ema_beta: f64,     // default 0.99 (EMA smoothing for Fisher estimation)
pub fisher_decay: f64,        // default 0.9 (decay of old Fisher between PLASTIC phases)
```

#### EWC Computation Details

##### Fisher lifecycle — explicit 3-step specification

The Fisher state consists of two components:
- `F_total`: accumulated Fisher diagonal (persists across PLASTIC phases)
- `F_ema`: running EMA of squared gradients (accumulates during PLASTIC only)

The lifecycle has three distinct steps tied to state transitions:

```
Step 1 — FROZEN → PLASTIC (wake):
    IF last_phase_reliable:
        F_total *= fisher_decay           // decay old to make room
    // ELSE: F_total unchanged — old Fisher preserved
    Reset F_ema to 0
    Reset plastic_step_counter to 0

Step 2 — During PLASTIC (every step):
    F_ema[i] = β_fisher * F_ema[i] + (1 - β_fisher) * g_raw[i]²
    plastic_step_counter += 1

Step 3 — PLASTIC → FROZEN (freeze):
    IF plastic_step_counter >= min_fisher_phase:
        F_total += F_ema                  // reliable data — add to total
        last_phase_reliable = true
    ELSE:
        last_phase_reliable = false       // unreliable — discard F_ema
    Snapshot θ
```

Where `min_fisher_phase = ceil(1 / (1 - fisher_ema_beta))`. With β=0.99, this
is 100 steps — the number of steps needed for the EMA to absorb ~63% of its
steady-state value. This threshold is **derived from the existing parameter**, not
a new configurable knob.

##### Short PLASTIC phase guard

If the PLASTIC phase lasts fewer than `min_fisher_phase` steps, the Fisher EMA
has not accumulated enough data to be reliable. In this case:

- **F_ema is discarded** — not added to F_total
- **fisher_decay is NOT applied** at the next wake — F_total is preserved intact

This prevents EWC protection from eroding through rapid FROZEN/PLASTIC oscillation.
If the dual-EWMA hysteresis (M2) causes the agent to briefly wake and re-freeze
(e.g., the fast EWMA temporarily crosses the slow baseline), the Fisher from the
previous long PLASTIC phase remains fully intact.

After N rapid oscillation cycles: `F_total` is unchanged → EWC protection = 100%.
When a long PLASTIC phase finally occurs, Fisher updates normally.

##### Fisher diagonal estimation (`F_ema`) via running EMA

The full Fisher Information Matrix is N×N (intractable). The diagonal approximation
computes one importance score per parameter using a running EMA of squared gradients,
updated every step during PLASTIC:

```
F_ema[i] = β_fisher * F_ema[i] + (1 - β_fisher) * g_raw[i]²
```

**Critical: post-clip, pre-scaling gradients.** The gradient `g_raw` is the clipped
local gradient computed in `Layer::backward()` at line 167 (`clip_vec(&mut grad,
GRAD_CLIP)`) — AFTER activation derivative and gradient clipping, but BEFORE
multiplication by `effective_lr`. This means:

- **Post-clip**: Fisher is bounded at `GRAD_CLIP² = 25.0` per parameter. Pre-clip
  gradients could include unstable spikes that were specifically truncated to prevent
  damage — including them would overweight parameters that happened to have gradient
  explosions.
- **Pre-scaling**: using gradients multiplied by `effective_lr` (which includes
  `surprise_scale * layer_decay * lr`) would bias Fisher toward the learning rate
  schedule rather than actual parameter importance, creating a positive feedback
  loop: high surprise → large scaled gradients → inflated Fisher → excessive EWC
  penalty → suppressed learning → persistent surprise.

**Why EMA instead of batch accumulation:**
- Weights recent gradients more heavily — near convergence is where parameter
  importance is most relevant
- Does not dilute with early-PLASTIC gradients computed at different θ values
- Always up-to-date — no timing dependency on transition
- Memory O(P) constant, independent of PLASTIC phase length

##### Weight snapshot (`θ_snapshot`)

At PLASTIC → FROZEN transition (Step 3), the agent clones all trainable parameters:

```
θ_snapshot = clone(θ)
```

This includes per-layer weights (`W`, `b`), `rezero_alpha` scalars, and skip
projection matrices (`W_proj`). The snapshot is the anchor point — EWC will
penalize deviations from these values during subsequent PLASTIC phases.

The snapshot is always taken at PLASTIC → FROZEN, regardless of whether the phase
was long or short. Even if Fisher was not updated (short phase), the weights may
have changed slightly and the snapshot should reflect the current state.

##### Fisher decay between environments

At FROZEN → PLASTIC transition (Step 1), if the previous PLASTIC phase was reliable
(long enough), `fisher_decay` is applied to make room for the new environment's
Fisher data. This prevents rigidity from growing without bound across multiple
environment transfers.

| Phase | F_env_A | F_env_B | F_env_C | Total rigidity |
|-------|---------|---------|---------|----------------|
| After env A | 1.0 | — | — | 1.0 |
| After env B | 0.9 | 1.0 | — | 1.9 |
| After env C | 0.81 | 0.9 | 1.0 | 2.71 |
| After env D | 0.73 | 0.81 | 0.9 | 2.44 + F_D |

If the agent returns to environment A, surprise rises, it relearns, and generates
fresh Fisher_A — the decayed old Fisher does not prevent re-learning.

Alternatives considered and rejected:
- **Accumulate (F = F_old + F_new)**: rigidity grows monotonically → network
  becomes unable to learn after N environments. Contradicts the vision of
  indefinite adaptability.
- **Replace (F = F_new)**: loses all protection for previous environments.
  No multi-environment consolidation.

##### Step 3 — Regularized update (during PLASTIC phase)

The EWC penalty is applied as a **post-update correction** after `Layer::backward`,
not inside it. This is correct because the EWC penalty is a per-weight local term
that does NOT propagate through the network — each weight's penalty depends only on
that weight, its Fisher, and its snapshot, not on other layers.

The injection point in `update_weights()` per layer:

```rust
// 0. Compute layer_decay (M3) and fold into surprise_scale
let layer_decay = if adaptive_consolidation {
    1.0 - adaptive_decay[i]                           // M3b
} else {
    consolidation_decay.powi((n_hidden - 1 - i) as i32) // M3a
};
let layer_surprise = surprise_scale * layer_decay;     // M3 folded in
let effective_lr = self.config.lr_weights * layer_surprise;

// 1. Compute EWC penalty from PRE-update weights (before backward modifies them)
let ewc_w = ewc_lambda * F_w[i] * (W[i] - W_snapshot[i]);   // matrix
let ewc_b = ewc_lambda * F_b[i] * (b[i] - b_snapshot[i]);   // vector

// 2. Normal backward pass — layer_decay reaches task gradient via layer_surprise
//    Layer::backward is NOT modified: it receives lr and layer_surprise as before,
//    computing effective_lr = lr * layer_surprise = lr * surprise_scale * layer_decay
let propagated = self.layers[i].backward(
    input, output, &delta, self.config.lr_weights, layer_surprise
);
// backward updates W and b internally via task gradient (with layer_decay applied)
// backward returns propagated delta for the next layer (independent of effective_lr)

// 3. Post-correction: apply EWC penalty (same effective_lr as task gradient)
self.backend.mat_scale_add(&mut self.layers[i].weights, &ewc_w, -effective_lr);
let bias_penalty = self.backend.vec_scale(&ewc_b, effective_lr);
self.layers[i].bias = self.backend.vec_sub(&self.layers[i].bias, &bias_penalty);

// 4. rezero_alpha and skip projection updates also use layer_surprise:
//    rezero_alpha[idx] -= effective_lr * grad_alpha;
//    mat_scale_add(proj, &dw_proj, -effective_lr);
```

Key design decisions:

- **Layer::backward is NOT modified.** The M3 layer_decay is folded into the
  `surprise_scale` argument passed to backward: `layer_surprise = surprise_scale *
  layer_decay`. Backward computes `effective_lr = lr * layer_surprise` internally,
  which equals `lr * surprise_scale * layer_decay`. This ensures both the task
  gradient and the EWC penalty use the same effective_lr. With `consolidation_decay
  = 1.0` (legacy), `layer_decay = 1.0` and `layer_surprise = surprise_scale` —
  identical to v2.0.0 behavior.
- **Pre-update θ for penalty computation.** The penalty uses θ BEFORE backward
  modifies it (step 1 before step 2). This is mathematically exact per the EWC
  formula: `θ -= effective_lr × (task_grad + λ × F × (θ - θ*))`.
- **Propagated gradient is clean.** The `W^T × grad` propagated to earlier layers
  does NOT include EWC penalty or effective_lr — it depends only on the clipped
  gradient and the weight matrix. Layer_decay does not affect gradient propagation.
- **EWC penalty is NOT gradient-clipped.** The penalty is a smooth quadratic term,
  not a noisy gradient. If the penalty is large, it SHOULD dominate — that is the
  protection mechanism. Maximum penalty magnitude: `λ × GRAD_CLIP² × WEIGHT_CLIP
  = 0.1 × 25.0 × 5.0 = 12.5`. At effective_lr ≈ 0.005: weight change = 0.0625
  per step, within normal operating ranges.
- **All per-layer updates use layer_surprise.** The rezero_alpha update, skip
  projection update, and any other per-layer weight modification within
  `update_weights()` use the same `effective_lr = lr * layer_surprise`, ensuring
  consistent layer_decay application across all trainable parameters of the layer.

The penalty acts as an **elastic spring** pulling each weight back toward its
consolidated value. The spring stiffness is proportional to the parameter's
importance (`F[i]`):

- **High F[i]** (critical for previous environment): strong resistance to change.
  The network can only move this weight if the task gradient is large enough to
  overcome the spring — meaning the new environment genuinely requires it.
- **Low F[i]** (unimportant for previous environment): the weight moves freely
  to accommodate the new task.
- **λ_ewc** controls global stiffness. Higher values prioritize preserving old
  knowledge; lower values prioritize adapting to the new environment. This parameter
  is exposed in the library config (TOML/CLI) and can be included in the GA genome
  as an evolvable parameter — agents with λ_ewc too high become petrified (cannot
  learn → low fitness → selected against), while agents with λ_ewc too low suffer
  catastrophic forgetting (lose prior knowledge → low fitness → selected against).
  The GA naturally finds the optimal rigidity for the environment.

##### F_total saturation bound

F_total accumulates across environments via `F_total += F_ema` with fisher_decay
applied at each cycle. The geometric series converges to:

```
F_total_max = F_ema_steady / (1 - fisher_decay) = F_ema / 0.1 = 10 × F_ema
```

Worst-case penalty magnitude (all parameters at maximum):
- F_ema = GRAD_CLIP² = 25.0, F_total = 250, drift = WEIGHT_CLIP = 5.0
- penalty = λ × F_total × drift = 0.1 × 250 × 5.0 = 125.0
- weight change = effective_lr × 125 = 0.01 × 125 = 1.25 per step

Typical values (27-neuron TTT network, 3 environment transitions):
- F_ema ≈ 0.5, F_total ≈ 1.36, drift ≈ 0.1
- penalty = 0.1 × 1.36 × 0.1 = 0.014 (~28% of typical task gradient)

The worst case requires all gradients at clip AND weights drifted to WEIGHT_CLIP
AND 10+ environment transitions — extremely unlikely for current network sizes.
If a specific environment produces excessive EWC penalties, reduce `ewc_lambda`
directly or let the GA evolve it to the optimal value.

Note: EWC penalty is inside the `effective_lr` multiplication. In FROZEN state
(`effective_lr ≈ scale_floor = 0.0`), both task gradient and EWC penalty are zero —
nothing moves. EWC only matters during PLASTIC phases when the network is actively
learning and needs protection against forgetting.

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

| Data              | Size                        | When updated               |
|-------------------|-----------------------------|----------------------------|
| `θ_snapshot`      | 1× all trainable parameters | PLASTIC → FROZEN transition |
| `F_ema`           | 1× f64 per parameter        | Every step during PLASTIC   |

For current network sizes (layers of 18-27 neurons), this roughly doubles the weight
memory — negligible. For larger networks on a future GPU backend, storage should be
profiled.

##### Integration with M2 (hysteresis) and M3 (layer decay)

The full update formula combining all mechanisms:

```
For layer i, parameter j:
    // M3 — layer decay
    IF adaptive_consolidation:
        layer_decay = 1.0 - adaptive_decay[i]     // from M3b per-layer surprise EMA
    ELSE:
        layer_decay = consolidation_decay^(n_hidden - 1 - i)  // from M3a exponential

    // M3 folded into surprise_scale for Layer::backward
    layer_surprise = surprise_scale * layer_decay
    effective_lr   = lr * layer_surprise           // = lr * surprise_scale * layer_decay

    // M4 — EWC penalty (pre-update θ)
    ewc_penalty    = λ_ewc * F[i,j] * (θ[i,j] - θ_snapshot[i,j])

    // Task gradient applied by Layer::backward (uses layer_surprise as surprise_scale)
    // EWC penalty applied as post-correction (uses same effective_lr)
    θ[i,j]        -= effective_lr * (task_grad[i,j] + ewc_penalty)
```

Both task gradient and EWC penalty are scaled by the same `effective_lr`, which
includes `layer_decay`. This ensures M3 layer-wise consolidation applies uniformly
to all weight updates for a given layer, not just the EWC penalty.

When the hysteresis gate is FROZEN: `surprise_scale ≈ scale_floor (0.0)`, so
`effective_lr = 0` regardless of EWC — the spring is irrelevant because nothing
moves. EWC only matters during PLASTIC phases when the network is actively learning
a new environment and needs to protect previously consolidated weights.

**Note**: EWC adds per-parameter storage (Fisher diagonal + snapshot), roughly
doubling memory for weights. For the current network sizes (layers of 18-27 neurons)
this is negligible. For larger networks with a future GPU backend, this should be
considered.

## Mechanism Interaction

The four mechanisms form a layered system. Actor and critic run the same pipeline
independently, each with its own plasticity signal:

```
ACTOR                                    CRITIC
─────                                    ──────
PC Surprise (RMS prediction error)       |TD error|
    │                                        │
    ▼                                        ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  M2: Dual-EWMA Gate      │          │  M2: Dual-EWMA Gate      │
│  (actor FROZEN/PLASTIC)  │          │  (critic FROZEN/PLASTIC) │
└──────────┬──────────────┘          └──────────┬──────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  M1: Scale Mapping       │          │  M1: Scale Mapping       │
│  surprise → actor scale  │          │  |td_error| → critic     │
└──────────┬──────────────┘          └────────┬────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  M3: Layer Decay         │          │  M3a: Layer Decay        │
│  (M3a or M3b)            │          │  (fixed exponential only)│
└──────────┬──────────────┘          └────────┬────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│  M4: EWC Regularization  │          │  M4: EWC Regularization  │
│  (actor Fisher EMA)      │          │  (critic Fisher EMA)     │
└──────────┬──────────────┘          └────────┬────────────────┘
           │                                  │
           ▼                                  ▼
  actor effective_lr                 critic effective_lr
  per layer per parameter            per layer per parameter
```

## Configuration

All new parameters have defaults that reproduce v2.0.0 behavior:

Actor and critic each have their own hysteresis state machine and consolidation
parameters. Shared parameters (M1 scale range, M4 EWC strength) are at the agent
level.

```rust
pub struct PcActorCriticConfig {
    // ... existing fields ...

    // M1 — Scale range (shared)
    pub scale_floor: f64,            // default 0.0
    pub scale_ceil: f64,             // default 2.0

    // M2 — Dual-EWMA Hysteresis (per-network, same defaults)
    pub hysteresis: bool,          // default false (legacy linear mapping)
    pub fast_window: usize,        // default 10 (fast EWMA window)
    pub slow_window: usize,        // default 50 (slow EWMA baseline window)
    pub wake_fraction: f64,        // default 0.5 (fast > slow × 1.5 → PLASTIC)
    pub sleep_fraction: f64,       // default 0.3 (fast < slow × 0.7 → FROZEN)

    // M3 — Layer consolidation (per-network)
    // Actor:
    pub consolidation_decay: f64,              // default 1.0 (no decay, legacy)
    pub adaptive_consolidation: bool,          // default false (use M3a exponential)
    pub consolidation_ema_beta: f64,           // default 0.99 (only when adaptive)
    pub consolidation_sigmoid_k: f64,          // default 10.0 (only when adaptive)
    pub consolidation_error_threshold: f64,    // default 0.05 (only when adaptive)
    // Critic:
    pub critic_consolidation_decay: f64,       // default 1.0 (no decay, legacy)

    // M4 — EWC (shared strength, per-network state)
    pub ewc_lambda: f64,             // default 0.0 (disabled)
    pub fisher_ema_beta: f64,        // default 0.99 (EMA smoothing for Fisher)
    pub fisher_decay: f64,           // default 0.9 (decay of old Fisher between phases)

    // Actor-critic coupling (optional safety mechanism)
    pub actor_wakes_critic: bool,              // default false (fully independent)
    pub actor_wakes_critic_threshold: usize,   // default 1000 (FROZEN steps before forced wake)
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

[agent.critic]
consolidation_decay = 1.0
```

Continuous learning mode with fixed decay (M3a):

```toml
[agent.actor]
scale_floor = 0.0
scale_ceil = 2.0
hysteresis = true
fast_window = 10
slow_window = 50
wake_fraction = 0.5
sleep_fraction = 0.3
consolidation_decay = 0.5
ewc_lambda = 0.1
fisher_ema_beta = 0.99
fisher_decay = 0.9

hidden_layers = [
    { size = 27, activation = "softsign" },
    { size = 27, activation = "softsign" },
    { size = 18, activation = "softsign" },
]

[agent.critic]
consolidation_decay = 0.5
```

Continuous learning mode with adaptive decay (M3b, actor only):

```toml
[agent.actor]
scale_floor = 0.0
scale_ceil = 2.0
hysteresis = true
fast_window = 10
slow_window = 50
wake_fraction = 0.5
sleep_fraction = 0.3
adaptive_consolidation = true
consolidation_ema_beta = 0.99
consolidation_sigmoid_k = 10.0
consolidation_error_threshold = 0.05
ewc_lambda = 0.1
fisher_ema_beta = 0.99
fisher_decay = 0.9

# consolidation_decay is IGNORED when adaptive_consolidation = true
hidden_layers = [
    { size = 27, activation = "softsign" },
    { size = 27, activation = "softsign" },
    { size = 18, activation = "softsign" },
]

[agent.critic]
consolidation_decay = 0.5    # critic always uses M3a
```

All parameters accept CLI override (e.g., `--scale-floor 0.0`,
`--consolidation-decay 0.5`, `--fisher-decay 0.9`) for experimentation without
modifying config files.

## GA Integration (CCA Crossover)

The existing CCA crossover (`PcActorCritic::crossover`) creates a child agent by
blending the weights of two parents using Canonical Correlation Analysis for neuron
alignment. Continuous learning adds per-agent mutable state (EWMA, hysteresis state,
Fisher, snapshots, buffers) that must be handled correctly after crossover.

### Principle

A crossover child is a **new agent** with novel weights (a blend of two parents).
It has no experience, no convergence history, and no consolidated knowledge. All
continuous learning state must be initialized to a clean PLASTIC state, identical
to a freshly constructed agent. The child starts learning from scratch — the only
inheritance from its parents is the weight values themselves.

### State reset contract

The `crossover()` method must initialize all continuous learning fields as follows:

```rust
// After blending actor and critic weights from parents:

// M2 — Dual-EWMA Hysteresis (per-network)
actor_fast_ewma: 0.0,                     // no signal history
actor_fast_ewma_k: 0,                     // warmup counter at zero
actor_slow_ewma: 0.0,
actor_slow_ewma_k: 0,
actor_plasticity_state: PlasticityState::Plastic,  // start learning
critic_fast_ewma: 0.0,
critic_fast_ewma_k: 0,
critic_slow_ewma: 0.0,
critic_slow_ewma_k: 0,
critic_plasticity_state: PlasticityState::Plastic,

// M4 — EWC (per-network)
actor_fisher_ema: vec![0.0; actor_num_params],     // no importance data
actor_weight_snapshot: None,                        // no anchor point
critic_fisher_ema: vec![0.0; critic_num_params],
critic_weight_snapshot: None,

// Buffers
surprise_buffer: VecDeque::new(),          // already done in v2.0.0
td_error_buffer: VecDeque::new(),          // new for critic hysteresis

// Step state
step_state: None,                          // no previous step
```

The current `crossover()` already resets `surprise_buffer: VecDeque::new()` (line
360 of `pc_actor_critic.rs`). The new fields follow the same pattern.

### What crossover does NOT reset

| Component | Inherited from | Reason |
|-----------|---------------|--------|
| Weights (W, b) | Blended from parents | The purpose of crossover |
| `rezero_alpha` | Blended from parents | Part of residual weights |
| Skip projections (`W_proj`) | Blended from parents | Part of residual weights |
| Config | `child_config` parameter | Caller decides child's configuration |
| ActivationCaches | Not stored in agent | External, used only during crossover alignment |

### M3b per-layer EMA

If the actor uses M3b (adaptive consolidation), each layer maintains an EMA of its
prediction error. After crossover, these are reset to zero — the child's layers have
novel weights and their prediction error history is unknown:

```rust
actor_layer_error_ema: vec![0.0; n_hidden],   // reset per-layer EMA
```

During the child's initial PLASTIC phase, the EMA warms up naturally as the PC
loop runs and produces per-layer prediction errors.

### GA lifecycle with continuous learning

```
1. Population of agents in environment
2. Each agent runs step() for E episodes
   → Continuous learning governs plasticity internally
   → Some agents converge (FROZEN), others still learning (PLASTIC)
3. Selection: rank by fitness (e.g., win rate, depth reached)
4. Crossover: blend top agents via CCA
   → Child starts in PLASTIC with clean state
   → Child's weights are a blend — may be better or worse than parents
5. Child runs step() for E episodes
   → If blend was good: converges fast (low surprise quickly), freezes
   → If blend was bad: high surprise persists, keeps adapting
6. Repeat from (3)
```

The continuous learning mechanisms are fully compatible with GA because:
- Crossover produces weights only — all plasticity state is agent-specific
- A child naturally starts in PLASTIC and converges based on its own performance
- EWC does not interfere — no Fisher or snapshot exists for a new child
- The child's first PLASTIC→FROZEN transition creates its first EWC anchor

### No changes to CCA mechanics

The CCA crossover algorithm (neuron alignment via SVD, Hungarian matching, weight
blending) operates exclusively on weight matrices and ActivationCaches. None of the
continuous learning state participates in or affects the alignment or blending
process. The crossover is a pure function of weights and activation history.

### Serialization

The existing serializer (`serializer.rs`) persists weights and config via JSON.
Continuous learning adds mutable state (EWMAs, Fisher, snapshots, hysteresis state)
that must be handled during save/load:

- **Load without CL state** (e.g., a v2.0.0 model or weights-only export):
  initialize all continuous learning state to clean PLASTIC defaults — identical
  to the crossover reset contract above. The agent starts learning from scratch
  with the loaded weights.
- **Load with CL state** (full checkpoint): resume exactly where the agent left
  off — FROZEN/PLASTIC state, EWMA values, Fisher diagonals, weight snapshots,
  and step counters are all restored.

The serialization format and field selection are defined during TDD implementation.
The `#[serde(skip, default)]` pattern already used for the `backend` field provides
the mechanism for backward-compatible deserialization of new state fields.

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

### Plasticity Signal

The critic uses **|TD error|** as its own plasticity signal, independent from the
actor's surprise. The TD error is already computed in `learn_continuous`
(`pc_actor_critic.rs:578`):

```rust
let td_error = target - v_s;
```

| Signal | Measures | Network |
|--------|----------|---------|
| PC surprise (RMS prediction error) | How well the internal model predicts across layers | Actor |
| \|TD error\| | How well the value function predicts expected returns | Critic |

The two signals are independent because they measure fundamentally different things.
The actor can have low surprise (internal model is consistent) while the critic has
high TD error (value estimates are wrong) — for example, after a reward function
change that does not alter state dynamics. Conversely, the actor can have high
surprise (novel states) while the critic generalizes well to those states (low TD
error).

#### TD error bootstrap blindspot (known limitation)

|TD error| has a fundamental limitation inherited from TD learning: it uses the
critic's own estimates in the bootstrap target:

```
td_error = r + γ × V(s') - V(s)
```

If V(s) and V(s') are both wrong but **consistently** wrong (e.g., both ≈ 50 when
they should be ≈ 0.5), the TD error is small — the errors cancel. The critic
believes it is accurate because there is no discrepancy between consecutive state
values. This can occur during environment transfer when reward scale decreases
significantly (e.g., from [-100,100] to [-1,1]): the stale V estimates are
internally consistent at the old scale, producing near-zero TD error despite being
globally incorrect.

In practice, environment transfer also changes the state distribution — the critic
visits states for which its estimates are more variable, causing TD error to
eventually rise. The bootstrap blindspot is worst in the specific case where state
dynamics are identical and only reward magnitude changes. This is inherent to TD
learning and not introduced by the continuous learning design.

#### Actor-wakes-critic coupling (optional safety mechanism)

For environment transfer scenarios where the bootstrap blindspot may prevent the
critic from detecting changes, an optional unidirectional coupling is available:
when the actor enters PLASTIC and the critic has been FROZEN for longer than a
configurable threshold, the critic is forced to PLASTIC.

```
IF actor transitions FROZEN → PLASTIC
   AND critic is FROZEN
   AND critic_frozen_steps >= actor_wakes_critic_threshold:
       Force critic to PLASTIC
```

`critic_frozen_steps` increments on every call to `step()` while the critic is in
FROZEN state, regardless of whether a learning update occurs on that step. It resets
to 0 when the critic enters PLASTIC (by its own sleep condition or by this coupling).

This is asymmetric — the actor can wake the critic, but NOT the reverse. The actor's
PC surprise is a more reliable environment change detector (it measures internal
model consistency, not bootstrapped estimates), so it serves as a safety signal for
the critic.

When the critic is forced to PLASTIC by this coupling:
- The critic's own dual-EWMA is not modified — it continues tracking |TD error|
- The critic's Fisher short-phase guard still applies (if the forced PLASTIC phase
  is short, Fisher is preserved)
- The critic returns to FROZEN via its own sleep condition (fast < slow × sleep_fraction)

```rust
pub actor_wakes_critic: bool,              // default false (independent hysteresis)
pub actor_wakes_critic_threshold: usize,   // default 1000 (FROZEN steps before forced wake)
```

```toml
[agent]
actor_wakes_critic = false                 # default: fully independent
actor_wakes_critic_threshold = 1000        # steps critic must be FROZEN before actor can wake it
```

```bash
--actor-wakes-critic --actor-wakes-critic-threshold 500
```

When disabled (default), actor and critic hysteresis are fully independent as
specified in M2. Enable only for environments where reward scale changes are
expected during transfer.

### Mechanism Applicability

#### M1 — Scale Range: applies

The critic computes its own `critic_surprise_scale` from |TD error| using the same
linear interpolation formula as the actor (M1), with its own floor and ceiling:

```
critic_effective_lr = critic_lr * critic_surprise_scale
```

When |TD error| is consistently low, `critic_surprise_scale ≈ scale_floor` and the
critic freezes. When |TD error| rises, the critic learns.

#### M2 — Hysteresis: applies (independent state machine)

The critic has its own FROZEN/PLASTIC state machine driven by |TD error|, with its
own adaptive buffer and thresholds. The critic's state transitions are fully
independent from the actor's:

| Scenario | Actor state | Critic state | Behavior |
|----------|-------------|--------------|----------|
| Both learning | PLASTIC | PLASTIC | Normal training |
| Actor converged first | FROZEN | PLASTIC | Critic catches up to stable policy |
| Critic converged first | PLASTIC | FROZEN | Critic waits while actor explores |
| Both converged | FROZEN | FROZEN | Stable agent, no drift |
| Reward function change | FROZEN | PLASTIC | Critic adapts, actor unaffected |

Caveat: TD error is a lagging indicator of actor policy changes. When the actor
enters PLASTIC, the critic's TD error may not spike immediately. This lag is
self-correcting (TD error rises as the policy diverges) and is inherent to TD
learning, not introduced by this design.

#### M3a — Fixed Exponential Decay: applies

The critic has its own `consolidation_decay` parameter (independent from the actor's),
applied with the same exponential formula:

```
critic_effective_lr[i] = critic_lr * critic_surprise_scale * critic_consolidation_decay^(n_critic_hidden - 1 - i)
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

#### M4 — EWC: applies (independent state)

The critic maintains its own EWC state, fully separate from the actor. The same
3-step Fisher lifecycle applies (see M4 specification), with the critic's own
`F_total`, `F_ema`, `plastic_step_counter`, and `last_phase_reliable` flag:

- **Fisher EMA**: `F_critic_ema[i]` accumulated from the critic's own raw gradients
  (∂loss/∂θ_critic, pre-scaling) during the critic's PLASTIC phases
- **Short phase guard**: same `min_fisher_phase` threshold applies — short critic
  PLASTIC phases preserve old Fisher, no decay applied
- **Snapshot**: `θ_critic_snapshot` taken at the critic's own PLASTIC → FROZEN
  transition (independent from the actor's transition)
- **Penalty**: applied during critic PLASTIC phases:

```
critic_total_grad[i] = critic_task_grad[i] + ewc_lambda * F_critic[i] * (θ_critic[i] - θ_critic_snapshot[i])
```

The `ewc_lambda` is shared between actor and critic. The `fisher_ema_beta` and
`fisher_decay` are also shared. The Fisher diagonal already normalizes the EWC
penalty per-parameter — a weight with large gradients produces large Fisher values,
so the penalty scales automatically regardless of the gradient magnitude difference
between actor (REINFORCE) and critic (MSE). A shared lambda controls the global
rigidity level for both networks.

**Future exploration**: if experiments show that actor and critic require different
rigidity (e.g., the critic needs to adapt faster than the actor after environment
transfer), a `critic_ewc_lambda` override can be introduced without breaking changes.
This is subject to experimental validation — do not add preemptively.

### Critic Configuration

```toml
[agent.critic]
consolidation_decay = 0.5    # default 1.0 (no decay, legacy)
# Hysteresis parameters for the critic's own |TD error| state machine:
# fast_window, slow_window, wake_fraction, sleep_fraction
# default to the same values as the actor but are independently configurable.
# ewc_lambda, fisher_ema_beta, fisher_decay are shared at agent level.
```

### Summary

| Mechanism | Actor | Critic |
|-----------|-------|--------|
| Plasticity signal | PC surprise (RMS prediction error) | \|TD error\| |
| M1 (scale range) | Actor surprise → actor scale | \|TD error\| → critic scale |
| M2 (hysteresis) | Independent FROZEN/PLASTIC | Independent FROZEN/PLASTIC |
| M3a (fixed decay) | `consolidation_decay` (actor) | `consolidation_decay` (critic, independent) |
| M3b (adaptive decay) | Per-layer prediction error EMA | Not applicable (future exploration) |
| M4 (EWC) | Actor snapshots + Fisher EMA | Critic snapshots + Fisher EMA (independent) |

 
