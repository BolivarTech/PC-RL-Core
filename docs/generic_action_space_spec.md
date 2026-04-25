# Generic Action Space — v4.0.0 Specification

**Version:** 2.0 (rewrite against v3.0.0 baseline)
**Date:** 2026-04-25
**Author:** Julian Bolivar
**Crate:** `pc-rl-core`
**Status:** Spec-ready for SBTDD `/deep-plan`. Target release: **v4.0.0 (BREAKING)**.
**Predecessor spec:** v1.0 (2026-XX), preserved at git tag `pre-v4-spec` for context.
**Coordination:** Q1 (return type) intersects with GpuLinAlg Phase 2 — see §13.

---

## 1. Overview

PC-RL-Core's defining ambition is to be a **general-purpose reinforcement
learning framework** built around deliberative predictive coding. The
discrete-action specialization that has carried the project through v3.x
must be widened so the same actor-critic loop can drive continuous control,
multi-discrete grouping, and hybrid action spaces — without architectural
forks.

This document specifies the v4.0.0 release that introduces a **generic
action space** abstraction. It is a planned BREAKING change per SemVer
2.0.0 (replay-buffer schema and step-API surface change), with an explicit
backward-compatible migration path for v3.x discrete consumers.

The spec is written as the SBTDD spec-behavior input. Implementation plan
is generated downstream via `/deep-plan` against this document.

---

## 2. Motivation

The v3.0.0 architecture assumes a **discrete action space**: the actor
produces logits, the library applies softmax + sampling/argmax, and
returns a single `usize` action index. This assumption is embedded in the
public API (`step_masked`, `act`), the replay buffer schema
(`ReplayTransition::action: usize`), the policy gradient
(`delta[action] -= 1.0`), the masking primitives (`softmax_masked`,
`argmax_masked`), and the distillation anchors (KL on softmax(logits)).

This works for board games like Tic-Tac-Toe but excludes a large class of
target environments:

| Environment | Output interpretation | Current support (v3.0.0) |
|-------------|----------------------|--------------------------|
| Board game (TTT) | N neurons = N positions, pick one | **Yes** (discrete) |
| Robot locomotion | K neurons = K joint torques (all active simultaneously) | **No** |
| Continuous control (CartPole, MuJoCo) | M neurons = M-dim action vector | **No** |
| Multi-discrete (e.g. tool + intensity) | Groups of neurons, one selection per group | **No** |
| Hybrid (discrete + continuous) | Mixed groups in one output layer | **No** |

### 2.1 The diagonal problem (canonical example)

Robot with output neurons `[forward, back, right, left]` and activations
`[0.8, 0.1, 0.7, 0.05]` should move **diagonally forward-right** — both
`forward` and `right` simultaneously. v3.0.0 forces a single selection
(argmax → `forward`), discarding the multi-dimensional structure the
consumer encoded into the topology.

The library does not — and should not — interpret topology. The consumer
designs the network and knows what each output neuron means. The library
must expose the raw output and let the consumer interpret it.

---

## 3. Current architecture (v3.0.0 baseline)

This section refreshes the obsolete v2.0.0 baseline of the predecessor
spec. All references below are valid as of HEAD = `v3.0.0`.

### 3.1 Public action-related API

| Method | Signature | Module |
|--------|-----------|--------|
| `infer` | `(&self, input: &[f64]) -> InferResult<L>` | `pc_actor_critic::mod` |
| `act` | `(&mut self, state, valid_actions, mode: SelectionMode) -> (usize, InferResult<L>)` | `pc_actor_critic::mod` |
| `step` | `(&mut self, state, reward, terminal) -> usize` | `pc_actor_critic::mod` |
| `step_masked` | `(&mut self, state, valid, reward, done) -> Result<usize, PcError>` | `pc_actor_critic::mod` |
| `learn` (deprecated) | `(&mut self, trajectory) -> f64` | `pc_actor_critic::mod` |
| `learn_continuous` | trajectory-style episodic, returns `Result<f64, PcError>` | `pc_actor_critic::mod` |
| `learn_continuous_inner` | `(step: &LearnStep<'_, L>) -> Result<f64, PcError>` (internal) | `pc_actor_critic::mod` |
| `replay_learn` | `(&mut self, batch_size: usize) -> Result<(), PcError>` | `pc_actor_critic::mod` |

### 3.2 Where `usize` is structurally embedded

| Concern | Location | Form |
|---------|----------|------|
| Return type | `act`, `step`, `step_masked` | `usize` |
| Replay schema | `ReplayTransition::action` | `usize` |
| Policy gradient | `learn_continuous_inner` | `delta[action] -= 1.0` |
| Action masking | `LearnStep::valid_actions: &[usize]` | discrete index list |
| Softmax/argmax/sample | `matrix::{softmax_masked, argmax_masked, sample_from_probs}` | over discrete indices |
| Distillation KL | Polyak/Frozen anchor losses | `KL(softmax(live), softmax(target))` |

### 3.3 Pipelines that touch the action

```
            +---- Online step ----+
state ---> infer -> y_conv -> y_conv/temperature -> softmax_masked -> sample/argmax -> action: usize
                |                                                            |
                +-> latent_concat -> critic.forward -> V(s) ---+             |
                                                              advantage      |
                                                                |            |
                                                  policy_grad: delta[action] -= 1.0
                                                                |
                                          actor.update_with_decay(delta * advantage * effective_actor_scale_for_mode)


            +---- Replay step ----+
ReplayTransition (state, action: usize, reward, next_state, done, valid_actions)
                |
            same gradient construction with LearnMode::Replay; effective_actor_scale_for_mode honors scale_floor_replay and skip_kl bypass.

            +---- Distillation (Polyak / Frozen) ----+
live actor pi(s) = softmax(logits_live)
target actor pi*(s) = softmax(logits_target)   (Polyak EMA or Frozen snapshot)
KL_loss = sum(pi(s) * log(pi(s) / pi*(s)))     # discrete KL — undefined for raw continuous output
```

The KL-distillation step is the most action-space-coupled regularizer in
the codebase: **it presupposes a discrete probability distribution**.

### 3.4 v3.0.0 architectural elements relevant to this spec

| Element | Introduced | Effect on action space |
|---------|------------|------------------------|
| `LearnMode::Online` / `Replay` | v2.2.0 | Branch-free dispatch in `learn_continuous_inner`; mode flows via `LearnStep` |
| Hysteresis state machine | v2.1.0 | Surprise-driven plasticity; **independent** of action space |
| `effective_actor_scale_for_mode` | v2.2.1 | Mode-aware actor scale; gates magnitude, not direction |
| `effective_critic_scale_for_mode` | v3.0.0 | Mode-aware critic scale; same indifference |
| Polyak target / Frozen anchor | v2.2.0 | KL distillation — **action-space coupled** |
| EWC Fisher diagonal | v2.1.0 | Fisher of `g_raw²` — sourced from policy gradient, mode-dependent |
| Replay buffer (dual compartment) | v2.2.0 | Schema includes `action: usize` — **must migrate** |
| Cross-wake coupling | v2.1.0+ | State-machine concern only; action-space orthogonal |
| TD(n) n-step returns | v2.1.0 | Reward accumulation; action-space orthogonal |

---

## 4. Resolved decisions (was: open questions)

The five open questions from the predecessor spec are resolved here as
binding architectural decisions. They feed `/deep-plan` directly.

### 4.1 Q1 — Return type: `Vec<f64>` (host) or `L::Vector` (backend-native)?

**DECISION: `Vec<f64>` for the public surface. Add a `_raw_device` escape
hatch returning `L::Vector` for performance-critical GPU consumers.**

Rationale:
- 99% of downstream consumers want host data (write to file, send over
  network, multiply by Python tensor, log to console).
- A blanket `L::Vector` return would force every consumer to call
  `vec_to_vec` regardless of backend — a no-op on CpuLinAlg, a host
  transfer on GpuLinAlg either way.
- The `_raw_device` variant gives GPU consumers a zero-copy path when
  the action is fed back into another GPU kernel without ever touching
  the host.

This decision **must be re-validated** when GpuLinAlg Phase 2 lands —
specifically that `vec_to_vec` cost is acceptable per-step on Quadro P1000
target hardware (§13).

### 4.2 Q2 — Temperature in continuous mode

**DECISION: Temperature applies as pre-output scaling on the actor's
final pre-activation in continuous mode. Default 1.0 (no scaling).
Behavior preserved for discrete mode.**

In discrete: `softmax(y_conv / temperature)` — controls sharpness.
In continuous: `output = activation(pre_activation / temperature)` —
controls saturation depth (e.g. for tanh, lower temperature pushes
outputs toward saturation; higher temperature compresses toward zero).

Validation rule: `temperature > 0.0 && temperature.is_finite()`. NaN/Inf
rejected at construction.

### 4.3 Q3 — Entropy regularization in continuous mode

**DECISION: Continuous mode uses a Gaussian policy with **fixed σ**
(config: `policy_sigma`). Entropy of a fixed-σ Gaussian is constant
w.r.t. policy parameters; entropy gradient is therefore zero. The
existing `entropy_coeff` is silently inert in continuous mode (no
runtime cost; no validation rejection).**

Rationale:
- Fixed σ is the simplest stochastic continuous policy with a
  well-defined log-likelihood and gradient.
- Learned σ (state-dependent) is deferred to v5.0.0 — requires either
  doubling the output dim or adding a separate variance head, plus
  changes to inference/serialization. Out of scope for v4.0.0.
- Output variance regularization (alternative) would require a fresh
  derivation; no clear win over fixed-σ for the primary use cases.

The policy:
```
pi(a | s) = N(mu(s), sigma^2 * I)        // mu(s) = actor output
log pi(a | s) = -((a - mu)^T (a - mu)) / (2 sigma^2) - constant
nabla_theta log pi = (a - mu) / sigma^2  *  nabla_theta mu(s)
```

The learned sample is `a = mu(s) + sigma * eps`, `eps ~ N(0, I)`. The
sample is what the consumer executes; (a - mu) is what the library
needs to compute the gradient.

Default `policy_sigma = 0.1` for continuous mode. Discrete mode ignores
the field.

### 4.4 Q4 — Hybrid action spaces (e.g. discrete tool + continuous force)

**DECISION: Out of scope for v4.0.0. Tracked for v5.0.0+.**

v4.0.0 supports **pure** ActionSpace::Discrete OR **pure**
ActionSpace::Continuous. Hybrid would require:
- Output layer partitioning into per-group activation functions
- Per-group gradient mode dispatch
- Multi-objective replay schema

Defer until at least one v4.0.0 consumer reports a real-world hybrid
need with a concrete environment.

### 4.5 Q5 — `valid_actions` mask in continuous mode

**DECISION: `valid_actions` is a discrete-only concept. The continuous
API drops the mask parameter entirely. Constraints are enforced by the
output activation function (sigmoid for `[0, 1]`, tanh for `[-1, 1]`),
or by the environment rejecting/clipping invalid outputs upstream of
the agent.**

Replay's `ReplayTransition` schema (per §6) ensures continuous
transitions don't carry a stale `valid_actions: Vec<usize>` — the
field is only present in the Discrete variant.

---

## 5. v4.0.0 architecture

### 5.1 Core abstraction: `ActionSpace` enum

```rust
/// Action space configuration. Determines the policy gradient form,
/// the public step API, and the replay buffer transition schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSpace {
    /// Discrete: actor logits → softmax → sample/argmax over a finite set.
    /// REINFORCE policy gradient. Replay stores `Action::Discrete(usize)`.
    /// This is the v3.x default; preserved bit-for-bit when selected.
    Discrete,
    /// Continuous: actor output is the mean μ(s) of a fixed-σ Gaussian
    /// policy. Sampled action `a = μ(s) + σ·ε`. Gradient via the standard
    /// Gaussian-policy log-likelihood. Replay stores `Action::Continuous(Vec<f64>)`.
    Continuous,
}

impl Default for ActionSpace {
    /// Default `Discrete` to preserve v3.x behavior for unconfigured users.
    fn default() -> Self { ActionSpace::Discrete }
}
```

### 5.2 Public API split

The discrete and continuous flows are surfaced as **separate methods**
on `PcActorCritic`. No runtime dispatch on a generic `step()`. Type
safety: the consumer's call site reflects the action space.

Discrete (compatible with v3.x — only the precondition changes):

```rust
impl<L: LinAlg> PcActorCritic<L> {
    /// Discrete-mode step. Same shape as v3.x `step_masked`.
    /// **Precondition:** `config.action_space == ActionSpace::Discrete`.
    /// Violation returns `PcError::ConfigValidation`.
    pub fn step_masked(
        &mut self,
        state: &[f64],
        valid_actions: &[usize],
        reward: f64,
        done: bool,
    ) -> Result<usize, PcError> { /* unchanged behavior */ }
}
```

Continuous (new in v4.0.0):

```rust
impl<L: LinAlg> PcActorCritic<L> {
    /// Continuous-mode step. Returns the sampled action vector
    /// `a = μ(s) + σ·ε` of length `actor.output_size`.
    /// **Precondition:** `config.action_space == ActionSpace::Continuous`.
    /// Violation returns `PcError::ConfigValidation`.
    pub fn step_continuous(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<Vec<f64>, PcError>;

    /// Same as `step_continuous` but returns the device-native vector
    /// without host transfer. Zero-copy on backends where the actor
    /// computes natively in device memory (GpuLinAlg). Identical
    /// behavior to `step_continuous` on CpuLinAlg.
    pub fn step_continuous_raw_device(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<L::Vector, PcError>;

    /// Inference-only continuous output (no gradient, no learning).
    /// Returns the policy mean μ(s); sampling is the consumer's choice
    /// when bypassing `step_continuous`.
    pub fn act_continuous(&mut self, state: &[f64]) -> Vec<f64>;
}
```

The deprecated `step` (without mask) is removed in v4.0.0 — its
discrete-only semantics conflict with the multi-mode design. Migration
path: replace `step` with `step_masked` (already available since
v2.0.0).

### 5.3 Policy gradient — the only gradient-path change

The gradient direction is the **only** algorithmic change between
modes. Magnitude (scale) is unchanged: `effective_actor_scale_for_mode`
already gates by hysteresis, replay-mode opt-in, and surprise.

**Discrete (REINFORCE — unchanged from v3.x):**
```
pi      = softmax(y_conv / temperature)
delta   = pi
delta[action_taken] -= 1.0
delta  *= advantage    // advantage = td_error
```

**Continuous (Gaussian log-likelihood gradient — new):**
```
mu      = output_activation(pre_activation / temperature)        // = y_conv
a       = mu + sigma * eps                                       // sampled, eps ~ N(0, I)
delta_mu = (a - mu) / sigma^2                                    // d log pi(a|s) / d mu
delta   = delta_mu * advantage                                   // pre-multiply by td_error
// delta then propagates through output activation derivative as usual.
```

In code, the dispatch lives inside `learn_continuous_inner`:

```rust
let delta = match self.config.action_space {
    ActionSpace::Discrete => {
        // existing v3.x code
        let mut d = pi.clone();
        d[step.action_taken] -= 1.0;
        d
    }
    ActionSpace::Continuous => {
        // new: (a - mu) / sigma^2
        let mut d = vec![0.0; self.actor.output_size];
        for j in 0..d.len() {
            d[j] = (step.action_continuous[j] - mu[j]) / (self.config.policy_sigma.powi(2));
        }
        d
    }
};
// Common path beyond this point — magnitude and propagation unchanged.
let scaled_delta = delta_scaled_by_advantage_and_actor_scale(delta, advantage, mode);
self.actor.update_with_decay(...);
```

The `LearnStep` bundle (introduced v2.0.0, extended v2.2.0) gains an
action variant to carry both forms:

```rust
pub(crate) enum StepAction<'a> {
    Discrete { action: usize, valid_actions: &'a [usize] },
    Continuous { action: &'a [f64] },
}

pub(crate) struct LearnStep<'a, L: LinAlg> {
    pub state: &'a [f64],
    pub infer: InferResult<L>,
    pub action: StepAction<'a>,    // <-- replaces `action: usize` + `valid_actions: &[usize]`
    pub reward: f64,
    pub next_state: &'a [f64],
    pub done: bool,
    pub mode: LearnMode,
}
```

### 5.4 Replay buffer schema migration

```rust
/// v4.0.0 — generic action variant. Replaces `action: usize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Discrete(usize),
    Continuous(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTransition {
    pub state: Vec<f64>,
    pub action: Action,                      // <-- v4 change
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub valid_actions: Option<Vec<usize>>,   // <-- Some(...) only for Action::Discrete
}
```

The `valid_actions` field becomes `Option<Vec<usize>>` because it
genuinely doesn't apply in the continuous case. Validation rule:
`action == Action::Discrete(_) <=> valid_actions.is_some()`. Mismatch
rejected at `replay_buffer.push` time with `PcError::ConfigValidation`.

### 5.5 Distillation interaction (Polyak / Frozen anchors)

The KL-distillation regularizers (`distillation_lambda_polyak`,
`distillation_lambda_frozen`) presuppose a discrete categorical
distribution. They have **no natural definition** for raw continuous
output without committing to a probabilistic interpretation.

**v4.0.0 decision:** distillation regularizers are **disabled in
continuous mode**. Validation:

```rust
fn validate_config(config: &PcActorCriticConfig) -> Result<(), PcError> {
    // ... existing rules ...
    if config.action_space == ActionSpace::Continuous {
        if config.distillation_lambda_polyak > 0.0 {
            return Err(PcError::ConfigValidation(
                "distillation_lambda_polyak is not supported in continuous \
                 action space (KL is undefined for raw output). Set to 0.0 \
                 or use ActionSpace::Discrete. L2-anchored continuous \
                 distillation is tracked for v5.0.0."
                    .into(),
            ));
        }
        if config.distillation_lambda_frozen > 0.0 {
            return Err(PcError::ConfigValidation(
                "distillation_lambda_frozen is not supported in continuous \
                 action space — same reason as Polyak. Tracked for v5.0.0."
                    .into(),
            ));
        }
    }
    Ok(())
}
```

This is a strict-validation gate: continuous + nonzero distillation is
rejected at construction, not silently zeroed. Future work (v5.0.0+):
L2-anchored continuous distillation (replace KL with
`||μ_live - μ_anchor||²`).

### 5.6 EWC / Fisher diagonal in continuous mode

The Fisher diagonal is `E[g_raw²]`. v4.0.0 sources `g_raw` from the
mode-appropriate policy gradient (REINFORCE for discrete, Gaussian
log-likelihood for continuous). The Fisher lifecycle (accumulate,
decay, snapshot) is **identical** across modes — only the source of
`g_raw` changes.

No public API impact. The `is_online` gate remains; no replay
contamination of Fisher EMA.

### 5.7 Inference path & PC convergence

Unchanged. The actor's iterative bottom-up/top-down inference loop
produces `y_conv` regardless of action space. The critic still
receives `latent_concat` (concatenation of all hidden activations) +
state.

### 5.8 Hysteresis & cross-wake

Unchanged. Hysteresis is driven by PC surprise (RMS prediction error
across layers) — independent of how `y_conv` is interpreted
downstream.

### 5.9 What does NOT change

- PC inference loop semantics
- Critic architecture (still V(s), MLP)
- Surprise scoring & buffer
- Continuous learning M1-M4 (scale range, hysteresis, layer decay, EWC)
- CCA crossover (operates on weight matrices)
- TD(n) n-step return computation
- GAE eligibility traces
- Cross-wake coupling thresholds & state-machine transitions
- Replay buffer dual-compartment FIFO + positive-reward filter

---

## 6. Configuration changes (`PcActorCriticConfig`)

Two new fields, both `#[serde(default)]` to preserve v3.x save-file
compatibility:

```rust
pub struct PcActorCriticConfig {
    // ... v3.0.0 fields unchanged ...

    /// v4.0.0 — action space discriminator. Default `Discrete` preserves
    /// v3.x behavior for save files written before this field existed.
    #[serde(default)]
    pub action_space: ActionSpace,

    /// v4.0.0 — Gaussian policy std-dev for continuous mode. Ignored when
    /// `action_space == Discrete`. Must be `> 0.0 && finite`. Default 0.1.
    #[serde(default = "default_policy_sigma")]
    pub policy_sigma: f64,
}

fn default_policy_sigma() -> f64 { 0.1 }
```

Validation rules (added to `validate_config`):

| Rule | Error |
|------|-------|
| `action_space == Continuous` && `policy_sigma <= 0.0` or NaN/Inf | `policy_sigma must be > 0 and finite when action_space is Continuous` |
| `action_space == Continuous` && `distillation_lambda_polyak > 0.0` | rejection per §5.5 |
| `action_space == Continuous` && `distillation_lambda_frozen > 0.0` | rejection per §5.5 |
| `action_space == Continuous` && `entropy_coeff > 0.0` | **WARN-only** (no rejection): documented as inert, no runtime cost |

The entropy_coeff warning is logged once per agent construction via
`tracing::warn!` (or stderr if no tracing configured) — does not block
construction.

---

## 7. Public API surface — final contract

### 7.1 New methods (continuous mode)

```rust
pub fn step_continuous(&mut self, state: &[f64], reward: f64, done: bool)
    -> Result<Vec<f64>, PcError>;

pub fn step_continuous_raw_device(&mut self, state: &[f64], reward: f64, done: bool)
    -> Result<L::Vector, PcError>;

pub fn act_continuous(&mut self, state: &[f64]) -> Vec<f64>;
```

### 7.2 Modified methods (discrete mode — existing API)

```rust
// Precondition tightened: action_space must be Discrete.
// Bit-equivalent behavior to v3.x when precondition holds.
pub fn step_masked(&mut self, state, valid, reward, done)
    -> Result<usize, PcError>;

pub fn act(&mut self, state, valid_actions, mode: SelectionMode)
    -> Result<(usize, InferResult<L>), PcError>;     // <-- now Result
```

### 7.3 Removed methods

```rust
// REMOVED in v4.0.0 — was deprecated since v2.0.0.
// Migration: use step_masked with full action set.
pub fn step(&mut self, state, reward, terminal) -> usize;
```

### 7.4 Replay & learn methods

```rust
pub fn replay_learn(&mut self, batch_size: usize) -> Result<(), PcError>;
// Unchanged signature; internal dispatch on transition variant.
```

---

## 8. Migration path (v3.x → v4.0.0)

### 8.1 Discrete consumers (default)

**No code change required** if the consumer:
- never set `action_space` (defaults to `Discrete`),
- uses `step_masked` (not the removed `step`),
- treats `act()` return as `(usize, InferResult)` (now `Result<...>`).

The only call-site change is unwrapping the new `Result` on `act()`:

```rust
// v3.x
let (action, infer) = agent.act(&state, &valid, SelectionMode::Play);

// v4.0.0
let (action, infer) = agent.act(&state, &valid, SelectionMode::Play)?;
```

### 8.2 Removing deprecated `step()`

```rust
// v3.x (deprecated, removed in v4)
let action = agent.step(&state, reward, done);

// v4.0.0
let valid: Vec<usize> = (0..agent.actor.output_size).collect();
let action = agent.step_masked(&state, &valid, reward, done)?;
```

### 8.3 Adopting continuous mode

```rust
let mut config = my_existing_v3_config;
config.action_space = ActionSpace::Continuous;
config.policy_sigma = 0.1;
config.distillation_lambda_polyak = 0.0;   // required in continuous
config.distillation_lambda_frozen = 0.0;   // required in continuous
config.actor.output_activation = Activation::Tanh;  // recommended for bounded actions

let mut agent = PcActorCritic::new(backend, config, seed)?;

loop {
    let action = agent.step_continuous(&state, reward, done)?;
    let next_state = env.apply(&action);
    state = next_state;
}
```

### 8.4 Replay save-file compatibility

v3.x saves with `replay_buffer` populated use the legacy
`action: usize` schema. v4.0.0 deserialization auto-converts to
`Action::Discrete(usize)` via custom serde implementation:

```rust
impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        // Try untagged v4 form first (Discrete | Continuous variant).
        // Fall back to bare usize for v3.x compatibility.
        // ... custom impl ...
    }
}
```

A serializer round-trip test locks this on a frozen v3.x JSON fixture.

---

## 9. Implementation phases (SBTDD)

This section is the input for `/deep-plan`. Five phases, lockstep
RED → GREEN → REFACTOR cycles per CLAUDE.local.md §1.

### Phase 1 — `ActionSpace` enum + config validation

**Scope:** Add the enum and `policy_sigma` to `PcActorCriticConfig`.
Add validation rules (continuous + sigma, continuous + distillation,
continuous + entropy-warn). No behavior change yet — `action_space`
isn't consulted by any production path.

**Tests:** ~6 RED tests covering all validation rules + sentinel
defaults + serde backward compat.

### Phase 2 — `Action` enum + `ReplayTransition` schema migration

**Scope:** Replace `ReplayTransition::action: usize` with
`Action::Discrete | Continuous`. Custom serde Deserialize for legacy
support. `valid_actions` becomes `Option<Vec<usize>>`.

**Tests:** ~5 RED tests. Includes a frozen v3.x JSON fixture
deserialization round-trip (parallel to v3.0.0
`test_pre_v3_json_loads_with_default_critic_floor_replay`).

### Phase 3 — `LearnStep::action` variant + gradient dispatch

**Scope:** Replace `LearnStep::action: usize` + `valid_actions: &[usize]`
with `StepAction` enum. Add the Gaussian-policy gradient branch in
`learn_continuous_inner`. Discrete branch preserved bit-for-bit.

**Tests:** ~8 RED tests. Includes a regression-equivalence test:
v3.x discrete step_masked vs v4.0.0 discrete step_masked must produce
bit-identical weight trajectories on a fixed seed.

### Phase 4 — `step_continuous` / `step_continuous_raw_device` /
`act_continuous` public API

**Scope:** Wire the new public methods. Sample `a = μ + σ·ε` using the
agent's RNG (deterministic under seed). Pre-condition checks
(`action_space` must match the called API).

**Tests:** ~7 RED tests. Includes:
- Continuous step on a 4-D action space sanity test.
- Determinism under fixed seed.
- Gradient direction correctness (perturb action away from μ, check
  gradient pulls back proportional to advantage).
- Pre-condition violation: `step_continuous` on Discrete agent rejected.

### Phase 5 — Documentation & migration guide

**Scope:** CHANGELOG `[4.0.0]` BREAKING section, README migration
subsection (paired with v3 migration), rustdoc on all new public API.

**Tests:** None new. Existing 21 doctests must remain green; new
doctests for `step_continuous` / `act_continuous` / `policy_sigma`.

---

## 10. Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|---------|------------|------------|
| R1 | Continuous gradient direction wrong (sign, scale) | **High** | Medium | Phase 3 gradient-direction test on a hand-derivable 1-D example before any 2+ D test |
| R2 | Replay schema migration breaks legacy save files | High | Low | Frozen v3.x fixture round-trip test in Phase 2; custom Deserialize fallback |
| R3 | Distillation rejection breaks v3.x consumers who set `lambda_polyak > 0` and later flip to Continuous | Medium | Low | Validation error message points to the workaround (set lambda to 0); migration guide explicit |
| R4 | RNG sampling cost dominates step latency in continuous mode | Low | Medium | `policy_sigma * randn` is O(output_dim) per step — negligible for typical 4-32 output sizes; revisit if benchmarks show >5% overhead |
| R5 | Fisher diagonal in continuous shows different convergence vs discrete (different `g_raw²` distribution) | Low | High (expected) | Document that continuous Fisher numbers are not directly comparable to discrete; existing Fisher tests cover the lifecycle, not absolute values |
| R6 | `policy_sigma` is a fixed scalar; multi-dim actions with different scales (e.g. wheel torque + steering angle) get the same exploration noise | Medium | Medium | Document; recommend output normalization in the consumer's network. Per-dim sigma deferred to v5.0.0 (same scope as learned σ) |
| R7 | GpuLinAlg interaction with `step_continuous_raw_device` adds host-device coupling | Medium | Low | Q1 decision (§4.1) carves the escape hatch deliberately; re-validate after GpuLinAlg lands |
| R8 | Removal of `step()` (deprecated since v2.0.0) catches consumers who never migrated | Low | Low | CHANGELOG migration table shows the trivial replacement; deprecation warning has been on for 2+ major versions |

---

## 11. Out of scope (v5.0.0+)

- **Hybrid action spaces** (Q4 from v1 spec): per-group action mode
  dispatch. Tracked.
- **Learned σ** (state-dependent variance): doubles output dim or adds
  variance head. Tracked.
- **Per-dim σ vector**: anisotropic exploration. Tracked.
- **L2-anchored continuous distillation**: KL replacement for
  Polyak/Frozen anchors. Tracked.
- **Multi-discrete grouping** (Q4 partial): output partitioned into
  argmax groups with shared backbone. Tracked.
- **Action-conditional critic** (Q(s,a) instead of V(s)): enables true
  DPG / DDPG. Major surgery; tracked.
- **Continuous-mode entropy regularizer** (output variance, KL to
  prior): only relevant once learned σ exists.

---

## 12. SemVer rationale

v4.0.0 is a major bump per SemVer 2.0.0 because:

1. **Removed public method:** `step()` (deprecated since v2.0.0) is
   removed. Any consumer still calling it gets a compile error.
2. **Changed return type:** `act()` returns `Result<...>` instead of
   `(usize, InferResult<L>)`. Pattern-match call sites must add `?`.
3. **Changed replay schema:** `ReplayTransition::action: usize` becomes
   `action: Action`. Save files are deserialization-compatible (custom
   Deserialize handles the fallback), but any consumer that
   pattern-matches on the field type breaks.
4. **Tightened preconditions:** existing `step_masked` now returns an
   error if `action_space != Discrete`. Behavior change in unchanged
   code if a v3.x consumer flips `action_space` to `Continuous` without
   migrating their step calls.

The same Cargo-resolves-major-boundary safety-net rule that bounded
v3.0.0's blast radius applies here: downstream `pc-rl-core = "3"` users
do **not** auto-receive v4. They must update their `Cargo.toml` to
`pc-rl-core = "4"` to opt into the breaking change.

---

## 13. Coordination with GpuLinAlg Phase 2

The two next-major-version workstreams interact at exactly two points:

### 13.1 `step_continuous_raw_device` return type

**Decision (binding):** the method exists on the public API regardless
of which workstream lands first. Body:

- If GpuLinAlg lands first (most likely path per project memory):
  `step_continuous_raw_device` returns the actor's native
  `L::Vector` directly. CpuLinAlg's `Vector = Vec<f64>` makes the
  method bit-equivalent to `step_continuous`. GpuLinAlg's
  `Vector = CudaSlice<f64>` makes it zero-copy device-side.
- If v4.0.0 lands first: the method returns `L::Vector` against the
  current CpuLinAlg trait — no behavior change for existing
  consumers, just a new escape hatch that's bit-equivalent to
  `step_continuous` on CpuLinAlg until GpuLinAlg implements `Vector`
  natively.

### 13.2 Sampling RNG locality

`a = μ + σ·ε` requires generating `ε ~ N(0, I)`. The RNG is host-side
(`rand::StdRng` per the existing actor). On GpuLinAlg, the noise is
generated on the host and uploaded to device for the addition.
Acceptable for typical output sizes (4-64 dims); a future optimization
could use a device-side RNG kernel (cuRAND) — out of scope for v4.0.0.

### 13.3 Recommended sequencing

**GpuLinAlg Phase 2 first**, then v4.0.0 generic action space:

- GpuLinAlg crystallizes the `L::Vector` semantics (size, alignment,
  device locality) before v4.0.0 commits to `step_continuous_raw_device`.
- v4.0.0's continuous-mode test suite gets to exercise both backends
  on day one — discovers backend-specific gradient bugs early.
- Conversely if v4.0.0 lands first, GpuLinAlg Phase 2 needs to add the
  continuous-mode kernels alongside the existing CPU port — wider
  scope per single SBTDD plan.

The user's stated goal — *generic action space is the next implementation
priority* — is honored by adopting this spec as the v4.0.0 SBTDD input
**now**, while explicitly sequencing the implementation order
GpuLinAlg → v4.0.0.

---

## 14. Validation gates

Per `CLAUDE.local.md` §0.2, every commit must pass:

```bash
cargo test                              # All pass
cargo clippy --tests -- -D warnings     # 0 warnings
cargo fmt --check                       # Clean
cargo build --release                   # No warnings
cargo doc --no-deps                     # No warnings
cargo audit                             # No vulnerabilities
```

Plus, at the end of Phase 4:

```bash
cargo nextest run --run-ignored only -E 'test(phase2_stress_scenario_rollback_recovery)'
# Smoke gate must remain green — generic action space must not regress
# v2.2.0 stress recovery.
```

---

## 15. Pre-merge review (two-loop)

Per the v2.2.1 / v3.0.0 precedent:

- **Loop 1:** `superpowers:requesting-code-review` against `BASE..HEAD`
  diff. Threshold: clean-to-go.
- **Loop 2:** `magi:magi` code-review of the same diff. Threshold:
  GO WITH CAVEATS or higher. Conditions applied per CLAUDE.local.md §6.

Caspar (Critic) will likely emphasize:
- Replay schema migration completeness (frozen-fixture round-trip).
- Removal of `step()` blast radius (downstream consumers).
- Continuous gradient sign correctness — request hand-derivable
  1-D test.

---

## 16. Summary

| Metric | Value |
|--------|-------|
| Total LOC code estimate | ~250 (config + ActionSpace + Action enum + dispatch + new public methods) |
| Total LOC tests estimate | ~600 (5 phases × ~6 tests average + serde fixtures) |
| Total LOC docs estimate | ~150 (CHANGELOG + README + rustdoc) |
| Phases | 5 (config → replay → gradient → public API → docs) |
| Open architectural questions | 0 (all 5 from v1 spec resolved in §4) |
| Coordination dependencies | GpuLinAlg Phase 2 (recommended to land first) |
| Release | v4.0.0 BREAKING per SemVer 2.0.0 |
| Migration cost (typical discrete consumer) | 1-2 line changes (`act()` `?`, swap `step()` → `step_masked()`) |
| Migration cost (continuous adopter) | new code path; no removal needed |
