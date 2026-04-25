# Generic Action Space — v4.0.0 Specification

**Version:** 2.0 (rewrite against v3.0.0 baseline)
**Date:** 2026-04-25
**Author:** Julian Bolivar
**Crate:** `pc-rl-core`
**Status:** Spec-ready for SBTDD `/deep-plan`. Target release: **v4.0.0 (BREAKING)**.
**Predecessor spec:** v1.0 (2026-XX), preserved at git tag `pre-v4-spec` for context.
**Coordination:** v4.0.0 is **CPU-first** — must function correctly on
`CpuLinAlg` with no GPU dependencies. GpuLinAlg Phase 2 is a separate
enhancement workstream that lands **after** v4.0.0 and broadens the
backend, not the action-space contract. See §13 for the (small) interaction
surface.

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
hatch returning `L::Vector` for forward-compatibility with future GPU
backends.**

Rationale:
- v4.0.0 is **CPU-first**. The decision must work cleanly on
  `CpuLinAlg` with zero forward-references to backends that don't
  exist yet.
- 99% of downstream consumers want host data (write to file, send over
  network, multiply by Python tensor, log to console). `Vec<f64>` is
  the right ergonomic default.
- The `_raw_device` variant exists in the v4.0.0 API surface so that
  when GpuLinAlg eventually lands, no API surgery is needed — just an
  efficient implementation. On `CpuLinAlg`, `L::Vector = Vec<f64>`,
  so `_raw_device` is bit-equivalent to the regular variant. **No GPU
  is required for v4.0.0 implementation, testing, or release.**
- A blanket `L::Vector` return would force every CPU consumer to call
  `vec_to_vec` for a no-op clone — over-engineering for the dominant
  use case.

The escape hatch is forward-looking, not blocking. It can also be
**deferred to a v4.x minor release** if Phase 4 of the implementation
plan finds it adds non-trivial complexity to the CPU-only path.

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

**Brainstorm decision (Q5+Q7):** `act_continuous` mirrors the discrete
`act` API exactly, taking a `SelectionMode` parameter. `Play` returns
deterministic μ (no RNG advance); `Training` returns sampled
`μ + σ·ε` (RNG advances).

**Brainstorm decision (Q6):** every entry-point method does a
runtime hard-error precondition check on `config.action_space`. Cost
is ~1ns (one enum comparison) per call — strictly trivial vs the PC
inference loop. Aligns with the CLAUDE.md global "no silent failures"
rule. Symmetric across `step_masked`, `step_continuous`, `act`, and
`act_continuous`.

Discrete (compatible with v3.x — only the precondition changes):

```rust
impl<L: LinAlg> PcActorCritic<L> {
    /// Discrete-mode step. Same shape as v3.x `step_masked`.
    /// **Precondition:** `config.action_space == ActionSpace::Discrete`.
    /// Returns `PcError::ConfigValidation` on mismatch (loud, not silent).
    pub fn step_masked(
        &mut self,
        state: &[f64],
        valid_actions: &[usize],
        reward: f64,
        done: bool,
    ) -> Result<usize, PcError> { /* unchanged behavior + precondition guard */ }

    /// Discrete-mode inference. Same shape as v3.x `act`.
    /// **Precondition:** `config.action_space == ActionSpace::Discrete`.
    pub fn act(
        &mut self,
        state: &[f64],
        valid_actions: &[usize],
        mode: SelectionMode,
    ) -> Result<(usize, InferResult<L>), PcError>;
}
```

Continuous (new in v4.0.0):

```rust
impl<L: LinAlg> PcActorCritic<L> {
    /// Continuous-mode step. Returns the sampled action vector
    /// `a = μ(s) + σ·ε` of length `actor.output_size`.
    /// **Precondition:** `config.action_space == ActionSpace::Continuous`.
    /// Always samples (Training mode is implicit — `step_*` paths are
    /// learning paths, never deterministic).
    pub fn step_continuous(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<Vec<f64>, PcError>;

    /// Same as `step_continuous` but returns the device-native vector
    /// without host transfer. Forward-compat hook for GpuLinAlg
    /// (Phase 2). On `CpuLinAlg`, `L::Vector = Vec<f64>` so this is
    /// bit-equivalent to `step_continuous`. May be deferred to v4.x
    /// minor if it complicates the CPU-only path during implementation.
    pub fn step_continuous_raw_device(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<L::Vector, PcError>;

    /// Inference-only continuous output. Mirrors discrete `act` ergonomics.
    /// `Play` mode returns the policy mean μ(s) deterministically (no
    /// RNG advance). `Training` mode returns `μ + σ·ε` (RNG advances).
    /// Returns `(action_vec, InferResult<L>)` paralleling discrete `act`.
    /// **Precondition:** `config.action_space == ActionSpace::Continuous`.
    pub fn act_continuous(
        &mut self,
        state: &[f64],
        mode: SelectionMode,
    ) -> Result<(Vec<f64>, InferResult<L>), PcError>;
}
```

The deprecated `step` (without mask) is removed in v4.0.0 — its
discrete-only semantics conflict with the multi-mode design. Migration
path: replace `step` with `step_masked` (already available since
v2.0.0).

The deprecated `learn(trajectory)` is **kept** in v4.0.0 with a
Discrete-only validation guard (per brainstorm Q7 / item 7). Continuous
trajectory learning is not supported in v4.0.0; consumers using
continuous mode must use `step_continuous` in a loop. `learn()` and
`TrajectoryStep` are scheduled for proper audit-and-removal in v5.0.0.
Note that `learn()` return type bumps to `Result<f64, PcError>` (was
`f64`) to surface the new precondition error — a sub-breaking change
already absorbed by v4.0.0 BREAKING.

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

**Brainstorm decisions (Q1, Q2, Q8):**

- **Q1 — `Action` enum is `#[serde(untagged)]`.** JSON token-type
  unambiguously discriminates: integer → `Discrete(usize)`, array →
  `Continuous(Vec<f64>)`. v3.x JSON `"action": 5` deserializes to
  `Action::Discrete(5)` automatically without any custom impl.
- **Q2 — Cross-mode contamination is rejected at `push` time.** Buffer
  stores its `action_space` (or derives from config); pushing an
  `Action::Continuous(_)` into a `Discrete`-configured buffer (or vice
  versa) returns `PcError::ConfigValidation`. `load_agent` extends the
  same validation — buffer schema must match the loaded config's
  `action_space`, else load fails loudly.
- **Q8 — `valid_actions: Option<Vec<usize>>` uses plain
  `#[serde(default)]`.** No custom `deserialize_with` needed; serde
  automatically wraps v3.x bare `[0, 1, 2]` into `Some([0, 1, 2])` via
  Some-elision. Validation enforces the `is_some() iff Discrete`
  invariant in `push`.

```rust
/// v4.0.0 — generic action variant. Replaces `action: usize`.
/// `#[serde(untagged)]` is binding per brainstorm Q1; do not change
/// without a fresh ABI migration plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Action {
    /// JSON form: bare integer (e.g. `5`). Compatible with v3.x.
    Discrete(usize),
    /// JSON form: bare array of f64 (e.g. `[0.1, 0.7, -0.3]`).
    Continuous(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTransition {
    pub state: Vec<f64>,
    pub action: Action,                      // <-- v4 change
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    /// Some(mask) iff Discrete; None for Continuous (mask concept N/A).
    /// `#[serde(default)]` covers all three v3.x → v4 migration cases:
    ///   * legacy bare `[0,1,2]` → `Some([0,1,2])` via serde Some-elision
    ///   * v4 explicit `null`     → `None`
    ///   * v4 absent (continuous) → `None` via default
    /// **Brainstorm Q8:** no custom `deserialize_with` is needed — empirically
    /// verified that serde's Some-elision handles the legacy form natively.
    #[serde(default)]
    pub valid_actions: Option<Vec<usize>>,
}
```

Validation rule (enforced at `replay_buffer.push`):

```rust
match (&transition.action, &transition.valid_actions) {
    (Action::Discrete(_),   Some(_)) => Ok(()),
    (Action::Continuous(_), None)    => Ok(()),
    _ => Err(PcError::ConfigValidation(
        "valid_actions must be Some for Discrete and None for Continuous".into(),
    )),
}
```

Cross-mode buffer protection (per brainstorm Q2): `replay_buffer.push`
also validates `transition.action` matches the buffer's
`action_space`, derived from the agent's `config.action_space` at
construction. `load_agent` validates `loaded_buffer.action_space ==
loaded_config.action_space`; mismatch returns
`PcError::ConfigValidation` with explicit guidance to either restore
matching config or clear the buffer before reload.

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
                 or use ActionSpace::Discrete."
                    .into(),
            ));
        }
        if config.distillation_lambda_frozen > 0.0 {
            return Err(PcError::ConfigValidation(
                "distillation_lambda_frozen is not supported in continuous \
                 action space — same reason as Polyak."
                    .into(),
            ));
        }
    }
    Ok(())
}
```

This is a strict-validation gate: continuous + nonzero distillation is
rejected at construction, not silently zeroed.

**On L2-anchored continuous distillation:** an alternative formulation
replacing KL with `||μ_live − μ_anchor||²` is conceptually plausible but
**experimental**. There is no settled theory for how an L2 anchor on
the actor's mean interacts with PC inference convergence, hysteresis
state-machine signals, or the self-recovery toolkit semantics. It is
**not on the v5.0.0 roadmap**. If/when explored, the work belongs on
an experimental branch (e.g. `experimental/continuous-l2-distillation`)
with its own empirical validation cycle before any release commitment.

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

Hysteresis state-machine **mechanism** is unchanged. Both signals are
action-space agnostic:
- **Actor signal:** PC surprise score (RMS prediction error across PC
  layers) — depends only on inference convergence dynamics, not on
  output interpretation.
- **Critic signal:** `|td_error|` magnitude — reward-space, not
  action-space.

**Brainstorm Q4 — continuous-mode operational note:** The empirical
distribution of `td_error` may shift in continuous mode because the
Gaussian-policy gradient form changes weight-update dynamics. The
`adaptive_surprise = true` default (introduced in v2.1.0) recalibrates
the FROZEN/PLASTIC thresholds dynamically from the observed
`td_error_buffer` (mean ± std), absorbing this distribution shift
without consumer intervention.

If hysteresis behaves unexpectedly under continuous (frequent
oscillation, never freezing, never thawing):

1. Verify `adaptive_surprise = true` (v3.x default; check that you
   haven't disabled it).
2. If still problematic, manually tune `surprise_low` /
   `surprise_high` against observed `td_error` magnitudes.
3. As a last resort, disable hysteresis (`actor_hysteresis = false` /
   `critic_hysteresis = false`) — recall this also disables the v3.0.0
   self-recovery toolkit.

A smoke test in Phase 4 verifies that under
`action_space = Continuous` + `adaptive_surprise = true` +
`actor_hysteresis = true`, at least one FROZEN ↔ PLASTIC transition is
observed within 200 step calls (confirms the state machine activates
under continuous-mode dynamics).

Cross-wake coupling thresholds and force-transition logic are
unchanged.

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

### 5.10 Continuous-mode operational tuning (mitigation matrix)

Per brainstorm Q3, GAE / TD(n) machinery works unchanged in continuous
mode. The Gaussian-policy gradient `(a − μ) / σ²` has magnitude
~`1/σ²` per output dim relative to the discrete REINFORCE form;
WEIGHT_CLIP=5.0 and GRAD_CLIP=5.0 envelope the typical operating
range for `σ ∈ [0.05, 1.0]`.

If consumers observe pathological behavior in continuous mode, the
following mitigations are recommended in priority order:

| Symptom | Primary mitigation | Secondary | Tertiary |
|---|---|---|---|
| WEIGHT_CLIP saturation observed | ↑ `policy_sigma` (smaller gradient magnitude) | ↓ `lr_weights` | ↓ `td_steps` or ↓ GAE `λ` (faster trace decay) |
| Slow learning / weights barely moving | ↓ `policy_sigma` (larger gradient signal) | ↑ `lr_weights` | Verify rewards are in `[-1, 1]` range |
| Hysteresis oscillating (frequent FROZEN ↔ PLASTIC) | Confirm `adaptive_surprise = true` | Manually retune `surprise_low` / `surprise_high` | Disable hysteresis if exploring |
| Hysteresis never freezing | Confirm `adaptive_surprise = true`; check `surprise_low` not too low | ↑ `actor_sleep_fraction` / `critic_sleep_fraction` | — |
| Polyak/Frozen distillation rejected at construction | Set `distillation_lambda_polyak = 0.0` and `distillation_lambda_frozen = 0.0` (continuous mode requirement, §5.5) | — | — |

Telemetry for empirical diagnosis (already exposed in v3.x):
- `replay_clamp_count` — increments when the critic's td_error clamp
  fires; sustained increase signals magnitude problems.
- `actor_frozen_steps` / `critic_frozen_steps` — visibility into
  hysteresis duty cycle.
- `td_error_buffer` — adaptive surprise reads this; a scripted dump
  reveals the magnitude distribution.

The CHANGELOG `[4.0.0]` entry must include this mitigation matrix
verbatim — consumers adopting continuous mode need it as a first-line
reference.

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

// Per brainstorm Q5+Q7: mirrors discrete `act` with SelectionMode.
// Play  → returns μ(s) deterministically (no RNG advance).
// Train → returns μ + σ·ε (RNG advances).
pub fn act_continuous(&mut self, state: &[f64], mode: SelectionMode)
    -> Result<(Vec<f64>, InferResult<L>), PcError>;
```

### 7.2 Modified methods (discrete mode — existing API)

```rust
// Precondition tightened (Q6): runtime hard-error if action_space != Discrete.
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
// Buffer push validates Action variant matches buffer's action_space (Q2).

// Per brainstorm Q7 / item 7: kept as deprecated, Discrete-only.
// Return type bumps to Result<f64, PcError> for the precondition error.
#[deprecated(since = "2.1.0", note = "use step_masked in a loop")]
pub fn learn(&mut self, trajectory: &[TrajectoryStep<L>]) -> Result<f64, PcError>;
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
`action: usize` schema. v4.0.0 auto-converts via the `#[serde(untagged)]`
discriminator on `Action` (Q1 brainstorm decision):

```rust
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum Action {
    Discrete(usize),       // matches JSON integer (v3.x bare form)
    Continuous(Vec<f64>),  // matches JSON array of floats (v4 only)
}
```

No custom `Deserialize` impl is needed — JSON token-type
(`Number` vs `Array`) discriminates unambiguously between the two
variants. A serializer round-trip test in Phase 2 locks this against
a frozen v3.x JSON fixture (mirroring the v3.0.0 precedent
`test_pre_v3_json_loads_with_default_critic_floor_replay`).

The `valid_actions` field (now `Option<Vec<usize>>`) uses plain
`#[serde(default)]` (Q8 brainstorm decision); serde's Some-elision
wraps the legacy bare `[0, 1, 2]` form into `Some([0, 1, 2])`
automatically — no custom `deserialize_with` needed.

---

## 9. Implementation phases (SBTDD)

This section is the input for `/deep-plan`. Five phases, lockstep
RED → GREEN → REFACTOR cycles per CLAUDE.local.md §1.

### Phase 1 — `ActionSpace` enum + config validation + symmetric preconditions

**Scope:** Add `ActionSpace` enum and `policy_sigma` to
`PcActorCriticConfig`. Validation rules:
- continuous + `policy_sigma <= 0.0` or NaN/Inf → reject
- continuous + `distillation_lambda_polyak > 0.0` → reject
- continuous + `distillation_lambda_frozen > 0.0` → reject
- continuous + `entropy_coeff > 0.0` → warn-only (no rejection)

Add the symmetric runtime precondition guard in the four entry points
(per Q6): `step_masked`, `step_continuous`, `act`, `act_continuous`.
The guards return `PcError::ConfigValidation` if `action_space`
doesn't match the called method. `step_continuous` and
`act_continuous` are stubs that return the error in this phase
(real implementation lands Phase 4).

**Tests:** ~8 RED tests covering all validation rules + sentinel
defaults + serde backward compat + the four precondition guards.

### Phase 2 — `Action` enum + `ReplayTransition` schema migration

**Scope:** Replace `ReplayTransition::action: usize` with
`#[serde(untagged)] Action::Discrete | Continuous`. Add the
cross-mode `push` validation (Q2): the buffer rejects transitions
whose `Action` variant doesn't match the buffer's `action_space`,
and rejects transitions where `valid_actions.is_some() != is_discrete`.
`load_agent` validates buffer-config consistency on load. No custom
`Deserialize` impl needed — untagged + Some-elision suffice.

**Tests:** ~6 RED tests. Includes:
- A frozen v3.x JSON fixture deserialization round-trip (parallel to
  v3.0.0 `test_pre_v3_json_loads_with_default_critic_floor_replay`).
- Cross-mode `push` rejection (Discrete buffer + Continuous transition
  → error).
- `load_agent` rejection on mismatched buffer schema.
- Validation rule on `valid_actions` is_some() ↔ Discrete.

### Phase 3 — `LearnStep::action` variant + gradient dispatch

**Scope:** Replace `LearnStep::action: usize` + `valid_actions: &[usize]`
with `StepAction` enum. Add the Gaussian-policy gradient branch in
`learn_continuous_inner`. Discrete branch preserved bit-for-bit.

**Tests:** ~9 RED tests. Includes:
- A 1-D hand-derivable gradient-direction test (R1 mitigation):
  perturb `a` away from `μ`, check gradient direction is `(a − μ)/σ²`
  and weight update direction multiplies advantage correctly.
- Multi-dim (4-D) gradient direction sanity.
- Discrete bit-equivalence: v3.x seeded run vs v4.0.0 Discrete seeded
  run produces identical weight trajectories after N=100 steps.
- TD(n=5) + Continuous + σ=0.1: weights remain finite, no clip
  saturation > 5% over 100 steps (Q3 sanity).
- GAE(λ=0.95) + Continuous: trace decay observed, weights remain
  finite.

### Phase 4 — `step_continuous` / `step_continuous_raw_device` /
`act_continuous` public API + hysteresis smoke

**Scope:** Wire the public continuous methods. `step_continuous`
samples `a = μ + σ·ε` using the agent's RNG (deterministic under
seed). `act_continuous(state, mode)` returns `(action, InferResult)`:
Play=μ, Training=μ+σ·ε. The precondition guards from Phase 1 graduate
from stub-error to functional dispatch.

**Tests:** ~9 RED tests. Includes:
- Continuous step on a 4-D action space sanity test.
- Determinism under fixed seed (two identical runs → identical action
  sequences).
- `act_continuous(_, Play)` reproducibility (no RNG advance, two
  consecutive calls return identical μ).
- `act_continuous(_, Training)` divergence (two consecutive calls
  return different sampled actions).
- Hysteresis smoke (Q4): `action_space=Continuous` +
  `actor_hysteresis=true` + `adaptive_surprise=true` → at least one
  FROZEN ↔ PLASTIC transition observed within 200 step calls.
- WEIGHT_CLIP non-saturation (Q3): 100 continuous steps with σ=0.1,
  clip-binding count remains under 5% of total updates.

### Phase 5 — Documentation & migration guide

**Scope:** CHANGELOG `[4.0.0]` BREAKING section, README migration
subsection (paired with v3 migration), rustdoc on all new public API.

**Tests:** None new. Existing 21 doctests must remain green; new
doctests for `step_continuous` / `act_continuous` / `policy_sigma`.

---

## 10. Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|---------|------------|------------|
| R1 | Continuous gradient direction wrong (sign, scale) | **High** | Medium | Phase 3 hand-derivable 1-D gradient test before any multi-dim test |
| R2 | Replay schema migration breaks legacy save files | High | Low | Frozen v3.x fixture round-trip in Phase 2. Untagged enum (Q1) + Some-elision (Q8) — no custom impl needed; failure surface is minimal |
| R3 | Distillation rejection breaks v3.x consumers who set `lambda_polyak > 0` and later flip to Continuous | Medium | Low | Validation error message points to workaround (set lambda to 0); migration guide explicit. Self-recovery toolkit availability is documented as Discrete-only in v4.0.0 |
| R4 | RNG sampling cost dominates step latency in continuous mode | Low | Medium | `policy_sigma * randn` is O(output_dim) per step — negligible for typical 4-32 output sizes; revisit if benchmarks show >5% overhead |
| R5 | Fisher diagonal in continuous shows different convergence vs discrete (different `g_raw²` distribution) | Low | High (expected) | Document that continuous Fisher numbers are not directly comparable to discrete; existing Fisher tests cover the lifecycle, not absolute values |
| R6 | `policy_sigma` is a fixed scalar; multi-dim actions with different scales (e.g. wheel torque + steering angle) get the same exploration noise | Medium | Medium | Document; recommend output normalization in the consumer's network. Per-dim sigma deferred to v5.0.0 (same scope as learned σ) |
| R7 | GpuLinAlg interaction with `step_continuous_raw_device` adds host-device coupling | Low | Low | Q1 decision (§4.1) carves the escape hatch deliberately; CPU-first sequencing means GpuLinAlg arrives later and can validate against v4.0.0 contract |
| R8 | Removal of `step()` (deprecated since v2.0.0) catches consumers who never migrated | Low | Low | CHANGELOG migration table shows the trivial replacement; deprecation warning has been on for 2+ major versions |
| R9 | Hysteresis thresholds (`surprise_low`/`high`) tuned empirically for discrete may misbehave in continuous | Low | Medium | Brainstorm Q4 mitigation: `adaptive_surprise=true` (default) recalibrates thresholds dynamically. Phase 4 smoke test verifies state machine activates within 200 steps. Static-threshold consumers (`adaptive_surprise=false`) get a doc warning |
| R10 | Cross-mode buffer contamination silently corrupts gradients | High | Low | Brainstorm Q2: strict reject in `replay_buffer.push` + `load_agent` validation. Phase 2 tests cover both rejection paths |
| R11 | Consumer calls wrong entry-point method for their mode (e.g. `step_masked` on Continuous agent) | Medium | Medium | Brainstorm Q6: runtime hard-error on every call. ~1ns overhead. Loud failure with workaround in error message. Phase 1 tests cover all four guards |

---

## 11. Out of scope (v5.0.0+)

- **Hybrid action spaces** (Q4 from v1 spec): per-group action mode
  dispatch. Tracked.
- **Learned σ** (state-dependent variance): doubles output dim or adds
  variance head. Tracked.
- **Per-dim σ vector**: anisotropic exploration. Tracked.
- **L2-anchored continuous distillation**: KL replacement for
  Polyak/Frozen anchors. **Experimental — no release commitment.**
  If pursued, lives on an experimental branch with its own validation
  cycle. Self-recovery toolkit (`rollback_soft` / `rollback_hard` /
  `champion_update`) remains effectively unavailable in Continuous
  mode until and unless this experiment validates.
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
   `action: Action` (`#[serde(untagged)]` enum, Q1 brainstorm decision).
   v3.x save files deserialize automatically via JSON token-type
   discrimination — no custom impl needed — but any consumer that
   pattern-matches on the field type at compile time breaks.
   `valid_actions` becomes `Option<Vec<usize>>`; v3.x's bare
   `[0,1,2]` form auto-wraps to `Some([0,1,2])` via Some-elision (Q8).
4. **Tightened preconditions:** existing `step_masked` now returns an
   error if `action_space != Discrete`. Behavior change in unchanged
   code if a v3.x consumer flips `action_space` to `Continuous` without
   migrating their step calls.

The same Cargo-resolves-major-boundary safety-net rule that bounded
v3.0.0's blast radius applies here: downstream `pc-rl-core = "3"` users
do **not** auto-receive v4. They must update their `Cargo.toml` to
`pc-rl-core = "4"` to opt into the breaking change.

---

## 13. Forward compatibility with GpuLinAlg

v4.0.0 lands **before** GpuLinAlg Phase 2. The generic-action-space
contract is fully implementable, testable, and shippable on
`CpuLinAlg` alone — no GPU required at any stage of the v4.0.0 SBTDD
cycle (spec, plan, RED, GREEN, REFACTOR, MAGI gates, merge,
crates.io publish).

This section documents the (small) forward-compatibility surface so
GpuLinAlg can later slot in without API churn.

### 13.1 `step_continuous_raw_device` is a forward-compat hook

The method exists in the v4.0.0 public API for the explicit purpose of
**not having to break compatibility later**. On `CpuLinAlg` it is
bit-equivalent to `step_continuous` (because `CpuLinAlg::Vector = Vec<f64>`).
When GpuLinAlg eventually implements `Vector = CudaSlice<f64>`, the
same method offers a zero-copy device-side action vector — no public
API change, no rev bump.

If Phase 4 of the implementation plan finds that adding this hook adds
non-trivial complexity to the CPU-only path, it can be **deferred to
v4.x** (a future minor release) and added when GpuLinAlg actually lands.
The decision is a Phase 4 implementation question, not a v4.0.0 spec
requirement.

### 13.2 Sampling RNG locality

`a = μ + σ·ε` requires `ε ~ N(0, I)`. The RNG is host-side
(`rand::StdRng`, the existing actor RNG). This is a **CPU-only
implementation** in v4.0.0 — no GPU consideration enters the
sampling logic.

When GpuLinAlg lands, the host-generated noise will be uploaded to
device alongside `μ` for the addition. Acceptable for typical output
sizes (4-64 dims). Device-side RNG (cuRAND or equivalent) is a
GpuLinAlg-internal optimization, not a v4.0.0 concern.

### 13.3 What v4.0.0 contains for GpuLinAlg's later benefit

| Item | Required in v4.0.0? | Notes |
|------|---------------------|-------|
| `ActionSpace` enum + serde | **YES** — core feature | Independent of backend |
| `Action` enum (replay schema) | **YES** — core feature | Independent of backend |
| `step_continuous` returning `Vec<f64>` | **YES** — core feature | Independent of backend |
| `step_continuous_raw_device` returning `L::Vector` | **Optional** — forward-compat hook | Can defer to v4.x if it complicates the CPU-only path |
| Gaussian-policy gradient | **YES** — core feature | Implemented and tested on CpuLinAlg |
| Distillation rejection in continuous | **YES** — validation rule | Independent of backend |
| Per-dim policy_sigma | **NO** — v5+ | Whether deferred to v5 or to a v4.x minor is independent of GPU work |

### 13.4 Out of scope: any GPU implementation work

This spec **does not** describe, depend on, or block any GPU
implementation. v4.0.0 ships on CpuLinAlg. A separate spec
(`sbtdd/spec-behavior-gpu.md`, per project memory) will drive the
GpuLinAlg Phase 2 workstream when it begins. That spec must port
both the v3.0.0 baseline AND the v4.0.0 continuous-mode kernels —
but that's its problem, not v4.0.0's.

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
| Total LOC code estimate | ~280 (config + ActionSpace + Action enum + dispatch + 4 entry-point preconditions + new public methods) |
| Total LOC tests estimate | ~700 (Phase 1: 8 + Phase 2: 6 + Phase 3: 9 + Phase 4: 9 + serde fixtures) |
| Total LOC docs estimate | ~180 (CHANGELOG + README + rustdoc + mitigation matrix) |
| Phases | 5 (config+preconditions → replay → gradient → public API+smoke → docs) |
| Open architectural questions | 0 (all 5 from v1 spec resolved in §4 + 9 brainstorm items in §5/§9/§10) |
| Backend dependency | **CpuLinAlg only** — no GPU required for spec, plan, implementation, or release |
| Coordination dependencies | None blocking. GpuLinAlg Phase 2 (separate workstream) lands later as a backend enhancement; v4.0.0 surface is forward-compatible with it via §13.1 |
| Release | v4.0.0 BREAKING per SemVer 2.0.0 |
| Migration cost (typical discrete consumer) | 1-2 line changes (`act()` `?`, swap `step()` → `step_masked()`) |
| Migration cost (continuous adopter) | new code path; no removal needed |
| Brainstorm decisions locked | 9 items (§4 + §5.2/5.4/5.8/5.10 + §7 + §9 + §10) |
