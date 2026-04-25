# pc-rl-core feature spec — Self-recovery mechanisms for continuous learning

**Status:** Proposed (awaiting implementation plan)
**Author:** PC-TicTacToe experimentation (downstream consumer)
**Date:** 2026-04-13
**Target branch:** `continuos_learning`
**Estimated effort:** Phase 1 (1-2 days), Phase 2 (3-5 days), both including tests
**Relationship to GA roadmap:** Complementary; see "Relationship to GA framing" section

---

## Executive summary

Empirical evidence from PC-TicTacToe stress test experiments (see
`docs/pc_rl_core_cross_wake_bug.md` and the "Stress Test Experiments" section
of the downstream `CLAUDE.md`) shows that **the current CL stack in pc-rl-core
is purely defensive**: hysteresis, consolidation decay, and EWC can delay
catastrophic forgetting but cannot reverse it. Once a champion cascades from
fitness 0.95 (functional) to 0.40 (total collapse), no configuration of the
existing CL mechanisms recovers it.

This spec proposes **two new mechanisms** aimed at enabling genuine
self-recovery — i.e., gradient signals that actively pull a degraded agent
back toward the functional attractor basin without external intervention:

1. **Phase 1: Self-distillation with delayed target** — maintain a soft
   snapshot of the agent from N steps ago and regularize the current policy
   against it. Simple, defensive, ~1-2 days of work.
2. **Phase 2: Dual-compartment replay buffer** — store positive-reward
   trajectories from both training and recent stress, periodically replay
   them as off-policy gradient updates. More complex, stronger theoretical
   grounding, ~3-5 days of work.

Phase 2 is conditional on Phase 1's results: implement delayed target first,
evaluate, and only proceed to replay buffer if the simpler mechanism is
insufficient to prevent cascade.

**Non-goals:** curriculum replay, model-based planning, imitation
regularization, restart on catastrophe, GA population management (see
"Non-goals" section).

---

## Context & motivation

### Downstream problem statement

The PC-TicTacToe project trains a DPC agent via curriculum-based
`find-champion` to achieve fitness 0.95 (100% draws against minimax depth 9).
This champion is then subjected to a `stress-test` phase that calls
`step_masked()` against random-depth opponents. Under all currently-tested CL
configurations:

| Config | Cascade to 0.40 | Median cascade episode |
|--------|-----------------|------------------------|
| CL-off | 100% (3/3 replicates) | ~116k |
| EWC-only | 100% (3/3 replicates) | ~114k |
| CL-balanced+EWC | 100% (2/2 replicates) | ~404k |
| CL-max (effective permanent freeze) | 0% (fitness held at 0.95) | — |

CL-max works by setting `actor_wake_fraction = 2.0, actor_sleep_fraction = 0.005`
which produces ~99% FROZEN state for the actor. This is not recovery — it is
disabling learning entirely. The other configs all cascade, eventually.

The mechanistic analysis (six compounding effects: P2 reward ceiling = 0,
TD gradient direction = "away from bad" not "toward good", softmax
exploration noise against perfect-play opponents, shared parameters between
P1 and P2, entropy-biased loss landscape, training-time curriculum absent
during stress) is documented in finding #11 of the downstream `CLAUDE.md`.
The short form: **TD learning in DPC has no mechanism to identify and move
toward the known-good policy basin once it has drifted out of it. The current
CL stack can only slow the drift; it cannot reverse it.**

### Why these two mechanisms specifically

Of five candidate self-recovery mechanisms considered (curriculum replay,
dual-compartment replay buffer, model-based planning, imitation
regularization, restart on catastrophe), these two are the most viable under
the constraint "agent repairs itself without external intervention":

- **Delayed target distillation** is the simplest. It requires no external
  data, no environment modification, and no new architectural subsystems.
  The only cost is a second copy of the actor weights and one extra forward
  pass per step. If this is sufficient on its own, we avoid the complexity of
  replay buffers.
- **Dual-compartment replay buffer** is the strongest theoretically. It uses
  the agent's own successful past experience as a source of constructive
  gradients. It requires a new architectural subsystem but fits naturally as
  an addition to `step_masked()`.

The other three candidates are rejected:
- **Curriculum replay** modifies the environment, not the agent — not
  self-recovery in the strict sense.
- **Model-based planning** would require ~500× compute overhead for
  tree-search rollouts, prohibitive for long stress tests.
- **Imitation regularization** is structurally equivalent to EWC with a
  different anchor; empirical data shows EWC alone does not prevent cascade,
  so this would likely produce the same negative result.
- **Restart on catastrophe** is state replacement, not recovery.

### Relationship to GA framing

The downstream project's broader roadmap (see `CLAUDE.md` section
"Architectural reinterpretation") delegates the stability-plasticity dilemma
across levels: individuals are frozen (no adaptation), the population is
evolved via GA (adaptation via selection). Under that framing, champions
should not be learning continuously at all.

**These experiments are not in tension with the GA framing.** They test
whether individual-level self-recovery is achievable as a scientific
question. Three possible outcomes and their implications:

1. **Both mechanisms succeed** → individual self-recovery is possible;
   reconsider the GA framing; simpler solution exists for some use cases.
2. **Only Phase 1 succeeds** → delayed target is a useful training-time
   stability mechanism even under GA; include it in the find-champion
   training loop.
3. **Neither succeeds** → the stability-plasticity dilemma at the individual
   level is empirically unsolved in DPC; this strengthens the case for
   GA-based adaptation as architecturally necessary, not just preferred.

All three outcomes produce actionable results. The cost of running this
experiment (1-2 weeks of implementation + a few days of compute) is small
relative to the architectural decision it informs.

---

## Phase 1: Self-distillation with delayed target

### Core idea

Maintain a "lagging snapshot" of the agent's actor network — a copy of the
weights from some steps ago — and add a KL-divergence regularization term to
the actor's loss that penalizes divergence from the lagging policy. The
lagging snapshot is updated via a soft Polyak averaging rule (like DQN target
networks) so that it tracks the current network slowly.

**The intuition:** when stress gradients push the current actor away from the
functional policy, the lagging snapshot — which was the current actor a few
hundred or thousand steps ago, before the drift accumulated — provides a
pull-back signal toward "where you were recently". It is the agent using its
own past as a teacher.

### Mathematical formulation

Let `π_θ(·|s)` be the current actor policy with parameters `θ`, and
`π_θ'(·|s)` be the delayed target policy with parameters `θ'`. Define the
per-step loss:

```
L_actor = L_td(θ) + α · D_KL(π_θ(·|s) || π_θ'(·|s))
```

Where:
- `L_td(θ)` is the standard TD loss already computed by `step_masked()`.
- `α` is `distillation_lambda`, a scalar hyperparameter (typical range
  0.01 to 1.0).
- `D_KL` is the KL divergence between the softmax-normalized output
  distributions of the two networks over valid actions.

The gradient of the KL term with respect to `θ` is:

```
∇_θ D_KL(π_θ || π_θ') = Σ_a [π_θ(a|s) · (log π_θ(a|s) - log π_θ'(a|s) + 1)] · ∇_θ log π_θ(a|s)
```

In practice, for a softmax policy over `n` valid actions, this reduces to an
extra backward pass through the current network with the KL gradient as the
upstream error. The delayed network is used only for forward inference — no
gradient flows through it.

The delayed target parameters are updated via soft Polyak averaging after
each actor update:

```
θ' ← τ · θ + (1 - τ) · θ'
```

Where `τ` is `delayed_target_tau`, a small scalar (typical range 0.001 to
0.01). Small `τ` means the target lags far behind; large `τ` means it
approaches a hard copy.

An alternative is hard synchronization every N steps:

```
if step_count % sync_interval == 0:
    θ' ← θ
```

Both options should be configurable; default to soft Polyak with
`tau = 0.005` (target lags ~200 steps behind in effect, similar to `delay_steps
= 200` if implemented as a hard delay).

### Proposed pc-rl-core API

Add a new field to `PcActorCritic<L>`:

```rust
pub struct PcActorCritic<L: LinAlg> {
    // existing fields ...

    /// Delayed target actor for self-distillation regularization.
    /// None when `distillation_lambda == 0.0` (feature disabled).
    delayed_actor: Option<PcActor<L>>,
    /// Step counter for hard sync mode (used when tau == 0 and
    /// delayed_target_sync_interval > 0).
    delayed_sync_counter: u64,
}
```

Add configuration fields to `PcActorCriticConfig`:

```rust
pub struct PcActorCriticConfig {
    // existing fields ...

    /// Coefficient for the KL(current || delayed_target) regularization
    /// in the actor loss. Set to 0.0 to disable self-distillation entirely.
    /// Default: 0.0.
    #[serde(default)]
    pub distillation_lambda: f64,

    /// Polyak averaging rate for the delayed target update.
    /// After each actor gradient step:
    ///     θ_delayed ← tau · θ_current + (1 - tau) · θ_delayed
    /// Value of 0.0 combined with nonzero `delayed_target_sync_interval`
    /// enables hard-sync mode instead of soft averaging.
    /// Range: [0.0, 1.0]. Default: 0.005.
    #[serde(default = "default_delayed_target_tau")]
    pub delayed_target_tau: f64,

    /// Hard synchronization interval (in steps). When > 0 AND
    /// delayed_target_tau == 0.0, copy θ → θ_delayed every N steps.
    /// When 0, hard-sync mode is disabled (use soft Polyak via tau instead).
    /// Default: 0.
    #[serde(default)]
    pub delayed_target_sync_interval: u64,
}
```

### Integration into `step_masked()`

The delayed target extension hooks into the existing actor update path. The
proposed integration point is immediately after the standard TD gradient
computation but before applying the gradient to the weights. Pseudocode:

```rust
// Inside step_masked(), after computing the standard TD actor loss:

if self.config.distillation_lambda > 0.0 {
    if let Some(ref delayed) = self.delayed_actor {
        // Forward pass through delayed target to get reference distribution
        let delayed_logits = delayed.forward(&state, &valid_actions)?;
        let delayed_log_probs = log_softmax(&delayed_logits);

        // Current actor already has its forward pass cached from TD gradient
        let current_log_probs = log_softmax(&current_logits);

        // KL gradient flows back through the current actor only
        let kl_loss = kl_divergence(&current_log_probs, &delayed_log_probs);
        let kl_gradient = compute_kl_gradient(&current_log_probs, &delayed_log_probs);

        // Scale by distillation_lambda and add to the existing TD gradient
        // (both share the same backward pass target; they accumulate)
        self.actor.accumulate_gradient(kl_gradient * self.config.distillation_lambda);
    }
}

// Apply gradient as usual (weights already updated by the TD path, or will be
// after this accumulation — exact placement depends on the existing gradient
// application order in step_masked)

// After actor weights have been updated, update the delayed target:
if let Some(ref mut delayed) = self.delayed_actor {
    if self.config.delayed_target_tau > 0.0 {
        // Soft Polyak update
        delayed.polyak_update_from(&self.actor, self.config.delayed_target_tau);
    } else if self.config.delayed_target_sync_interval > 0 {
        // Hard sync on schedule
        self.delayed_sync_counter += 1;
        if self.delayed_sync_counter >= self.config.delayed_target_sync_interval {
            delayed.copy_weights_from(&self.actor);
            self.delayed_sync_counter = 0;
        }
    }
}
```

**Important interactions with existing mechanisms:**

- **Hysteresis:** the KL term should be applied even when the actor is FROZEN,
  because FROZEN in DPC means "weights do not update" but the delayed
  snapshot can still be read for reference. Actually, a FROZEN actor means
  `step_masked()` skips the gradient application entirely for the actor, in
  which case the KL computation is wasted. Recommendation: **skip the KL
  computation entirely when `actor_hysteresis.is_some() && state == Frozen`**,
  for efficiency.
- **EWC:** EWC and distillation are additive regularizers on the actor loss.
  They should coexist without special handling. If both are active, the total
  loss is `L_td + ewc_lambda · L_fisher + distillation_lambda · L_kl`. This is
  correct behavior — EWC pulls toward training-initial anchors, distillation
  pulls toward delayed-current anchors. Both are "stay close to something
  known-good", just with different anchors.
- **Fisher update:** the delayed target should NOT be a source of Fisher
  information. Fisher is computed from gradients of the TD loss only. This
  separation must be explicit.

### New polyak_update_from helper

`PcActor<L>` does not currently expose a polyak-update method. This must be
added. Proposed signature in `src/pc_actor.rs`:

```rust
impl<L: LinAlg> PcActor<L> {
    /// Performs an in-place soft (Polyak) update toward another actor:
    ///     self.weights ← tau · other.weights + (1 - tau) · self.weights
    ///
    /// # Errors
    /// Returns PcError::DimensionMismatch if topologies differ.
    pub fn polyak_update_from(
        &mut self,
        other: &PcActor<L>,
        tau: f64,
    ) -> Result<(), PcError> {
        // Implementation: iterate layers, for each layer:
        //   weights[i] = tau * other.weights[i] + (1 - tau) * weights[i]
        //   bias[i]    = tau * other.bias[i]    + (1 - tau) * bias[i]
        // Validate topology match upfront.
    }

    /// Performs an in-place hard copy from another actor:
    ///     self.weights ← other.weights
    /// Equivalent to `polyak_update_from(other, 1.0)` but avoids
    /// unnecessary arithmetic.
    pub fn copy_weights_from(&mut self, other: &PcActor<L>) -> Result<(), PcError> {
        // Direct weight copy via backend primitives.
    }
}
```

These helpers are independently useful beyond the distillation feature — for
example, they could be reused by the GA crossover operators or by any future
"teacher-student" training setup.

### Configuration validation

Add to `PcActorCriticConfig::validate_cl()` or equivalent:

```rust
if config.distillation_lambda < 0.0 {
    return Err(PcError::ConfigValidation(format!(
        "distillation_lambda ({}) must be >= 0.0",
        config.distillation_lambda
    )));
}
if config.delayed_target_tau < 0.0 || config.delayed_target_tau > 1.0 {
    return Err(PcError::ConfigValidation(format!(
        "delayed_target_tau ({}) must be in [0.0, 1.0]",
        config.delayed_target_tau
    )));
}
// Mutual exclusion: either soft Polyak OR hard sync, not both
if config.distillation_lambda > 0.0 {
    let uses_polyak = config.delayed_target_tau > 0.0;
    let uses_sync = config.delayed_target_sync_interval > 0;
    if uses_polyak && uses_sync {
        return Err(PcError::ConfigValidation(
            "delayed_target_tau and delayed_target_sync_interval are mutually exclusive; \
             set one to zero".to_string(),
        ));
    }
    if !uses_polyak && !uses_sync {
        return Err(PcError::ConfigValidation(
            "distillation_lambda > 0 requires either delayed_target_tau > 0 \
             or delayed_target_sync_interval > 0".to_string(),
        ));
    }
}
```

### Serialization

The delayed target must be persisted with the agent. Add to the serializer:

```rust
pub struct SaveFile {
    // existing fields ...
    /// Delayed target actor weights. None when distillation is disabled.
    #[serde(default)]
    pub delayed_actor_weights: Option<PcActorWeights>,
    /// Delayed target sync counter. Only meaningful in hard-sync mode.
    #[serde(default)]
    pub delayed_sync_counter: u64,
}
```

Update `save_agent` and `load_agent` accordingly. Default behavior for legacy
save files without the new fields: create delayed_actor from current weights
at load time (i.e., the restored agent starts with current == delayed).

### Unit tests to add

In `src/pc_actor_critic/mod.rs` tests module:

1. **`test_distillation_disabled_reproduces_baseline`**
   Set `distillation_lambda = 0.0`, run 100 steps with a known seed, verify
   the weight trajectory is byte-identical to a run without the feature.

2. **`test_polyak_update_tau_zero_no_change`**
   `delayed.polyak_update_from(current, tau=0)` must leave delayed unchanged.

3. **`test_polyak_update_tau_one_full_copy`**
   `delayed.polyak_update_from(current, tau=1)` must produce
   `delayed.weights == current.weights`.

4. **`test_polyak_update_partial_interpolation`**
   With `tau=0.5`, after one update delayed should be exactly the midpoint
   between initial delayed and current.

5. **`test_copy_weights_from_exact`**
   `delayed.copy_weights_from(current)` must produce byte-exact equality.

6. **`test_polyak_update_rejects_topology_mismatch`**
   Two actors with different hidden_layer sizes → PcError::DimensionMismatch.

7. **`test_hard_sync_fires_on_schedule`**
   Configure `delayed_target_tau=0, sync_interval=10`. Run 25 steps. Assert
   the delayed target was copy-synced at step 10 and step 20 but not at step
   15.

8. **`test_distillation_gradient_pulls_toward_delayed`**
   Construct a scenario where current and delayed policies differ on a known
   state. Run one `step_masked` with `distillation_lambda = 1.0`. Assert the
   current policy's output distribution on that state moved measurably
   toward the delayed policy's distribution.

9. **`test_distillation_zero_when_current_equals_delayed`**
   When current == delayed, the KL gradient is zero and should not perturb
   the TD update. Run with `distillation_lambda = 1000.0` (high) when current
   is initialized as a copy of delayed; weight delta over one step should
   match the baseline TD-only delta within numerical tolerance.

10. **`test_distillation_skipped_when_actor_frozen`**
    When actor_hysteresis is active and state == Frozen, the delayed target
    forward pass should be skipped for efficiency. Verify via a perf-regression
    test or an instrumentation flag that the forward pass was not called.

11. **`test_serialization_round_trip_preserves_delayed_target`**
    Save an agent with distillation enabled and a non-trivial delayed target
    (not equal to current). Load it. Verify `agent.delayed_actor` is byte-exact
    to pre-save state.

12. **`test_serialization_legacy_file_creates_delayed_from_current`**
    Load a save file from before this feature existed (no `delayed_actor_weights`
    field). Verify the loaded agent has `delayed_actor` initialized to a copy
    of current actor, not None, when distillation is enabled in the config.

### Success criteria (Phase 1)

After implementation and unit tests pass, the downstream PC-TicTacToe project
will run a stress test comparison:

| Config | Expected result |
|--------|-----------------|
| CL-balanced+EWC (baseline, current) | Cascade to 0.40 around ep ~400k (per existing replicates) |
| CL-balanced+EWC + delayed target (λ=0.1, τ=0.005) | **Test**: does cascade happen? |
| CL-balanced+EWC + delayed target (λ=0.5, τ=0.005) | Higher λ — expected to be more protective but may limit learning |
| CL-balanced+EWC + delayed target (λ=0.1, hard sync 1000) | Alternative delay mode |

**Phase 1 is a SUCCESS if any of the following hold:**

- Fitness held at 0.95 for the full 500k run (no first drop).
- First drop occurs but fitness recovers to 0.95 at some point during the run.
- Cascade to 0.40 does not occur within 500k episodes (significant improvement
  over the ~400k baseline).
- Time at 0.40 is less than 5% of run duration (vs ~17% for CL-balanced+EWC).

**Phase 1 is a PARTIAL SUCCESS if:**

- Cascade still happens but is delayed significantly (>600k episodes).
- Time at 0.40 is reduced (e.g., to <10%) but not eliminated.

**Phase 1 is a FAILURE if:**

- Cascade timing is unchanged from baseline (~400k).
- Fitness trajectory is indistinguishable from CL-balanced+EWC.
- Or worse: delayed target interacts destructively with EWC and cascade happens
  earlier than baseline.

On SUCCESS, consider stopping — the simpler mechanism is sufficient. On
PARTIAL SUCCESS or FAILURE, proceed to Phase 2.

---

## Phase 2: Dual-compartment replay buffer (conditional on Phase 1 results)

### Core idea

Two ring buffers of past `(state, action, reward, next_state, done)`
transitions:

- **Buffer A (training memory):** filled during `find-champion`. Stores
  transitions from successful episodes only (reward ≥ 0). Read-only during
  stress test. Provides the "original good policy" signal.
- **Buffer B (recent success memory):** filled during stress test. FIFO
  eviction. Same "reward ≥ 0" filter. Captures adaptations the agent discovers
  during stress that it should not forget.

A new `replay_learn(batch_size)` method samples `batch_size/2` transitions
from A and `batch_size/2` from B, recomputes TD targets using the current
critic, and applies a gradient update. Called periodically during stress test
(e.g., every 100 episodes) in addition to the normal step-by-step learning.

The key property: **the transitions in the buffer all had non-negative reward
in their original context**, so the gradients they induce in the current
policy are constructive on average. This is the "positive signal injection"
that pure TD learning in a stress environment lacks.

### Why two compartments

A naive single-buffer replay has a distribution shift problem: if the buffer
is only filled during training, its contents become stale after many stress
episodes (the current agent may disagree with stored Q-values). If filled
only during stress, early stress noise contaminates the memory.

The two-compartment design gives both properties:

- **A acts as an immutable anchor** — its transitions represent "the behavior
  of a champion that worked". Always available, never drifts.
- **B acts as a live cache** — its transitions represent "recent good
  moments", captured even if they are transient.
- **Sampling 50/50 from both** balances anchor stability with current
  relevance.

### Mathematical formulation

Given a sampled batch of `N` transitions `{(s_i, a_i, r_i, s'_i, d_i)}_{i=1}^N`:

For each transition, compute the **current-critic TD target**:

```
y_i = r_i + γ · V_φ(s'_i) · (1 - d_i)
```

Where `V_φ` is the current critic (not the stored critic at time of capture).
The critic loss for the replay batch is:

```
L_critic_replay = (1/N) · Σ_i (V_φ(s_i) - y_i)²
```

The actor loss for the replay batch uses policy gradient with the replay
advantage:

```
A_i = y_i - V_φ(s_i)
L_actor_replay = -(1/N) · Σ_i [A_i · log π_θ(a_i | s_i)]
```

Both losses are added to the gradient accumulators and applied in the same
way as normal TD updates. The delayed target and EWC regularizers, if active,
ALSO apply to the replay update (they are properties of the actor loss
function, not of the data source).

**Off-policy bias:** importance sampling ratios would strictly be required
for unbiased off-policy gradients, but in DPC with a mostly-deterministic
`SelectionMode::Play` mode this bias is typically small. Leave importance
sampling out of the initial implementation; if experiments show instability,
add it in a follow-up.

### Proposed pc-rl-core API

Add a new module `src/pc_actor_critic/replay.rs`:

```rust
//! Dual-compartment replay buffer for off-policy learning from stored
//! positive-reward trajectories.

use serde::{Deserialize, Serialize};

/// A single transition stored in the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTransition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub valid_actions: Vec<usize>,
}

/// Dual-compartment replay buffer with two FIFO ring buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// "A" compartment: training memories. Read-only after training ends.
    pub training_memories: std::collections::VecDeque<ReplayTransition>,
    training_capacity: usize,

    /// "B" compartment: recent stress-test successes. FIFO eviction.
    pub recent_memories: std::collections::VecDeque<ReplayTransition>,
    recent_capacity: usize,

    /// Whether new pushes go to the training compartment (find-champion phase)
    /// or the recent compartment (stress test phase). Toggled via
    /// `seal_training_memories()`.
    pub training_phase: bool,

    /// If true, only transitions with reward >= 0 are stored. Recommended.
    pub positive_only: bool,
}

impl ReplayBuffer {
    pub fn new(training_capacity: usize, recent_capacity: usize, positive_only: bool) -> Self { ... }

    /// Push a new transition. Routed to training_memories or recent_memories
    /// based on training_phase. If positive_only is true and transition reward
    /// is negative, the transition is silently dropped.
    pub fn push(&mut self, transition: ReplayTransition) { ... }

    /// Seal the training memories — after this call, all future pushes go to
    /// recent_memories. Typically called at end of find-champion.
    pub fn seal_training_memories(&mut self) {
        self.training_phase = false;
    }

    /// Sample `batch_size` transitions with a 50/50 split between
    /// training_memories and recent_memories. If either compartment is empty,
    /// sample entirely from the non-empty one. Returns fewer than batch_size
    /// if both are smaller than the requested split.
    pub fn sample(&self, batch_size: usize, rng: &mut StdRng) -> Vec<&ReplayTransition> { ... }

    /// Number of transitions currently stored across both compartments.
    pub fn total_len(&self) -> usize { ... }
}
```

Add to `PcActorCritic<L>`:

```rust
pub struct PcActorCritic<L: LinAlg> {
    // existing fields ...
    replay_buffer: Option<ReplayBuffer>,
}

impl<L: LinAlg> PcActorCritic<L> {
    /// Records the current transition into the replay buffer if one is active.
    /// Called internally by step_masked() after a transition completes.
    fn record_transition(&mut self, transition: ReplayTransition) { ... }

    /// Seals the training compartment of the replay buffer. Typically called
    /// by the downstream project at the end of find-champion.
    pub fn seal_replay_training_memories(&mut self) {
        if let Some(ref mut buf) = self.replay_buffer {
            buf.seal_training_memories();
        }
    }

    /// Samples a batch from the replay buffer and applies an off-policy
    /// gradient update using current-critic TD targets. Does nothing if the
    /// buffer is empty or not configured. Callable from the downstream
    /// project during stress test or training loops.
    ///
    /// # Errors
    /// Returns PcError on numerical or backend failures.
    pub fn replay_learn(&mut self, batch_size: usize) -> Result<(), PcError> { ... }
}
```

### Configuration

```rust
pub struct PcActorCriticConfig {
    // existing fields ...

    /// Capacity of the training-phase replay compartment.
    /// 0 disables the replay buffer entirely. Default: 0.
    #[serde(default)]
    pub replay_training_capacity: usize,

    /// Capacity of the recent-stress replay compartment.
    /// 0 disables stress-phase recording. Default: 0.
    #[serde(default)]
    pub replay_recent_capacity: usize,

    /// If true, only transitions with reward >= 0 are stored. Default: true.
    #[serde(default = "default_replay_positive_only")]
    pub replay_positive_only: bool,

    /// Batch size for each replay_learn() call. Default: 64.
    #[serde(default = "default_replay_batch_size")]
    pub replay_batch_size: usize,
}
```

### Integration into downstream stress test loop

In PC-TicTacToe's `StressTester::run()` (file `src/training/stress_test.rs`),
add a periodic replay call:

```rust
while !self.stop_flag.load(Ordering::Acquire) {
    self.run_stress_episode(&mut minimax, agent_side);
    episode += 1;

    // New: periodic replay every N episodes
    if episode % self.stress_config.replay_interval as u64 == 0 {
        let _ = self.agent.replay_learn(self.stress_config.replay_batch_size);
    }

    // ... existing checkpoint logic ...
}
```

This requires new config fields in `StressTestSection` in PC-TicTacToe:

```toml
[stress_test]
# ... existing fields ...

# Replay buffer controls. Set replay_interval = 0 to disable.
replay_interval = 100       # Replay every 100 episodes during stress
replay_batch_size = 64      # (mirrors pc-rl-core's config)
```

### Training-phase integration

During `find-champion`, the replay buffer is filled automatically by
`step_masked()` if enabled. No changes required in the training loop beyond
adding the config fields. At the end of `find-champion`, the downstream
project should call `agent.seal_replay_training_memories()` before calling
`save_agent()` so that the buffer state is persisted in the training phase.

### Unit tests to add

In `src/pc_actor_critic/replay.rs`:

1. **`test_replay_buffer_empty_sample_returns_empty`**
2. **`test_replay_buffer_push_respects_training_phase`**
3. **`test_replay_buffer_positive_only_filter`**
4. **`test_replay_buffer_fifo_eviction_recent_compartment`**
5. **`test_replay_buffer_seal_routes_new_pushes_to_recent`**
6. **`test_replay_buffer_sample_50_50_split`** — with 100 training + 100 recent, sampling batch=50 should give ~25 from each
7. **`test_replay_buffer_sample_fallback_when_one_empty`** — if training is empty, all samples come from recent, and vice versa
8. **`test_replay_buffer_serialization_round_trip`**

In `src/pc_actor_critic/mod.rs` tests module:

9. **`test_replay_learn_no_buffer_no_op`** — when replay_buffer is None, replay_learn is a no-op.
10. **`test_replay_learn_updates_weights`** — with a non-empty buffer and known batch, verify the actor and critic weights change measurably after replay_learn.
11. **`test_replay_learn_does_not_mutate_buffer`** — after replay_learn, the buffer contents are unchanged.
12. **`test_replay_learn_coexists_with_ewc`** — with EWC enabled, replay_learn applies both TD gradient and EWC penalty.
13. **`test_replay_learn_coexists_with_distillation`** — with delayed target enabled, replay_learn applies both TD gradient and distillation pull.
14. **`test_replay_learn_respects_hysteresis`** — when actor_hysteresis.state == Frozen, replay_learn should NOT update actor weights (only critic). Verify separation.
15. **`test_seal_training_memories_is_idempotent`**

### Success criteria (Phase 2)

Same structure as Phase 1, with the added dimension of combining both
mechanisms. Four configurations to compare against the CL-balanced+EWC
baseline:

| Config | Expected behavior |
|--------|-------------------|
| CL-balanced+EWC (baseline) | Cascade ~400k (existing data) |
| CL-balanced+EWC + delayed target | Phase 1 result |
| CL-balanced+EWC + replay buffer | Phase 2 alone |
| CL-balanced+EWC + delayed target + replay buffer | Combined — best case |

**Phase 2 is a SUCCESS if:**

- Replay-buffer-only is a clear improvement over Phase 1 alone, OR
- Combined (both mechanisms) produces recovery behavior: fitness oscillating
  between 0.70 and 0.95 during stress rather than monotonic decline, OR
- Combined extends cascade beyond any practical horizon (>2M episodes).

**Phase 2 is a FAILURE if:**

- Replay buffer alone has no effect beyond delayed target alone.
- Combined configuration is not measurably better than either individually.
- Or worse: replay buffer destabilizes training (off-policy bias too large).

On FAILURE of both phases, the negative result is itself valuable: it
justifies the GA-based adaptation architecture empirically by ruling out the
simpler individual-level solutions.

---

## Experimental protocol

### Phase 1 validation (delayed target)

After Phase 1 implementation in pc-rl-core:

1. **Unit tests pass** in pc-rl-core.
2. **Integration test in PC-TicTacToe**: build a tiny config with
   `distillation_lambda = 0.1, delayed_target_tau = 0.005`, run
   `find-champion --iterations 1` and verify the champion is saved with
   delayed target state.
3. **Single-shot stress test**: load the champion, run
   `stress-test -c config_stress_cl_balanced_distillation.toml` for 500k
   episodes. Record cascade behavior.
4. **If single-shot looks promising (no cascade or delayed cascade)**: run 3
   replicates to confirm.
5. **If single-shot looks like the baseline**: Phase 1 is inconclusive;
   proceed to Phase 2.

### Phase 2 validation (replay buffer)

After Phase 2 implementation in pc-rl-core:

1. **Unit tests pass**.
2. **Retrain a champion** with replay buffer enabled in `find-champion`
   so the training compartment is populated. This is important because the
   existing `champion.json` does not have a replay buffer attached.
3. **Single-shot stress test** with replay buffer only (no delayed target).
4. **Single-shot stress test** with replay buffer + delayed target combined.
5. **3 replicates each** of the promising configs.

### What to measure per stress test run

For each run, extract from the CSV:
- First drop episode (fitness 0.95 → 0.70 transition)
- Cascade episode (fitness 0.70 → 0.40 transition) or "never"
- Recovery events (fitness 0.70 → 0.95 transitions — these are the new signal)
- Time at each bucket (0.95, 0.70, 0.40)
- Actor frozen %, critic frozen %, total hysteresis transitions
- Max fitness, min fitness, final fitness

The `scripts/run_stress_replicates.py` orchestrator in PC-TicTacToe already
extracts most of these metrics; the "recovery events" count will need a
small addition to the analyzer.

---

## Non-goals

The following are explicitly NOT part of this spec and should not be
implemented as part of Phase 1 or Phase 2:

1. **Curriculum replay** — modifying the stress environment to intercalate
   easy opponents. This changes the environment, not the agent, and
   therefore does not test self-recovery. Rejected earlier in the design
   conversation.

2. **Model-based planning (MCTS rollouts, imagination)** — would add ~500×
   compute overhead per episode, making 500k-episode runs infeasible.
   Architecturally a much larger change. Better suited for a separate future
   effort if any.

3. **Imitation regularization (fixed-anchor L2 penalty)** — structurally
   equivalent to EWC with a different anchor. Empirical data already shows
   EWC alone does not prevent cascade, so this would likely produce the same
   negative result. Implementing it would duplicate the existing EWC code
   path with minimal new information.

4. **Restart on catastrophe detection** — is state replacement, not
   recovery. Not consistent with the "self-recovery" goal.

5. **GA population management** — the downstream roadmap's end state, but
   this spec is a test of whether individual-level mechanisms can substitute
   for or complement the GA layer. The GA itself is a higher-level,
   separately-scoped effort.

6. **Prioritized replay (TD-error-weighted sampling)** — a well-known
   improvement over uniform replay, but adds implementation complexity.
   Start with uniform 50/50 and add prioritization only if uniform sampling
   is empirically insufficient.

7. **Importance sampling corrections for off-policy replay bias** — in
   `SelectionMode::Play` the policy is nearly deterministic, so the bias is
   small. Add in a follow-up if instability is observed.

8. **Target network for the critic** — a separate critic target network
   (like DQN) would also stabilize value estimation, but adds another copy
   of the critic. Out of scope here; the delayed target is actor-only in
   this spec.

---

## Relationship to existing pc-rl-core features

### EWC interaction

Both mechanisms coexist with EWC (`ewc_lambda > 0`) without special handling.
All three regularizers (EWC, distillation, replay) contribute additively to
the actor loss. Key property to preserve: **each regularizer only affects
what it should** — EWC touches only parameters with Fisher > 0, distillation
applies to the full parameter set, replay samples from its own data.

Configuration sanity check: if all three are enabled simultaneously, the
total regularization pressure on the actor could be high enough to prevent
all learning. Recommended defaults for combined mode:

```toml
ewc_lambda = 0.1              # same as current
distillation_lambda = 0.1     # moderate
delayed_target_tau = 0.005    # slow drift
replay_interval = 200         # less frequent
replay_batch_size = 32        # smaller batch
```

### Hysteresis interaction

Both mechanisms should respect the hysteresis FROZEN state. When the actor
is FROZEN:

- **Delayed target:** skip the KL computation entirely (no point computing a
  gradient when it won't be applied).
- **Replay buffer:** skip only the actor portion of the replay update; still
  update the critic (critic has its own hysteresis state and may be PLASTIC
  while the actor is FROZEN).

This separation is the same as the existing handling in `step_masked`: the
actor and critic can be in different plasticity states at the same time and
should be updated independently.

### Cross-wake bug dependency

Phase 1 and Phase 2 can be implemented and tested independently of the
cross-wake deadlock bug documented in `docs/pc_rl_core_cross_wake_bug.md`.
However, **the downstream stress test experiments to validate these
mechanisms are contaminated by the deadlock bug when run for > ~300k
episodes**. Recommendation: fix the cross-wake bug first (or in parallel),
then validate Phase 1 and Phase 2 against a deadlock-free baseline.

If time pressure prevents fixing cross-wake first, Phase 1 and Phase 2
validation runs should be limited to ≤ 250k episodes to stay below the
observed deadlock onset, at the cost of not seeing the long-horizon behavior.

### Serialization compatibility

Both features add new optional fields to the save file format. Legacy save
files without these fields must load cleanly with the new features
initialized to default (delayed target = copy of current actor, replay
buffer = None). This preserves backward compatibility for existing
`champion.json` files trained before these features existed.

---

## Acceptance criteria summary

**Phase 1 is complete when:**

- [ ] `PcActorCriticConfig` has new fields: `distillation_lambda`,
      `delayed_target_tau`, `delayed_target_sync_interval`
- [ ] `PcActorCritic` has `delayed_actor: Option<PcActor<L>>` field
- [ ] `PcActor` has `polyak_update_from` and `copy_weights_from` methods
- [ ] `step_masked` applies KL regularization when enabled
- [ ] All 12 unit tests in "Unit tests to add" pass
- [ ] `validate_config` rejects invalid combinations (both sync modes on, negative lambda, tau out of range)
- [ ] Save/load preserves delayed target state
- [ ] Downstream PC-TicTacToe can build against the new API without errors
- [ ] Stress test smoke test runs to completion with distillation enabled

**Phase 2 is complete when:**

- [ ] New module `replay.rs` with `ReplayBuffer` and `ReplayTransition`
- [ ] `PcActorCritic` has `replay_buffer: Option<ReplayBuffer>` field
- [ ] `replay_learn(batch_size)` method implemented
- [ ] `seal_replay_training_memories()` method implemented
- [ ] `step_masked` auto-records transitions when buffer is configured
- [ ] All 15 unit tests in "Unit tests to add" pass
- [ ] Save/load preserves replay buffer state (both compartments)
- [ ] Downstream PC-TicTacToe's StressTester can call `replay_learn` periodically
- [ ] Stress test smoke test runs to completion with replay buffer enabled
- [ ] Combined (distillation + replay) smoke test runs to completion

**Experimentation is complete when** (downstream, not pc-rl-core's responsibility):

- [ ] At least 3 replicates of CL-balanced+EWC+distillation at 500k episodes
- [ ] At least 3 replicates of CL-balanced+EWC+replay at 500k episodes (if Phase 2 runs)
- [ ] At least 3 replicates of combined at 500k episodes (if Phase 2 runs)
- [ ] Comparative analysis with the 2026-04-13 baseline replicates
- [ ] Documentation update in downstream `CLAUDE.md` finding #11 or new finding #12 with the new data

---

## References

- `docs/pc_rl_core_cross_wake_bug.md` — upstream bug affecting long-running stress tests
- Downstream `CLAUDE.md` section "Stress Test Experiments" — baseline data and mechanism analysis
- Downstream `CLAUDE.md` finding #11 — mechanistic explanation of why CL degrades without recovery
- Downstream `CLAUDE.md` section "Architectural reinterpretation" — GA framing context
- `src/training/stress_test.rs` — downstream stress test implementation (where replay_learn will be called)
- `src/pc_actor_critic/mod.rs:process_hysteresis` — existing hysteresis state machine (interaction point)
- `src/pc_actor_critic/mod.rs:step_masked` — primary integration point for both phases
- `src/pc_actor.rs` — where `polyak_update_from` and `copy_weights_from` must be added
- `src/serializer.rs` — where save/load extensions must be applied

---

## Questions for the implementing agent

Before producing the detailed implementation plan, please confirm or decide:

1. **Soft vs hard delay as default:** this spec defaults to soft Polyak
   (`tau = 0.005`). Is that consistent with your style preferences, or do
   you prefer hard sync as default? (No strong downstream preference.)

2. **Where the KL gradient flows through:** current actor only, delayed is
   frozen. Confirm this is the standard distillation semantics you'll
   implement.

3. **Replay update location:** should `replay_learn` reuse the existing
   `step_masked` internal learning path (same actor/critic update code) or
   have its own specialized path? Reuse is simpler and more consistent;
   a specialized path could optimize for batches but duplicates logic.

4. **Replay buffer storage format:** `Vec<f64>` per state field is
   straightforward but memory-inefficient for large state spaces. For
   TicTacToe (9 floats per state) this is fine. Would you prefer a more
   compact representation for future scalability, or is `Vec<f64>` acceptable?

5. **Training phase toggle mechanism:** currently the spec uses an explicit
   `seal_training_memories()` call at end of training. An alternative is an
   auto-toggle based on whether the agent is in inference mode vs training
   mode. The explicit toggle is more predictable; comment if you prefer
   otherwise.

6. **Phase ordering flexibility:** if you discover during Phase 1
   implementation that it is trivially small (< 1 day), is it acceptable
   to include a minimal Phase 2 skeleton in the same commit batch even
   without full implementation, to reduce integration friction later?
