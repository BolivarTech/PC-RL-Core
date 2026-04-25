# Changelog

## [4.0.0] - 2026-04-25

### Breaking changes

- **Generic action space.** `pc-rl-core` now supports two action
  spaces selected via `PcActorCriticConfig::action_space`:
  - `ActionSpace::Discrete` (default) — preserves v3.x behavior bit-
    for-bit. REINFORCE policy gradient on softmax(logits).
  - `ActionSpace::Continuous` — fixed-σ Gaussian policy `a = μ + σ·ε`
    with descent-direction gradient `(μ − a) / σ²`.

  **Affected v3.x consumers:** if you never set `action_space`, you
  default to `Discrete` and observe **no** behavior change. If you
  explicitly set `Continuous`, you adopt the new code path.

- **Removed `PcActorCritic::step()`.** The deprecated discrete step
  method (deprecated since v2.0.0) is removed. Migration: replace
  with `step_masked` and pass the full action set:

  ```rust
  // v3.x (removed in v4)
  let action = agent.step(&state, reward, done);

  // v4.0.0
  let valid: Vec<usize> = (0..agent.actor.output_size).collect();
  let action = agent.step_masked(&state, &valid, reward, done)?;
  ```

- **`act()` return type bumps to `Result<...>`.** Pattern-match
  call sites must add `?`:

  ```rust
  // v3.x
  let (action, infer) = agent.act(&state, &valid, SelectionMode::Play);
  // v4.0.0
  let (action, infer) = agent.act(&state, &valid, SelectionMode::Play)?;
  ```

- **`learn(trajectory)` now returns `Result<f64, PcError>`** (was
  `f64`). Discrete-only — calling on Continuous agent returns
  `ConfigValidation`. Method remains deprecated since v2.1.0;
  scheduled for removal in v5.0.0.

- **Replay schema migration.** `ReplayTransition::action: usize`
  becomes `action: Action` (`#[serde(untagged)]` enum,
  `Discrete(usize) | Continuous(Vec<f64>)`). v3.x save files
  deserialize automatically — no consumer action required for
  loading. Pattern-match call sites that read the field as `usize`
  break at compile time.

- **`ReplayTransition::valid_actions: Vec<usize>` → `Option<Vec<usize>>`.**
  v3.x bare `[0,1,2]` form auto-wraps via serde Some-elision.
  Validation enforces `is_some() iff Discrete`.

- **`ReplayBuffer::push` now returns `Result<(), PcError>`** (was
  `()`). Validates Action variant matches buffer's action_space and
  the `valid_actions` invariant. Cross-mode contamination rejected.

### Added

- `ActionSpace` enum (`Discrete | Continuous`) in
  `pc_actor_critic::config`.
- `PcActorCriticConfig::policy_sigma: f64` — Gaussian policy std-dev
  for continuous mode. Default 0.1.
- `PcActorCritic::step_continuous(state, reward, done) -> Result<Vec<f64>, PcError>`
  — continuous-mode learning step.
- `PcActorCritic::step_continuous_raw_device(state, reward, done) -> Result<L::Vector, PcError>`
  — forward-compat hook for future GpuLinAlg backends.
- `PcActorCritic::act_continuous(state, mode: SelectionMode) -> Result<(Vec<f64>, InferResult<L>), PcError>`
  — continuous inference. Play returns μ deterministically; Training
  samples μ + σ·ε.
- `Action` enum in `pc_actor_critic::replay` — replay transition
  action variant.
- Runtime precondition guards on all 4 entry points (`step_masked`,
  `step_continuous`, `act`, `act_continuous`) reject mismatched
  `action_space`.

### Notes

- **Self-recovery toolkit (`rollback_soft`/`rollback_hard`/
  `champion_update`) remains available only in Discrete mode.**
  Continuous + distillation lambda > 0 is rejected at validation
  (KL undefined for raw output). L2-anchored continuous distillation
  is experimental future work, no release commitment.

- **Continuous-mode operational tuning.** Mitigation matrix for
  empirical issues:

  | Symptom | Primary mitigation | Secondary | Tertiary |
  |---|---|---|---|
  | WEIGHT_CLIP saturation | ↑ `policy_sigma` | ↓ `lr_weights` | ↓ `td_steps` or ↓ `gae_lambda` |
  | Slow learning | ↓ `policy_sigma` | ↑ `lr_weights` | verify rewards in `[-1, 1]` |
  | Hysteresis oscillating | confirm `adaptive_surprise=true` | retune `surprise_low`/`high` | disable hysteresis |
  | Distillation rejected | `distillation_lambda_polyak/frozen = 0.0` | — | — |

- **GpuLinAlg coordination:** v4.0.0 is CPU-first. `step_continuous_raw_device`
  is the forward-compat hook. GpuLinAlg Phase 2 (separate workstream)
  will port both v3.0.0 baseline AND v4.0.0 continuous-mode kernels.

- **SemVer rationale:** breaking changes (removed `step()`, replay
  schema migration, `act()` return type, `learn()` return type)
  warrant the major bump per SemVer 2.0.0. Consumers `pc-rl-core = "3"`
  do not auto-receive v4 — must explicitly bump `Cargo.toml`.

## [3.0.0] - 2026-04-25

### Breaking changes

- **`critic_hysteresis` is now actually enforced on critic weight
  updates.** Prior to v3.0.0, setting `critic_hysteresis = true`
  caused the framework to track a state machine for the critic but
  the tracked state was NEVER consulted at the weight-update site —
  the critic kept learning based on `|td_error|` magnitude alone,
  regardless of FROZEN/PLASTIC label. This was an architectural
  asymmetry with the actor (whose hysteresis HAS been enforced since
  v2.0.0). From v3.0.0 onward, `critic_hysteresis.state` IS enforced:
  when FROZEN, the critic's effective learning rate is clamped to
  `scale_floor` on the online path, and to `critic_floor_replay`
  (with the same tri-state sentinel semantics as the v2.2.1
  `scale_floor_replay`) on the replay path.

  **Affected consumers:** anyone running with
  `critic_hysteresis = true` who implicitly relied on the critic
  continuing to learn during FROZEN windows. The "critic stops
  learning" window is bounded by cross-wake coupling
  (`actor_wakes_critic`, default `true`, threshold `1000`), so under
  default coupling the maximum FROZEN-critic window is ~1000 steps
  per cycle. Workloads where the critic is FROZEN for a small
  fraction of training time will see negligible regression; workloads
  with high FROZEN-duty-cycle on the critic should expect a
  measurable slowdown in critic convergence unless they apply the
  migration below.

  **Migration table:**

  | Pre-v3.0.0 setting | v3.0.0 equivalent (preserve behavior) |
  |---|---|
  | `critic_hysteresis = false` | `critic_hysteresis = false` (no change, no regression) |
  | `critic_hysteresis = true`, replay active, want **closest approximation** to v2.2.x replay learning | add `critic_floor_replay = scale_ceil` (typically `2.0` — matches v2.2.x's td-magnitude-driven upper-band scale) |
  | `critic_hysteresis = true`, replay active, want **mild recovery** (less aggressive than v2.2.x) | add `critic_floor_replay = 0.3` (rule-of-thumb operating point; smaller values clamp tighter) |
  | `critic_hysteresis = true`, want TRUE stress protection (new behavior) | no change needed — v3.0.0 default delivers the symmetric protection that the field always intended |
  | `critic_hysteresis = true`, cross-wake disabled, critic stuck FROZEN | enable cross-wake OR set `critic_floor_replay` high (≥ `scale_ceil`) |

  See README "Migration from v2.2.x to v3.0.0" subsection for a
  ready-to-copy config snippet showing the recommended paired
  `(scale_floor_replay, critic_floor_replay)` opt-in.

### Added

- `PcActorCriticConfig::critic_floor_replay: f64` — opt-in override
  for replay-path **critic** learning-rate floor under FROZEN
  `critic_hysteresis`. Mirror of v2.2.1 `scale_floor_replay` for the
  critic side. Default `-1.0` (sentinel) honours the new gate (no
  critic update under FROZEN-replay); strict-positive values opt
  the critic into replay-driven learning even during FROZEN. Same
  tri-state sentinel semantics, same validation rule
  (`[0.0, 10 × scale_ceil]` finite or `~-1.0`), same upper-bound
  rationale documented on the field.
- `PcActorCritic::effective_critic_scale_for_mode` (`pub(crate)`) —
  mode-aware critic scale resolution. Honours
  `critic_hysteresis.state == Frozen` per the new gate; falls
  through to `critic_surprise_scale` for PLASTIC and disabled-
  hysteresis paths (no behavior change vs v2.2.x for those cases).

### Notes

- **Pairing recommendation:** for coherent actor-critic dynamics
  under FROZEN-replay, set `scale_floor_replay` and
  `critic_floor_replay` IN LOCKSTEP. Recommended symmetric pairs:

  - `(-1.0, -1.0)` (default): full stress protection — neither
    network updates during FROZEN windows.
  - `(0.3, 0.3)`: mild symmetric recovery — both networks learn
    from stored transitions under FROZEN.
  - `(1.0, 1.0)`: aggressive symmetric recovery — both networks
    learn at full magnitude.

  Asymmetric opt-in such as `(0.3, -1.0)` is allowed for tuning
  flexibility but produces actor-critic desynchronization (one
  network moves based on stored transitions while the other stays
  FROZEN-gated). Use only if you have a specific reason to update
  one network and not the other; lock the asymmetric semantic in
  with regression tests if you do.

- **SemVer rationale:** because `critic_hysteresis = true` consumers
  observe a behavioural change in unchanged code, this release is
  major per SemVer 2.0.0. Cargo respects the major boundary —
  downstream `pc-rl-core = "2"` users will NOT receive v3.0.0
  automatically; they must update `Cargo.toml` to `pc-rl-core = "3"`
  explicitly to adopt the breaking change.

- **Internal refactor:** `is_replay_floor_sentinel` and
  `validate_replay_floor` helpers added to dedupe the floor-field
  logic shared between actor and critic. No public API impact.

## [2.2.1] - 2026-04-24

### Added

- `PcActorCriticConfig::scale_floor_replay: f64` — opt-in override
  for replay-path actor learning-rate floor under FROZEN hysteresis.
  Default `-1.0` (sentinel) preserves v2.2.0 behavior exactly;
  values `> 0.0` let consumers enable replay-driven actor updates
  even during hysteresis FROZEN stress. When opted in (strict
  positive), the `skip_kl` gate is also bypassed so Polyak and
  Frozen KL anchors contribute to the replay update. The value
  `0.0` is accepted but functionally identical to the default,
  reserved for documentary use.

### Fixed

- Polyak target EMA no longer drifts toward the live actor during
  prolonged hysteresis FROZEN windows. Previously the EMA update
  ran unconditionally, causing `θ_polyak → θ_live_frozen` after
  `~1/polyak_tau` FROZEN steps and degrading `rollback_soft()` to
  a no-op. The EMA now advances only when the actor weights
  actually change (`s_scale > 0`), preserving the target's
  reference value across FROZEN windows. No config change needed;
  behavior is strictly better for any configuration that uses the
  Polyak target (`distillation_lambda_polyak > 0`).
- Config validation now rejects non-finite `scale_floor_replay`
  (NaN, ±Infinity) and values greater than `10 × scale_ceil` at
  both `PcActorCritic::new` and `apply_config` paths. Pre-v2.2.1
  code accepted these silently, producing undefined-behavior
  weight updates. This closes MAGI Melchior + Caspar Checkpoint 2
  iter 2 hardening requirement.

### Notes

- **Adding a public field to `PcActorCriticConfig` is technically a
  SemVer-minor concern.** Downstream consumers using the builder
  pattern (`PcActorCriticConfig { ..., ..Default::default() }`) or
  serde deserialization are unaffected. Consumers using exhaustive
  struct literals (`PcActorCriticConfig { field1: x, field2: y,
  ..., <all fields listed> }`) will get a missing-field compile
  error on upgrade — the fix is to add `scale_floor_replay: -1.0`
  to the literal or switch to `..Default::default()`. We ship this
  as a patch because the correct remediation is mechanical and
  non-behavioral.
- **Serde unknown-fields policy:** `PcActorCriticConfig` does NOT
  use `#[serde(deny_unknown_fields)]`. Legacy save files that lack
  the new field deserialize cleanly via `#[serde(default)]`;
  forward compatibility with future fields is also preserved
  (unknown fields are silently ignored on load). This policy is
  unchanged from v2.0.0+.
- **Polyak-target dynamics shift under replay opt-in.** Consumers
  running with `distillation_lambda_polyak > 0` AND the new
  `scale_floor_replay > 0.0` opt-in should expect a measurable
  change in Polyak-target behavior. The EMA semantic is "target
  tracks actor movement, not plasticity label": under the default
  sentinel, FROZEN actors don't move during replay and the target
  doesn't advance; under the opt-in, the actor DOES move during
  replay and the Polyak target tracks those replay-driven changes.
  This is symmetric with the online `scale_floor > 0` case and
  intentional — the target becomes partially shaped by the replay
  compartment, not only by on-policy trajectories. No action
  required if you keep the `-1.0` default; reconsider your
  `polyak_tau` if you opt in and observe target lag differently
  from v2.2.0.

## [2.2.0] - 2026-04-19

This release bundles everything since the v2.0.0 tag: the Continuous
Learning feature set (originally planned as v2.1.0 but never tagged)
plus the full Phase 1 + Phase 2 Self-Recovery toolkit.

### Breaking Changes

- `PcActorCritic::seal_replay_training_memories()` now returns
  `Result<(), PcError>` instead of `()`. The method surfaces a
  `PcError::ConfigValidation` on a buffer-less agent
  (`replay_training_capacity == 0` at construction and no subsequent
  `apply_config` allocation) so it matches the symmetric signature of
  `clear_recent_memories()`. Previously it was a silent no-op on
  misconfiguration, which made pipeline-wiring bugs invisible.
  Migration: `agent.seal_replay_training_memories();` →
  `agent.seal_replay_training_memories()?;` (or `.unwrap()` in tests).

### Added

- Self-distillation with dual anchors for self-recovery during continuous learning.
  **The two anchors serve fundamentally different purposes (MAGI R4 W5 — read
  this before tuning either lambda):**
  - **Polyak-tracked target (`distillation_lambda_polyak`, `polyak_tau`)** —
    purpose: **noise smoothing.** The Polyak target tracks the live actor with
    a lag of approximately `1/polyak_tau` steps (~200 at the default
    `polyak_tau = 0.005`). Use case: a soft inertia against short-term
    gradient noise, preventing the live actor from oscillating around a local
    minimum. **NOT useful for cascade recovery** — by the time a slow
    cascade has been detected (typically over hundreds of steps), the Polyak
    target has tracked the cascade itself and `rollback_soft()` recovers
    only a small fraction of the degradation. Use `rollback_soft()` for
    short-window noise spikes, not for sustained drift.
  - **Frozen champion anchor (`distillation_lambda_frozen`)** — purpose:
    **cascade recovery.** The frozen anchor is **immutable** between explicit
    `champion_update()` calls. It does not track live drift at all.
    Use case: provide a stable rollback target that the consumer can fall
    back to via `rollback_hard()` when fitness signals indicate sustained
    degradation. This is the workhorse for catastrophic-forgetting recovery.
    Loaded automatically from saved actor weights at `load_agent()` time, or
    auto-initialized from the live actor on legacy save files.
  - **Asymmetry summary table:**

    | | Polyak target | Frozen champion |
    |---|---|---|
    | Updates after every step | yes (`τ` blend) | no (only `champion_update`) |
    | Tracks live drift | yes (with lag) | no |
    | Recovery method | `rollback_soft` | `rollback_hard` |
    | Detection window | ~`1/polyak_tau` steps | unbounded |
    | Use case | noise smoothing | cascade recovery |
    | KL coefficient | `distillation_lambda_polyak` | `distillation_lambda_frozen` |

  - Both anchors contribute additive KL regularization terms to the actor
    loss when their respective lambdas are non-zero. They can be enabled
    independently or together; the defaults are 0.0 (no slot allocation).
  - New public methods: `rollback_soft()` (live ← polyak), `rollback_hard()`
    (live ← frozen, polyak ← frozen), `champion_update()` (frozen ← live).
  - **`rollback_hard` has a load-bearing cooldown contract** (~100-200 steps
    at `gamma = 0.99`). See the rustdoc on `rollback_hard` for the full
    rationale; calling it in a tight loop in an automated pipeline can
    trigger a positive feedback loop that deepens the EWC-protection gap.
  - New helpers `PcActor::polyak_update_from()` and `PcActor::copy_weights_from()`
    are independently useful for GA crossover and teacher-student setups.
- Backward-compatible serialization: legacy save files without the new anchor
  weights load cleanly with anchors auto-initialized from the live actor.
- Dual-compartment replay buffer for off-policy learning from stored positive-reward
  trajectories.
  - Two FIFO ring buffers: training memories (compartment A, immutable after
    `seal_replay_training_memories()`) and recent stress successes (compartment B,
    FIFO eviction).
  - New method `replay_learn(batch_size)` samples 50/50 from both compartments and
    applies off-policy TD updates via `learn_continuous_inner` with `replay_mode = true`.
  - Auto-records transitions in `step_masked()` when buffer is configured.
  - EWC, Polyak/Frozen distillation regularizers all apply to replay updates
    automatically (single update path, controlled by `replay_mode` flag).
  - Replay updates do NOT contaminate online state: GAE trace, hysteresis counters,
    and `td_error_buffer` are preserved unchanged.
  - New method `clear_recent_memories()` discards compartment B without
    touching compartment A. Use during cascade-recovery in self-play setups
    where compartment B may be contaminated by transitions from a degraded
    policy.
  - New accessor `replay_clamp_count() -> u64` — monotonic telemetry for
    the number of `replay_learn` updates where the td_error clamp
    bound. Sustained incrementing is the leading indicator that
    off-policy drift is approaching its envelope. The counter also
    increments on non-finite raw td_error (±Inf), so monitoring
    dashboards surface catastrophic saturation events.
  - Save/load now persists `replay_clamp_count`,
    `steps_since_last_rollback_hard`, and `rollback_hard_cooldown_steps`
    so monitoring and the cooldown gate survive save/load cycles.
    Legacy pre-Phase-2 files deserialize with bootstrap defaults.

### Refactored

- `learn_continuous_inner` gains two new parameters to support off-policy
  replay without code duplication: `mode: LearnMode` and
  `pre_td_error: Option<f64>`. All existing online callers pass
  `LearnMode::Online` and `pre_td_error: None`; zero behavior change to
  non-replay paths.
- Rollback/champion control methods (`rollback_soft`, `rollback_hard`,
  `champion_update`, `set_rollback_hard_cooldown`) extracted into a
  dedicated `src/pc_actor_critic/control.rs` submodule. Shared helpers
  `reset_actor_transient_state` and `clear_actor_fisher_ema` live in
  the parent module. Public API unchanged; internal reorganisation
  improves maintainability of the ~11k-line `mod.rs`.
- Anchor slot allocation (Polyak + Frozen) centralised in a single
  `Self::allocate_anchor_slots` helper applied at every constructor
  and `apply_config` site. Closes the MAGI Gate A DRY warning.
- Fisher-EMA gradient clip in `accumulate_actor_fisher_ema` now
  references `crate::matrix::GRAD_CLIP` instead of a hardcoded 5.0
  literal, matching the rest of the crate.

### Note on tuning

The self-recovery mechanism shipped in this release is a **toolkit, not a
pre-tuned solution** (MAGI R3 W5). The dual-anchor distillation
(`distillation_lambda_polyak`, `distillation_lambda_frozen`, `polyak_tau`),
the replay buffer capacities (`replay_training_capacity`,
`replay_recent_capacity`), the EWC strength (`ewc_lambda`), and the
hysteresis thresholds form a 7+ dimensional hyperparameter surface whose
optimal values depend strongly on the consumer's environment dynamics,
reward sparsity, and drift profile. The defaults shipped in
`PcActorCriticConfig` (e.g., `polyak_tau = 0.005`, `lambda = 0.0` for both
distillation terms) are **conservative starting points**, not validated
recommendations. Downstream consumers MUST conduct their own empirical
tuning sweeps and should expect 50-200 episodes of evaluation per
configuration to get a meaningful signal. The library does not (and
cannot) provide auto-tuning; that responsibility lives in the consumer.

### Fixed
- Cross-wake couplings in `PcActorCritic::process_hysteresis` now fire on
  sustained plastic state (not only on FROZEN→PLASTIC transitions), preventing
  a deadlock where both networks converge to stable equilibria and neither
  cross-wake can trigger. Affects configurations with `actor_hysteresis = true`,
  `critic_hysteresis = true`, and at least one bidirectional coupling enabled.

### Changed
- Under bidirectional coupling + EWC, `handle_fisher_wake` now fires on
  sustained-path cross-wake events (previously fired only on natural
  FROZEN→PLASTIC transitions). This is a behavior change for downstream EWC
  experiments: Fisher refresh cadence may differ under long-running
  configurations where cross-wake firings were previously deadlocked.
  No config migration needed; existing `ClState` checkpoints remain compatible.

## [2.1.0] - 2026-04-11

### Added
- **Continuous Learning (CL)** via intrinsic surprise modulation:
  - M1: Configurable surprise-to-learning-rate scale range (`scale_floor`/`scale_ceil`)
  - M2: Dual-EWMA hysteresis with automatic FROZEN/PLASTIC transitions
  - M3a: Fixed exponential layer-wise consolidation decay
  - M3b: Adaptive sigmoid layer-wise decay driven by per-layer prediction error EMA
  - M4: EWC regularization with Fisher diagonal lifecycle (decay/accumulate/merge)
  - Logits reversal for Fisher estimation (optional)
  - NaN safety guards: EwmaTracker, learn_continuous (td_error), push_surprise, push_td_error
  - ClState serialization with `PartialEq` + `Default` build-then-compare
- **TD(n)** generic n-step temporal difference returns:
  - `td_steps: usize` config field (0=TD(0) default, >=2 for multi-step)
  - Buffer + terminal flush architecture with `gamma^n` bootstrap
  - Pre-computed V(s) in flush via `pre_v_s` to avoid stale-estimate bias
  - O(K) suffix-sum for flush returns
  - NaN guard at buffer insertion; `td_steps=1` rejected at construction
- **Bidirectional hysteresis coupling**:
  - `critic_wakes_actor` coupling symmetric to existing `actor_wakes_critic`
  - Both default `true` (active when hysteresis enabled)
  - `actor_frozen_steps` counter symmetric to `critic_frozen_steps`
  - EWMA k reset on coupling-forced wake prevents immediate re-freeze
  - Anti-cascade proven safe (Frozen guard + warmup guard)
- `step()` and `step_masked()` unified step API with internal TD(0)/TD(n) learning
- `step_masked()` returns `Result<usize, PcError>` (was panic on empty valid_actions)
- `compute_decay_factors()` single source of truth (extracted from 3x duplication)
- Monolith `pc_actor_critic.rs` refactored into 6-file directory submodule:
  `config.rs`, `ewma.rs`, `hysteresis.rs`, `fisher.rs`, `trajectory.rs`, `mod.rs`
- 524 tests (was 392), 21 doctests

### Changed
- `PcActorCriticConfig` has ~15 new serde-default fields (all backward compatible)
- `Matrix` now derives `Default` + `PartialEq`
- `ClState`, `FisherStateSerialized`, `EwmaTrackerSerialized` derive `Default` + `PartialEq`
- `EwmaTracker.k` can be reset to 0 on coupling-forced wake (doc updated)
- `actor_wakes_critic` default changed from `false` to `true`

### Fixed
- M3b `layer_error_ema` was never updated during learning (now updates in `learn_continuous`)
- NaN reward propagates silently through weight updates (td_error guard added)
- NaN/Inf silently corrupts EwmaTracker (finite guard added)

## [2.0.0] - 2026-04-05

### Breaking Changes
- `LinAlg` trait methods now require `&self` (instance methods instead of static)
- `vec_as_slice` removed from `LinAlg` trait (31 methods remain)
- All struct constructors (`PcActorCritic::new`, `PcActor::new`, `MlpCritic::new`) now take `backend: L` as first parameter
- `load_agent` and `load_agent_generic` now take `backend` as second parameter
- `PcActorCritic::from_parts` now takes `backend: L` as last parameter
- Generic functions in `matrix.rs` (`cca_neuron_alignment`, `standardize_columns`) now take `backend: &L` as first parameter
- `LinAlg` backends used with serde-derived structs must implement `Default`

### Migration Guide
1. Create a backend instance: `let backend = CpuLinAlg::new();`
2. Pass it to constructors: `PcActorCritic::new(backend, config, seed)`
3. Pass it to load functions: `load_agent("path.json", CpuLinAlg::new())`
4. Replace `L::method(args)` with `backend.method(args)` in generic code
5. Update calls to `cca_neuron_alignment(&backend, ...)`, `standardize_columns(&backend, ...)`, etc.
6. Custom `LinAlg` backends must implement `Default` for serde compatibility
7. Serialization format is unchanged — v1.x JSON files load in v2.0

### Added
- `CpuLinAlg::new()` constructor
- `impl Default for CpuLinAlg`
- `backend: L` field on `Layer`, `PcActor`, `MlpCritic`, `PcActorCritic`

## [1.2.3] - 2026-04-04

- Resolve skip projection validation, SVD doc, NaN sort, and buffer size guard

## [1.2.2] - 2026-04-03

- Standalone library crate restructuring
- CD release pipeline with tag-version validation

## [1.2.1] - 2026-04-02

- Adaptive surprise with configurable buffer
- Serde defaults for all config fields

## [1.2.0] - 2026-04-01

- Golub-Kahan SVD O(n^3) replacing Jacobi O(n^4)

## [1.1.0] - 2026-03-30

- CCA crossover with Hungarian matching for GA evolution

## [1.0.0] - 2026-03-25

- Initial release: PC Actor-Critic with predictive coding inference
