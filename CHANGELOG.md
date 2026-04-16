# Changelog

## [Unreleased]

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
