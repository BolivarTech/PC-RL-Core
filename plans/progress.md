# Ralph Progress Log

Started: 2026-04-09
Task: Continuous Learning via Intrinsic Surprise Modulation (pc-rl-core v2.1.0)

## Codebase Patterns

- Rust library crate, all structs generic over `L: LinAlg`
- Tests in `#[cfg(test)] mod tests { ... }` at end of each file
- Helpers: `make_rng()` -> `StdRng::seed_from_u64(42)`, `make_backend()` -> `CpuLinAlg::new()`
- Error type: `PcError` (ConfigValidation, DimensionMismatch, Serialization, Io)
- Config: `#[serde(default = "fn_name")]` pattern
- Constants: WEIGHT_CLIP=5.0, GRAD_CLIP=5.0

## Key Files

- `src/pc_actor_critic.rs` — Main agent (most changes here)
- `src/pc_actor.rs` — Actor with update_weights_hybrid()
- `src/mlp_critic.rs` — Critic with update()
- `src/layer.rs` — Layer::backward() (DO NOT MODIFY signature)
- `src/serializer.rs` — JSON persistence
- `src/error.rs` — PcError enum

## References

- Spec: `sbtdd/spec-behavior.md`
- Plan: `sbtdd/claude-plan.md`
- TDD stubs: `sbtdd/claude-plan-tdd.md`
- Section files: `sbtdd/sections/section-01-*.md` through `section-07-*.md`

---

## Task 001: Unified step() API — COMPLETED (iteration 1)

**Implementation was already present** from a prior session. The step(), step_masked(), reset_step() methods and step_inner() helper were fully implemented with all 12 tests passing.

**What was done this iteration:**
- Added `#[allow(deprecated)]` to 7 existing tests calling `learn()` that were producing compile warnings
- Fixed in `pc_actor_critic.rs`: test_learn_single_step_trajectory, test_learn_multi_step_uses_stored_hidden_states, test_entropy_regularization_prevents_policy_collapse, test_learn_improves_policy_for_rewarded_action, test_agent_crossover_child_can_learn
- Fixed in `serializer.rs`: test_roundtrip_preserves_modified_rezero_alpha, test_roundtrip_preserves_skip_projections_directly
- Ran `cargo fmt` to fix formatting on the existing step() code

**Patterns:**
- `step_inner()` is a shared helper that both `step()` and `step_masked()` delegate to
- Transient state fields (state_prev, action_prev, infer_prev, valid_actions_prev) are Option types, None on construction
- Terminal flag clears all transient state after learning

## Task 002: Configurable Scale Range (M1) — COMPLETED (iteration 2)

**Changes:**
- Added `scale_floor` (default 0.0) and `scale_ceil` (default 2.0) config fields to `PcActorCriticConfig`
- Added serde default functions `default_scale_floor()` and `default_scale_ceil()`
- Added validation in `new()`: floor >= 0, ceil > floor
- Updated `surprise_scale()` to use config fields instead of hardcoded 0.1/2.0
- Updated test helper `default_config()` with `scale_floor: 0.1` for v2.0.0 backward compat
- Updated serializer test helper similarly
- Updated docstring example to include new fields
- 7 new tests, 411 total passing

**Patterns:**
- Default `scale_floor` is 0.0 (true freeze), but test helper uses 0.1 to preserve existing test assertions
- `scale_ceil == scale_floor` is rejected (degenerate case) — ceil must be strictly greater
- Serde defaults ensure backward compat with legacy JSON (missing fields default to 0.0/2.0)

## Task 003: EwmaTracker & PlasticityState — COMPLETED (iteration 3)

**Changes:**
- Added `PlasticityState` enum (Frozen/Plastic) with `#[default]` on Plastic, full serde support
- Added `EwmaTracker` struct with warmup arithmetic mean → steady-state EMA transition
- Formula: `A(k) = A(k-1) + (V - A(k-1)) / min(N, k)` — warmup is exact mean, steady-state is EMA with alpha=1/N
- Exported both types from `lib.rs`
- Added doctest example for `EwmaTracker`
- 8 new tests, 419 total passing

**Patterns:**
- Types placed at module scope in `pc_actor_critic.rs`, above `PcActorCriticConfig`
- All fields `pub` matching codebase convention for config/state structs
- `k: u64` is monotonic counter that never resets
- Clippy requires `#[default]` attribute on enum variant instead of manual `Default` impl

## Task 004: Dual-EWMA Hysteresis Integration — COMPLETED (iteration 4)

**Changes:**
- Added `HysteresisState` struct with dual-EWMA state machine (fast/slow EWMAs, wake/sleep fractions, warmup guard)
- Added 12 config fields for actor/critic hysteresis + actor_wakes_critic coupling (all `#[serde(default)]`)
- Added 7 new fields to `PcActorCritic`: actor/critic hysteresis (Option), plastic step counters, critic_frozen_steps, td_error_buffer, last_td_error
- Added validation: wake_fraction > 0, sleep_fraction in (0.0, 1.0) when hysteresis enabled
- Added `critic_surprise_scale()` — identical to `surprise_scale()` but reads from td_error_buffer
- Added `effective_actor_scale()` — returns scale_floor when actor FROZEN, else surprise_scale()
- Added `process_hysteresis()` — updates state machines, handles transitions, coupling, counters
- Modified `learn_continuous()` to use effective_actor_scale and push |td_error| to buffer
- Modified `step_inner()` to call process_hysteresis after learning
- Updated crossover() and from_parts() to include new fields (clean defaults)
- Exported `HysteresisState` from lib.rs
- 15 new tests, 434 total passing

**Patterns:**
- HysteresisState placed at module scope after EwmaTracker, before PcActorCriticConfig
- Hysteresis disabled by default (actor_hysteresis/critic_hysteresis=false → Option::None)
- `min_initial_plastic` set to `slow_window` at construction (Phase 4 will add Fisher term: max(slow_window, min_fisher_phase))
- Counter increment happens BEFORE transition check in process_hysteresis (pre-transition counting)
- Transition resets counter to 0 on the transition step (counter starts accumulating next step)
- actor_wakes_critic coupling checked after both state machines update, only on actor FROZEN→PLASTIC transition
- td_error_buffer uses same surprise_buffer_size for capacity

## Task 005: Layer-Wise Consolidation Decay (M3) — COMPLETED (iteration 5)

**Changes:**
- Added 6 config fields: consolidation_decay, critic_consolidation_decay, adaptive_consolidation, consolidation_ema_beta, consolidation_sigmoid_k, consolidation_error_threshold
- Added serde defaults and validation for all new fields (range checks, conditional validation for M3b)
- Added actor_decay_factors, critic_decay_factors (Vec<f64>) precomputed in new() using `decay^(n-1-i)` formula
- Added layer_error_ema (Vec<f64>) for M3b adaptive sigmoid decay (actor only)
- Added effective_actor_decay() method: returns M3b sigmoid factors or M3a fixed factors
- Modified PcActor::update_weights() and update_weights_hybrid() to accept &[f64] decay_factors
- Per-layer surprise: `layer_surprise = surprise_scale * decay_factors[i]` for hidden layers; output layer always gets raw surprise_scale
- Added MlpCritic::update_with_decay() for critic with per-layer decay
- Updated learn_continuous() to pass actor/critic decay factors
- Updated deprecated learn() to pass actor decay factors
- Updated crossover() and from_parts() to include new fields (clean defaults)
- Updated doctest, test helpers (default_config, three_layer_config), serializer test helper
- 18 new tests, 452 total passing

**Patterns:**
- Decay factors precomputed at construction, not per-step — avoids powi() in hot path
- Empty decay_factors slice means no per-layer decay (backward compat)
- M3b overrides M3a for actor; critic always uses M3a (no PC inference for per-layer errors)
- Sigmoid formula: adaptive_decay = sigmoid(-k * (error - threshold)), effective = (1 - adaptive_decay)
- Cold start (EMAs=0): ~38% effective LR (partially protected until EMAs warm up)
- Single hidden layer: decay = 1.0 regardless of base (exponent is 0)
- Zero hidden layers: empty decay factors, safe no-op

## Task 006: EWC Regularization (M4) — COMPLETED (iteration 6)

**Changes:**
- Added `FisherState<L>` struct: per-layer Fisher information (f_total, f_ema, theta snapshots)
- Added 4 config fields: `ewc_lambda` (default 0.0), `fisher_decay` (0.9), `fisher_ema_beta` (0.99), `logits_reversal` (false)
- Added validation: ewc_lambda >= 0, fisher_decay in [0,1], fisher_ema_beta in (0,1) when ewc_lambda > 0
- Added `actor_fisher`, `critic_fisher`, `actor_last_phase_reliable`, `critic_last_phase_reliable` to PcActorCritic
- Fisher allocation only when ewc_lambda > 0 (zero overhead when disabled)
- 3-step Fisher lifecycle: `handle_fisher_wake()` (decay F_total), `handle_fisher_sleep()` (merge/discard F_ema, snapshot)
- min_fisher_phase = ceil(1/(1-beta)); updates hysteresis min_initial_plastic = max(slow_window, min_fisher_phase)
- `accumulate_actor_fisher_ema()`: gradient extraction via Approach 1 (activation derivative, hadamard, clip)
- `update_fisher_ema_layer()`: F_ema = beta * F_ema + (1-beta) * g_raw²
- `apply_actor_ewc_correction()`: W -= effective_lr * lambda * F_total * (W_pre - snapshot), then WEIGHT_CLIP
- Logits reversal: delta_fisher = softmax(-y_conv/T, valid) - one_hot(action)
- Integrated into `learn_continuous()`: Fisher EMA + EWC when enabled, standard backward otherwise
- Integrated into `process_hysteresis()`: Fisher lifecycle on FROZEN↔PLASTIC transitions
- Made `PcActor::skip_alpha_index()` pub(crate) for gradient extraction access
- Exported `FisherState` from lib.rs
- Updated doctest, all test helpers (default_config, three_layer_config, serializer)
- 31 new tests, 484 total passing (+ 21 doctests)

**Patterns:**
- Fisher state lives on PcActorCritic, not PcActor — avoids modifying Layer::backward
- Gradient extraction is a read-only pass mirroring update_weights_hybrid logic
- EWC correction is post-backward per-layer: captures pre-update W, runs backward, applies correction
- Propagated gradient is clean (no EWC contamination) — verified by comparing hidden layer deltas
- Diagonal Fisher approximation: F_ema_w[r][c] = beta * F_ema_w[r][c] + (1-beta) * grad[r]²
- WEIGHT_CLIP (5.0) applied via .clamp() after EWC correction
- Short PLASTIC phases (< min_fisher_phase) discard F_ema, don't apply fisher_decay on next wake
- F_total preserved through rapid oscillations (5 cycles verified)
- Logits reversal produces different Fisher EMA but same weight updates (no snapshot = no correction)

## Task 007: GA Crossover Reset & Serialization — COMPLETED (iteration 7)

**Changes:**
- Added serializable CL structs: `FisherStateSerialized`, `EwmaTrackerSerialized`, `HysteresisStateSerialized`, `ClState`
- Extended `SaveFile` with `#[serde(default)] cl_state: Option<ClState>` for backward compat
- Added `PcActorCritic::to_cl_state()` — extracts all CL state into CPU-side serializable types
- Added `PcActorCritic::restore_cl_state()` — restores CL state from deserialized `ClState`
- Updated `save_agent()` to include `cl_state` in SaveFile
- Updated `load_agent_generic()` to call `restore_cl_state()` when `cl_state` is present
- Crossover already resets CL state to clean defaults (verified with new tests)
- 10 new tests (3 crossover reset + 6 serialization + 1 default config), 494 total passing

**Patterns:**
- `to_cl_state()` returns `None` when no CL features are active (zero overhead for v2.0.0 behavior)
- `restore_cl_state()` is called post-`from_parts()` — doesn't change `from_parts` API
- Legacy JSON (no `cl_state` field) loads as `None` → agent gets clean PLASTIC defaults via `from_parts`
- Fisher state serialization mirrors `to_weights()` pattern: L::Matrix → Matrix via element-wise copy
- Transient step state (state_prev, action_prev, infer_prev, valid_actions_prev) is NOT serialized
- All serializable structs use `#[serde(default)]` on optional fields for forward compat
