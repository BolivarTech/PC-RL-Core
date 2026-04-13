# `apply_config()` -- Runtime Configuration Update

## Status

**Approved** -- Ready for TDD implementation plan.

## Motivation

When training in non-stationary environments (curriculum learning, domain
transitions), the consumer needs to change learning parameters mid-training
without losing learned weights. The current API requires either:

1. Creating a new agent (loses weights), or
2. Manually mutating `agent.config` fields (bypasses validation, does not
   rebuild derived state like hysteresis machines or decay factors).

A stress test exposed this concretely: loading a serialized agent with
`from_parts()` ignores CL config changes because the saved config overrides
the caller's intent. `apply_config()` provides a validated, atomic way to
reconfigure an agent while preserving its learned representations.

### Use Cases

| Scenario | What changes | What must survive |
|----------|-------------|-------------------|
| Curriculum advancement | surprise thresholds, hysteresis windows | weights, topology |
| Environment transition | gamma, scale_floor/ceil, CL params | weights, topology |
| EWC phase change | ewc_lambda, fisher_decay | weights, Fisher state rebuilt |
| TD(0) -> TD(n) switch | td_steps | weights, topology |
| GAE enable/disable | gae_lambda | weights, topology |
| Hyperparameter sweep | any learning param | weights, topology |

## Design Decisions

These were resolved during the brainstorming session:

| Question | Decision | Rationale |
|----------|----------|-----------|
| Topology mismatch | **C: Error** -- reject if topology differs | Prevents silent corruption; consumer must create new agent for topology change |
| CL state on reconfig | **C3: Reset total** | Clean slate prevents stale EWC/hysteresis from poisoning new regime; door open to C2 (selective) in future |
| Field scope | **D1c: CL + TD/GAE + gamma + entropy** | Actor/critic lr, temperature, alpha are per-network concerns -- future setter API |
| Validation sharing | **E2: Extract `validate_config()`** | DRY -- `new()` and `apply_config()` share identical validation logic |

## API Surface

### Primary Method

```rust
/// Applies a new configuration to the agent, preserving weights and topology.
///
/// Validates that the new config's topology (actor hidden layers, output size,
/// critic hidden layers) matches the current agent. All continuous learning
/// state is reset to provide a clean baseline for the new configuration.
///
/// # What changes
///
/// - Gamma, surprise thresholds, surprise buffer size
/// - Scale floor/ceil (M1)
/// - Hysteresis parameters (M2) -- state machines rebuilt from scratch
/// - Consolidation decay (M3a/M3b) -- factors recomputed
/// - EWC parameters (M4) -- Fisher state reallocated if lambda changes
/// - TD(n) steps -- buffer cleared and resized
/// - GAE lambda -- trace resized
/// - Entropy coefficient
/// - Logits reversal flag
/// - Bidirectional coupling flags and thresholds
///
/// # What does NOT change
///
/// - Actor weights, biases, residual parameters (alpha_rz, skip projections)
/// - Critic weights and biases
/// - Network topology (layer count, sizes, activations)
/// - Actor lr_weights, alpha, tol, min_steps, max_steps, temperature, local_lambda
/// - Critic lr
/// - RNG state
/// - Backend
///
/// # Errors
///
/// Returns `PcError::ConfigValidation` if:
/// - The new config fails standard validation (same rules as `new()`)
/// - The new config's actor topology does not match the current actor
/// - The new config's critic topology does not match the current critic
///
/// # Future API
///
/// Per-network learning rates, temperature, and inference parameters will be
/// exposed through dedicated setter methods on `PcActor` and `MlpCritic`:
/// - `actor.set_lr(f64)`, `critic.set_lr(f64)`
/// - `actor.set_temperature(f64)`
/// - `actor.set_alpha(f64)`, `actor.set_tol(f64)`, `actor.set_max_steps(usize)`
///
/// These are out of scope for v2.1.x and will be added in a future minor release.
pub fn apply_config(&mut self, config: PcActorCriticConfig) -> Result<(), PcError>
```

### Extracted Validation

```rust
/// Validates a `PcActorCriticConfig` for internal consistency.
///
/// Checks gamma range, surprise buffer size, scale floor/ceil ordering,
/// hysteresis fractions, consolidation decay bounds, EWC parameter bounds,
/// td_steps validity, and gae_lambda/td_steps mutual exclusion.
///
/// Shared by `new()` and `apply_config()` to ensure identical validation
/// rules. Does NOT validate topology match (that requires an existing agent).
///
/// # Errors
///
/// Returns `PcError::ConfigValidation` with a descriptive message on failure.
fn validate_config(config: &PcActorCriticConfig) -> Result<(), PcError>
```

### Topology Validation

```rust
/// Validates that a new config's network topology matches the current agent.
///
/// Checks:
/// - Actor input size matches
/// - Actor hidden layer count matches
/// - Actor hidden layer sizes match (per layer)
/// - Actor output size matches
/// - Critic input size matches
/// - Critic hidden layer count matches
/// - Critic hidden layer sizes match (per layer)
///
/// Activation functions are NOT checked because they are part of the
/// topology/architecture and changing them mid-training is almost certainly
/// a bug, but they don't affect weight dimensions.
///
/// # Errors
///
/// Returns `PcError::ConfigValidation` with a message identifying which
/// layer or dimension mismatches.
fn validate_topology_match(&self, config: &PcActorCriticConfig) -> Result<(), PcError>
```

## Field-by-Field Update Table

### Fields updated by `apply_config()`

| Config field | Derived state rebuilt | Notes |
|-------------|---------------------|-------|
| `gamma` | -- | Direct copy |
| `surprise_low` | -- | Direct copy |
| `surprise_high` | -- | Direct copy |
| `adaptive_surprise` | -- | Direct copy |
| `surprise_buffer_size` | `surprise_buffer` cleared | Buffer reset to empty, new cap from config |
| `entropy_coeff` | -- | Direct copy |
| `scale_floor` | -- | Direct copy |
| `scale_ceil` | -- | Direct copy |
| `actor_hysteresis` | `actor_hysteresis` rebuilt or set to None | Fresh HysteresisState |
| `actor_fast_window` | via actor_hysteresis rebuild | -- |
| `actor_slow_window` | via actor_hysteresis rebuild | -- |
| `actor_wake_fraction` | via actor_hysteresis rebuild | -- |
| `actor_sleep_fraction` | via actor_hysteresis rebuild | -- |
| `critic_hysteresis` | `critic_hysteresis` rebuilt or set to None | Fresh HysteresisState |
| `critic_fast_window` | via critic_hysteresis rebuild | -- |
| `critic_slow_window` | via critic_hysteresis rebuild | -- |
| `critic_wake_fraction` | via critic_hysteresis rebuild | -- |
| `critic_sleep_fraction` | via critic_hysteresis rebuild | -- |
| `actor_wakes_critic` | -- | Direct copy |
| `actor_wakes_critic_threshold` | -- | Direct copy |
| `critic_wakes_actor` | -- | Direct copy |
| `critic_wakes_actor_threshold` | -- | Direct copy |
| `consolidation_decay` | `actor_decay_factors` recomputed | Via `compute_decay_factors()` |
| `critic_consolidation_decay` | `critic_decay_factors` recomputed | Via `compute_decay_factors()` |
| `adaptive_consolidation` | `layer_error_ema` reinitialized | Via `compute_decay_factors()` |
| `consolidation_ema_beta` | -- | Direct copy (used at runtime) |
| `consolidation_sigmoid_k` | -- | Direct copy (used at runtime) |
| `consolidation_error_threshold` | -- | Direct copy (used at runtime) |
| `ewc_lambda` | `actor_fisher`, `critic_fisher` reallocated | Fresh FisherState per layer (or empty Vec if lambda=0) |
| `fisher_decay` | -- | Direct copy |
| `fisher_ema_beta` | min_fisher_phase recalculated | Affects hysteresis warmup |
| `logits_reversal` | -- | Direct copy |
| `td_steps` | `td_buffer` cleared | Fresh VecDeque |
| `gae_lambda` | `actor_trace` resized | `vec![0.0; output_size]` or empty |

### Fields NOT updated (topology / per-network params)

| Config field | Reason | Future API |
|-------------|--------|------------|
| `actor.input_size` | Topology | -- |
| `actor.output_size` | Topology | -- |
| `actor.hidden_layers` | Topology | -- |
| `actor.output_activation` | Topology | -- |
| `actor.lr_weights` | Per-network | `actor.set_lr()` |
| `actor.alpha` | Per-network | `actor.set_alpha()` |
| `actor.tol` | Per-network | `actor.set_tol()` |
| `actor.min_steps` | Per-network | `actor.set_min_steps()` |
| `actor.max_steps` | Per-network | `actor.set_max_steps()` |
| `actor.temperature` | Per-network | `actor.set_temperature()` |
| `actor.local_lambda` | Per-network | `actor.set_local_lambda()` |
| `actor.synchronous` | Per-network | -- |
| `actor.residual` | Topology | -- |
| `actor.rezero_init` | Topology | -- |
| `critic.input_size` | Topology | -- |
| `critic.hidden_layers` | Topology | -- |
| `critic.output_activation` | Topology | -- |
| `critic.lr` | Per-network | `critic.set_lr()` |

### Transient state reset on `apply_config()`

| Field | Reset value | Reason |
|-------|-------------|--------|
| `surprise_buffer` | `VecDeque::new()` | Old surprise stats irrelevant under new config |
| `state_prev` | `None` | Step pipeline reset |
| `action_prev` | `None` | Step pipeline reset |
| `infer_prev` | `None` | Step pipeline reset |
| `valid_actions_prev` | `None` | Step pipeline reset |
| `actor_plastic_step_counter` | `0` | CL total reset |
| `critic_plastic_step_counter` | `0` | CL total reset |
| `critic_frozen_steps` | `0` | CL total reset |
| `actor_frozen_steps` | `0` | CL total reset |
| `td_error_buffer` | `VecDeque::new()` | Old TD errors irrelevant |
| `last_td_error` | `0.0` | CL total reset |
| `actor_last_phase_reliable` | `false` | CL total reset |
| `critic_last_phase_reliable` | `false` | CL total reset |
| `td_buffer` | `VecDeque::new()` | Config may change td_steps |
| `actor_trace` | `vec![0.0; trace_len]` | Config may change gae_lambda |

## Implementation Strategy

### Step 1: Extract `validate_config()`

Move the validation logic from `new()` (lines 192-324) into a standalone
`fn validate_config(config: &PcActorCriticConfig) -> Result<(), PcError>`.
The `new()` method calls this function instead of inlining the checks.

Behavioral equivalence: `new()` must produce identical errors for identical
invalid configs before and after extraction.

### Step 2: Implement `validate_topology_match()`

```rust
fn validate_topology_match(&self, config: &PcActorCriticConfig) -> Result<(), PcError> {
    // Actor checks
    if self.config.actor.input_size != config.actor.input_size {
        return Err(PcError::ConfigValidation(format!(
            "actor input_size mismatch: current {} vs new {}",
            self.config.actor.input_size, config.actor.input_size
        )));
    }
    let cur_ah = &self.config.actor.hidden_layers;
    let new_ah = &config.actor.hidden_layers;
    if cur_ah.len() != new_ah.len() {
        return Err(PcError::ConfigValidation(format!(
            "actor hidden layer count mismatch: current {} vs new {}",
            cur_ah.len(), new_ah.len()
        )));
    }
    for (i, (cur, new)) in cur_ah.iter().zip(new_ah.iter()).enumerate() {
        if cur.size != new.size {
            return Err(PcError::ConfigValidation(format!(
                "actor hidden layer {} size mismatch: current {} vs new {}",
                i, cur.size, new.size
            )));
        }
    }
    if self.config.actor.output_size != config.actor.output_size {
        return Err(PcError::ConfigValidation(format!(
            "actor output_size mismatch: current {} vs new {}",
            self.config.actor.output_size, config.actor.output_size
        )));
    }

    // Critic checks
    if self.config.critic.input_size != config.critic.input_size {
        return Err(PcError::ConfigValidation(format!(
            "critic input_size mismatch: current {} vs new {}",
            self.config.critic.input_size, config.critic.input_size
        )));
    }
    let cur_ch = &self.config.critic.hidden_layers;
    let new_ch = &config.critic.hidden_layers;
    if cur_ch.len() != new_ch.len() {
        return Err(PcError::ConfigValidation(format!(
            "critic hidden layer count mismatch: current {} vs new {}",
            cur_ch.len(), new_ch.len()
        )));
    }
    for (i, (cur, new)) in cur_ch.iter().zip(new_ch.iter()).enumerate() {
        if cur.size != new.size {
            return Err(PcError::ConfigValidation(format!(
                "critic hidden layer {} size mismatch: current {} vs new {}",
                i, cur.size, new.size
            )));
        }
    }

    Ok(())
}
```

### Step 3: Implement `apply_config()`

```rust
pub fn apply_config(&mut self, config: PcActorCriticConfig) -> Result<(), PcError> {
    // 1. Validate new config internally
    Self::validate_config(&config)?;

    // 2. Validate topology match
    self.validate_topology_match(&config)?;

    // 3. Recompute derived state
    let (actor_decay_factors, critic_decay_factors, layer_error_ema) =
        Self::compute_decay_factors(&config);
    let trace_len = Self::gae_trace_len(&config);

    // 4. Rebuild hysteresis state machines
    let actor_hysteresis = if config.actor_hysteresis {
        Some(HysteresisState { /* fresh */ })
    } else {
        None
    };
    let critic_hysteresis = if config.critic_hysteresis {
        Some(HysteresisState { /* fresh */ })
    } else {
        None
    };

    // 5. Update min_initial_plastic for Fisher warmup
    // (same logic as new())

    // 6. Reallocate Fisher state
    let actor_fisher = if config.ewc_lambda > 0.0 {
        // allocate fresh FisherState per actor layer
    } else {
        Vec::new()
    };
    let critic_fisher = if config.ewc_lambda > 0.0 {
        // allocate fresh FisherState per critic layer
    } else {
        Vec::new()
    };

    // 7. Apply all fields atomically
    self.config = config;
    self.surprise_buffer = VecDeque::new();
    self.state_prev = None;
    self.action_prev = None;
    self.infer_prev = None;
    self.valid_actions_prev = None;
    self.actor_hysteresis = actor_hysteresis;
    self.critic_hysteresis = critic_hysteresis;
    self.actor_plastic_step_counter = 0;
    self.critic_plastic_step_counter = 0;
    self.critic_frozen_steps = 0;
    self.actor_frozen_steps = 0;
    self.td_error_buffer = VecDeque::new();
    self.last_td_error = 0.0;
    self.actor_decay_factors = actor_decay_factors;
    self.critic_decay_factors = critic_decay_factors;
    self.layer_error_ema = layer_error_ema;
    self.actor_fisher = actor_fisher;
    self.critic_fisher = critic_fisher;
    self.actor_last_phase_reliable = false;
    self.critic_last_phase_reliable = false;
    self.td_buffer = VecDeque::new();
    self.actor_trace = vec![0.0; trace_len];

    Ok(())
}
```

## Interaction with Existing Features

| Feature | Impact | Notes |
|---------|--------|-------|
| `step()` / `step_masked()` | Step pipeline reset | First call after `apply_config()` acts as if episode just started |
| `learn()` (REINFORCE) | Unaffected | Episodic learning uses trajectory, not transient state |
| `crossover()` | Unaffected | Creates new agent from parents |
| `save()` / `load()` | Config from `apply_config()` is saved | Load + `apply_config()` is the intended pattern for reconfiguration |
| `from_parts()` | Does not call `apply_config()` | Consumer should call `apply_config()` after `from_parts()` to apply new CL config |
| Surprise buffer | Cleared | Old statistics irrelevant under new thresholds |
| TD(n) buffer | Cleared | Partially accumulated returns are discarded |
| GAE trace | Resized | Trace zeroed; first step starts fresh accumulation |

## Error Messages

All validation errors use `PcError::ConfigValidation(String)` with descriptive
messages identifying the specific constraint violated:

```
"actor input_size mismatch: current 9 vs new 4"
"actor hidden layer count mismatch: current 3 vs new 2"
"actor hidden layer 1 size mismatch: current 27 vs new 18"
"actor output_size mismatch: current 9 vs new 4"
"critic input_size mismatch: current 36 vs new 18"
"critic hidden layer count mismatch: current 1 vs new 2"
"critic hidden layer 0 size mismatch: current 36 vs new 24"
```

## Future Work

### C2: Selective CL Reset (deferred)

A future enhancement could allow selective reset of CL subsystems:

```rust
pub struct ApplyConfigOptions {
    pub reset_hysteresis: bool,  // default true
    pub reset_fisher: bool,      // default true
    pub reset_surprise: bool,    // default true
}
pub fn apply_config_with(&mut self, config: PcActorCriticConfig, opts: ApplyConfigOptions)
    -> Result<(), PcError>
```

This is not needed for v2.1.x. The total reset (C3) is the safe default.

### Per-Network Setters (deferred)

Individual setters for per-network parameters that `apply_config()` does not
touch:

```rust
// On PcActorCritic:
pub fn set_actor_lr(&mut self, lr: f64) -> Result<(), PcError>
pub fn set_critic_lr(&mut self, lr: f64) -> Result<(), PcError>
pub fn set_temperature(&mut self, temperature: f64) -> Result<(), PcError>
pub fn set_actor_alpha(&mut self, alpha: f64) -> Result<(), PcError>
pub fn set_actor_max_steps(&mut self, max_steps: usize) -> Result<(), PcError>
```

These are trivial forwarding methods with validation. Not needed for v2.1.x.

## Testing Strategy

Tests follow TDD Red-Green-Refactor:

1. **validate_config extraction** -- Existing `new()` validation tests must pass
   unchanged after extraction (behavioral equivalence).
2. **validate_topology_match** -- Tests for: matching topology (ok), layer count
   mismatch, layer size mismatch, output size mismatch, critic mismatch.
3. **apply_config happy path** -- Apply new CL config, verify config field updated,
   weights unchanged, CL state reset.
4. **apply_config topology reject** -- Verify error on topology mismatch.
5. **apply_config + step()** -- Verify agent works correctly after reconfiguration
   (step pipeline reset, new parameters in effect).
6. **apply_config EWC transition** -- Enable/disable EWC, verify Fisher state
   allocated/deallocated.
7. **apply_config TD/GAE switch** -- Switch td_steps and gae_lambda, verify
   buffers resized correctly.
