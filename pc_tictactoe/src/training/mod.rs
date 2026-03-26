// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Training pipelines for the PC Actor-Critic agent.
//!
//! - [`trainer`] — Episode-based training with curriculum learning.
//! - [`continuous`] — Continuous training with surprise-based immediate updates.

pub mod continuous;
pub mod experiment;
pub mod trainer;
