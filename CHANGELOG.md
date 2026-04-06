# Changelog

## [2.0.0] - 2026-04-05

### Breaking Changes
- `LinAlg` trait methods now require `&self` (instance methods instead of static)
- `vec_as_slice` removed from `LinAlg` trait (31 methods remain)
- All struct constructors (`PcActorCritic::new`, `PcActor::new`, `MlpCritic::new`) now take `backend: L` as first parameter
- `load_agent` and `load_agent_generic` now take `backend` as second parameter
- `PcActorCritic::from_parts` now takes `backend: L` as last parameter
- Generic functions in `matrix.rs` (`cca_neuron_alignment`, `standardize_columns`, `compute_matrix_sqrtm`, `generic_svd_alignment`) now take `backend: &L` as first parameter
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

- Resolve MAGI findings N1-N4: skip projection validation, SVD doc, NaN sort, buffer size guard

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
