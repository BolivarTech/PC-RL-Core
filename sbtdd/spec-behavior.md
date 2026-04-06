# Especificacion: Refactor LinAlg Trait a Instance Methods (v2.0)

## Objetivo

Transformar el trait `LinAlg` de metodos estaticos (sin `&self`) a metodos de instancia (`&self`) para permitir que los backends lleven estado interno (e.g., `Arc<CudaDevice>` en GPU). Eliminar `vec_as_slice` del trait. Agregar campo `backend: L` a todos los structs genericos. Este cambio es prerequisito para implementar `GpuLinAlg` con soporte multi-GPU.

## Requerimientos funcionales (SDD)

### R1: Trait LinAlg con `&self`
- Todos los 32 metodos actuales del trait pasan de `fn method(args)` a `fn method(&self, args)`
- Eliminar `vec_as_slice` del trait (31 metodos resultantes)
- Los tipos asociados `Vector` y `Matrix` no cambian
- Los bounds del trait (`Clone + Send + Sync + 'static`) no cambian

### R2: CpuLinAlg adaptado
- `CpuLinAlg` sigue siendo un unit struct (`struct CpuLinAlg;`)
- Implementar `CpuLinAlg::new() -> Self` como constructor explicito
- Todos los metodos reciben `&self` pero no lo usan (zero-cost, el compilador optimiza)
- `vec_as_slice` se elimina del trait pero sigue disponible como metodo directo en `Vec<f64>` via `Deref`

### R3: Structs genericos con campo `backend`
- `Layer<L>`: agregar campo `backend: L`
- `PcActor<L>`: agregar campo `backend: L`
- `MlpCritic<L>`: agregar campo `backend: L`
- `PcActorCritic<L>`: agregar campo `backend: L`
- El backend se pasa como primer argumento de `new()`:
  - `PcActorCritic::new(backend: L, config: PcActorCriticConfig, seed: u64) -> Result<Self, PcError>`
  - Igual para `PcActor::new()`, `MlpCritic::new()`, `Layer` (interno)

### R4: Llamadas internas migradas
- Todo uso de `L::method(args)` cambia a `self.backend.method(args)` en los structs
- Los type aliases (`PcActorCpu`, `MlpCriticCpu`, etc.) se mantienen
- Funciones libres en `matrix.rs` (`softmax_masked`, `argmax_masked`, etc.) que no pertenecen al trait no cambian

### R5: Serializer adaptado
- `save_agent` y `load_agent` reciben el backend como parametro
- `from_weights` recibe el backend como parametro
- `to_weights` no necesita el backend (extrae datos a CPU types)
- El formato JSON de serializacion no cambia (backward compatible)

### R6: CCA crossover adaptado
- `PcActor::crossover` recibe el backend
- `MlpCritic::crossover` recibe el backend
- Las funciones de alineacion CCA (`cca_neuron_alignment`) no cambian (operan sobre `Matrix` CPU)

## Restricciones

- Zero cambio en logica o algoritmos — solo refactor de firmas
- `CpuLinAlg` debe ser zero-cost (ZST, `&self` eliminado por el compilador)
- Todos los tests existentes deben pasar con cambio mecanico (agregar `CpuLinAlg::new()`)
- El formato de serializacion JSON debe ser backward compatible (un model.json de v1.2.3 debe cargarse en v2.0)
- No agregar dependencias nuevas

## Comportamiento esperado (BDD)

### Escenario 1: Construccion de agente CPU
- **Dado** una configuracion valida de `PcActorCriticConfig`
- **Cuando** se invoca `PcActorCritic::new(CpuLinAlg::new(), config, 42)`
- **Entonces** retorna `Ok(agent)` con el backend `CpuLinAlg` almacenado internamente

### Escenario 2: Construccion de agente con config invalida
- **Dado** una configuracion con `gamma = -0.1`
- **Cuando** se invoca `PcActorCritic::new(CpuLinAlg::new(), config, 42)`
- **Entonces** retorna `Err(PcError::ConfigValidation)` con mensaje que contiene "gamma"

### Escenario 3: Operaciones del trait usan &self
- **Dado** una instancia de `CpuLinAlg::new()`
- **Cuando** se invoca `backend.zeros_vec(10)`
- **Entonces** retorna un vector de 10 elementos, todos cero

### Escenario 4: mat_vec_mul via instancia
- **Dado** una instancia de `CpuLinAlg`, una matriz 3x3 y un vector de 3 elementos
- **Cuando** se invoca `backend.mat_vec_mul(&m, &v)`
- **Entonces** retorna el producto matrix-vector correcto

### Escenario 5: vec_as_slice eliminado del trait
- **Dado** el trait `LinAlg`
- **Cuando** se intenta llamar `backend.vec_as_slice(&v)` en codigo generico
- **Entonces** falla en compilacion (metodo no existe en el trait)

### Escenario 6: Serializacion round-trip con backend
- **Dado** un agente creado con `PcActorCritic::new(CpuLinAlg::new(), config, 42)`
- **Cuando** se guarda con `save_agent(&agent, "test.json")` y se carga con `load_agent("test.json", CpuLinAlg::new())`
- **Entonces** el agente cargado produce las mismas acciones que el original para los mismos inputs

### Escenario 7: Crossover con backend
- **Dado** dos agentes creados con `CpuLinAlg::new()` y caches de activacion
- **Cuando** se invoca crossover entre ambos
- **Entonces** el hijo hereda el backend del padre y produce acciones validas

### Escenario 8: Backend se propaga a sub-componentes
- **Dado** un agente creado con `PcActorCritic::new(CpuLinAlg::new(), config, 42)`
- **Cuando** se accede al actor y al critic internamente
- **Entonces** ambos tienen `backend: CpuLinAlg` almacenado

### Escenario 9: ZST (zero-sized type) para CpuLinAlg
- **Dado** `CpuLinAlg` como unit struct
- **Cuando** se agrega como campo `backend` a un struct
- **Entonces** `std::mem::size_of::<CpuLinAlg>()` es 0 — no agrega memoria

### Escenario 10: Backward compatibility de serializacion
- **Dado** un archivo `model.json` generado con v1.2.3 (sin campo backend)
- **Cuando** se carga con `load_agent("model.json", CpuLinAlg::new())` en v2.0
- **Entonces** carga exitosamente, el backend se asigna del parametro

## Lo que NO debe hacer

- No cambiar la logica de ningun algoritmo (PC inference, backprop, CCA, SVD)
- No cambiar el formato JSON de serializacion
- No agregar `Default` a los config structs
- No cambiar los tipos asociados `Vector` y `Matrix` de CpuLinAlg
- No agregar dependencias nuevas
- No modificar `golub_kahan.rs` internamente (solo adaptar las llamadas a LinAlg si las hay)
