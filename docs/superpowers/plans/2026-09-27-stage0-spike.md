# tinyloop Stage-0 Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether `mlx-rs` 0.32 can train this model on the M4. Implement the three hard kernels (Gated DeltaNet chunked scan, NSA-style block-sparse attention, gather-based MoE), prove each correct in forward pass and gradients against a naive reference, and measure 16k-token throughput.

**Architecture:** A single-crate library, `tinyloop`. Kernels are pure functions over `mlx_rs::Array` in `src/model/`, and test helpers live in `src/util.rs`. Correctness tests live in `tests/reference.rs`. A benchmark binary `src/bin/spike_bench.rs` prints timings and an estimated full-model tokens/s. There are no model modules, parameters, optimiser, or data yet; those belong to plan 2, written after this spike's results.

**Tech Stack:** Rust 1.97, `mlx-rs = "=0.32.0"` (MLX with Metal), `anyhow`.

**Spec:** `docs/specs/2026-09-26-tinyloop-design.md`, sections 2.2, 3.4, and 7 (tests 1–4 plus the throughput benchmark).

## Global Constraints

- Rust only. `mlx-rs` pinned to `=0.32.0`. No other ML dependencies.
- **Every `cargo` command in this repo is a heavy job.** `mlx-sys` compiles MLX C++ through cmake during `cargo check`, `cargo test`, and `cargo build`. Before the first cargo command of a session, message the other Claude sessions (`ListAgents`, then `SendMessage` to each) and wait if a Laguna job is running. Never run cargo while current-41's overnight run is active.
- Never write outside `tinyloop/`. Never write to `lagunaLab/`.
- MLX memory limit: 8 GB (`mlx_rs::memory::set_memory_limit(8 << 30)`) in the benchmark.
- MLX must build in Release mode even for `cargo test` (debug-mode MLX is unusably slow). This is enforced through `[profile.dev.build-override]` in Task 1.
- Kernel dimension conventions (every task): `[B, H, T, D]`; `Hq = G · Hkv`; `T` is a multiple of the block/chunk size; the DeltaNet chunk is a power of two.
- Test tolerances (fp32): forward ≤ 1e-4 absolute, gradients ≤ 1e-3 absolute, finite differences ≤ 1e-2 relative.

## Review Focus

1. Early tokens that have no complete compressed block yet: the output must be finite (the zero-logit sink), not NaN. Test in Task 4.
2. A sequence with fewer blocks than `topk` (e.g. T = 32, block = 16, topk = 3): selection clamps `topk` and still works. Test in Task 4.
3. Strong decay (`g` around −20 per token): the chunked DeltaNet must not overflow to inf/NaN, because the upper-triangle masking happens before `exp`. Test in Task 3.
4. `T` not a multiple of the chunk or block size: fail loudly with a clear message, not a reshape error deep inside MLX. `#[should_panic]` tests in Tasks 3 and 4.
5. bf16 inputs: no NaN, and output close to fp32. The benchmark in Task 7 reports the maximum difference and NaN count per kernel.

---

### Task 0: Prerequisites (user action, no code)

`mlx-sys` needs Apple's `metal` shader compiler. This Mac has only Command Line Tools (`xcrun: error: unable to find utility "metal"`).

- [ ] **Step 1: User installs Xcode** from the App Store (about 10+ GB download), then runs:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

- [ ] **Step 2: Verify**

Run: `xcrun -f metal && cmake --version`
Expected: a path ending in `/metal`, then `cmake version 4.x`.

- [ ] **Step 3: Coordination check.** Message the other sessions, and confirm current-41 has sent its overnight end message and that no Laguna load is active.

---

### Task 1: Crate scaffold and mlx-rs capability smoke test

Proves the build works, that MLX builds in Release mode, and that every op the kernels depend on has a working gradient: `take_along_axis`, `gather_mm`, `stop_gradient`, `argpartition_axis`, and the memory limit. If `gather_mm` has no gradient, record that and switch Task 5 to dense experts (see the Task 5 note).

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `src/lib.rs`, `src/util.rs`, `src/model/mod.rs`, `tests/smoke.rs`

**Interfaces:**
- Produces (`src/util.rs`, used by every later task):
  - `pub type R<T> = Result<T, mlx_rs::error::Exception>;`
  - `pub fn dims4(a: &Array) -> [i32; 4]`
  - `pub fn positions(n: i32) -> Array` (int32 `[n]`, values `0..n`)
  - `pub fn randn(shape: &[i32], seed: u64) -> R<Array>`
  - `pub fn rand_uniform(lo: f32, hi: f32, shape: &[i32], seed: u64) -> R<Array>`
  - `pub fn max_abs_diff(a: &Array, b: &Array) -> R<f32>`
  - `pub fn assert_close(name: &str, a: &Array, b: &Array, atol: f32)`
  - `pub fn grads_of<F: Fn(&[Array]) -> R<Array>>(f: &F, args: &[Array], cot: &Array) -> R<(Array, Vec<Array>)>`: the value and gradients of `sum(f(args) * cot)` with respect to every argument.
  - `pub fn fd_directional<F: Fn(&[Array]) -> R<Array>>(f: &F, args: &[Array], cot: &Array, i: usize, dir: &Array, eps: f32) -> R<f32>`

- [ ] **Step 1: Write `Cargo.toml`, `.gitignore`, and the module skeleton**

`Cargo.toml`:
```toml
[package]
name = "tinyloop"
version = "0.1.0"
edition = "2021"

[dependencies]
mlx-rs = "=0.32.0"
anyhow = "1"

# mlx-sys builds MLX in Debug when its build script has debug_assertions;
# force Release MLX for every profile (tests would otherwise be unusably slow).
[profile.dev.build-override]
opt-level = 3
debug-assertions = false

[profile.dev]
opt-level = 1

[profile.release]
opt-level = 3
lto = "thin"
```

`.gitignore`:
```
/target
```

`src/lib.rs`:
```rust
pub mod model;
pub mod util;
```

`src/model/mod.rs`:
```rust
//! Model kernels. The spike only has pure functions; modules with parameters arrive in plan 2.
```

`src/util.rs`:
```rust
//! Shared helpers: result alias, shapes, seeded random tensors, test comparisons, gradients.

use mlx_rs::{
    error::Exception,
    random,
    transforms::value_and_grad_with_argnums,
    Array,
};

pub type R<T> = Result<T, Exception>;

pub fn dims4(a: &Array) -> [i32; 4] {
    let s = a.shape();
    assert_eq!(s.len(), 4, "expected a 4-d array, got shape {s:?}");
    [s[0], s[1], s[2], s[3]]
}

/// int32 `[n]` holding `0..n`.
pub fn positions(n: i32) -> Array {
    Array::from_iter(0..n, &[n])
}

pub fn randn(shape: &[i32], seed: u64) -> R<Array> {
    let key = random::key(seed)?;
    random::normal::<f32>(shape, None, None, &key)
}

pub fn rand_uniform(lo: f32, hi: f32, shape: &[i32], seed: u64) -> R<Array> {
    let key = random::key(seed)?;
    random::uniform::<_, f32>(lo, hi, shape, &key)
}

pub fn max_abs_diff(a: &Array, b: &Array) -> R<f32> {
    let d = (a - b).abs()?.max(None)?;
    d.eval()?;
    Ok(d.item::<f32>())
}

pub fn assert_close(name: &str, a: &Array, b: &Array, atol: f32) {
    assert_eq!(a.shape(), b.shape(), "{name}: shape mismatch");
    let d = max_abs_diff(a, b).unwrap();
    assert!(d <= atol, "{name}: max abs diff {d} > {atol}");
}

/// Value and gradients of `sum(f(args) * cot)` with respect to every argument.
pub fn grads_of<F: Fn(&[Array]) -> R<Array>>(
    f: &F,
    args: &[Array],
    cot: &Array,
) -> R<(Array, Vec<Array>)> {
    let argnums: Vec<i32> = (0..args.len() as i32).collect();
    let loss = |a: &[Array]| -> R<Vec<Array>> { Ok(vec![(f(a)? * cot).sum(None)?]) };
    let (v, g) = value_and_grad_with_argnums(loss, argnums.as_slice())(args)?;
    Ok((v.into_iter().next().unwrap(), g))
}

/// Central-difference derivative of `sum(f(args) * cot)` along `dir` for argument `i`.
pub fn fd_directional<F: Fn(&[Array]) -> R<Array>>(
    f: &F,
    args: &[Array],
    cot: &Array,
    i: usize,
    dir: &Array,
    eps: f32,
) -> R<f32> {
    let at = |sign: f32| -> R<f32> {
        let mut a = args.to_vec();
        a[i] = &a[i] + &(dir * (sign * eps));
        let v = (f(&a)? * cot).sum(None)?;
        v.eval()?;
        Ok(v.item::<f32>())
    };
    Ok((at(1.0)? - at(-1.0)?) / (2.0 * eps))
}
```

- [ ] **Step 2: Write the smoke tests**

`tests/smoke.rs`:
```rust
use mlx_rs::{
    memory,
    ops::{argpartition_axis, gather_mm, indexing::{IndexOp, take_along_axis}, matmul},
    stop_gradient, Array, Dtype,
};
use tinyloop::util::{assert_close, grads_of, randn};

#[test]
fn grad_of_square() {
    let x = randn(&[5], 1).unwrap();
    let f = |a: &[Array]| Ok(&a[0] * &a[0]);
    let cot = Array::ones::<f32>(&[5]).unwrap();
    let (_, g) = grads_of(&f, &[x.clone()], &cot).unwrap();
    assert_close("d/dx x^2", &g[0], &(&x * 2.0), 1e-5);
}

#[test]
fn take_along_axis_has_gradient() {
    let x = randn(&[4, 6], 2).unwrap();
    let idx = Array::from_slice(&[0u32, 5, 1, 1, 2, 3, 4, 0], &[4, 2]);
    let f = |a: &[Array]| take_along_axis(&a[0], &idx, -1);
    let cot = Array::ones::<f32>(&[4, 2]).unwrap();
    let (_, g) = grads_of(&f, &[x], &cot).unwrap();
    // row 1 picks column 1 twice, so its gradient is 2 there
    let expected = Array::from_slice(
        &[1f32, 0., 0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0.],
        &[4, 6],
    );
    assert_close("take_along_axis grad", &g[0], &expected, 1e-6);
}

#[test]
fn gather_mm_matches_matmul_with_gradients() {
    let x = randn(&[6, 1, 1, 8], 3).unwrap();
    let w = randn(&[3, 8, 5], 4).unwrap();
    let idx = Array::from_slice(&[0u32, 2, 1, 1, 2, 0, 0, 0, 1, 2, 2, 1], &[6, 2]);
    let gathered = |a: &[Array]| gather_mm(&a[0], &a[1], None::<&Array>, &idx, false);
    // weights picked per (token, slot) by fancy indexing on the expert axis, then a batched matmul
    let reference = |a: &[Array]| matmul(&a[0], &a[1].index(&idx)); // [6,1,1,8] @ [6,2,8,5]
    let cot = randn(&[6, 2, 1, 5], 5).unwrap();
    let (y1, g1) = grads_of(&gathered, &[x.clone(), w.clone()], &cot).unwrap();
    let (y2, g2) = grads_of(&reference, &[x, w], &cot).unwrap();
    assert_close("gather_mm value", &y1, &y2, 1e-4);
    assert_close("gather_mm dx", &g1[0], &g2[0], 1e-4);
    assert_close("gather_mm dw", &g1[1], &g2[1], 1e-4);
}

#[test]
fn stop_gradient_blocks_gradient() {
    let x = randn(&[3], 6).unwrap();
    let f = |a: &[Array]| Ok(&stop_gradient(&a[0])? * &a[0]);
    let cot = Array::ones::<f32>(&[3]).unwrap();
    let (_, g) = grads_of(&f, &[x.clone()], &cot).unwrap();
    assert_close("stop_gradient", &g[0], &x, 1e-6); // d/dx (c * x) = c = x
}

#[test]
fn argpartition_gives_topk_indices() {
    let s = Array::from_slice(&[0.1f32, 0.9, 0.3, 0.7], &[1, 4]);
    let idx = argpartition_axis(&s.negative().unwrap(), 1, -1).unwrap().index((.., ..2));
    // as_dtype copies into a fresh contiguous array; as_slice on a strided view is unsafe to trust
    let mut v: Vec<i32> = idx.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
    v.sort();
    assert_eq!(v, vec![1, 3]);
}

#[test]
fn memory_limit_can_be_set() {
    memory::set_memory_limit(8 << 30).unwrap();
    assert!(memory::memory_limit().unwrap() >= 8 << 30);
}
```

- [ ] **Step 3: Coordinate, then run the tests (the first MLX build is heavy: about 10–20 min, several GB of RAM)**

Message the other sessions first. Then run: `cargo test --test smoke 2>&1 | tail -20`
Expected: `test result: ok. 6 passed`. Compile errors against the `mlx-rs` API are possible, because this code was written from reading the 0.32 source without compiling it. Fix call signatures to match the compiler's suggestion, without changing test semantics.

- [ ] **Step 4: Verify MLX was built in Release mode**

Run: `grep -h "CMAKE_BUILD_TYPE" target/debug/build/mlx-sys-*/output | head -2`
Expected: contains `Release`. If it says `Debug`, the `build-override` did not apply. Stop and report.

- [ ] **Step 5: If `gather_mm_matches_matmul_with_gradients` fails with a missing-VJP error**, record the exact error in `docs/spike-results.md` under "gather_mm", mark the test `#[ignore = "gather_mm has no VJP in mlx-rs 0.32"]`, and continue. Task 5 then uses its dense-experts fallback.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock .gitignore src tests/smoke.rs
git commit -m "feat: scaffold crate and mlx-rs capability smoke tests"
```

---

### Task 2: Unit-lower-triangular inverse

`(I + L)⁻¹` for strictly-lower `L`, computed by recursive 2×2 block inversion. This is O(log C) depth, fully differentiable, runs on the GPU, and avoids the cancellation problems of a Neumann series. `mlx_rs::linalg::tri_inv` is CPU-only and not used.

**Files:**
- Create: `src/model/deltanet.rs` (starts with this function)
- Modify: `src/model/mod.rs`
- Test: `tests/reference.rs`

**Interfaces:**
- Produces: `pub fn unit_lower_inverse(l: &Array) -> R<Array>`. `l` has shape `[..., C, C]`, strictly lower-triangular, and `C` is a power of two.

- [ ] **Step 1: Write the failing test**

`tests/reference.rs`:
```rust
use mlx_rs::{ops::{matmul, tril}, Array};
use tinyloop::model::deltanet::unit_lower_inverse;
use tinyloop::util::{assert_close, randn};

#[test]
fn unit_lower_inverse_inverts() {
    let l = (tril(&randn(&[3, 16, 16], 10).unwrap(), -1).unwrap()) * 0.5;
    let eye = Array::eye::<f32>(16, None, None).unwrap();
    let inv = unit_lower_inverse(&l).unwrap();
    let prod = matmul(&(&eye + &l), &inv).unwrap();
    assert_close("(I+L)·inv", &prod, &(&eye * &Array::ones::<f32>(&[3, 1, 1]).unwrap()), 1e-4);
}
```

- [ ] **Step 2: Run it and confirm it fails.** `cargo test --test reference unit_lower` gives a compile error, because `deltanet` is not defined.

- [ ] **Step 3: Implement**

`src/model/mod.rs`:
```rust
//! Model kernels. The spike only has pure functions; modules with parameters arrive in plan 2.

pub mod deltanet;
```

`src/model/deltanet.rs`:
```rust
//! Gated DeltaNet recurrence (Qwen3-Next style) in two equivalent forms.
//!
//! Shapes: q, k: [B, H, T, Dk] (caller L2-normalises k and q, and scales q),
//! v: [B, H, T, Dv], beta: [B, H, T] in (0, 1), g: [B, H, T] = log(alpha) <= 0.
//! Both forms return (out [B, H, T, Dv], final state [B, H, Dk, Dv]).

use mlx_rs::{
    ops::{concatenate, indexing::{Ellipsis, IndexOp}, matmul, ones_like, stack, zeros_like},
    Array,
};

use crate::util::R;

/// (I + L)^-1 for strictly lower-triangular L over the last two axes (size a power of two),
/// by recursive block inversion: inv([[A, 0], [C, D]]) = [[A⁻¹, 0], [-D⁻¹·C·A⁻¹, D⁻¹]].
pub fn unit_lower_inverse(l: &Array) -> R<Array> {
    let c = l.shape()[l.ndim() - 1];
    if c == 1 {
        return ones_like(l);
    }
    let h = c / 2;
    let a = l.index((Ellipsis, ..h, ..h));
    let cc = l.index((Ellipsis, h.., ..h));
    let d = l.index((Ellipsis, h.., h..));
    let both = unit_lower_inverse(&stack(&[a, d], 0)?)?; // invert both diagonal blocks at once
    let (ai, di) = (both.index(0), both.index(1));
    let bl = matmul(&matmul(&di, &cc)?, &ai)?.negative()?;
    let top = concatenate(&[ai.clone(), zeros_like(&ai)?], -1)?;
    let bot = concatenate(&[bl, di], -1)?;
    concatenate(&[top, bot], -2)
}
```

- [ ] **Step 4: Run the test and confirm it passes.** `cargo test --test reference unit_lower` gives `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model tests/reference.rs
git commit -m "feat: differentiable unit-lower-triangular inverse"
```

---

### Task 3: Gated DeltaNet, recurrent reference and chunked form

**Files:**
- Modify: `src/model/deltanet.rs`
- Test: `tests/reference.rs`

**Interfaces:**
- Consumes: `unit_lower_inverse`, `util::{R, dims4}`
- Produces:
  - `pub fn delta_rule_recurrent(q: &Array, k: &Array, v: &Array, beta: &Array, g: &Array) -> R<(Array, Array)>`
  - `pub fn delta_rule_chunked(q: &Array, k: &Array, v: &Array, beta: &Array, g: &Array, chunk: i32) -> R<(Array, Array)>`
  - `pub fn l2norm(x: &Array) -> R<Array>` (normalises the last axis)

- [ ] **Step 1: Write the failing tests** (append to `tests/reference.rs`)

```rust
use tinyloop::model::deltanet::{delta_rule_chunked, delta_rule_recurrent, l2norm};
use tinyloop::util::{fd_directional, grads_of, rand_uniform};

/// B=1, H=2, T=128, Dk=Dv=16 inputs; `g_lo` sets the strongest per-token log-decay.
fn delta_inputs(seed: u64, g_lo: f32) -> Vec<Array> {
    let q = &l2norm(&randn(&[1, 2, 128, 16], seed).unwrap()).unwrap() * 0.25; // 1/sqrt(16)
    let k = l2norm(&randn(&[1, 2, 128, 16], seed + 1).unwrap()).unwrap();
    let v = randn(&[1, 2, 128, 16], seed + 2).unwrap();
    let beta = rand_uniform(0.05, 0.95, &[1, 2, 128], seed + 3).unwrap();
    let g = rand_uniform(g_lo, 0.0, &[1, 2, 128], seed + 4).unwrap();
    vec![q, k, v, beta, g]
}

fn chunked(a: &[Array]) -> tinyloop::util::R<Array> {
    Ok(delta_rule_chunked(&a[0], &a[1], &a[2], &a[3], &a[4], 32)?.0)
}
fn recurrent(a: &[Array]) -> tinyloop::util::R<Array> {
    Ok(delta_rule_recurrent(&a[0], &a[1], &a[2], &a[3], &a[4])?.0)
}

#[test]
fn deltanet_chunked_matches_recurrent_forward_and_state() {
    let x = delta_inputs(20, -0.5);
    let (o1, s1) = delta_rule_chunked(&x[0], &x[1], &x[2], &x[3], &x[4], 32).unwrap();
    let (o2, s2) = delta_rule_recurrent(&x[0], &x[1], &x[2], &x[3], &x[4]).unwrap();
    assert_close("deltanet out", &o1, &o2, 1e-4);
    assert_close("deltanet state", &s1, &s2, 1e-4);
}

#[test]
fn deltanet_chunked_matches_recurrent_gradients() {
    let x = delta_inputs(30, -0.5);
    let cot = randn(&[1, 2, 128, 16], 35).unwrap();
    let (_, g1) = grads_of(&chunked, &x, &cot).unwrap();
    let (_, g2) = grads_of(&recurrent, &x, &cot).unwrap();
    for (i, name) in ["dq", "dk", "dv", "dbeta", "dg"].iter().enumerate() {
        assert_close(name, &g1[i], &g2[i], 1e-3);
    }
}

#[test]
fn deltanet_chunked_gradients_match_finite_differences() {
    let x = delta_inputs(40, -0.5);
    let cot = randn(&[1, 2, 128, 16], 45).unwrap();
    let (_, g) = grads_of(&chunked, &x, &cot).unwrap();
    for i in 0..5 {
        let dir = &randn(x[i].shape(), 50 + i as u64).unwrap() * 0.1;
        let fd = fd_directional(&chunked, &x, &cot, i, &dir, 1e-2).unwrap();
        let ad = (&g[i] * &dir).sum(None).unwrap().item::<f32>();
        assert!((fd - ad).abs() <= 1e-2 * ad.abs().max(1.0), "arg {i}: fd {fd} vs autodiff {ad}");
    }
}

#[test]
fn deltanet_chunked_survives_strong_decay() {
    let x = delta_inputs(60, -20.0);
    let (o, s) = delta_rule_chunked(&x[0], &x[1], &x[2], &x[3], &x[4], 32).unwrap();
    let (o2, _) = delta_rule_recurrent(&x[0], &x[1], &x[2], &x[3], &x[4]).unwrap();
    assert!(!o.is_nan().unwrap().any(None).unwrap().item::<bool>(), "NaN in output");
    assert!(!s.is_inf().unwrap().any(None).unwrap().item::<bool>(), "inf in state");
    assert_close("strong decay out", &o, &o2, 1e-4);
}

#[test]
#[should_panic(expected = "multiple of chunk")]
fn deltanet_chunked_rejects_ragged_length() {
    let x = randn(&[1, 1, 100, 16], 70).unwrap();
    let b = rand_uniform(0.1, 0.9, &[1, 1, 100], 71).unwrap();
    let _ = delta_rule_chunked(&x, &x, &x, &b, &(&b * -0.1), 32);
}
```

If `is_nan`, `is_inf`, or `any` have different names in `mlx-rs` 0.32, use the equivalent: the check is "no NaN in the output, no inf in the state".

- [ ] **Step 2: Run the tests and confirm they fail.** `cargo test --test reference deltanet` gives a compile error, because the functions are not defined.

- [ ] **Step 3: Implement** (append to `src/model/deltanet.rs`, and extend its `use` list with `cumsum, r#where, tril` and `crate::util::dims4`)

```rust
pub fn l2norm(x: &Array) -> R<Array> {
    let n = (x * x).sum_axis(-1, true)?;
    Ok(x * &(n + 1e-6).rsqrt()?)
}

/// Token-by-token reference. Slow; used only by tests.
pub fn delta_rule_recurrent(q: &Array, k: &Array, v: &Array, beta: &Array, g: &Array) -> R<(Array, Array)> {
    let [b, h, t, dk] = dims4(q);
    let dv = v.shape()[3];
    let mut s = Array::zeros::<f32>(&[b, h, dk, dv])?.as_dtype(q.dtype())?;
    let mut outs = Vec::with_capacity(t as usize);
    for i in 0..t {
        let qi = q.index((.., .., i)).expand_dims(-2)?; // [B,H,1,Dk]
        let ki = k.index((.., .., i)); // [B,H,Dk]
        let vi = v.index((.., .., i)).expand_dims(-2)?; // [B,H,1,Dv]
        let bi = beta.index((.., .., i)).reshape(&[b, h, 1, 1])?;
        let ai = g.index((.., .., i)).exp()?.reshape(&[b, h, 1, 1])?;
        s = &s * &ai;
        let v_old = matmul(&ki.expand_dims(-2)?, &s)?; // [B,H,1,Dv]
        let delta = (&vi - &v_old) * &bi;
        s = &s + &matmul(&ki.expand_dims(-1)?, &delta)?; // rank-1 write
        outs.push(matmul(&qi, &s)?);
    }
    Ok((concatenate(&outs, 2)?, s))
}

/// Chunkwise-parallel form (WY representation), matching `delta_rule_recurrent`.
pub fn delta_rule_chunked(
    q: &Array,
    k: &Array,
    v: &Array,
    beta: &Array,
    g: &Array,
    chunk: i32,
) -> R<(Array, Array)> {
    let [b, h, t, dk] = dims4(q);
    let dv = v.shape()[3];
    assert!(t % chunk == 0, "sequence length {t} must be a multiple of chunk {chunk}");
    assert!(chunk.count_ones() == 1, "chunk {chunk} must be a power of two");
    let (n, c) = (t / chunk, chunk);
    let qc = q.reshape(&[b, h, n, c, dk])?;
    let kc = k.reshape(&[b, h, n, c, dk])?;
    let vc = v.reshape(&[b, h, n, c, dv])?;
    let bc = beta.reshape(&[b, h, n, c, 1])?;
    let gc = cumsum(&g.reshape(&[b, h, n, c])?, -1, None, None)?; // inclusive, within each chunk
    let k_beta = &kc * &bc;
    let v_beta = &vc * &bc;

    // decay[i, j] = exp(g_i - g_j) for j <= i, else 0. Mask before exp: upper entries overflow.
    let ones = Array::ones::<f32>(&[c, c])?;
    let lower = tril(&ones, 0)?.gt(&Array::from_f32(0.0))?;
    let strict = tril(&ones, -1)?;
    let diff = gc.expand_dims(-1)? - gc.expand_dims(-2)?; // [B,H,N,C,C]
    let decay = r#where(&lower, &diff, &Array::from_f32(f32::NEG_INFINITY))?.exp()?;

    let l = matmul(&k_beta, &kc.swap_axes(-1, -2)?)? * &decay * &strict;
    let t_inv = unit_lower_inverse(&l)?; // (I + L)^-1
    let u = matmul(&t_inv, &v_beta)?;
    let w = matmul(&t_inv, &(&k_beta * &gc.exp()?.expand_dims(-1)?))?;
    let intra = matmul(&qc, &kc.swap_axes(-1, -2)?)? * &decay; // causal, diagonal included

    let mut s = Array::zeros::<f32>(&[b, h, dk, dv])?.as_dtype(q.dtype())?;
    let mut outs = Vec::with_capacity(n as usize);
    for i in 0..n {
        let (qi, ki, gi) = (qc.index((.., .., i)), kc.index((.., .., i)), gc.index((.., .., i)));
        let v_new = u.index((.., .., i)) - matmul(&w.index((.., .., i)), &s)?;
        let o_inter = matmul(&(&qi * &gi.exp()?.expand_dims(-1)?), &s)?;
        outs.push(o_inter + matmul(&intra.index((.., .., i)), &v_new)?);
        let g_last = gi.index((.., .., c - 1)); // [B,H]
        let k_dec = &ki * &(g_last.expand_dims(-1)? - &gi).exp()?.expand_dims(-1)?;
        s = &s * &g_last.exp()?.reshape(&[b, h, 1, 1])? + matmul(&k_dec.swap_axes(-1, -2)?, &v_new)?;
    }
    Ok((stack(&outs, 2)?.reshape(&[b, h, t, dv])?, s))
}
```

- [ ] **Step 4: Run the tests and confirm they pass.** `cargo test --test reference deltanet` gives `5 passed`. If the forward pass matches but gradients differ beyond 1e-3 while the finite-difference test passes, the recurrent reference's gradients are the suspect: confirm with `fd_directional` on `recurrent`.

- [ ] **Step 5: Commit**

```bash
git add src/model/deltanet.rs tests/reference.rs
git commit -m "feat: gated DeltaNet chunked scan with recurrent reference"
```

---

### Task 4: NSA-style sparse attention

Three branches. (a) Compressed: mean-pooled 64-token blocks, visible only once complete, with a zero-logit sink. (b) Selected: top-k blocks per (KV group, query block), gathered. (c) Sliding window, block-framed. There are dense masked references for (b) and (c). Scalable-softmax query scaling applies to (a) and (b) only.

**Files:**
- Create: `src/model/sparse_attn.rs`
- Modify: `src/model/mod.rs` (add `pub mod sparse_attn;`)
- Test: `tests/reference.rs`

**Interfaces:**
- Consumes: `util::{R, dims4, positions}`
- Produces:
  - `pub struct SparseCfg { pub block: i32, pub topk: i32, pub window: i32 }`
  - `pub fn scale_queries(q: &Array, ssmax: &Array) -> R<Array>` (`ssmax: [Hq]`)
  - `pub fn compressed_and_select(q: &Array, k: &Array, v: &Array, cfg: &SparseCfg) -> R<(Array, Array)>` returns `(o_cmp [B,Hq,T,D], idx [B,Hkv,T/block,K] uint32)`
  - `pub fn selected_attn(q, k, v, idx: &Array, cfg) -> R<Array>` and `pub fn selected_attn_dense(q, k, v, idx, cfg) -> R<Array>`
  - `pub fn window_attn(q, k, v, cfg) -> R<Array>` and `pub fn window_attn_dense(q, k, v, cfg) -> R<Array>`
  - `pub fn sparse_attention(q, k, v, gates: &Array, ssmax: &Array, cfg) -> R<Array>` (`gates: [B,Hq,T,3]`, pre-sigmoid)

- [ ] **Step 1: Write the failing tests** (append to `tests/reference.rs`)

```rust
use tinyloop::model::sparse_attn::{
    compressed_and_select, scale_queries, selected_attn, selected_attn_dense, sparse_attention,
    window_attn, window_attn_dense, SparseCfg,
};

const CFG: SparseCfg = SparseCfg { block: 16, topk: 3, window: 32 };

/// B=1, Hq=4, Hkv=2, D=16 with sequence length t.
fn attn_inputs(t: i32, seed: u64) -> Vec<Array> {
    vec![
        &randn(&[1, 4, t, 16], seed).unwrap() * 0.25,
        randn(&[1, 2, t, 16], seed + 1).unwrap(),
        randn(&[1, 2, t, 16], seed + 2).unwrap(),
    ]
}

#[test]
fn selection_always_includes_own_block_and_no_future_block() {
    let x = attn_inputs(128, 80);
    let (_, idx) = compressed_and_select(&x[0], &x[1], &x[2], &CFG).unwrap();
    let idx = idx.as_dtype(mlx_rs::Dtype::Int32).unwrap(); // contiguous copy
    let v = idx.as_slice::<i32>(); // [1, 2, 8, 3], row-major
    for h in 0..2 {
        for qb in 0..8i32 {
            let row = &v[(h * 8 + qb as usize) * 3..(h * 8 + qb as usize) * 3 + 3];
            assert!(row.contains(&qb), "head {h} block {qb} misses itself: {row:?}");
            if qb >= 2 {
                assert!(row.iter().all(|&j| j <= qb), "head {h} block {qb} picked a future block: {row:?}");
            }
        }
    }
}

#[test]
fn selected_gather_matches_dense_forward_and_gradients() {
    let x = attn_inputs(128, 90);
    let (_, idx) = compressed_and_select(&x[0], &x[1], &x[2], &CFG).unwrap();
    let gather = |a: &[Array]| selected_attn(&a[0], &a[1], &a[2], &idx, &CFG);
    let dense = |a: &[Array]| selected_attn_dense(&a[0], &a[1], &a[2], &idx, &CFG);
    let cot = randn(&[1, 4, 128, 16], 95).unwrap();
    let (y1, g1) = grads_of(&gather, &x, &cot).unwrap();
    let (y2, g2) = grads_of(&dense, &x, &cot).unwrap();
    assert_close("selected out", &y1, &y2, 1e-4);
    for (i, n) in ["dq", "dk", "dv"].iter().enumerate() {
        assert_close(n, &g1[i], &g2[i], 1e-3);
    }
}

#[test]
fn window_gather_matches_dense_forward_and_gradients() {
    let x = attn_inputs(128, 100);
    let gather = |a: &[Array]| window_attn(&a[0], &a[1], &a[2], &CFG);
    let dense = |a: &[Array]| window_attn_dense(&a[0], &a[1], &a[2], &CFG);
    let cot = randn(&[1, 4, 128, 16], 105).unwrap();
    let (y1, g1) = grads_of(&gather, &x, &cot).unwrap();
    let (y2, g2) = grads_of(&dense, &x, &cot).unwrap();
    assert_close("window out", &y1, &y2, 1e-4);
    for (i, n) in ["dq", "dk", "dv"].iter().enumerate() {
        assert_close(n, &g1[i], &g2[i], 1e-3);
    }
}

#[test]
fn sparse_attention_is_finite_for_early_tokens_and_short_sequences() {
    for t in [32, 128] {
        // t = 32 gives 2 blocks < topk 3, and tokens 0..15 have no complete block
        let x = attn_inputs(t, 110);
        let gates = randn(&[1, 4, t, 3], 111).unwrap();
        let ssmax = Array::from_slice(&[0.3f32, 0.4, 0.5, 0.6], &[4]);
        let f = |a: &[Array]| sparse_attention(&a[0], &a[1], &a[2], &a[3], &a[4], &CFG);
        let cot = randn(&[1, 4, t, 16], 112).unwrap();
        let (y, g) = grads_of(&f, &[x[0].clone(), x[1].clone(), x[2].clone(), gates, ssmax], &cot).unwrap();
        assert!(!y.is_nan().unwrap().any(None).unwrap().item::<bool>(), "t={t}: NaN value");
        for (i, gi) in g.iter().enumerate() {
            assert!(!gi.is_nan().unwrap().any(None).unwrap().item::<bool>(), "t={t}: NaN grad {i}");
        }
    }
}

#[test]
fn scale_queries_grows_with_position() {
    let q = Array::ones::<f32>(&[1, 1, 4, 1]).unwrap();
    let s = scale_queries(&q, &Array::from_slice(&[1.0f32], &[1])).unwrap();
    let v = s.as_slice::<f32>();
    assert!((v[0] - 2f32.ln()).abs() < 1e-5 && v[3] > v[2] && v[2] > v[1]);
}

#[test]
#[should_panic(expected = "multiple of block")]
fn sparse_attention_rejects_ragged_length() {
    let x = attn_inputs(100, 120);
    let _ = compressed_and_select(&x[0], &x[1], &x[2], &CFG);
}
```

- [ ] **Step 2: Run the tests and confirm they fail.** `cargo test --test reference sparse selected window selection scale` gives a compile error.

- [ ] **Step 3: Implement.** Add `pub mod sparse_attn;` to `src/model/mod.rs`, then create `src/model/sparse_attn.rs`:

```rust
//! NSA-style sparse attention: compressed, selected and sliding-window branches.
//!
//! q: [B, Hq, T, D]; k, v: [B, Hkv, T, D]; Hq = G·Hkv; T and window are multiples of `block`.
//! Block selection is shared by the G query heads of a KV group and by the `block` queries of a
//! query block (MoBA-style), so the gather is [B, Hkv, T/block, topk·block, D].

use mlx_rs::{
    ops::{
        argpartition_axis, broadcast_to, concatenate, indexing::{take_along_axis, Ellipsis, IndexOp},
        matmul, r#where, sigmoid, softmax_axes, stack, zeros_like,
    },
    stop_gradient, Array, Dtype,
};

use crate::util::{dims4, positions, R};

pub struct SparseCfg {
    pub block: i32,
    pub topk: i32,
    pub window: i32,
}

fn neg_inf() -> Array {
    Array::from_f32(f32::NEG_INFINITY)
}

fn masked_softmax(scores: &Array, mask: &Array) -> R<Array> {
    softmax_axes(&r#where(mask, scores, &neg_inf())?, &[-1], None)
}

/// Scalable-softmax scaling for branches (a) and (b): q · ssmax[h] · ln(t + 2) / sqrt(D).
pub fn scale_queries(q: &Array, ssmax: &Array) -> R<Array> {
    let [_, hq, t, d] = dims4(q);
    let len = (positions(t).as_dtype(Dtype::Float32)? + 2.0).log()?;
    let f = ssmax.reshape(&[1, hq, 1, 1])? * len.reshape(&[1, 1, t, 1])? / (d as f32).sqrt();
    Ok(q * &f)
}

/// [B, Hq, T, D] -> [B, Hkv, NQB, G·block, D]
fn group_queries(q: &Array, hkv: i32, bk: i32) -> R<Array> {
    let [b, hq, t, d] = dims4(q);
    let (g, nb) = (hq / hkv, t / bk);
    q.reshape(&[b, hkv, g, nb, bk, d])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hkv, nb, g * bk, d])
}

/// Inverse of `group_queries`.
fn ungroup(o: &Array, hq: i32, bk: i32) -> R<Array> {
    let s = o.shape();
    let (b, hkv, nb, d) = (s[0], s[1], s[2], s[4]);
    o.reshape(&[b, hkv, nb, hq / hkv, bk, d])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hq, nb * bk, d])
}

/// Absolute position of each grouped query row: [NQB, G·block, 1].
fn query_pos(nb: i32, g: i32, bk: i32) -> R<Array> {
    broadcast_to(&positions(nb * bk).reshape(&[nb, 1, bk])?, &[nb, g, bk])?.reshape(&[nb, g * bk, 1])
}

/// Branch (a) plus block selection. q must already be scaled (`scale_queries`).
pub fn compressed_and_select(q: &Array, k: &Array, v: &Array, cfg: &SparseCfg) -> R<(Array, Array)> {
    let [b, hq, t, d] = dims4(q);
    let hkv = k.shape()[1];
    let (g, bk) = (hq / hkv, cfg.block);
    assert!(t % bk == 0, "sequence length {t} must be a multiple of block {bk}");
    let nb = t / bk;
    let kc = k.reshape(&[b, hkv, nb, bk, d])?.mean_axis(3, None)?; // [B,Hkv,NB,D]
    let vc = v.reshape(&[b, hkv, nb, bk, d])?.mean_axis(3, None)?;
    let qg = q.reshape(&[b, hkv, g, t, d])?;
    let s = matmul(&qg, &kc.expand_dims(2)?.swap_axes(-1, -2)?)?; // [B,Hkv,G,T,NB]
    // block j is visible to token t once complete: (j + 1)·block <= t + 1
    let visible = ((positions(nb).reshape(&[1, nb])? + 1) * bk).le(&(positions(t).reshape(&[t, 1])? + 1))?;
    let s = r#where(&visible, &s, &neg_inf())?;
    // zero-logit sink: tokens with no complete block get a zero output instead of NaN
    let sink = zeros_like(&s.index((Ellipsis, ..1)))?;
    let p = softmax_axes(&concatenate(&[s, sink], -1)?, &[-1], None)?.index((Ellipsis, ..nb));
    let o_cmp = matmul(&p, &vc.expand_dims(2)?)?.reshape(&[b, hq, t, d])?;

    // importance of kv block j for query block i, summed over the group's heads and the block's queries
    let imp = stop_gradient(&p)?.reshape(&[b, hkv, g, nb, bk, nb])?.sum_axes(&[2, 4], None)?;
    let (qb, kb) = (positions(nb).reshape(&[nb, 1])?, positions(nb).reshape(&[1, nb])?);
    let imp = r#where(
        &kb.eq(&qb)?,
        &Array::from_f32(1e9), // own block: always selected
        &r#where(&kb.gt(&qb)?, &Array::from_f32(-1e9), &imp)?, // future blocks: last resort, masked later
    )?;
    let kk = cfg.topk.min(nb);
    let idx = argpartition_axis(&imp.negative()?, kk - 1, -1)?.index((Ellipsis, ..kk));
    Ok((o_cmp, idx))
}

/// Branch (b): attention inside the selected blocks, gather implementation.
pub fn selected_attn(q: &Array, k: &Array, v: &Array, idx: &Array, cfg: &SparseCfg) -> R<Array> {
    let [b, hq, t, d] = dims4(q);
    let hkv = k.shape()[1];
    let (g, bk) = (hq / hkv, cfg.block);
    let (nb, kk) = (t / bk, idx.shape()[3]);
    let ix = broadcast_to(&idx.reshape(&[b, hkv, nb * kk, 1, 1])?, &[b, hkv, nb * kk, bk, d])?;
    let gather = |x: &Array| -> R<Array> {
        take_along_axis(&x.reshape(&[b, hkv, nb, bk, d])?, &ix, 2)?.reshape(&[b, hkv, nb, kk * bk, d])
    };
    let (ks, vs) = (gather(k)?, gather(v)?);
    let s = matmul(&group_queries(q, hkv, bk)?, &ks.swap_axes(-1, -2)?)?; // [B,Hkv,NQB,G·bk,K·bk]
    let kpos = (idx.as_dtype(Dtype::Int32)?.reshape(&[b, hkv, nb, kk, 1])? * bk
        + positions(bk).reshape(&[1, 1, 1, 1, bk])?)
    .reshape(&[b, hkv, nb, 1, kk * bk])?;
    let mask = kpos.le(&query_pos(nb, g, bk)?)?;
    ungroup(&matmul(&masked_softmax(&s, &mask)?, &vs)?, hq, bk)
}

/// Dense attention with a boolean mask broadcastable to [B, Hkv, G, T, T].
fn dense_attn(q: &Array, k: &Array, v: &Array, mask: &Array) -> R<Array> {
    let [b, hq, t, d] = dims4(q);
    let hkv = k.shape()[1];
    let qg = q.reshape(&[b, hkv, hq / hkv, t, d])?;
    let s = matmul(&qg, &k.expand_dims(2)?.swap_axes(-1, -2)?)?;
    matmul(&masked_softmax(&s, mask)?, &v.expand_dims(2)?)?.reshape(&[b, hq, t, d])
}

/// Dense reference for branch (b): full T×T scores, masked to the selected blocks and causality.
pub fn selected_attn_dense(q: &Array, k: &Array, v: &Array, idx: &Array, cfg: &SparseCfg) -> R<Array> {
    let [b, _, t, _] = dims4(q);
    let hkv = k.shape()[1];
    let bk = cfg.block;
    let (nb, kk) = (t / bk, idx.shape()[3]);
    let chosen = idx
        .as_dtype(Dtype::Int32)?
        .reshape(&[b, hkv, nb, kk, 1])?
        .eq(&positions(nb).reshape(&[1, 1, 1, 1, nb])?)?
        .as_dtype(Dtype::Float32)?
        .sum_axis(3, None)?
        .gt(&Array::from_f32(0.0))?; // [B,Hkv,NQB,NB]
    let tok = broadcast_to(&chosen.reshape(&[b, hkv, nb, 1, nb, 1])?, &[b, hkv, nb, bk, nb, bk])?
        .reshape(&[b, hkv, 1, t, t])?;
    let causal = positions(t).reshape(&[1, t])?.le(&positions(t).reshape(&[t, 1])?)?;
    dense_attn(q, k, v, &tok.logical_and(&causal)?)
}

/// Branch (c): sliding window, block-framed gather implementation.
pub fn window_attn(q: &Array, k: &Array, v: &Array, cfg: &SparseCfg) -> R<Array> {
    let [b, hq, t, d] = dims4(q);
    let hkv = k.shape()[1];
    let (g, bk) = (hq / hkv, cfg.block);
    assert!(cfg.window % bk == 0, "window {} must be a multiple of block {bk}", cfg.window);
    let (nb, nw) = (t / bk, cfg.window / bk);
    let frames = |x: &Array| -> R<Array> {
        let pad = Array::zeros::<f32>(&[b, hkv, nw, bk, d])?.as_dtype(x.dtype())?;
        let padded = concatenate(&[pad, x.reshape(&[b, hkv, nb, bk, d])?], 2)?;
        let views: Vec<Array> = (0..=nw).map(|o| padded.index((.., .., o..o + nb))).collect();
        stack(&views, 3)?.reshape(&[b, hkv, nb, (nw + 1) * bk, d])
    };
    let (kw, vw) = (frames(k)?, frames(v)?);
    let s = matmul(&group_queries(q, hkv, bk)?, &kw.swap_axes(-1, -2)?)?;
    let kpos = ((positions(nb).reshape(&[nb, 1, 1])? - nw + positions(nw + 1).reshape(&[1, nw + 1, 1])?) * bk
        + positions(bk).reshape(&[1, 1, bk])?)
    .reshape(&[nb, 1, (nw + 1) * bk])?;
    let qpos = query_pos(nb, g, bk)?;
    let mask = kpos
        .ge(&Array::from_int(0))?
        .logical_and(&kpos.le(&qpos)?)?
        .logical_and(&(&qpos - &kpos).lt(&Array::from_int(cfg.window))?)?;
    ungroup(&matmul(&masked_softmax(&s, &mask)?, &vw)?, hq, bk)
}

/// Dense reference for branch (c).
pub fn window_attn_dense(q: &Array, k: &Array, v: &Array, cfg: &SparseCfg) -> R<Array> {
    let t = q.shape()[2];
    let (qp, kp) = (positions(t).reshape(&[t, 1])?, positions(t).reshape(&[1, t])?);
    let mask = kp.le(&qp)?.logical_and(&(&qp - &kp).lt(&Array::from_int(cfg.window))?)?;
    dense_attn(q, k, v, &mask)
}

/// Full S-layer attention core. gates: [B, Hq, T, 3] pre-sigmoid; ssmax: [Hq].
pub fn sparse_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    gates: &Array,
    ssmax: &Array,
    cfg: &SparseCfg,
) -> R<Array> {
    let d = q.shape()[3];
    let qs = scale_queries(q, ssmax)?;
    let (o_cmp, idx) = compressed_and_select(&qs, k, v, cfg)?;
    let o_sel = selected_attn(&qs, k, v, &idx, cfg)?;
    let o_win = window_attn(&(q / (d as f32).sqrt()), k, v, cfg)?;
    let gs = sigmoid(gates)?;
    Ok(&o_cmp * &gs.index((Ellipsis, 0..1))
        + &o_sel * &gs.index((Ellipsis, 1..2))
        + &o_win * &gs.index((Ellipsis, 2..3)))
}
```

- [ ] **Step 4: Run the tests and confirm they pass.** `cargo test --test reference` gives all tests passing (6 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/model tests/reference.rs
git commit -m "feat: NSA-style sparse attention with dense references"
```

---

### Task 5: MoE routing and gather-based experts

**Files:**
- Create: `src/model/moe.rs`
- Modify: `src/model/mod.rs` (add `pub mod moe;`)
- Test: `tests/reference.rs`

**Interfaces:**
- Consumes: `util::{R, positions}`
- Produces:
  - `pub struct Experts { pub w1: Array, pub w3: Array, pub w2: Array }` (`w1`, `w3`: `[E, D, H]`; `w2`: `[E, H, D]`)
  - `pub fn route(x: &Array, router: &Array, bias: &Array, k: i32) -> R<(Array, Array)>` returns `(idx [N,K] uint32, weights [N,K])`
  - `pub fn experts_gather(x: &Array, idx: &Array, w: &Array, e: &Experts) -> R<Array>` and `pub fn experts_dense(...)` with the same signature
  - `pub fn swiglu(x: &Array, w1: &Array, w3: &Array, w2: &Array) -> R<Array>` (the shared expert)

**Fallback:** if Task 1 recorded that `gather_mm` has no VJP, `experts_gather` must call `experts_dense`. The benchmark then measures dense cost, 4× the routed FLOPs at top-2 of 8. Note this in the results.

- [ ] **Step 1: Write the failing tests** (append to `tests/reference.rs`)

```rust
use tinyloop::model::moe::{experts_dense, experts_gather, route, Experts};

fn moe_args(seed: u64) -> Vec<Array> {
    vec![
        randn(&[64, 32], seed).unwrap(),                    // x
        &randn(&[32, 8], seed + 1).unwrap() * 0.2,          // router
        &randn(&[8, 32, 48], seed + 2).unwrap() * 0.1,      // w1
        &randn(&[8, 32, 48], seed + 3).unwrap() * 0.1,      // w3
        &randn(&[8, 48, 32], seed + 4).unwrap() * 0.1,      // w2
    ]
}

fn moe_with(a: &[Array], gathered: bool) -> tinyloop::util::R<Array> {
    let bias = Array::zeros::<f32>(&[8])?;
    let (idx, w) = route(&a[0], &a[1], &bias, 2)?;
    let e = Experts { w1: a[2].clone(), w3: a[3].clone(), w2: a[4].clone() };
    if gathered { experts_gather(&a[0], &idx, &w, &e) } else { experts_dense(&a[0], &idx, &w, &e) }
}

#[test]
fn moe_gather_matches_dense_forward_and_gradients() {
    let x = moe_args(130);
    let cot = randn(&[64, 32], 136).unwrap();
    let (y1, g1) = grads_of(&|a: &[Array]| moe_with(a, true), &x, &cot).unwrap();
    let (y2, g2) = grads_of(&|a: &[Array]| moe_with(a, false), &x, &cot).unwrap();
    assert_close("moe out", &y1, &y2, 1e-4);
    for (i, n) in ["dx", "drouter", "dw1", "dw3", "dw2"].iter().enumerate() {
        assert_close(n, &g1[i], &g2[i], 1e-3);
    }
}

#[test]
fn routing_bias_steers_selection_but_not_weights_gradient_path() {
    let x = moe_args(140);
    let bias = Array::from_slice(&[0f32, 0., 0., 100., 0., 0., 0., 0.], &[8]);
    let (idx, w) = route(&x[0], &x[1], &bias, 2).unwrap();
    let idx = idx.as_dtype(mlx_rs::Dtype::Int32).unwrap(); // contiguous copy
    let rows = idx.as_slice::<i32>().chunks(2).filter(|r| r.contains(&3)).count();
    assert_eq!(rows, 64, "every token must pick the biased expert");
    let sums = w.sum_axis(-1, None).unwrap();
    assert_close("weights normalised", &sums, &Array::ones::<f32>(&[64]).unwrap(), 1e-5);
}
```

- [ ] **Step 2: Run the tests and confirm they fail.** `cargo test --test reference moe routing` gives a compile error.

- [ ] **Step 3: Implement.** Add `pub mod moe;` to `src/model/mod.rs`, then create `src/model/moe.rs`:

```rust
//! MoE FFN: sigmoid router, selection bias for aux-loss-free balancing, top-k SwiGLU experts.

use mlx_rs::{
    nn::silu,
    ops::{argpartition_axis, gather_mm, indexing::{take_along_axis, IndexOp}, matmul, sigmoid},
    stop_gradient, Array, Dtype,
};

use crate::util::{positions, R};

pub struct Experts {
    pub w1: Array, // [E, D, H]
    pub w3: Array, // [E, D, H]
    pub w2: Array, // [E, H, D]
}

/// Top-k experts per token. The bias only affects which experts are picked; the weights come
/// from the unbiased scores, renormalised over the picked experts.
pub fn route(x: &Array, router: &Array, bias: &Array, k: i32) -> R<(Array, Array)> {
    let scores = sigmoid(&matmul(x, router)?)?; // [N, E]
    let pick = stop_gradient(&(&scores + bias))?;
    let idx = argpartition_axis(&pick.negative()?, k - 1, -1)?.index((.., ..k)); // [N, K]
    let w = take_along_axis(&scores, &idx, -1)?;
    let w = &w / &w.sum_axis(-1, true)?;
    Ok((idx, w))
}

/// Routed experts via gather_mm: only the picked experts are computed.
pub fn experts_gather(x: &Array, idx: &Array, w: &Array, e: &Experts) -> R<Array> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let xe = x.reshape(&[n, 1, 1, d])?;
    let h = silu(&gather_mm(&xe, &e.w1, None::<&Array>, idx, false)?)?
        * gather_mm(&xe, &e.w3, None::<&Array>, idx, false)?; // [N, K, 1, H]
    let y = gather_mm(&h, &e.w2, None::<&Array>, idx, false)?.squeeze_axes(&[-2])?; // [N, K, D]
    (y * w.expand_dims(-1)?).sum_axis(1, None)
}

/// Reference: every expert on every token, combined with the same sparse weights.
pub fn experts_dense(x: &Array, idx: &Array, w: &Array, e: &Experts) -> R<Array> {
    let n_exp = e.w1.shape()[0];
    let xe = x.expand_dims(0)?; // [1, N, D]
    let h = silu(&matmul(&xe, &e.w1)?)? * matmul(&xe, &e.w3)?; // [E, N, H]
    let y = matmul(&h, &e.w2)?; // [E, N, D]
    let onehot = idx
        .as_dtype(Dtype::Int32)?
        .expand_dims(-1)?
        .eq(&positions(n_exp))?
        .as_dtype(Dtype::Float32)?; // [N, K, E]
    let comb = (onehot * w.expand_dims(-1)?).sum_axis(1, None)?; // [N, E]
    (y * comb.transpose()?.expand_dims(-1)?).sum_axis(0, None)
}

/// Dense SwiGLU (the always-on shared expert).
pub fn swiglu(x: &Array, w1: &Array, w3: &Array, w2: &Array) -> R<Array> {
    matmul(&(silu(&matmul(x, w1)?)? * matmul(x, w3)?), w2)
}
```

- [ ] **Step 4: Run the tests and confirm they pass.** `cargo test --test reference moe routing` gives `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model tests/reference.rs
git commit -m "feat: MoE routing and gather-based experts with dense reference"
```

---

### Task 6: Truncated-backprop loop helper

**Files:**
- Modify: `src/model/mod.rs`
- Test: `tests/reference.rs`

**Interfaces:**
- Produces: `pub fn run_loop(core: impl FnMut(&Array) -> R<Array>, s0: &Array, r: usize, k: usize) -> R<Array>`. It applies `core` `r` times, and gradients flow through only the last `k` iterations (none reach `s0` when `k >= r`).

- [ ] **Step 1: Write the failing test** (append to `tests/reference.rs`)

```rust
use mlx_rs::ops::tanh;
use tinyloop::model::run_loop;

#[test]
fn run_loop_truncates_gradients_to_last_k_iterations() {
    let w0 = &randn(&[4, 4], 150).unwrap() * 0.5;
    let e = randn(&[1, 4], 151).unwrap();
    let s0 = randn(&[1, 4], 152).unwrap();
    let cot = Array::ones::<f32>(&[1, 4]).unwrap();
    let step = |s: &Array, w: &Array| tanh(&(matmul(s, w).unwrap() + &e));

    // autodiff through the truncated loop (r = 5, k = 2)
    let trunc = |a: &[Array]| run_loop(|s| step(s, &a[0]), &s0, 5, 2);
    let (_, g) = grads_of(&trunc, &[w0.clone()], &cot).unwrap();

    // finite differences where only the last 2 iterations see the perturbed weights
    let split = |a: &[Array]| -> tinyloop::util::R<Array> {
        let mut s = s0.clone();
        for _ in 0..3 { s = step(&s, &w0)?; }
        for _ in 0..2 { s = step(&s, &a[0])?; }
        Ok(s)
    };
    let dir = randn(&[4, 4], 153).unwrap();
    let fd = fd_directional(&split, &[w0.clone()], &cot, 0, &dir, 1e-2).unwrap();
    let ad = (&g[0] * &dir).sum(None).unwrap().item::<f32>();
    assert!((fd - ad).abs() <= 1e-2 * ad.abs().max(1.0), "fd {fd} vs autodiff {ad}");

    // and truncation really removed something: full backprop differs
    let full = |a: &[Array]| run_loop(|s| step(s, &a[0]), &s0, 5, 5);
    let (_, gf) = grads_of(&full, &[w0.clone()], &cot).unwrap();
    assert!(tinyloop::util::max_abs_diff(&g[0], &gf[0]).unwrap() > 1e-4);
}
```

- [ ] **Step 2: Run the test and confirm it fails.** `cargo test --test reference run_loop` gives a compile error.

- [ ] **Step 3: Implement** (append to `src/model/mod.rs`)

```rust
use mlx_rs::{stop_gradient, Array};

use crate::util::R;

/// Apply `core` r times from s0. Gradients flow through only the last k iterations;
/// with k >= r the cut sits before the first iteration, so none reach s0.
pub fn run_loop(mut core: impl FnMut(&Array) -> R<Array>, s0: &Array, r: usize, k: usize) -> R<Array> {
    let mut s = s0.clone();
    for i in 0..r {
        if i == r.saturating_sub(k) {
            s = stop_gradient(&s)?;
        }
        s = core(&s)?;
    }
    Ok(s)
}
```

- [ ] **Step 4: Run the test and confirm it passes.** `cargo test --test reference run_loop` gives `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model/mod.rs tests/reference.rs
git commit -m "feat: truncated-backprop loop helper"
```

---

### Task 7: Throughput benchmark and go/no-go report

Measures forward+backward time at T = 16,384 for one G layer, one S layer and one MoE layer at spec sizes, in fp32 and bf16. It then estimates full-model training tokens/s and records the decision.

**Files:**
- Create: `src/bin/spike_bench.rs`, `docs/spike-results.md`

**Interfaces:**
- Consumes: `deltanet::{delta_rule_chunked, l2norm}`, `sparse_attn::{sparse_attention, SparseCfg}`, `moe::{route, experts_gather, swiglu, Experts}`, `util::{grads_of, randn, rand_uniform, max_abs_diff, R}`

- [ ] **Step 1: Write the benchmark**

`src/bin/spike_bench.rs`:
```rust
//! Stage-0 spike benchmark: fwd+bwd time per layer type at T = 16k, fp32 vs bf16,
//! and an estimate of full-model training throughput. Heavy job: coordinate first.

use std::time::Instant;

use mlx_rs::{memory, transforms::eval, Array, Dtype};
use tinyloop::model::{
    deltanet::{delta_rule_chunked, l2norm},
    moe::{experts_gather, route, swiglu, Experts},
    sparse_attn::{sparse_attention, SparseCfg},
};
use tinyloop::util::{grads_of, max_abs_diff, rand_uniform, randn, R};

const T: i32 = 16_384;

/// Median fwd+bwd milliseconds over 5 timed runs (after 2 warm-ups), peak MB, and the value.
fn bench<F: Fn(&[Array]) -> R<Array>>(f: &F, args: &[Array]) -> anyhow::Result<(f64, f64, Array)> {
    let out = f(args)?;
    let cot = randn(out.shape(), 999)?.as_dtype(out.dtype())?;
    memory::reset_peak_memory()?;
    let mut times = Vec::new();
    for i in 0..7 {
        let t0 = Instant::now();
        let (v, g) = grads_of(f, args, &cot)?;
        eval(g.iter().chain(std::iter::once(&v)))?;
        if i >= 2 {
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((times[2], memory::peak_memory()? as f64 / 1e6, out))
}

fn cast(args: &[Array], dt: Dtype) -> anyhow::Result<Vec<Array>> {
    Ok(args.iter().map(|a| a.as_dtype(dt)).collect::<Result<_, _>>()?)
}

fn main() -> anyhow::Result<()> {
    memory::set_memory_limit(8 << 30)?;

    // G: 4 heads × 128, chunk 64
    let g_args = vec![
        &l2norm(&randn(&[1, 4, T, 128], 1)?)? * (1.0 / (128f32).sqrt()),
        l2norm(&randn(&[1, 4, T, 128], 2)?)?,
        randn(&[1, 4, T, 128], 3)?,
        rand_uniform(0.05, 0.95, &[1, 4, T], 4)?,
        rand_uniform(-0.1, 0.0, &[1, 4, T], 5)?,
    ];
    let g_fn = |a: &[Array]| Ok(delta_rule_chunked(&a[0], &a[1], &a[2], &a[3], &a[4], 64)?.0);

    // S: Hq 8, Hkv 2, d 64, block 64, top-16, window 512
    let cfg = SparseCfg { block: 64, topk: 16, window: 512 };
    let s_args = vec![
        randn(&[1, 8, T, 64], 6)?,
        randn(&[1, 2, T, 64], 7)?,
        randn(&[1, 2, T, 64], 8)?,
        randn(&[1, 8, T, 3], 9)?,
        Array::from_slice(&[0.1f32; 8], &[8]),
    ];
    let s_fn = |a: &[Array]| sparse_attention(&a[0], &a[1], &a[2], &a[3], &a[4], &cfg);

    // MoE: d 512, 8 experts of hidden 512, top-2, plus a shared expert of hidden 512
    let m_args = vec![
        randn(&[T, 512], 10)?,
        &randn(&[512, 8], 11)? * 0.05,
        &randn(&[8, 512, 512], 12)? * 0.04,
        &randn(&[8, 512, 512], 13)? * 0.04,
        &randn(&[8, 512, 512], 14)? * 0.04,
        &randn(&[512, 512], 15)? * 0.04,
        &randn(&[512, 512], 16)? * 0.04,
        &randn(&[512, 512], 17)? * 0.04,
    ];
    let m_fn = |a: &[Array]| -> R<Array> {
        let bias = Array::zeros::<f32>(&[8])?;
        let (idx, w) = route(&a[0], &a[1], &bias, 2)?;
        let e = Experts { w1: a[2].clone(), w3: a[3].clone(), w2: a[4].clone() };
        Ok(experts_gather(&a[0], &idx, &w, &e)? + swiglu(&a[0], &a[5], &a[6], &a[7])?)
    };

    println!("layer,dtype,fwd_bwd_ms,peak_mb,max_abs_diff_vs_f32,nan_count");
    let mut ms = [0f64; 3];
    for (li, (name, args, f)) in [
        ("G", &g_args, &g_fn as &dyn Fn(&[Array]) -> R<Array>),
        ("S", &s_args, &s_fn),
        ("MoE", &m_args, &m_fn),
    ]
    .into_iter()
    .enumerate()
    {
        let (t32, p32, o32) = bench(&f, args)?;
        ms[li] = t32;
        println!("{name},f32,{t32:.1},{p32:.0},0,0");
        let (t16, p16, o16) = bench(&f, &cast(args, Dtype::Bfloat16)?)?;
        let o16 = o16.as_dtype(Dtype::Float32)?;
        let nans = o16.is_nan()?.as_dtype(Dtype::Int32)?.sum(None)?.item::<i32>();
        println!("{name},bf16,{t16:.1},{p16:.0},{:.4},{nans}", max_abs_diff(&o16, &o32)?);
    }

    // main config: prelude GG + coda GG (4 G, 4 MoE); core GGGS (3 G, 1 S, 4 MoE);
    // mean r = 6 with the last 4 iterations differentiated; a forward-only pass ≈ 1/3 of fwd+bwd.
    let [g, s, m] = ms;
    let core = 3.0 * g + s + 4.0 * m;
    let step_ms = 4.0 * (g + m) + 4.0 * core + 2.0 * core / 3.0;
    let tok_s = T as f64 / (step_ms * 1.3 / 1e3); // ×1.3 for projections, norms, embedding, optimiser
    println!("estimated_step_ms_per_16k,{step_ms:.0}");
    println!("estimated_train_tokens_per_s,{tok_s:.0}");
    Ok(())
}
```

If `cargo build` rejects the `&dyn Fn` array (a closure type-inference edge case), unroll the loop into three explicit blocks. Behaviour must stay identical.

- [ ] **Step 2: Coordinate, then run.** Message the other sessions (peak memory is about 6–8 GB). Run: `cargo run --release --bin spike_bench 2>&1 | tee /tmp/spike_bench.csv`
Expected: 6 layer rows plus 2 estimate rows, and no panic.

- [ ] **Step 3: Write `docs/spike-results.md`** with:
  - the full test output summary (`cargo test --release 2>&1 | grep "test result"`);
  - the benchmark CSV, verbatim;
  - the `gather_mm` VJP status from Task 1;
  - the **decision**, from these rules:
    - **GO** if all reference tests pass and `estimated_train_tokens_per_s ≥ 1000`. Precision: bf16 if bf16 is faster, has `nan_count = 0` and `max_abs_diff < 0.05` for all three layers; otherwise fp32.
    - **INVESTIGATE** if the tests pass but tokens/s is between 400 and 1000. List the slowest layer and whether MLX `compile` (plan 2) could close the gap.
    - **NO-GO** if a kernel cannot pass its reference test, or tokens/s is below 400. Recommend the `burn` fallback (spec section 7) and cite the failing evidence.

- [ ] **Step 4: Commit**

```bash
git add src/bin/spike_bench.rs docs/spike-results.md
git commit -m "feat: spike throughput benchmark and go/no-go results"
```

---

## After this plan

Plan 2 is written only after `docs/spike-results.md` says GO. It covers model modules with parameters (projections, norms, conv, gates, and the MoE bias update), the full Huginn loop, the optimiser, checkpointing, and the data pipeline. The spec's test 4 (a finite-difference check of the *full model*) also moves there. This spike covers finite differences at the kernel and loop level only.
