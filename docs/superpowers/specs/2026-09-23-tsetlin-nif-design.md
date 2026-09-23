# Tsetlin Machine inference library for Elixir (Rustler NIF)

Date: 2026-09-23
Status: approved (design), pending implementation plan

## 1. Purpose and scope

`tsetlin_runner` is an Elixir library that runs **inference (`predict` only)**
for Tsetlin Machine classifiers trained by the Julia project at
`/home/xabi/work/julia/tsetlin_world` (which vendors
`BooBSD/Tsetlin.jl`). It is meant to be embedded in resource-constrained
targets, specifically a **Nerves firmware for the Raspberry Pi Zero
(1st generation, ARMv6, single core, 512MB RAM)**.

Explicitly out of scope for this spec:

- Training (`train!`/`feedback!`) — models are trained in Julia only.
- Porting feature extraction (`ground_features`, `binarize_frame`, the
  gridworld simulation, the A* teacher, visualization). The Elixir library
  only consumes an already-computed boolean feature vector.
- Automated cross-language (Julia vs Elixir) regression testing in CI.

## 2. Architecture

```
tsetlin_world (Julia, existing repo)
  └── export_tm(tm, path)          # new: compile(tm) + write portable binary

tsetlin_runner (Elixir, this repo)
  ├── native/tsetlin_nif/          # Rust crate (Rustler)
  │     - parses the exported binary format
  │     - holds the compiled model as a Rustler ResourceArc
  │     - predict(model_resource, packed_bits) -> class label (i64)
  ├── lib/tsetlin_runner.ex        # public Elixir API
  │     - load(path) :: {:ok, model} | {:error, reason}
  │     - predict(model, packed_bits) :: {:ok, integer} | {:error, reason}
  │     - pack_bits([boolean]) :: binary
  └── lib/tsetlin_runner/native.ex # Rustler.NIF raw bindings module
```

`tsetlin_runner` is a plain Elixir library (a Hex-style dependency), not a
Nerves firmware project itself. A separate Nerves firmware project (with
`nerves_system_rpi0` etc.) depends on it and cross-compiles it as part of
its own `mix firmware`.

## 3. Model export format

`Tsetlin.jl`'s existing `compile(tm)` already strips training-only state
(`positive_clauses`/`negative_clauses`, the raw automata) and keeps only
the `*_included_literals` bitmask matrices — exactly what inference needs.
The exporter calls `compile(tm)` and dumps those matrices as-is (Julia
matrices are column-major, and the column slice `check_clause` reads is
already contiguous in that layout, so no transposition is needed on
export or import).

V1 does **not** export the `*_included_literals_idx` matrices (the
indexed-lookup optimization in `Tsetlin.jl`). The Rust NIF implements the
non-indexed `check_clause`/`vote` path. This is simple and fast enough for
the feature vector sizes involved (tens to ~1500 bits).

### Binary layout (`.tmbin`, little-endian)

```
magic       "TSTM"           4 bytes
version     u32 = 1
kind        u8   0 = bool (2 implicit classes, 1 shared clause block)
                 1 = general (N classes, N clause blocks)
clause_size u32  (input feature vector length, in bits)
chunks_size u32  (= ceil(clause_size / 64))
classes_num u32
ta_clauses  u32  (clauses per polarity per class)
lf          i64  (leniency factor — see below)
classes[]   i64 * classes_num   (labels; bool kind uses [1, 0] by convention)
blocks[]    1 block if kind=0, classes_num blocks if kind=1
  each block = 4 matrices of u64[chunks_size * ta_clauses], in order:
    positive_included_literals
    positive_included_literals_inverted
    negative_included_literals
    negative_included_literals_inverted
```

`lf` is `TMClassifier.LF` from Julia. It is required at inference time, not
just during training: `check_clause` returns `max(0, LF - mismatch_count)`
for each clause, and `vote()` sums that value (not a plain 0/1) across
clauses to produce `pos`/`neg`. `predict()` compares those sums directly,
so `LF` directly parameterizes every prediction and must travel with the
model. (`T`, `S`, `L` are training-only and are correctly excluded.)

Endianness is not negotiated: both the Julia export host and the Rust
NIF's runtime targets (x86_64 dev machine, ARMv6 Raspberry Pi Zero) are
little-endian.

Class labels are always encoded as `i64`. This covers every model
observed in the source project (`GroundTM` uses `Int` labels `1`/`2`;
`Bool`-classified models map to `1`/`0`). Any richer label semantics
(e.g. mapping `1` back to `:traversable`) is the caller's responsibility
in Elixir.

## 4. Inference data flow

1. The Elixir caller already has a boolean feature vector (produced by
   whatever feature-extraction code the caller uses — out of scope here).
2. `TsetlinRunner.pack_bits/1` (pure Elixir, no NIF) packs it into the
   same bit layout as Julia's `TMInput(x::AbstractArray{Bool})`: bit
   `i-1` of the input maps to bit `(i-1) mod 64` of chunk `floor((i-1)/64)`
   (LSB-first per 64-bit word).
3. `TsetlinRunner.predict/2` calls into the NIF with the loaded model
   resource and the packed binary. The NIF runs `vote()` per class block
   (or the single block for `kind=0`) and returns the winning class label.
4. Errors are returned as tagged tuples, never raised, for the two
   caller-facing failure modes: `{:error, :invalid_format}` (bad magic or
   unsupported version at `load/1`) and `{:error, :bit_length_mismatch}`
   (packed input size doesn't match the model's `chunks_size` at
   `predict/2`).

## 5. Rustler / Nerves cross-compilation

- Rust target for the Raspberry Pi Zero: `arm-unknown-linux-gnueabihf`
  (Tier-2 triple for ARMv6 Linux, hardfloat — the correct match for the
  Pi 1/Zero's BCM2835/ARM1176JZF-S).
- `tsetlin_runner` stays a plain Elixir library; the consuming Nerves
  firmware project cross-compiles it (and its NIF) as part of its own
  `MIX_TARGET=rpi0 mix firmware`. No separate precompiled-artifact
  download step (e.g. `rustler_precompiled`) is needed for V1 — Nerves
  always cross-compiles from the build host, it never compiles on the
  device itself.
- Nerves exports `CC`, `CROSSCOMPILE`, `TARGET_ARCH`, etc. during the
  build, but Rustler/Cargo don't read those directly. A small build-glue
  script in `native/tsetlin_nif/` translates `MIX_TARGET` into a Rust
  target triple via an explicit, documented table, then points Cargo's
  linker at the `CC` Nerves provides for that target:

  ```
  rpi0, rpi   -> arm-unknown-linux-gnueabihf   (ARMv6 hardfloat)
  rpi2        -> armv7-unknown-linux-gnueabihf
  rpi3, rpi3a -> armv7-unknown-linux-gnueabihf
  rpi4        -> armv7-unknown-linux-gnueabihf
  bbb         -> armv7-unknown-linux-gnueabihf
  host        -> native host triple (no cross-compilation)
  ```

  This table is deliberately explicit rather than inferred from
  `TARGET_ARCH` alone, so adding a new Nerves target later is a one-line
  change.
- On a normal (non-Nerves) dev machine, the library compiles and tests
  natively with no special setup — the crate has no dependencies besides
  `rustler`.
- `predict/2` is scheduled as a Rustler **DirtyCpu** NIF. The Pi Zero is
  single-core at 1GHz; running `vote()` on the normal (non-dirty)
  scheduler risks stalling the only BEAM scheduler thread for the
  duration of a possibly-nontrivial loop.

## 6. Testing and error handling

- **Rust**: `cargo test` unit tests for `check_clause`/`vote` against
  small, hand-written literal matrices with known expected results. Runs
  on the host, no BEAM required.
- **Elixir**: unit tests for `pack_bits/1` (verifies the LSB-first bit
  layout against known small vectors) and an end-to-end integration test
  (`load/1` + `predict/2`) against a small hand-crafted `.tmbin` fixture
  checked into the repo.
- **Cross-language parity**: no automated Julia-vs-Elixir test in this
  repo (adding a Julia toolchain to this devenv purely for that would be
  disproportionate). Parity is validated manually once, when the Julia
  exporter is written: run `predict()` in Julia and
  `TsetlinRunner.predict/2` in Elixir on the same exported model and
  input, and confirm they agree.
- The loaded model lives off the BEAM heap (Rust `ResourceArc`), which
  keeps GC pressure low on the Pi Zero's 512MB budget. It is reclaimed
  automatically when the Elixir reference is garbage-collected; no
  explicit `unload` API.
- Error surface is intentionally small: `{:error, :invalid_format}` and
  `{:error, :bit_length_mismatch}`, both returned (never raised) from the
  public API.

## 7. Explicitly deferred (not in this spec)

- The Julia-side `export_tm` function's exact code (implementation detail
  of the Julia repo, not this one — the format above is the contract).
- Support for models with more than the two `kind` variants described
  above, or for the indexed literal-lookup optimization.
- Any Nerves firmware project itself (this repo only produces the
  library that such a project would depend on).
