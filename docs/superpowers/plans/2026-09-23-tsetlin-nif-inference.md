# Tsetlin Machine Inference Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tsetlin_runner`, an Elixir library that loads a compiled
Tsetlin Machine (exported from the Julia `tsetlin_world` project) and runs
`predict` on it via a Rustler NIF, cross-compilable for a Nerves
Raspberry Pi Zero (rpi0) target.

**Architecture:** A Rust crate (`native/tsetlin_nif`) parses a small
versioned binary format into an in-memory model and implements the
Tsetlin Machine's `check_clause`/`vote`/`predict` inference math exactly
as `Tsetlin.jl` does it. A thin Elixir layer (`TsetlinRunner`) exposes
`load/1`, `predict/2`, and a pure-Elixir `pack_bits/1` helper, backed by a
raw NIF binding module (`TsetlinRunner.Native`). A pure mapping module
(`TsetlinRunner.Target`, defined in `mix.exs`) resolves Nerves'
`MIX_TARGET` to the Rust target triple Cargo should cross-compile for.

**Tech Stack:** Elixir ~> 1.18, Rustler (Rust NIFs), Rust 2021 edition.

**Spec:** `docs/superpowers/specs/2026-09-23-tsetlin-nif-design.md`

## Global Constraints

- Inference only in this repo — no `train!`/`feedback!` port.
- No feature-extraction port — callers supply an already-computed boolean
  feature vector; this library starts at `pack_bits/1`.
- No automated Julia-vs-Elixir regression test in this repo (per spec
  §6, parity is validated manually once the Julia exporter exists).
- Rust target triple for the Raspberry Pi Zero: `arm-unknown-linux-gnueabihf`.
- V1 does not export or use the `*_included_literals_idx` matrices — only
  the non-indexed `check_clause`/`vote` path is implemented.
- The model file format is versioned (`version = 1` today). Any parse
  failure (bad magic, unsupported version, invalid `kind`, truncated
  data) must surface as `{:error, :invalid_format}` — never raise, never
  crash the BEAM.
- `predict/2` is a Rustler **DirtyCpu** NIF (spec §5 — the Pi Zero is
  single-core and must not have its one BEAM scheduler stalled).
- `LF` (leniency factor) travels with every exported model and is used by
  `check_clause` at inference time; `T`, `S`, `L` are training-only and
  are not part of the format.

## Review Focus

- **Corrupted/malformed `.tmbin` file** (bad magic, unsupported version,
  invalid `kind` byte, or a file truncated mid-matrix) — must return
  `{:error, :invalid_format}`, not crash. Owned by Task 2 (Rust parser
  unit tests) and Task 5 (`load/1` integration test).
- **Wrong-length input at predict time** (packed bits size doesn't match
  the model's `chunks_size`) — must return `{:error, :bit_length_mismatch}`,
  not panic/segfault on an out-of-bounds read. Owned by Task 4/5.
- **Feature vectors spanning more than one 64-bit chunk** (every
  realistic model, e.g. GroundTM's ~1447 bits, is multi-chunk) — a
  fixture using only 2 bits would hide an off-by-one in chunk indexing.
  Owned by Task 1 (`pack_bits/1`, chunk-boundary test) and Task 3
  (`vote`, multi-clause/multi-chunk test).
- **Vote ties between classes** — `predict` must keep the first class
  whose vote it saw when a later class scores an equal (not strictly
  greater) margin, matching `Tsetlin.jl`'s `is_better = v > best_vote`.
  Owned by Task 3 and Task 5.
- **The `kind = 0` (bool) layout**, which has a single shared clause
  block instead of one block per class — structurally different from
  `kind = 1` and easy to get wrong if only the general case is tested.
  Owned by Task 3.

---

## File Structure

```
tsetlin_runner/
├── devenv.nix                          # modify: enable languages.rust
├── mix.exs                             # create: project + deps + MIX_TARGET->Cargo wiring
├── .formatter.exs                      # create: standard mix new output
├── lib/
│   ├── tsetlin_runner.ex               # create: public API (load/1, predict/2, pack_bits/1)
│   └── tsetlin_runner/
│       └── native.ex                   # create: raw Rustler binding module
├── native/
│   └── tsetlin_nif/
│       ├── Cargo.toml                  # create
│       └── src/
│           ├── lib.rs                  # create: Rustler NIF entry points, resource type
│           ├── format.rs               # create: .tmbin binary parser
│           └── tsetlin.rs              # create: check_clause / vote / predict math
└── test/
    ├── test_helper.exs                 # create: standard mix new output
    ├── tsetlin_runner_test.exs         # create: pack_bits/1 + load/predict integration tests
    └── tsetlin_runner_target_test.exs  # create: MIX_TARGET -> Cargo target mapping tests
```

---

### Task 1: Project scaffold + `pack_bits/1`

**Files:**
- Modify: `devenv.nix`
- Create: `mix.exs`
- Create: `.formatter.exs`
- Create: `lib/tsetlin_runner.ex`
- Create: `test/test_helper.exs`
- Create: `test/tsetlin_runner_test.exs`

**Interfaces:**
- Produces: `TsetlinRunner.pack_bits([boolean()]) :: binary()` — every
  later task that constructs a packed-bits input (Task 5's integration
  tests) uses this exact name and signature.

- [ ] **Step 1: Enable Rust in devenv (needed starting Task 2, harmless now)**

Edit `devenv.nix`, adding the Rust language next to the existing Elixir one:

```nix
  # https://devenv.sh/languages/
  languages.elixir.enable = true;
  languages.rust.enable = true;
```

- [ ] **Step 2: Create the Mix project files**

Create `mix.exs`:

```elixir
defmodule TsetlinRunner.MixProject do
  use Mix.Project

  def project do
    [
      app: :tsetlin_runner,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:rustler, "~> 0.34"}
    ]
  end
end
```

Create `.formatter.exs`:

```elixir
[
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"]
]
```

Create `test/test_helper.exs`:

```elixir
ExUnit.start()
```

- [ ] **Step 3: Write the failing tests for `pack_bits/1`**

Create `test/tsetlin_runner_test.exs`:

```elixir
defmodule TsetlinRunnerTest do
  use ExUnit.Case, async: true

  describe "pack_bits/1" do
    test "packs an empty list into an empty binary" do
      assert TsetlinRunner.pack_bits([]) == <<>>
    end

    test "packs fewer than 64 bits into a single little-endian u64, LSB first" do
      assert TsetlinRunner.pack_bits([true, true]) == <<3::little-64>>
      assert TsetlinRunner.pack_bits([true, false]) == <<1::little-64>>
      assert TsetlinRunner.pack_bits([false, true]) == <<2::little-64>>
      assert TsetlinRunner.pack_bits([false, false]) == <<0::little-64>>
    end

    test "spans multiple 64-bit chunks without losing bit 64" do
      # 64 `false` bits (chunk 0 == 0), then a single `true` bit, which
      # must land as bit 0 of chunk 1 (index 64 overall) -- a naive
      # chunking bug would drop it or shift it into the wrong chunk.
      bits = List.duplicate(false, 64) ++ [true]

      assert TsetlinRunner.pack_bits(bits) == <<0::little-64, 1::little-64>>
    end
  end
end
```

- [ ] **Step 4: Run the tests to confirm they fail**

Run: `devenv shell -- mix deps.get && devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: compile error — `TsetlinRunner` module (and `pack_bits/1`) does
not exist yet.

- [ ] **Step 5: Implement `pack_bits/1`**

Create `lib/tsetlin_runner.ex`:

```elixir
defmodule TsetlinRunner do
  @moduledoc """
  Runs inference for Tsetlin Machine classifiers exported from the Julia
  `tsetlin_world` project.
  """

  import Bitwise

  @doc """
  Packs a list of booleans into the little-endian, LSB-first, 64-bit-chunk
  layout Tsetlin Machine models expect (matching `Tsetlin.jl`'s `TMInput`
  bit layout). Bit `i` of the input (0-indexed) becomes bit `rem(i, 64)`
  of chunk `div(i, 64)`.
  """
  @spec pack_bits([boolean()]) :: binary()
  def pack_bits(bits) when is_list(bits) do
    bits
    |> Enum.chunk_every(64)
    |> Enum.map(&pack_chunk/1)
    |> IO.iodata_to_binary()
  end

  defp pack_chunk(chunk_bits) do
    value =
      chunk_bits
      |> Enum.with_index()
      |> Enum.reduce(0, fn
        {true, index}, acc -> acc ||| 1 <<< index
        {false, _index}, acc -> acc
      end)

    <<value::unsigned-little-64>>
  end
end
```

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: `3 tests, 0 failures`

- [ ] **Step 7: Commit**

```bash
git add devenv.nix mix.exs .formatter.exs lib/tsetlin_runner.ex test/test_helper.exs test/tsetlin_runner_test.exs
git commit -m "Scaffold tsetlin_runner Mix project and implement pack_bits/1"
```

---

### Task 2: Rust crate scaffold + `.tmbin` binary parser

**Files:**
- Create: `native/tsetlin_nif/Cargo.toml`
- Create: `native/tsetlin_nif/src/lib.rs` (stub only; real NIF wiring is Task 4)
- Create: `native/tsetlin_nif/src/format.rs`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (for Task 3 and Task 4): `format::Kind` (`Bool` | `General`),
  `format::ClauseBlock { positive_included_literals: Vec<u64>,
  positive_included_literals_inverted: Vec<u64>,
  negative_included_literals: Vec<u64>,
  negative_included_literals_inverted: Vec<u64> }`, `format::Model {
  kind: Kind, clause_size: u32, chunks_size: u32, classes_num: u32,
  ta_clauses: u32, lf: i64, classes: Vec<i64>, blocks: Vec<ClauseBlock> }`,
  `format::FormatError` (`InvalidMagic`, `UnsupportedVersion(u32)`,
  `InvalidKind(u8)`, `Truncated`), `format::parse(bytes: &[u8]) ->
  Result<Model, FormatError>`.

This task only builds and tests the Rust crate on its own (`cargo test`);
it is not yet wired into Elixir/Rustler (that's Task 4).

- [ ] **Step 1: Scaffold the crate**

Create `native/tsetlin_nif/Cargo.toml`:

```toml
[package]
name = "tsetlin_nif"
version = "0.1.0"
edition = "2021"

[lib]
name = "tsetlin_nif"
crate-type = ["cdylib", "rlib"]

[dependencies]
rustler = "0.34"
```

(`rlib` is included alongside `cdylib` so `cargo test` can compile and run
unit tests against this crate directly, in addition to the `cdylib` that
Rustler loads as a NIF.)

Create a placeholder `native/tsetlin_nif/src/lib.rs` so the crate builds
while `format.rs` is developed:

```rust
pub mod format;
```

- [ ] **Step 2: Write the failing tests for the binary format parser**

Create `native/tsetlin_nif/src/format.rs`:

```rust
use std::convert::TryInto;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Bool,
    General,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseBlock {
    pub positive_included_literals: Vec<u64>,
    pub positive_included_literals_inverted: Vec<u64>,
    pub negative_included_literals: Vec<u64>,
    pub negative_included_literals_inverted: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub kind: Kind,
    pub clause_size: u32,
    pub chunks_size: u32,
    pub classes_num: u32,
    pub ta_clauses: u32,
    pub lf: i64,
    pub classes: Vec<i64>,
    pub blocks: Vec<ClauseBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidKind(u8),
    Truncated,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.pos + n > self.data.len() {
            return Err(FormatError::Truncated);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, FormatError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn read_matrix(cur: &mut Cursor, len: usize) -> Result<Vec<u64>, FormatError> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(cur.u64()?);
    }
    Ok(v)
}

pub fn parse(bytes: &[u8]) -> Result<Model, FormatError> {
    let mut cur = Cursor::new(bytes);

    let magic = cur.take(4)?;
    if magic != b"TSTM" {
        return Err(FormatError::InvalidMagic);
    }

    let version = cur.u32()?;
    if version != 1 {
        return Err(FormatError::UnsupportedVersion(version));
    }

    let kind = match cur.u8()? {
        0 => Kind::Bool,
        1 => Kind::General,
        other => return Err(FormatError::InvalidKind(other)),
    };

    let clause_size = cur.u32()?;
    let chunks_size = cur.u32()?;
    let classes_num = cur.u32()?;
    let ta_clauses = cur.u32()?;
    let lf = cur.i64()?;

    let mut classes = Vec::with_capacity(classes_num as usize);
    for _ in 0..classes_num {
        classes.push(cur.i64()?);
    }

    let block_count = match kind {
        Kind::Bool => 1,
        Kind::General => classes_num as usize,
    };
    let matrix_len = (chunks_size as usize) * (ta_clauses as usize);

    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        blocks.push(ClauseBlock {
            positive_included_literals: read_matrix(&mut cur, matrix_len)?,
            positive_included_literals_inverted: read_matrix(&mut cur, matrix_len)?,
            negative_included_literals: read_matrix(&mut cur, matrix_len)?,
            negative_included_literals_inverted: read_matrix(&mut cur, matrix_len)?,
        });
    }

    Ok(Model {
        kind,
        clause_size,
        chunks_size,
        classes_num,
        ta_clauses,
        lf,
        classes,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiny 2-bit, 2-class fixture also used by `tsetlin.rs`'s tests
    /// and by the Elixir integration test:
    /// - class 1 fires on input bits (true, true)
    /// - class 2 fires on input bits (false, false)
    /// - LF = 2, one clause per polarity per class, no negative literals.
    pub fn tiny_general_model_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TSTM");
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version
        bytes.push(1); // kind = General
        bytes.extend_from_slice(&2u32.to_le_bytes()); // clause_size
        bytes.extend_from_slice(&1u32.to_le_bytes()); // chunks_size
        bytes.extend_from_slice(&2u32.to_le_bytes()); // classes_num
        bytes.extend_from_slice(&1u32.to_le_bytes()); // ta_clauses
        bytes.extend_from_slice(&2i64.to_le_bytes()); // lf
        bytes.extend_from_slice(&1i64.to_le_bytes()); // classes[0] = 1
        bytes.extend_from_slice(&2i64.to_le_bytes()); // classes[1] = 2
        // class 1 block: positive literals = 0b11, rest 0
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        // class 2 block: positive_inverted literals = 0b11, rest 0
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes
    }

    #[test]
    fn parses_the_tiny_general_fixture() {
        let model = parse(&tiny_general_model_bytes()).unwrap();
        assert_eq!(model.kind, Kind::General);
        assert_eq!(model.clause_size, 2);
        assert_eq!(model.chunks_size, 1);
        assert_eq!(model.classes_num, 2);
        assert_eq!(model.ta_clauses, 1);
        assert_eq!(model.lf, 2);
        assert_eq!(model.classes, vec![1, 2]);
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(model.blocks[0].positive_included_literals, vec![3]);
        assert_eq!(model.blocks[1].positive_included_literals_inverted, vec![3]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = tiny_general_model_bytes();
        bytes[0] = b'X';
        assert_eq!(parse(&bytes), Err(FormatError::InvalidMagic));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = tiny_general_model_bytes();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(parse(&bytes), Err(FormatError::UnsupportedVersion(2)));
    }

    #[test]
    fn rejects_truncated_data() {
        let bytes = tiny_general_model_bytes();
        let truncated = &bytes[..bytes.len() - 4];
        assert_eq!(parse(truncated), Err(FormatError::Truncated));
    }
}
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `devenv shell -- bash -c "cd native/tsetlin_nif && cargo test"`
Expected: this actually compiles and passes immediately since step 2 wrote
both the implementation and the tests together (the parser is simple
enough to write directly with its tests, unlike the multi-step Elixir
task). Confirm all 4 tests in `format::tests` pass. If any fails, fix
`parse`/`Cursor` until they do — do not change the test expectations,
which are derived by hand from the fixture's byte layout above.

- [ ] **Step 4: Confirm the crate builds standalone**

Run: `devenv shell -- bash -c "cd native/tsetlin_nif && cargo test"`
Expected: `test result: ok. 4 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/Cargo.toml native/tsetlin_nif/src/lib.rs native/tsetlin_nif/src/format.rs
git commit -m "Add tsetlin_nif crate with .tmbin binary format parser"
```

---

### Task 3: Rust inference core (`check_clause` / `vote` / `predict`)

**Files:**
- Create: `native/tsetlin_nif/src/tsetlin.rs`
- Modify: `native/tsetlin_nif/src/lib.rs` (add `pub mod tsetlin;`)

**Interfaces:**
- Consumes: `format::{Kind, ClauseBlock, Model}` from Task 2, and
  `format::tests::tiny_general_model_bytes()` (made `pub(crate)` in this
  task) for the shared fixture.
- Produces (for Task 4): `tsetlin::predict(model: &format::Model, chunks:
  &[u64]) -> i64`.

This task replicates `Tsetlin.jl`'s `check_clause`/`vote`/`predict`
(non-indexed path) verbatim. It stays a pure `cargo test` task — no
Rustler/BEAM involved yet.

- [ ] **Step 1: Make the shared test fixture visible to this module**

In `native/tsetlin_nif/src/format.rs`, change the fixture helper's
visibility so `tsetlin.rs`'s tests can reuse it:

```rust
pub(crate) fn tiny_general_model_bytes() -> Vec<u8> {
```

(was `pub fn tiny_general_model_bytes`, inside `#[cfg(test)] mod tests`)

- [ ] **Step 2: Write the failing tests**

Create `native/tsetlin_nif/src/tsetlin.rs`:

```rust
use crate::format::{ClauseBlock, Kind, Model};

/// Mirrors `Tsetlin.jl`'s non-indexed `check_clause`: for each chunk,
/// `val` bits mark literal *mismatches* against the input, and the
/// clause's score is `max(0, LF - mismatch_count)`.
fn check_clause(chunks: &[u64], literals: &[u64], literals_inverted: &[u64], lf: i64) -> i64 {
    let mut c: i64 = 0;
    for n in 0..chunks.len() {
        let chunk = chunks[n];
        let lit = literals[n];
        let lit_inv = literals_inverted[n];
        let val = ((lit ^ lit_inv) & chunk) ^ lit;
        c += val.count_ones() as i64;
    }
    (lf - c).max(0)
}

/// Mirrors `Tsetlin.jl`'s `vote`: sums `check_clause` (not a plain 0/1)
/// across every clause column of a block, separately for the positive
/// and negative polarity.
fn vote(chunks: &[u64], block: &ClauseBlock, chunks_size: usize, ta_clauses: usize, lf: i64) -> (i64, i64) {
    let mut pos = 0i64;
    let mut neg = 0i64;
    for i in 0..ta_clauses {
        let start = i * chunks_size;
        let end = start + chunks_size;
        pos += check_clause(
            chunks,
            &block.positive_included_literals[start..end],
            &block.positive_included_literals_inverted[start..end],
            lf,
        );
        neg += check_clause(
            chunks,
            &block.negative_included_literals[start..end],
            &block.negative_included_literals_inverted[start..end],
            lf,
        );
    }
    (pos, neg)
}

/// Mirrors `Tsetlin.jl`'s two `predict` methods: the `Bool` kind compares
/// a single block's pos/neg vote; the general kind picks the class with
/// the highest `pos - neg` margin, keeping the *first* class seen on a
/// tie (`v > best_vote` is strict, matching `Tsetlin.jl`'s `is_better`).
pub fn predict(model: &Model, chunks: &[u64]) -> i64 {
    let chunks_size = model.chunks_size as usize;
    let ta_clauses = model.ta_clauses as usize;

    match model.kind {
        Kind::Bool => {
            let (pos, neg) = vote(chunks, &model.blocks[0], chunks_size, ta_clauses, model.lf);
            if pos > neg {
                model.classes[0]
            } else {
                model.classes[1]
            }
        }
        Kind::General => {
            let mut best_vote = i64::MIN;
            let mut best_class = model.classes[0];
            for (i, block) in model.blocks.iter().enumerate() {
                let (pos, neg) = vote(chunks, block, chunks_size, ta_clauses, model.lf);
                let v = pos - neg;
                if v > best_vote {
                    best_vote = v;
                    best_class = model.classes[i];
                }
            }
            best_class
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format;

    #[test]
    fn check_clause_scores_by_mismatch_count_against_lf() {
        // literals = 0b11 (positive literal on both bits), lf = 2.
        assert_eq!(check_clause(&[0b11], &[0b11], &[0], 2), 2); // 0 mismatches
        assert_eq!(check_clause(&[0b01], &[0b11], &[0], 2), 1); // 1 mismatch
        assert_eq!(check_clause(&[0b00], &[0b11], &[0], 2), 0); // 2 mismatches, floored at 0
    }

    fn predict_tiny(bits: [bool; 2]) -> i64 {
        let bytes = format::tiny_general_model_bytes();
        let model = format::parse(&bytes).unwrap();
        let chunk: u64 = (bits[0] as u64) | ((bits[1] as u64) << 1);
        predict(&model, &[chunk])
    }

    #[test]
    fn predicts_class_2_when_both_bits_false() {
        assert_eq!(predict_tiny([false, false]), 2);
    }

    #[test]
    fn predicts_class_1_when_both_bits_true() {
        assert_eq!(predict_tiny([true, true]), 1);
    }

    #[test]
    fn breaks_ties_in_favor_of_the_first_class() {
        // (true, false) and (false, true) both score pos=1/neg=2 for
        // BOTH classes (v = -1 for each) -- class 1 (index 0) must win
        // because Tsetlin.jl's `is_better = v > best_vote` is strict.
        assert_eq!(predict_tiny([true, false]), 1);
        assert_eq!(predict_tiny([false, true]), 1);
    }

    #[test]
    fn vote_sums_check_clause_across_multiple_chunks_and_clauses() {
        // clause_size=65 -> chunks_size=2 (every real model, e.g.
        // GroundTM's ~1447 bits, needs more than one chunk and more than
        // one clause; the tiny 1-chunk/1-clause fixture above can't catch
        // a bug in the `start = i * chunks_size` column slicing or in
        // looping over more than one chunk). Two positive clauses, no
        // negative literals: clause A requires bit 0 true, clause B
        // requires bit 64 true (the sole bit of the second chunk).
        // Column-major flattening: [clauseA_chunk0, clauseA_chunk1,
        // clauseB_chunk0, clauseB_chunk1].
        let block = ClauseBlock {
            positive_included_literals: vec![0b1, 0, 0, 0b1],
            positive_included_literals_inverted: vec![0, 0, 0, 0],
            negative_included_literals: vec![0, 0, 0, 0],
            negative_included_literals_inverted: vec![0, 0, 0, 0],
        };
        let lf = 3;
        let chunks_size = 2;
        let ta_clauses = 2;

        // neg is always 6: two empty (all-zero-literal) negative clause
        // columns each score LF=3 (0 mismatches against an empty mask).
        assert_eq!(vote(&[1, 0], &block, chunks_size, ta_clauses, lf), (5, 6));
        assert_eq!(vote(&[0, 1], &block, chunks_size, ta_clauses, lf), (5, 6));
        assert_eq!(vote(&[1, 1], &block, chunks_size, ta_clauses, lf), (6, 6));
        assert_eq!(vote(&[0, 0], &block, chunks_size, ta_clauses, lf), (4, 6));
    }

    #[test]
    fn bool_kind_uses_a_single_shared_block() {
        // kind=0, clause_size=2, chunks_size=1, classes_num=2 (implicit
        // true/false), ta_clauses=1, lf=2. One block: positive literals
        // require both bits true; no negative literals.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TSTM");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(0); // kind = Bool
        bytes.extend_from_slice(&2u32.to_le_bytes()); // clause_size
        bytes.extend_from_slice(&1u32.to_le_bytes()); // chunks_size
        bytes.extend_from_slice(&2u32.to_le_bytes()); // classes_num
        bytes.extend_from_slice(&1u32.to_le_bytes()); // ta_clauses
        bytes.extend_from_slice(&2i64.to_le_bytes()); // lf
        bytes.extend_from_slice(&1i64.to_le_bytes()); // classes[0] = true
        bytes.extend_from_slice(&0i64.to_le_bytes()); // classes[1] = false
        // single block
        bytes.extend_from_slice(&3u64.to_le_bytes()); // positive literals = 0b11
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        let model = format::parse(&bytes).unwrap();
        assert_eq!(model.blocks.len(), 1);

        // both bits true -> pos=2, neg=0 -> true (classes[0])
        assert_eq!(predict(&model, &[0b11]), 1);
        // both bits false -> pos=0, neg=0 (empty negative block always
        // scores LF, but there IS no negative clause weight here since
        // neg matrices are all-zero too) -> pos(0) > neg(0) is false -> false
        assert_eq!(predict(&model, &[0b00]), 0);
    }
}
```

- [ ] **Step 3: Wire the module into the crate root**

In `native/tsetlin_nif/src/lib.rs`, add:

```rust
pub mod format;
pub mod tsetlin;
```

- [ ] **Step 4: Run the tests**

Run: `devenv shell -- bash -c "cd native/tsetlin_nif && cargo test"`
Expected: `test result: ok. 10 passed; 0 failed` (4 from `format`, 6 from
`tsetlin`). If an assertion doesn't match, re-derive the expected value by
hand from `check_clause`'s formula above rather than adjusting the
implementation to fit a guessed value.

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/src/lib.rs native/tsetlin_nif/src/format.rs native/tsetlin_nif/src/tsetlin.rs
git commit -m "Implement Tsetlin Machine inference math (check_clause/vote/predict)"
```

---

### Task 4: Rustler NIF wiring

**Files:**
- Modify: `native/tsetlin_nif/Cargo.toml` (no change needed — `rustler`
  is already a dependency; listed here because this task is where it
  starts being used)
- Modify: `native/tsetlin_nif/src/lib.rs`
- Create: `lib/tsetlin_runner/native.ex`
- Modify: `test/tsetlin_runner_test.exs` (add a smoke test)

**Interfaces:**
- Consumes: `format::parse`, `tsetlin::predict` from Tasks 2-3.
- Produces (for Task 5): `TsetlinRunner.Native.load_model_nif(path ::
  String.t()) :: {:ok, reference()} | {:error, :invalid_format |
  :io_error}` and `TsetlinRunner.Native.predict_nif(model :: reference(),
  packed_bits :: binary()) :: {:ok, integer()} | {:error,
  :bit_length_mismatch}`.

- [ ] **Step 1: Write the failing smoke test**

Add to `test/tsetlin_runner_test.exs` (inside the existing `TsetlinRunnerTest`
module, alongside the `pack_bits/1` describe block):

```elixir
  describe "TsetlinRunner.Native (raw NIF bindings)" do
    test "load_model_nif/1 returns an io_error tuple for a missing file" do
      assert TsetlinRunner.Native.load_model_nif("/nonexistent/path.tmbin") ==
               {:error, :io_error}
    end
  end
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: compile error — `TsetlinRunner.Native` does not exist yet.

- [ ] **Step 3: Implement the Rustler NIF entry points**

Replace `native/tsetlin_nif/src/lib.rs` with:

```rust
pub mod format;
pub mod tsetlin;

use rustler::{Atom, Binary, Env, ResourceArc, Term};

mod atoms {
    rustler::atoms! {
        invalid_format,
        io_error,
        bit_length_mismatch,
    }
}

pub struct ModelResource(pub format::Model);

fn format_error_to_atom(_err: format::FormatError) -> Atom {
    atoms::invalid_format()
}

#[rustler::nif]
fn load_model_nif(path: String) -> Result<ResourceArc<ModelResource>, Atom> {
    let bytes = std::fs::read(&path).map_err(|_| atoms::io_error())?;
    let model = format::parse(&bytes).map_err(format_error_to_atom)?;
    Ok(ResourceArc::new(ModelResource(model)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn predict_nif(resource: ResourceArc<ModelResource>, bits: Binary) -> Result<i64, Atom> {
    let model = &resource.0;
    let expected_len = (model.chunks_size as usize) * 8;
    if bits.len() != expected_len {
        return Err(atoms::bit_length_mismatch());
    }

    let chunks: Vec<u64> = bits
        .as_slice()
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Ok(tsetlin::predict(model, &chunks))
}

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ModelResource, env);
    true
}

rustler::init!("Elixir.TsetlinRunner.Native", load = on_load);
```

- [ ] **Step 4: Create the Elixir NIF binding module**

Create `lib/tsetlin_runner/native.ex`:

```elixir
defmodule TsetlinRunner.Native do
  @moduledoc false
  use Rustler, otp_app: :tsetlin_runner, crate: "tsetlin_nif"

  def load_model_nif(_path), do: :erlang.nif_error(:nif_not_loaded)
  def predict_nif(_resource, _packed_bits), do: :erlang.nif_error(:nif_not_loaded)
end
```

- [ ] **Step 5: Run the tests**

Run: `devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: this compiles the Rust crate (first time will take a while —
`cargo` fetches and builds `rustler` and its dependencies) and all tests,
including the new smoke test, pass.

- [ ] **Step 6: Commit**

```bash
git add native/tsetlin_nif/src/lib.rs lib/tsetlin_runner/native.ex test/tsetlin_runner_test.exs
git commit -m "Wire tsetlin_nif into Elixir via Rustler (load_model_nif/predict_nif)"
```

---

### Task 5: Public API (`load/1`, `predict/2`) + end-to-end tests

**Files:**
- Modify: `lib/tsetlin_runner.ex`
- Modify: `test/tsetlin_runner_test.exs`

**Interfaces:**
- Consumes: `TsetlinRunner.Native.load_model_nif/1`,
  `TsetlinRunner.Native.predict_nif/2` from Task 4;
  `TsetlinRunner.pack_bits/1` from Task 1.
- Produces: `TsetlinRunner.load(path :: String.t()) :: {:ok,
  reference()} | {:error, :invalid_format | :io_error}` and
  `TsetlinRunner.predict(model :: reference(), packed_bits :: binary())
  :: {:ok, integer()} | {:error, :bit_length_mismatch}` — the library's
  full public surface.

- [ ] **Step 1: Write the failing end-to-end tests**

Add to `test/tsetlin_runner_test.exs`, replacing the `Native` smoke test's
describe block with a fuller one that also exercises the public API
end-to-end against the same tiny fixture used in the Rust tests:

```elixir
  describe "load/1 and predict/2 (end-to-end against a tiny fixture model)" do
    # Mirrors native/tsetlin_nif/src/format.rs's tiny_general_model_bytes():
    # 2-bit input, 2 classes, lf=2, one clause per polarity per class.
    # Class 1 requires both input bits true; class 2 requires both false.
    defp tiny_model_bytes do
      <<
        "TSTM",
        1::little-32,
        1::8,
        2::little-32,
        1::little-32,
        2::little-32,
        1::little-32,
        2::little-64,
        1::little-64,
        2::little-64,
        3::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        3::little-64,
        0::little-64,
        0::little-64
      >>
    end

    defp with_tiny_model(fun) do
      path = Path.join(System.tmp_dir!(), "tsetlin_runner_test_#{System.unique_integer([:positive])}.tmbin")
      File.write!(path, tiny_model_bytes())

      try do
        {:ok, model} = TsetlinRunner.load(path)
        fun.(model)
      after
        File.rm(path)
      end
    end

    test "load/1 returns an io_error tuple for a missing file" do
      assert TsetlinRunner.load("/nonexistent/path.tmbin") == {:error, :io_error}
    end

    test "load/1 returns an invalid_format tuple for garbage bytes" do
      path = Path.join(System.tmp_dir!(), "tsetlin_runner_garbage_#{System.unique_integer([:positive])}.tmbin")
      File.write!(path, <<0, 1, 2, 3>>)

      assert TsetlinRunner.load(path) == {:error, :invalid_format}
      File.rm(path)
    end

    test "predicts class 2 when both input bits are false" do
      with_tiny_model(fn model ->
        bits = TsetlinRunner.pack_bits([false, false])
        assert TsetlinRunner.predict(model, bits) == {:ok, 2}
      end)
    end

    test "predicts class 1 when both input bits are true" do
      with_tiny_model(fn model ->
        bits = TsetlinRunner.pack_bits([true, true])
        assert TsetlinRunner.predict(model, bits) == {:ok, 1}
      end)
    end

    test "breaks a vote tie in favor of the first class" do
      with_tiny_model(fn model ->
        assert TsetlinRunner.predict(model, TsetlinRunner.pack_bits([true, false])) == {:ok, 1}
        assert TsetlinRunner.predict(model, TsetlinRunner.pack_bits([false, true])) == {:ok, 1}
      end)
    end

    test "predict/2 returns a bit_length_mismatch tuple for the wrong input size" do
      with_tiny_model(fn model ->
        too_short = <<0::little-32>>
        assert TsetlinRunner.predict(model, too_short) == {:error, :bit_length_mismatch}
      end)
    end
  end
```

Remove the now-redundant `TsetlinRunner.Native (raw NIF bindings)`
describe block added in Task 4 — this new block supersedes it by testing
the same missing-file case through the public API.

- [ ] **Step 2: Run the tests to confirm the new ones fail**

Run: `devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: compile error or failures — `TsetlinRunner.load/1` and
`TsetlinRunner.predict/2` don't exist yet.

- [ ] **Step 3: Implement the public API**

In `lib/tsetlin_runner.ex`, add (keep the existing `pack_bits/1` and its
`import Bitwise`/`pack_chunk/1`):

```elixir
  alias TsetlinRunner.Native

  @doc """
  Loads a compiled Tsetlin Machine model (a `.tmbin` file produced by the
  Julia `tsetlin_world` project's exporter) from `path`.
  """
  @spec load(String.t()) :: {:ok, reference()} | {:error, :invalid_format | :io_error}
  def load(path) when is_binary(path) do
    Native.load_model_nif(path)
  end

  @doc """
  Runs inference for `packed_bits` (see `pack_bits/1`) against a `model`
  returned by `load/1`.
  """
  @spec predict(reference(), binary()) :: {:ok, integer()} | {:error, :bit_length_mismatch}
  def predict(model, packed_bits) when is_binary(packed_bits) do
    Native.predict_nif(model, packed_bits)
  end
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `devenv shell -- mix test test/tsetlin_runner_test.exs`
Expected: all tests pass (3 `pack_bits/1` tests + 6 end-to-end tests).

- [ ] **Step 5: Commit**

```bash
git add lib/tsetlin_runner.ex test/tsetlin_runner_test.exs
git commit -m "Add public load/1 and predict/2 API with end-to-end tests"
```

---

### Task 6: `MIX_TARGET` -> Rust cross-compilation target mapping

**Files:**
- Modify: `mix.exs`
- Create: `test/tsetlin_runner_target_test.exs`

**Interfaces:**
- Produces: `TsetlinRunner.Target.resolve(mix_target :: String.t() | nil)
  :: {:ok, :native} | {:ok, String.t()} | {:error, :unknown_target}` and
  `TsetlinRunner.Target.configure_cargo_target!/0 :: :ok` (raises on an
  unknown `MIX_TARGET`).

This module is defined inside `mix.exs` itself, not under `lib/` —
`mix.exs` is evaluated before anything in `lib/` is compiled, so a module
it needs at that point cannot live there. Defining it directly in
`mix.exs` (a valid `.exs` script that can hold multiple `defmodule`s) is
the standard way to make compile-time configuration logic like this both
usable in `mix.exs` and independently testable, since `mix.exs` is
already loaded (and the module already defined) by the time `mix test`
runs any test file.

- [ ] **Step 1: Write the failing tests**

Create `test/tsetlin_runner_target_test.exs`:

```elixir
defmodule TsetlinRunner.TargetTest do
  use ExUnit.Case, async: false

  describe "resolve/1" do
    test "maps known Nerves rpi targets to their Rust triples" do
      assert TsetlinRunner.Target.resolve("rpi0") == {:ok, "arm-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi") == {:ok, "arm-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi2") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi3") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi3a") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("rpi4") == {:ok, "armv7-unknown-linux-gnueabihf"}
      assert TsetlinRunner.Target.resolve("bbb") == {:ok, "armv7-unknown-linux-gnueabihf"}
    end

    test "treats nil and \"host\" as no cross-compilation" do
      assert TsetlinRunner.Target.resolve(nil) == {:ok, :native}
      assert TsetlinRunner.Target.resolve("host") == {:ok, :native}
    end

    test "reports an unknown target as an error instead of guessing" do
      assert TsetlinRunner.Target.resolve("some_future_board") == {:error, :unknown_target}
    end
  end

  describe "configure_cargo_target!/0" do
    setup do
      original = System.get_env("MIX_TARGET")
      original_cargo = System.get_env("CARGO_BUILD_TARGET")

      on_exit(fn ->
        if original, do: System.put_env("MIX_TARGET", original), else: System.delete_env("MIX_TARGET")
        if original_cargo, do: System.put_env("CARGO_BUILD_TARGET", original_cargo), else: System.delete_env("CARGO_BUILD_TARGET")
      end)

      :ok
    end

    test "leaves CARGO_BUILD_TARGET unset when MIX_TARGET is unset" do
      System.delete_env("MIX_TARGET")
      System.delete_env("CARGO_BUILD_TARGET")

      assert TsetlinRunner.Target.configure_cargo_target!() == :ok
      assert System.get_env("CARGO_BUILD_TARGET") == nil
    end

    test "sets CARGO_BUILD_TARGET for a known Nerves target" do
      System.put_env("MIX_TARGET", "rpi0")

      assert TsetlinRunner.Target.configure_cargo_target!() == :ok
      assert System.get_env("CARGO_BUILD_TARGET") == "arm-unknown-linux-gnueabihf"
    end

    test "raises for an unknown MIX_TARGET instead of silently building for the host" do
      System.put_env("MIX_TARGET", "some_future_board")

      assert_raise RuntimeError, ~r/unknown MIX_TARGET/, fn ->
        TsetlinRunner.Target.configure_cargo_target!()
      end
    end
  end
end
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `devenv shell -- mix test test/tsetlin_runner_target_test.exs`
Expected: compile error — `TsetlinRunner.Target` does not exist yet.

- [ ] **Step 3: Implement `TsetlinRunner.Target` in `mix.exs`**

Replace `mix.exs` with:

```elixir
defmodule TsetlinRunner.Target do
  @moduledoc """
  Maps a Nerves `MIX_TARGET` value to the Rust target triple Cargo should
  cross-compile the `tsetlin_nif` crate for. Defined here (rather than
  under `lib/`) because `mix.exs` needs it before anything in `lib/` has
  been compiled.
  """

  @mapping %{
    "rpi0" => "arm-unknown-linux-gnueabihf",
    "rpi" => "arm-unknown-linux-gnueabihf",
    "rpi2" => "armv7-unknown-linux-gnueabihf",
    "rpi3" => "armv7-unknown-linux-gnueabihf",
    "rpi3a" => "armv7-unknown-linux-gnueabihf",
    "rpi4" => "armv7-unknown-linux-gnueabihf",
    "bbb" => "armv7-unknown-linux-gnueabihf"
  }

  @spec resolve(String.t() | nil) :: {:ok, :native} | {:ok, String.t()} | {:error, :unknown_target}
  def resolve(nil), do: {:ok, :native}
  def resolve("host"), do: {:ok, :native}

  def resolve(mix_target) when is_binary(mix_target) do
    case Map.fetch(@mapping, mix_target) do
      {:ok, triple} -> {:ok, triple}
      :error -> {:error, :unknown_target}
    end
  end

  @doc """
  Reads `MIX_TARGET` from the OS environment and sets `CARGO_BUILD_TARGET`
  accordingly, so Rustler's `cargo build` cross-compiles automatically.
  Raises on an unrecognized `MIX_TARGET` rather than silently falling
  back to a host build.
  """
  @spec configure_cargo_target!() :: :ok
  def configure_cargo_target! do
    case resolve(System.get_env("MIX_TARGET")) do
      {:ok, :native} ->
        :ok

      {:ok, triple} ->
        System.put_env("CARGO_BUILD_TARGET", triple)
        :ok

      {:error, :unknown_target} ->
        raise "tsetlin_runner: unknown MIX_TARGET=#{inspect(System.get_env("MIX_TARGET"))}, " <>
                "add it to TsetlinRunner.Target's mapping table"
    end
  end
end

TsetlinRunner.Target.configure_cargo_target!()

defmodule TsetlinRunner.MixProject do
  use Mix.Project

  def project do
    [
      app: :tsetlin_runner,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:rustler, "~> 0.34"}
    ]
  end
end
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `devenv shell -- mix test test/tsetlin_runner_target_test.exs`
Expected: `7 tests, 0 failures`

- [ ] **Step 5: Run the full test suite as a final regression check**

Run: `devenv shell -- mix test`
Expected: all tests across both files pass, confirming `mix.exs`'s new
top-level `configure_cargo_target!()` call didn't break the normal host
build (this host has no `MIX_TARGET` set, so it hits the `:native`
no-op branch every time `mix` starts).

- [ ] **Step 6: Commit**

```bash
git add mix.exs test/tsetlin_runner_target_test.exs
git commit -m "Add MIX_TARGET -> Rust target triple mapping for Nerves cross-compilation"
```

---

## Out of scope / follow-up work (not part of this plan)

- The Julia-side `export_tm(tm, path)` function that writes `.tmbin`
  files lives in the `tsetlin_world` repo (`/home/xabi/work/julia/tsetlin_world`),
  not this one, and is not covered here (spec §7).
- Actually building and booting a Nerves `rpi0` firmware image that
  depends on this library is a separate project; Task 6 only makes the
  cross-compilation target selection correct and testable in isolation.
- Manual Julia-vs-Elixir prediction parity check (spec §6) happens once
  the Julia exporter exists.
