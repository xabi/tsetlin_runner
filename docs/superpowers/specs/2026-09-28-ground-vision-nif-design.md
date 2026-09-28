# Ground vision NIF: JPEG to classification grid, on-device

Date: 2026-09-28
Status: approved (design), pending implementation plan

## 1. Purpose and scope

The 2026-09-23 spec (`2026-09-23-tsetlin-nif-design.md`) explicitly
deferred feature extraction: `tsetlin_runner` only ran `predict/2` against
an already-computed boolean feature vector, produced by whatever the
caller used. On the actual robot, nothing produces that vector: the Julia
`tsetlin_world` project's `GroundTM.jl`/`Perception.jl`/`RealBridge.jl`
implement the whole camera-frame to feature-vector pipeline (JPEG decode,
box-average resize, luminance/green/edge feature maps, windowed bit
packing), and Julia does not run on the Nerves target. This spec picks up
that deferred item: a new NIF function that takes a raw JPEG frame (as
`Robomow.Camera` already captures) and a loaded model, and returns a
classification grid directly, replicating the Julia pipeline's math
exactly.

This is sub-project 1 of "local autonomy for the robot" (see the
2026-09-25/2026-09-28 discussion in the `robomow_poncho` repo's
`tsetlin_world/docs/superpowers/plans/2026-09-25-ground-tm-rectangular-patches.md`).
Sub-project 2 (driving decisions, camera/Robot wiring, the Range sensor
safety backstop) is a separate spec, written after this one is
implemented and validated.

Explicitly out of scope for this spec:

- Wiring this into `Robomow.Camera`/`Robomow.Robot` or any driving logic.
- The `Robomow.Range` sensor or any safety-stop behavior.
- Training, or anything that changes what a `.tmbin` model contains -- this
  spec only adds a new input path (JPEG bytes) ahead of the existing
  `predict/2`, using the same loaded `ResourceArc<ModelResource>`.
- Capturing frames in a non-JPEG format to skip JPEG decode entirely
  (noted as a possible future optimization if decode cost turns out to
  matter; not attempted here).

## 2. Architecture

```
tsetlin_runner (Elixir, this repo)
  native/tsetlin_nif/src/
    lib.rs           existing: load_model_nif, predict_nif
    tsetlin.rs       existing: check_clause/vote/predict
    format.rs        existing: .tmbin parsing
    jpeg.rs          new: JPEG bytes -> Vec<f32> RGB, normalized [0,1]
    resize.rs        new: box-average resize (_resize_box equivalent)
    features.rs      new: luminance/green/edge maps, windowed bits
    lib.rs           new: classify_frame_nif ties the above together
  lib/tsetlin_runner.ex   new: classify_frame/4 wraps the NIF call
```

`classify_frame_nif` is a new NIF entry point alongside the existing
`load_model_nif`/`predict_nif` -- it does not replace or change either.

## 3. Public interface

```rust
#[rustler::nif]
fn classify_frame_nif(
    resource: ResourceArc<ModelResource>,
    jpeg: Binary,
    out_w: u32,
    out_h: u32,
    radius: u32,
) -> Result<Vec<i64>, Atom>
```

```elixir
@spec classify_frame(reference(), binary(), pos_integer(), pos_integer(), non_neg_integer()) ::
        {:ok, [integer()]} | {:error, atom()}
def classify_frame(model, jpeg, out_w, out_h, radius) when
    is_reference(model) and is_binary(jpeg) and
    is_integer(out_w) and out_w > 0 and
    is_integer(out_h) and out_h > 0 and
    is_integer(radius) and radius >= 0 do
  Native.classify_frame_nif(model, jpeg, out_w, out_h, radius)
end
```

The returned list has `out_w * out_h` entries, row-major: index
`row * out_w + col` (0-indexed row/col), each either `1`
(`GROUND_TRAVERSABLE`) or `2` (`GROUND_BLOCKED`) -- the same class labels
`predict/2` already returns, read from the loaded model's `classes[]`
(not hardcoded), so a differently-labeled model still round-trips
correctly. This is a different order than `GroundTM.jl`'s column-major
`(col, row)` grids -- deliberately: this is a new Elixir/Rust surface, not
a Julia one, and row-major is the natural order for both languages'
callers. The doc comment on both the Rust function and the Elixir wrapper
states this explicitly, since a swapped axis or transposed order is
exactly the class of bug the `robomow_poncho` rectangular-patches work
spent a whole plan chasing.

`radius` is a parameter (not fixed to the deployed model's radius=8)
because `predict_nif` already takes the caller's word for the model's
`chunks_size`/`clause_size` via the packed bit length -- this function
must build a feature vector of the exact length the specific loaded
model expects, and radius is the caller's only lever to control that
(`ground_feature_len(radius)` must equal the model's `clause_size`, or
`predict` fails with `bit_length_mismatch` exactly as it does today for a
mismatched external caller).

## 4. Pipeline (replicates tsetlin_world exactly)

Each step below cites the Julia source it must match bit-for-bit (or
within float rounding -- see Testing).

1. JPEG decode to RGB. `jpeg::decode` (a pure-Rust crate, e.g.
   `zune-jpeg` -- no C toolchain dependency to cross-compile for
   `arm-unknown-linux-gnueabihf`). Output: 8-bit RGB, no color
   management/ICC handling (matches PIL's default `Image.open(...).convert("RGB")`
   used throughout the Python labeling script and the Julia PPM pipeline).
2. Normalize. `byte as f32 / 255.0` per channel -- linear, no gamma
   correction (`GroundTM.jl`'s `_read_ppm`: `data[i] / Float32(maxv)`).
3. Box-average resize to `out_w x out_h` (`RealBridge.jl`'s
   `_resize_box`/`_box_range`): for each output cell, average every
   source pixel whose box-range (computed the same way, floor-based,
   inclusive) falls in that cell; clamp each averaged channel to
   `[0.0, 1.0]`.
4. Feature maps over the resized `out_w x out_h` image (`Perception.jl`):
   - Luminance `Y = 0.299R + 0.587G + 0.114B`.
   - Green-vs-neutral `green = G - (R + B) / 2`.
   - Sobel 3x3 gradients `G_w`/`G_h` on `Y` (kernels below), zero-padded
     at the border (Julia's `_conv3x3` leaves border row/col at exactly
     `0.0`, computing only interior `2..W-1, 2..H-1` -- replicate the same
     border behavior, not a reflected/clamped convolution):
     ```
     G_w = [-1 -2 -1; 0 0 0; 1 2 1]    (vertical edges)
     G_h = [-1  0  1; -2 0 2; -1 0 1]  (horizontal edges)
     ```
   - `edge = |G_w| > 0.15 || |G_h| > 0.15` (`EDGE_THRESHOLD`).
5. Per-cell windowed features (`GroundTM.jl`'s `ground_features`): for
   output cell `(col, row)` (1-indexed to match the Julia loop exactly
   during porting, converted to 0-indexed only at the final output-array
   write), scan `dc, dr` in `[-radius, radius]` in that nested order (`dc`
   outer, `dr` inner -- the bit order must match, since it determines
   which bits a model trained in Julia expects where); for each
   `(dc, dr)`, clamp `col+dc` to `[1, out_w]` and `row+dr` to `[1, out_h]`
   independently, then emit 5 bits: `Y > 0.28`, `Y > 0.40`, `Y > 0.50`
   (`GRAY_THRESHOLDS`), `green > 0.05` (`GROUND_GREEN_THRESHOLD`), `edge`.
   After the window, emit 2 more bits: `row > 0.65 * out_h`,
   `row > 0.80 * out_h`. Total: `(2*radius+1)^2 * 5 + 2` bits, matching
   `ground_feature_len(radius)`.
6. Pack + predict. `tsetlin::predict(model, chunks: &[u64])` already takes
   unpacked `u64` chunks directly (see `tsetlin.rs`) -- `predict_nif`
   itself only reaches that shape by unpacking the pre-packed `Binary` an
   Elixir caller sent via `TsetlinRunner.pack_bits/1`. Since this pipeline
   computes the cell's bits in Rust to begin with, build the `Vec<u64>`
   chunks directly from the per-cell `bool` array (new helper, matching
   `pack_bits/1`'s exact bit order: bit `i` of the input -> bit `i mod 64`
   of chunk `i div 64`, LSB-first) and call `tsetlin::predict` with it --
   no intermediate `Binary`/byte round-trip, and no per-pixel Elixir/Rust
   crossing (every one of the `out_w * out_h` cells is handled inside this
   one NIF call).

## 5. Error handling

Mirrors `predict_nif`'s existing style -- tagged errors, never a panic
across the NIF boundary:

- Malformed/undecodable JPEG -> `{:error, :invalid_jpeg}`.
- `out_w == 0 || out_h == 0` -> `{:error, :invalid_dimensions}` (checked
  before decode, so a bad call fails fast without doing JPEG work first --
  same "fail before the expensive part" shape as the Python labeling
  script's `--img-width 0` guard).
- A per-cell feature vector whose length doesn't match the loaded
  model's `clause_size` (i.e. `radius` doesn't match what the model was
  trained with) -> `{:error, :bit_length_mismatch}`, the same atom
  `predict_nif` already returns for this condition -- checked once up
  front (feature length is the same for every cell), not per cell.

## 6. Testing

The core claim this spec makes is numeric parity with `GroundTM.jl`, not
just "it runs" -- that is the test suite's job to prove, not an
afterthought:

- Rust unit tests, no BEAM required:
  - `resize.rs`: box-average resize against small hand-computed matrices
    (a handful of pixels with known averages), including a degenerate
    1-pixel-source and a non-square `out_w != out_h` case (this project's
    actual deployed shape).
  - `features.rs`: luminance/green/Sobel/edge against hand-computed
    values on a tiny synthetic image; explicit check that border pixels
    of the Sobel maps are exactly `0.0`.
  - `jpeg.rs`: decode a small fixture JPEG (checked into the repo) and
    assert the decoded RGB matches known pixel values.
- Cross-language parity fixtures (the fixture that actually validates
  this spec's purpose): generate, once, from `tsetlin_world` in Julia --
  a small synthetic RGB image (e.g. a sky/grass gradient, the same shape
  used in `test/ground_tm.jl`'s rectangular tests), JPEG-encode it, and
  record `GroundTM.jl`'s `ground_features`/`ground_to_sim_frame` output
  for that exact image at a couple of `(out_w, out_h, radius)`
  combinations including the deployed `(32, 24, 8)`. Check both the JPEG
  fixture and the expected output grid into `tsetlin_runner`'s test
  fixtures. A `cargo test` then decodes that fixture JPEG through the new
  pipeline and asserts the classification grid matches exactly (feature
  bits are booleans off float comparisons against fixed thresholds, so
  exact equality is the right bar, not a tolerance -- a boundary case
  landing on the wrong side of a threshold due to a rounding difference
  is precisely the kind of bug this parity test exists to catch).
- Elixir integration test: `TsetlinRunner.classify_frame/4` against the
  same fixture JPEG and a small hand-crafted `.tmbin`, asserting the
  returned list's shape (`out_w * out_h` elements) and the row-major
  index mapping (spot-check a couple of known cells against the fixture's
  expected grid, not just the aggregate).
- On-device: no new on-device benchmark is required by this spec (the
  existing `Robomow.TsetlinBench` pattern already measures `predict/2`; a
  follow-up device benchmark of `classify_frame/4`'s end-to-end per-frame
  cost -- JPEG decode + resize + features, not just predict -- belongs to
  sub-project 2, once there's a real frame source to benchmark against).

## 7. Explicitly deferred (not in this spec)

- Raw (non-JPEG) frame capture to skip decode cost.
- Driving decisions, `Robomow.Camera`/`Robomow.Robot` wiring, the Range
  sensor safety backstop (sub-project 2).
- Any change to the `.tmbin` format or `predict_nif`'s existing contract.
- Multi-frame temporal smoothing or any state carried between calls --
  `classify_frame_nif` is stateless per call, exactly like `predict_nif`.
