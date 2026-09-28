# Ground Vision NIF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `classify_frame_nif`/`TsetlinRunner.classify_frame/4` entry point that takes a raw JPEG frame and a loaded Tsetlin Machine model and returns a classification grid directly, replicating `tsetlin_world`'s Julia feature-extraction pipeline exactly.

**Architecture:** Four new Rust modules inside `tsetlin_nif` (JPEG decode, box-average resize, feature maps, per-cell windowed features + bit packing), tied together by one new NIF function that never crosses the Elixir/Rust boundary more than once per frame. Correctness is proven against fixtures generated once from the real Julia pipeline, not just "it runs."

**Tech Stack:** Rust (Rustler NIF, existing crate `tsetlin_nif`), a pure-Rust JPEG decoder (no C toolchain dependency, for `arm-unknown-linux-gnueabihf` cross-compilation), Elixir (ExUnit).

**Spec:** `docs/superpowers/specs/2026-09-28-ground-vision-nif-design.md`

## Global Constraints

- Cross-compile target stays `arm-unknown-linux-gnueabihf` (unchanged); this plan's own TDD loop runs entirely with host `cargo test`/`mix test` — no Nerves firmware build is part of this plan.
- JPEG decode must be a pure-Rust crate (no C dependency to cross-compile).
- `classify_frame_nif` is a **new** NIF entry point. `predict_nif`/`load_model_nif`'s existing contracts are unchanged.
- Output: flat list, `out_w * out_h` entries, **row-major** (`index = row * out_w + col`, 0-indexed), values are the loaded model's own `classes[]` labels (never hardcoded `1`/`2`).
- Byte→float normalization is `byte as f32 / 255.0` — no gamma correction.
- Feature formulas must match `tsetlin_world` exactly: luminance `0.299R+0.587G+0.114B`, green `G-(R+B)/2`, Sobel kernels `[-1,-2,-1;0,0,0;1,2,1]` / `[-1,0,1;-2,0,2;-1,0,1]`, edge threshold `0.15`, gray thresholds `(0.28,0.40,0.50)`, green threshold `0.05`, position thresholds `row > 0.65*out_h` / `row > 0.80*out_h`.
- A NIF must never crash the BEAM: any internal panic (e.g. from the third-party JPEG decoder on adversarial input) is caught and converted to a tagged error.

## Review Focus

- **A genuine Rust panic inside JPEG decode must not crash the whole BEAM.** This is robot control software; the spec's error handling covers clean decode errors (`Result::Err`) but a malformed/adversarial byte sequence could also trigger a panic inside third-party decoder code. Task 1 wraps the decode call in `std::panic::catch_unwind`.
- **`radius=0` (the minimal case — a window is just the center pixel, 7 bits total) must produce exactly 7 bits, not silently 0 or the wrong count.** An off-by-one in loop bounds is easy to miss when the common case (radius=8) has a large, forgiving window. Task 4 tests it explicitly.
- **A `dc`/`dr` (column-offset/row-offset) transcription swap in the windowed-feature loop.** This is the exact bug class that took a whole plan to catch in `tsetlin_world`'s Julia code on 2026-09-25 (width used where height was needed). Task 4's test uses a single-hot-pixel asymmetric image specifically constructed so a swap changes the result.
- **`out_w`/`out_h` larger than the source image (upsampling) must not panic or read out of bounds**, even though it isn't a real deployment scenario (the camera is always downsized). Task 2 tests this explicitly — the assertion is "no panic, valid output," not a specific "correct" upsampled value.
- **A `radius` argument that doesn't match the loaded model's `clause_size`** (e.g. calling with the deployed radius=8 model but passing `radius=1`) must surface as `{:error, :bit_length_mismatch}`, not an out-of-bounds panic or — worse — a silently-wrong prediction from reading garbage past the end of a too-short bit vector. Task 5 tests this at the `classify_frame` level.

---

### Task 1: JPEG decode

**Files:**
- Modify: `native/tsetlin_nif/Cargo.toml`
- Create: `native/tsetlin_nif/src/jpeg.rs`
- Modify: `native/tsetlin_nif/src/lib.rs` (add `mod jpeg;`)
- Create: `native/tsetlin_nif/tests/fixtures/tiny_solid.jpg` (generated, not hand-written — see Step 2)

**Interfaces:**
- Produces: `pub struct RgbImage { pub width: u32, pub height: u32, pub data: Vec<f32> }` where `data[(y * width + x) as usize * 3 + c]` is channel `c` (0=R,1=G,2=B) of pixel `(x, y)`, already normalized to `[0.0, 1.0]`. `pub fn decode(jpeg_bytes: &[u8]) -> Result<RgbImage, DecodeError>` where `pub enum DecodeError { Malformed }`. Consumed by Task 2 (`resize_box` takes an `&RgbImage`).

- [ ] **Step 1: Write the failing test**

Add to `native/tsetlin_nif/src/jpeg.rs`:

```rust
use std::panic;

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    Malformed,
}

pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

pub fn decode(_jpeg_bytes: &[u8]) -> Result<RgbImage, DecodeError> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/tiny_solid.jpg");

    // Fixture is 16x16: rows 0-7 a solid reddish block (200,50,50), rows
    // 8-15 a solid greenish block (50,180,60) -- see Step 2 for how it was
    // generated. JPEG is lossy even at quality=100, so pixel values are
    // checked within a small tolerance, not bit-exact.
    fn approx(a: f32, b: u8, tol: f32) -> bool {
        (a - (b as f32 / 255.0)).abs() <= tol
    }

    #[test]
    fn decodes_known_dimensions_and_pixel_blocks() {
        let img = decode(FIXTURE).expect("fixture should decode");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);

        let top = 3 * ((2 * img.width + 2) as usize); // (x=2, y=2), red block
        assert!(approx(img.data[top], 200, 0.03));
        assert!(approx(img.data[top + 1], 50, 0.03));
        assert!(approx(img.data[top + 2], 50, 0.03));

        let bottom = 3 * ((2 * img.width + 12) as usize); // (x=2, y=12), green block
        assert!(approx(img.data[bottom], 50, 0.03));
        assert!(approx(img.data[bottom + 1], 180, 0.03));
        assert!(approx(img.data[bottom + 2], 60, 0.03));
    }

    #[test]
    fn returns_malformed_not_panic_for_truncated_bytes() {
        // First 10 bytes of a real JPEG is not a valid JPEG, but must not
        // panic -- this is the untrusted-input boundary (a camera frame).
        let truncated = &FIXTURE[..10];
        assert_eq!(decode(truncated), Err(DecodeError::Malformed));
    }
}
```

- [ ] **Step 2: Generate the fixture, then run the test to confirm it fails**

The fixture doesn't exist yet either — generate it first:

```bash
cd /home/xabi/work/elixir/tsetlin_runner
mkdir -p native/tsetlin_nif/tests/fixtures
devenv shell -- uv run --with pillow python3 -c "
from PIL import Image
im = Image.new('RGB', (16, 16))
px = im.load()
for x in range(16):
    for y in range(16):
        px[x, y] = (200, 50, 50) if y < 8 else (50, 180, 60)
im.save('native/tsetlin_nif/tests/fixtures/tiny_solid.jpg', quality=100, subsampling=0)
"
```

Run: `cd native/tsetlin_nif && cargo test jpeg::`
Expected: compiles, then panics at `unimplemented!()` in
`decodes_known_dimensions_and_pixel_blocks` (the first test to run).

- [ ] **Step 3: Add the JPEG decoder dependency**

```bash
cd native/tsetlin_nif
cargo add zune-jpeg
```

This resolves and pins whatever the current published version is — do
not hand-guess a version number.

- [ ] **Step 4: Implement `decode`**

Replace the `unimplemented!()` body. The exact method names below are
`zune-jpeg`'s typical decode-to-RGB API as of this writing; if
`cargo add` pulled a version whose API differs, check
`cargo doc -p zune-jpeg --open` and adjust — the contract that matters is
"JPEG bytes in, `(width, height, RGB u8 bytes)` out, `Err` (not a panic)
on malformed input", not these exact calls:

```rust
pub fn decode(jpeg_bytes: &[u8]) -> Result<RgbImage, DecodeError> {
    // catch_unwind is the untrusted-input boundary: a malformed/adversarial
    // camera frame must never crash the whole BEAM by panicking inside
    // third-party decoder code (see this plan's Global Constraints).
    let result = panic::catch_unwind(|| {
        let mut decoder = zune_jpeg::JpegDecoder::new(jpeg_bytes);
        let pixels = decoder.decode().map_err(|_| DecodeError::Malformed)?;
        let info = decoder.info().ok_or(DecodeError::Malformed)?;
        Ok::<_, DecodeError>((info.width as u32, info.height as u32, pixels))
    });

    let (width, height, pixels) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(DecodeError::Malformed), // caught panic
    };

    let data = pixels.iter().map(|&b| b as f32 / 255.0).collect();
    Ok(RgbImage { width, height, data })
}
```

Add `mod jpeg;` near the top of `native/tsetlin_nif/src/lib.rs`.

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `cargo test jpeg::`
Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/xabi/work/elixir/tsetlin_runner
git add native/tsetlin_nif/Cargo.toml native/tsetlin_nif/Cargo.lock \
  native/tsetlin_nif/src/jpeg.rs native/tsetlin_nif/src/lib.rs \
  native/tsetlin_nif/tests/fixtures/tiny_solid.jpg
git commit -m "tsetlin_nif: JPEG decode to normalized RgbImage, panic-safe"
```

---

### Task 2: Box-average resize

**Files:**
- Create: `native/tsetlin_nif/src/resize.rs`
- Modify: `native/tsetlin_nif/src/lib.rs` (add `mod resize;`)

**Interfaces:**
- Consumes: `jpeg::RgbImage` from Task 1.
- Produces: `pub fn resize_box(img: &RgbImage, out_w: u32, out_h: u32) -> RgbImage` (same `RgbImage` shape, now `out_w x out_h`). Consumed by Task 3 (`compute_feature_maps` takes the resized `RgbImage`).

- [ ] **Step 1: Write the failing tests**

Create `native/tsetlin_nif/src/resize.rs`:

```rust
use crate::jpeg::RgbImage;

/// Inclusive source-index range (0-indexed) that box-downsample output
/// index `i` (of `size_out` total) should average over, out of `size_in`
/// source pixels. Mirrors tsetlin_world's `_box_range` (RealBridge.jl),
/// translated from Julia's 1-indexed floor-based formula to 0-indexed.
fn box_range(i: u32, size_in: u32, size_out: u32) -> (u32, u32) {
    let i0 = ((i as u64 * size_in as u64) / size_out as u64) as u32;
    let i1_raw = (((i + 1) as u64 * size_in as u64) / size_out as u64) as u32;
    let i1 = i1_raw.saturating_sub(1).max(i0).min(size_in - 1);
    (i0.min(size_in - 1), i1)
}

pub fn resize_box(_img: &RgbImage, _out_w: u32, _out_h: u32) -> RgbImage {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, data: Vec<f32>) -> RgbImage {
        assert_eq!(data.len() as u32, w * h * 3);
        RgbImage { width: w, height: h, data }
    }

    #[test]
    fn averages_a_4x4_source_down_to_a_non_square_2x1() {
        // 4 columns x 4 rows, all channels equal to the column index (0..3)
        // scaled by 0.1, so each output cell's expected average is easy to
        // hand-check. out_w=2, out_h=1 deliberately non-square.
        let mut data = vec![0.0f32; 4 * 4 * 3];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = x as f32 * 0.1;
                let base = ((y * 4 + x) * 3) as usize;
                data[base] = v;
                data[base + 1] = v;
                data[base + 2] = v;
            }
        }
        let src = img(4, 4, data);
        let out = resize_box(&src, 2, 1);
        assert_eq!((out.width, out.height), (2, 1));
        // Left output column averages source columns 0-1 (v=0.0,0.1) -> 0.05
        assert!((out.data[0] - 0.05).abs() < 1e-6);
        // Right output column averages source columns 2-3 (v=0.2,0.3) -> 0.25
        assert!((out.data[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_1x1_source_upsampled_does_not_panic_and_copies_the_pixel() {
        let src = img(1, 1, vec![0.4, 0.5, 0.6]);
        let out = resize_box(&src, 3, 2);
        assert_eq!((out.width, out.height), (3, 2));
        for cell in 0..6 {
            let base = cell * 3;
            assert!((out.data[base] - 0.4).abs() < 1e-6);
            assert!((out.data[base + 1] - 0.5).abs() < 1e-6);
            assert!((out.data[base + 2] - 0.6).abs() < 1e-6);
        }
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test resize::`
Expected: panics at `unimplemented!()`.

- [ ] **Step 3: Implement `resize_box`**

```rust
pub fn resize_box(img: &RgbImage, out_w: u32, out_h: u32) -> RgbImage {
    let mut data = vec![0.0f32; (out_w * out_h * 3) as usize];
    for oy in 0..out_h {
        let (y0, y1) = box_range(oy, img.height, out_h);
        for ox in 0..out_w {
            let (x0, x1) = box_range(ox, img.width, out_w);
            let mut sum = [0.0f32; 3];
            let mut n = 0u32;
            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    let base = ((sy * img.width + sx) * 3) as usize;
                    sum[0] += img.data[base];
                    sum[1] += img.data[base + 1];
                    sum[2] += img.data[base + 2];
                    n += 1;
                }
            }
            let out_base = ((oy * out_w + ox) * 3) as usize;
            for c in 0..3 {
                data[out_base + c] = (sum[c] / n as f32).clamp(0.0, 1.0);
            }
        }
    }
    RgbImage { width: out_w, height: out_h, data }
}
```

- [ ] **Step 4: Run to confirm it passes**

Run: `cargo test resize::`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/src/resize.rs native/tsetlin_nif/src/lib.rs
git commit -m "tsetlin_nif: box-average resize (_resize_box equivalent)"
```

---

### Task 3: Feature maps (luminance, green, Sobel edges)

**Files:**
- Create: `native/tsetlin_nif/src/features.rs`
- Modify: `native/tsetlin_nif/src/lib.rs` (add `mod features;`)

**Interfaces:**
- Consumes: `jpeg::RgbImage` from Task 1 (specifically the resized output of Task 2's `resize_box`).
- Produces: `pub struct FeatureMaps { pub width: u32, pub height: u32, pub luminance: Vec<f32>, pub green: Vec<f32>, pub edge: Vec<bool> }` (each `Vec` indexed `y * width + x`, 0-indexed) and `pub fn compute_feature_maps(img: &RgbImage) -> FeatureMaps`. Consumed by Task 4 (`cell_bits` reads `FeatureMaps` fields).

- [ ] **Step 1: Write the failing tests**

Create `native/tsetlin_nif/src/features.rs`:

```rust
use crate::jpeg::RgbImage;

const GRAY_THRESHOLDS: [f32; 3] = [0.28, 0.40, 0.50];
const GREEN_THRESHOLD: f32 = 0.05;
const EDGE_THRESHOLD: f32 = 0.15;
const SOBEL_W: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
const SOBEL_H: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

pub struct FeatureMaps {
    pub width: u32,
    pub height: u32,
    pub luminance: Vec<f32>,
    pub green: Vec<f32>,
    pub edge: Vec<bool>,
}

pub fn compute_feature_maps(_img: &RgbImage) -> FeatureMaps {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, data: Vec<f32>) -> RgbImage {
        RgbImage { width: w, height: h, data }
    }

    #[test]
    fn luminance_and_green_match_hand_computed_values() {
        // Single pixel, R=0.5 G=0.6 B=0.1.
        // Y = 0.299*0.5 + 0.587*0.6 + 0.114*0.1 = 0.5096
        // green = 0.6 - (0.5+0.1)/2 = 0.3
        let src = img(1, 1, vec![0.5, 0.6, 0.1]);
        let maps = compute_feature_maps(&src);
        assert!((maps.luminance[0] - 0.5096).abs() < 1e-4);
        assert!((maps.green[0] - 0.3).abs() < 1e-4);
    }

    #[test]
    fn sobel_border_pixels_are_exactly_zero_edge() {
        // 3x3, one bright center pixel, rest dark -- tsetlin_world's
        // _conv3x3 only computes the interior (leaves the border at exactly
        // 0.0), so every border pixel's edge bit must be false regardless
        // of content, and the interior pixel is the only one that can be
        // true.
        let mut data = vec![0.0f32; 3 * 3 * 3];
        let center = ((1 * 3 + 1) * 3) as usize;
        data[center] = 1.0;
        data[center + 1] = 1.0;
        data[center + 2] = 1.0;
        let src = img(3, 3, data);
        let maps = compute_feature_maps(&src);
        for y in 0..3u32 {
            for x in 0..3u32 {
                if x == 1 && y == 1 {
                    continue;
                }
                assert!(!maps.edge[(y * 3 + x) as usize], "border ({x},{y}) must be false");
            }
        }
    }

    #[test]
    fn sobel_w_and_sobel_h_are_not_transposed() {
        // tsetlin_world's img/Y arrays are indexed [w, h] (dim1=column,
        // dim2=row) -- Julia's `_SOBEL_W`/`_SOBEL_H` literals are matrices
        // in that same convention, so a straightforward Rust port using a
        // row-major `y[row*width+col]` layout can easily end up applying
        // the wrong kernel to the wrong axis (a transpose). Verified against
        // the real Julia function (`TsetlinWorld._conv3x3`) on 2026-09-28:
        // a 5x5 luminance that varies only along the COLUMN axis (value =
        // column index, 1..5, constant per row) gives `_conv3x3(Y,
        // _SOBEL_W)[3,3] == 8.0` and `_conv3x3(Y, _SOBEL_H)[3,3] == 0.0` --
        // i.e. SOBEL_W responds to a column-direction gradient, SOBEL_H
        // does not.
        let w = 5u32;
        let mut luminance = vec![0.0f32; (w * w) as usize];
        for row in 0..w {
            for col in 0..w {
                luminance[(row * w + col) as usize] = (col + 1) as f32;
            }
        }
        let img_data: Vec<f32> = luminance.iter().flat_map(|&y| [y, y, y]).collect();
        let src = img(w, w, img_data);
        let maps = compute_feature_maps(&src);

        // Rust is 0-indexed; Julia's Y[3,3] (1-indexed, center of a 5x5) is
        // this test's (col=2, row=2).
        let center = (2 * w + 2) as usize;
        // compute_feature_maps only exposes the *thresholded* edge bool,
        // not the raw Sobel magnitude -- exercise the raw gradients
        // directly here via the crate-private conv3x3 helper so this test
        // can assert the exact value (8.0), not just "edge is true/false".
        let gw = conv3x3(&luminance, w, w, &SOBEL_W);
        let gh = conv3x3(&luminance, w, w, &SOBEL_H);
        assert!((gw[center] - 8.0).abs() < 1e-4, "gw[center] = {}", gw[center]);
        assert!((gh[center] - 0.0).abs() < 1e-4, "gh[center] = {}", gh[center]);
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test features::`
Expected: panics at `unimplemented!()`.

- [ ] **Step 3: Implement `compute_feature_maps`**

```rust
fn conv3x3(y: &[f32], width: u32, height: u32, kernel: &[[f32; 3]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; (width * height) as usize];
    if width < 3 || height < 3 {
        return out; // no interior to convolve; matches Julia's empty range
    }
    for row in 1..(height - 1) {
        for col in 1..(width - 1) {
            let mut s = 0.0f32;
            for kr in 0..3u32 {
                for kc in 0..3u32 {
                    let sy = row + kr - 1;
                    let sx = col + kc - 1;
                    s += y[(sy * width + sx) as usize] * kernel[kc as usize][kr as usize];
                }
            }
            out[(row * width + col) as usize] = s;
        }
    }
    out
}

pub fn compute_feature_maps(img: &RgbImage) -> FeatureMaps {
    let n = (img.width * img.height) as usize;
    let mut luminance = vec![0.0f32; n];
    let mut green = vec![0.0f32; n];
    for i in 0..n {
        let base = i * 3;
        let (r, g, b) = (img.data[base], img.data[base + 1], img.data[base + 2]);
        luminance[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        green[i] = g - (r + b) / 2.0;
    }
    let gw = conv3x3(&luminance, img.width, img.height, &SOBEL_W);
    let gh = conv3x3(&luminance, img.width, img.height, &SOBEL_H);
    let edge = (0..n)
        .map(|i| gw[i].abs() > EDGE_THRESHOLD || gh[i].abs() > EDGE_THRESHOLD)
        .collect();

    FeatureMaps { width: img.width, height: img.height, luminance, green, edge }
}
```

(`GRAY_THRESHOLDS`/`GREEN_THRESHOLD` are unused by this task's own code —
they're consumed by Task 4 — keep them here since they're this module's
constants per the spec; `#[allow(dead_code)]` if the compiler warns
before Task 4 lands.)

- [ ] **Step 4: Run to confirm it passes**

Run: `cargo test features::`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/src/features.rs native/tsetlin_nif/src/lib.rs
git commit -m "tsetlin_nif: luminance/green/Sobel-edge feature maps"
```

---

### Task 4: Per-cell windowed features and bit packing

**Files:**
- Create: `native/tsetlin_nif/src/window.rs`
- Modify: `native/tsetlin_nif/src/lib.rs` (add `mod window;`)

**Interfaces:**
- Consumes: `features::FeatureMaps` from Task 3.
- Produces: `pub fn cell_bits(maps: &FeatureMaps, col: u32, row: u32, radius: u32) -> Vec<bool>` (length `(2*radius+1)^2 * 5 + 2`) and `pub fn pack_bits(bits: &[bool]) -> Vec<u64>` (LSB-first, matching `TsetlinRunner.pack_bits/1`). Both consumed by Task 5 (`classify_frame_nif` calls `cell_bits` then `pack_bits` per grid cell, then `tsetlin::predict`).

- [ ] **Step 1: Write the failing tests**

Create `native/tsetlin_nif/src/window.rs`:

```rust
use crate::features::FeatureMaps;

pub fn cell_bits(_maps: &FeatureMaps, _col: u32, _row: u32, _radius: u32) -> Vec<bool> {
    unimplemented!()
}

pub fn pack_bits(_bits: &[bool]) -> Vec<u64> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_maps(width: u32, height: u32) -> FeatureMaps {
        FeatureMaps {
            width,
            height,
            luminance: vec![0.0; (width * height) as usize],
            green: vec![0.0; (width * height) as usize],
            edge: vec![false; (width * height) as usize],
        }
    }

    #[test]
    fn radius_zero_produces_exactly_seven_bits() {
        let maps = flat_maps(4, 4);
        let bits = cell_bits(&maps, 1, 1, 0);
        assert_eq!(bits.len(), 7); // (2*0+1)^2 * 5 + 2
    }

    #[test]
    fn does_not_swap_column_and_row_offsets() {
        // 5x5, luminance is 0 everywhere except one "hot" pixel at
        // (col=3, row=1). Classifying cell (col=2, row=2) with radius=1:
        // the only window offset that reaches the hot pixel is
        // (dc=+1, dr=-1) -> (col=3, row=1). If dc/dr were swapped inside
        // cell_bits, the implementation would instead look at
        // (dc=-1, dr=+1) -> (col=1, row=3), which is NOT hot, and every
        // window position's luminance bits would read as "low" -- so a
        // swap changes which of the 9 window groups (of 5 bits each)
        // reports Y > GRAY_THRESHOLDS.
        let w = 5u32;
        let mut luminance = vec![0.0f32; (w * w) as usize];
        luminance[(1 * w + 3) as usize] = 1.0; // (row=1, col=3)
        let maps = FeatureMaps {
            width: w,
            height: w,
            luminance,
            green: vec![0.0; (w * w) as usize],
            edge: vec![false; (w * w) as usize],
        };

        let bits = cell_bits(&maps, 2, 2, 1);
        assert_eq!(bits.len(), 47); // (2*1+1)^2 * 5 + 2

        // Window scan order is dc in -1..=1 outer, dr in -1..=1 inner
        // (matches GroundTM.jl's `for dc in -radius:radius, dr in
        // -radius:radius`). (dc=+1, dr=-1) is the 8th of 9 groups
        // (0-indexed group 7): dc=-1 -> groups 0-2, dc=0 -> groups 3-5,
        // dc=+1 -> groups 6-8; within dc=+1, dr=-1 is the first (group 6).
        let group6 = 6 * 5;
        assert!(bits[group6]); // Y=1.0 > 0.28
        assert!(bits[group6 + 1]); // Y=1.0 > 0.40
        assert!(bits[group6 + 2]); // Y=1.0 > 0.50

        // Every other group must be all-low (luminance 0.0 everywhere else).
        for g in 0..9 {
            if g == 6 {
                continue;
            }
            let base = g * 5;
            assert!(!bits[base], "group {g} should not see the hot pixel");
        }
    }

    #[test]
    fn packs_bits_lsb_first_matching_elixir_pack_bits() {
        assert_eq!(pack_bits(&[true, true]), vec![3u64]);
        assert_eq!(pack_bits(&[true, false]), vec![1u64]);
        let mut sixty_five = vec![false; 64];
        sixty_five.push(true);
        assert_eq!(pack_bits(&sixty_five), vec![0u64, 1u64]);
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test window::`
Expected: panics at `unimplemented!()`.

- [ ] **Step 3: Implement `cell_bits` and `pack_bits`**

```rust
const GRAY_THRESHOLDS: [f32; 3] = [0.28, 0.40, 0.50];
const GREEN_THRESHOLD: f32 = 0.05;

pub fn cell_bits(maps: &FeatureMaps, col: u32, row: u32, radius: u32) -> Vec<bool> {
    let r = radius as i64;
    let w = maps.width as i64;
    let h = maps.height as i64;
    let mut bits = Vec::with_capacity(((2 * radius + 1).pow(2) * 5 + 2) as usize);

    for dc in -r..=r {
        for dr in -r..=r {
            let cc = (col as i64 + dc).clamp(0, w - 1) as u32;
            let rr = (row as i64 + dr).clamp(0, h - 1) as u32;
            let idx = (rr * maps.width + cc) as usize;
            let y = maps.luminance[idx];
            bits.push(y > GRAY_THRESHOLDS[0]);
            bits.push(y > GRAY_THRESHOLDS[1]);
            bits.push(y > GRAY_THRESHOLDS[2]);
            bits.push(maps.green[idx] > GREEN_THRESHOLD);
            bits.push(maps.edge[idx]);
        }
    }

    bits.push((row as f32) > 0.65 * maps.height as f32);
    bits.push((row as f32) > 0.80 * maps.height as f32);
    bits
}

pub fn pack_bits(bits: &[bool]) -> Vec<u64> {
    bits.chunks(64)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u64, |acc, (i, &b)| if b { acc | (1 << i) } else { acc })
        })
        .collect()
}
```

- [ ] **Step 4: Run to confirm it passes**

Run: `cargo test window::`
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/src/window.rs native/tsetlin_nif/src/lib.rs
git commit -m "tsetlin_nif: per-cell windowed features and LSB-first bit packing"
```

---

### Task 5: `classify_frame_nif` wiring, Elixir wrapper, error handling

**Files:**
- Modify: `native/tsetlin_nif/src/lib.rs` (add `classify_frame_nif`)
- Modify: `lib/tsetlin_runner/native.ex` (add `classify_frame_nif` stub)
- Modify: `lib/tsetlin_runner.ex` (add `classify_frame/4`)
- Test: `test/tsetlin_runner_test.exs`

**Interfaces:**
- Consumes: `jpeg::decode` (Task 1), `resize::resize_box` (Task 2), `features::compute_feature_maps` (Task 3), `window::cell_bits`/`window::pack_bits` (Task 4), `tsetlin::predict` (existing).
- Produces: `TsetlinRunner.classify_frame(model, jpeg, out_w, out_h, radius) :: {:ok, [integer()]} | {:error, atom()}` — the spec's public interface. Not consumed by any later task in this plan (sub-project 2 is a separate plan).

- [ ] **Step 1: Write the failing Elixir tests**

Add to `test/tsetlin_runner_test.exs`, inside the existing
`TsetlinRunnerTest` module (after the `load/1 and predict/2` describe
block):

```elixir
  describe "classify_frame/4" do
    # A 7-bit model (radius=0: (2*0+1)^2*5+2 = 7), one clause per polarity,
    # so it always predicts class 1 regardless of input -- this task only
    # needs a structurally valid model, not a semantically meaningful one
    # (numeric correctness against Julia is a separate task).
    defp tiny_7bit_model_bytes do
      <<
        "TSTM",
        1::little-32,
        0::8,
        7::little-32,
        1::little-32,
        2::little-32,
        1::little-32,
        1::little-64,
        1::little-64,
        0::little-64,
        0::little-64,
        0::little-64,
        0::little-64
      >>
    end

    defp with_tiny_7bit_model(fun) do
      path =
        Path.join(System.tmp_dir!(), "tsetlin_runner_7bit_#{System.unique_integer([:positive])}.tmbin")

      File.write!(path, tiny_7bit_model_bytes())

      try do
        {:ok, model} = TsetlinRunner.load(path)
        fun.(model)
      after
        File.rm(path)
      end
    end

    @fixture_jpeg File.read!(
                     Path.join([
                       __DIR__,
                       "..",
                       "native/tsetlin_nif/tests/fixtures/tiny_solid.jpg"
                     ])
                   )

    test "returns a fully-classified out_w*out_h grid for a valid frame" do
      with_tiny_7bit_model(fn model ->
        assert {:ok, grid} = TsetlinRunner.classify_frame(model, @fixture_jpeg, 2, 2, 0)
        assert length(grid) == 4
        assert Enum.all?(grid, &(&1 in [0, 1]))
      end)
    end

    test "returns invalid_jpeg for malformed frame bytes" do
      with_tiny_7bit_model(fn model ->
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 2, 2, 0) ==
                 {:error, :invalid_jpeg}
      end)
    end

    test "returns invalid_dimensions for out_w=0 without attempting to decode" do
      with_tiny_7bit_model(fn model ->
        # Deliberately-invalid JPEG bytes: if dimensions were checked AFTER
        # decode, this would fail with :invalid_jpeg instead.
        assert TsetlinRunner.classify_frame(model, <<0, 1, 2, 3>>, 0, 2, 0) ==
                 {:error, :invalid_dimensions}
      end)
    end

    test "returns bit_length_mismatch when radius doesn't match the model's clause_size" do
      with_tiny_7bit_model(fn model ->
        # This model's clause_size is 7 (radius=0). radius=1 -> 47 bits.
        assert TsetlinRunner.classify_frame(model, @fixture_jpeg, 2, 2, 1) ==
                 {:error, :bit_length_mismatch}
      end)
    end
  end
```

- [ ] **Step 2: Run to confirm it fails**

Run: `mix test test/tsetlin_runner_test.exs`
Expected: `UndefinedFunctionError` for `TsetlinRunner.classify_frame/5` (or
similar — the function doesn't exist yet).

- [ ] **Step 3: Implement `classify_frame_nif`**

Add to `native/tsetlin_nif/src/lib.rs`, alongside the existing
`predict_nif`:

```rust
mod jpeg;
mod resize;
mod features;
mod window;

#[rustler::nif]
fn classify_frame_nif(
    resource: ResourceArc<ModelResource>,
    jpeg_bytes: Binary,
    out_w: u32,
    out_h: u32,
    radius: u32,
) -> Result<Vec<i64>, Atom> {
    if out_w == 0 || out_h == 0 {
        return Err(atoms::invalid_dimensions());
    }

    let decoded = jpeg::decode(jpeg_bytes.as_slice()).map_err(|_| atoms::invalid_jpeg())?;
    let resized = resize::resize_box(&decoded, out_w, out_h);
    let maps = features::compute_feature_maps(&resized);

    let model = &resource.0;
    let expected_len = ((2 * radius + 1) as usize).pow(2) * 5 + 2;
    let expected_chunks = model.chunks_size as usize;
    if expected_len.div_ceil(64) != expected_chunks {
        return Err(atoms::bit_length_mismatch());
    }

    let mut grid = Vec::with_capacity((out_w * out_h) as usize);
    for row in 0..out_h {
        for col in 0..out_w {
            let bits = window::cell_bits(&maps, col, row, radius);
            let chunks = window::pack_bits(&bits);
            grid.push(tsetlin::predict(model, &chunks));
        }
    }
    Ok(grid)
}
```

Add two new atoms to the existing `mod atoms { rustler::atoms! { ... } }`
block: `invalid_jpeg` and `invalid_dimensions` (alongside the existing
`invalid_format`, `io_error`, `bit_length_mismatch`).

Add to `native.ex`:

```elixir
  def classify_frame_nif(_resource, _jpeg, _out_w, _out_h, _radius),
    do: :erlang.nif_error(:nif_not_loaded)
```

Add to `tsetlin_runner.ex`:

```elixir
  @doc """
  Classifies every cell of an `out_w`x`out_h` grid from a raw JPEG frame in
  one call: JPEG decode, box-average resize, feature-map computation, and
  per-cell `predict/2` -- replicating `tsetlin_world`'s Julia pipeline
  exactly (see docs/superpowers/specs/2026-09-28-ground-vision-nif-design.md).

  Returns a flat, row-major list (`index = row * out_w + col`, 0-indexed)
  of the loaded model's own class labels -- this is a different order
  than `tsetlin_world`'s column-major Julia grids.

  `radius` must match whatever radius the loaded model was trained with
  (i.e. `ground_feature_len(radius)` must equal the model's `clause_size`),
  or this returns `{:error, :bit_length_mismatch}`.
  """
  @spec classify_frame(reference(), binary(), pos_integer(), pos_integer(), non_neg_integer()) ::
          {:ok, [integer()]} | {:error, atom()}
  def classify_frame(model, jpeg, out_w, out_h, radius)
      when is_reference(model) and is_binary(jpeg) and
             is_integer(out_w) and out_w > 0 and
             is_integer(out_h) and out_h > 0 and
             is_integer(radius) and radius >= 0 do
    Native.classify_frame_nif(model, jpeg, out_w, out_h, radius)
  end
```

- [ ] **Step 4: Run to confirm it passes**

Run: `mix test test/tsetlin_runner_test.exs`
Expected: all `classify_frame/4` tests pass, and the full suite
(`mix test`) still passes.

- [ ] **Step 5: Commit**

```bash
git add native/tsetlin_nif/src/lib.rs lib/tsetlin_runner/native.ex \
  lib/tsetlin_runner.ex test/tsetlin_runner_test.exs
git commit -m "tsetlin_runner: classify_frame/4 -- JPEG to classification grid in one call"
```

---

### Task 6: Cross-language parity fixture and validation

**Files:**
- Create (in `robomow_poncho/tsetlin_world`, then copied): a synthetic-image `.tmbin` + PPM + expected-grid fixture
- Create: `native/tsetlin_nif/tests/fixtures/ground_tm_parity.jpg`
- Create: `native/tsetlin_nif/tests/fixtures/ground_tm_parity.tmbin`
- Create: `native/tsetlin_nif/tests/fixtures/ground_tm_parity_expected.txt`
- Modify: `native/tsetlin_nif/src/lib.rs` (add the parity test)
- Test: `test/tsetlin_runner_test.exs` (Elixir integration test using the same fixtures)

**Interfaces:**
- Consumes: `classify_frame_nif` (Task 5) as a whole; this task adds no new production code, only fixtures and tests.

- [ ] **Step 1: Generate the fixtures from the real Julia pipeline**

From `/home/xabi/work/elixir/robomow_poncho/tsetlin_world`:

```bash
devenv shell -- julia --project=. -e '
using TsetlinWorld
using Random

cfg = GroundConfig(; img_size=16, n_frames=10, epochs=5, n_eval_frames=2,
    clauses=20, T=10, S=20, L=40, LF=10, checkpoint_path=nothing)
result = train_ground(cfg; rng=MersenneTwister(0))

W, H = 320, 240
photo = Array{Float32}(undef, W, H, 3)
for c in 1:W, r in 1:H
    if r < H ÷ 2
        photo[c, r, :] = [0.34f0, 0.47f0, 0.85f0] # sky
    else
        photo[c, r, :] = [0.32f0, 0.30f0, 0.12f0] # dry grass
    end
end

TsetlinWorld.write_ppm("/tmp/ground_tm_parity.ppm", photo)
TsetlinWorld.export_tm(result.tm, "/tmp/ground_tm_parity.tmbin")

out_w, out_h, radius = 32, 24, 8
small = TsetlinWorld._resize_box(photo, out_w, out_h)
maps = TsetlinWorld.ground_feature_maps(small)
open("/tmp/ground_tm_parity_expected.txt", "w") do io
    println(io, "$out_w $out_h $radius")
    for row in 1:out_h, col in 1:out_w
        bits = TsetlinWorld.ground_features(small, col, row; maps, radius)
        label = predict(result.tm, TMInput(bits))
        println(io, "$(col-1) $(row-1) $label")
    end
end
'
```

Convert the PPM to a JPEG (quality=100, no subsampling, to minimize
lossy drift at the sky/grass boundary — the boundary band is still
excluded from the exact-match assertion in Step 2, see the comment
there):

```bash
devenv shell -- uv run --with pillow python3 -c "
from PIL import Image
im = Image.open('/tmp/ground_tm_parity.ppm')
im.save('/tmp/ground_tm_parity.jpg', quality=100, subsampling=0)
"
```

Copy all three fixtures into `tsetlin_runner`:

```bash
cp /tmp/ground_tm_parity.jpg /tmp/ground_tm_parity.tmbin /tmp/ground_tm_parity_expected.txt \
  /home/xabi/work/elixir/tsetlin_runner/native/tsetlin_nif/tests/fixtures/
```

- [ ] **Step 2: Write the failing Rust parity test**

Add to `native/tsetlin_nif/src/lib.rs`, in a new `#[cfg(test)] mod parity_test` block (or extend the existing test module):

```rust
#[cfg(test)]
mod parity_tests {
    use super::*;
    use std::fs;

    #[test]
    fn matches_julia_ground_tm_pipeline() {
        let jpeg = fs::read("tests/fixtures/ground_tm_parity.jpg").unwrap();
        let model_bytes = fs::read("tests/fixtures/ground_tm_parity.tmbin").unwrap();
        let model = crate::format::parse(&model_bytes).expect("fixture model should parse");

        let expected_text = fs::read_to_string("tests/fixtures/ground_tm_parity_expected.txt").unwrap();
        let mut lines = expected_text.lines();
        let header: Vec<u32> = lines.next().unwrap().split(' ').map(|s| s.parse().unwrap()).collect();
        let (out_w, out_h, radius) = (header[0], header[1], header[2]);

        let decoded = crate::jpeg::decode(&jpeg).unwrap();
        let resized = crate::resize::resize_box(&decoded, out_w, out_h);
        let maps = crate::features::compute_feature_maps(&resized);

        // The sky/grass boundary (row = out_h/2 in the source, here row 12
        // of 24) sits inside a JPEG 8x8 block even at quality=100/no
        // subsampling -- a few pixels right at the transition can shift by
        // a rounding amount after the DCT round-trip, which can flip a
        // resize cell that averages both colors right at the 50/50 mark.
        // That is a property of JPEG, not this pipeline's math, so rows
        // within 1 of the transition are excluded from the exact-match
        // assertion below (still checked for being a valid label, not
        // skipped entirely).
        let boundary_rows = [(out_h / 2) - 1, out_h / 2];

        let mut mismatches = Vec::new();
        for line in lines {
            let parts: Vec<i64> = line.split(' ').map(|s| s.parse().unwrap()).collect();
            let (col, row, expected_label) = (parts[0] as u32, parts[1] as u32, parts[2]);

            let bits = crate::window::cell_bits(&maps, col, row, radius);
            let chunks = crate::window::pack_bits(&bits);
            let actual_label = crate::tsetlin::predict(&model, &chunks);

            assert!(actual_label == 1 || actual_label == 2, "invalid label at ({col},{row})");

            if boundary_rows.contains(&row) {
                continue;
            }
            if actual_label != expected_label {
                mismatches.push((col, row, expected_label, actual_label));
            }
        }

        assert!(mismatches.is_empty(), "parity mismatches: {mismatches:?}");
    }
}
```

- [ ] **Step 3: Run to confirm it fails**

Run: `cargo test parity_tests::`
Expected: fails (either a compile error if `crate::format::parse` isn't
`pub` yet — make it `pub(crate)` or `pub` if needed — or an assertion
failure if the pipeline has a bug).

- [ ] **Step 4: Fix whatever the failure reveals**

If it's a real mismatch, use `superpowers:systematic-debugging`: the
Review Focus items in this plan's header (bit order, radius=0, upsample)
are the most likely root causes if Tasks 1-4's own unit tests still pass
in isolation but this integration test fails. Do not weaken the
assertion (e.g. widening the boundary-row exclusion) without first
confirming via manual inspection that the mismatch is genuinely a JPEG
boundary artifact and not a real pipeline bug.

- [ ] **Step 5: Run to confirm it passes**

Run: `cargo test` (full suite)
Expected: all tests pass, including `parity_tests::matches_julia_ground_tm_pipeline`.

- [ ] **Step 6: Write the Elixir integration test**

Add to `test/tsetlin_runner_test.exs`:

```elixir
  describe "classify_frame/4 matches the Julia ground_tm pipeline" do
    test "spot-checks known cells against the Julia-computed fixture" do
      jpeg =
        File.read!(
          Path.join([__DIR__, "..", "native/tsetlin_nif/tests/fixtures/ground_tm_parity.jpg"])
        )

      model_path =
        Path.join([__DIR__, "..", "native/tsetlin_nif/tests/fixtures/ground_tm_parity.tmbin"])

      {:ok, model} = TsetlinRunner.load(model_path)

      assert {:ok, grid} = TsetlinRunner.classify_frame(model, jpeg, 32, 24, 8)
      assert length(grid) == 32 * 24

      # Spot-check a cell well inside the sky region (row 2 of 24, far from
      # the boundary) and one well inside the grass region (row 20 of 24).
      sky_idx = 2 * 32 + 16
      grass_idx = 20 * 32 + 16
      assert Enum.at(grid, sky_idx) in [1, 2]
      assert Enum.at(grid, grass_idx) in [1, 2]
      assert Enum.at(grid, sky_idx) != Enum.at(grid, grass_idx)
    end
  end
```

- [ ] **Step 7: Run to confirm it passes**

Run: `mix test`
Expected: full Elixir suite passes.

- [ ] **Step 8: Commit**

```bash
git add native/tsetlin_nif/tests/fixtures/ground_tm_parity.jpg \
  native/tsetlin_nif/tests/fixtures/ground_tm_parity.tmbin \
  native/tsetlin_nif/tests/fixtures/ground_tm_parity_expected.txt \
  native/tsetlin_nif/src/lib.rs test/tsetlin_runner_test.exs
git commit -m "tsetlin_nif: cross-language parity fixture and test against GroundTM.jl"
```

---
