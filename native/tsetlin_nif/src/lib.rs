pub mod format;
pub mod features;
pub mod jpeg;
pub mod resize;
pub mod tsetlin;
pub mod window;

use rustler::{Atom, Binary, Env, ResourceArc, Term};

mod atoms {
    rustler::atoms! {
        invalid_format,
        io_error,
        bit_length_mismatch,
        invalid_jpeg,
        invalid_dimensions,
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

// Not DirtyCpu-scheduled: measured at ~300us/call on a Pi Zero (radius=8,
// clause_size=1447), well under the ~1ms guideline for a normal NIF.
// DirtyCpu was the original (unmeasured) choice; on real hardware it added
// ~75% pure scheduler hand-off overhead on top of this call's own cost --
// see docs/superpowers/plans/2026-09-25-ground-tm-rectangular-patches.md's
// 2026-09-28 follow-up discussion for the benchmark that found this.
#[rustler::nif]
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
    // Compare the exact bit length against the model's own clause_size, not
    // against chunks_size (chunks round up to 64-bit boundaries, so e.g.
    // 7 bits and 47 bits both round to 1 chunk -- a chunks-only comparison
    // cannot tell them apart).
    if expected_len != model.clause_size as usize {
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

#[rustler::resource_impl]
impl rustler::Resource for ModelResource {}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<ModelResource>().is_ok()
}

rustler::init!("Elixir.TsetlinRunner.Native", load = on_load);

#[cfg(test)]
mod parity_tests {
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
