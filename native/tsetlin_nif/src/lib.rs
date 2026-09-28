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

#[rustler::resource_impl]
impl rustler::Resource for ModelResource {}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<ModelResource>().is_ok()
}

rustler::init!("Elixir.TsetlinRunner.Native", load = on_load);
