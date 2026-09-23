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

#[allow(unused_must_use, non_local_definitions)]
fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(ModelResource, env);
    true
}

rustler::init!("Elixir.TsetlinRunner.Native", load = on_load);
