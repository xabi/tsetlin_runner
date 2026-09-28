use std::panic;

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    Malformed,
}

// Generous over the deployed 320x240 camera capture (and the resize target,
// which is always smaller than the source) -- this exists only to reject a
// "JPEG bomb" (a tiny file whose header declares huge dimensions) before
// any large buffer is allocated, not to constrain real camera frames.
pub const MAX_DIM: usize = 2048;

#[derive(Debug, PartialEq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

pub fn decode(jpeg_bytes: &[u8]) -> Result<RgbImage, DecodeError> {
    // catch_unwind is the untrusted-input boundary: a malformed/adversarial
    // camera frame must never crash the whole BEAM by panicking inside
    // third-party decoder code (see this plan's Global Constraints).
    let result = panic::catch_unwind(|| {
        let cursor = zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg_bytes);
        let options = zune_jpeg::zune_core::options::DecoderOptions::default()
            .set_max_width(MAX_DIM)
            .set_max_height(MAX_DIM);
        let mut decoder = zune_jpeg::JpegDecoder::new_with_options(cursor, options);
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

        let top = 3 * ((2 * img.width + 2) as usize); // (y=2, x=2), red block
        assert!(approx(img.data[top], 200, 0.03));
        assert!(approx(img.data[top + 1], 50, 0.03));
        assert!(approx(img.data[top + 2], 50, 0.03));

        let bottom = 3 * ((12 * img.width + 2) as usize); // (y=12, x=2), green block
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

    #[test]
    fn rejects_a_frame_wider_than_max_dim_without_allocating_it() {
        // Final review (2026-09-28): decode() had no bound on the decoded
        // image's own declared dimensions, so a small JPEG claiming e.g.
        // 16384x16384 pixels would still fully decode into a multi-GB
        // buffer before this function ever sees `out_w`/`out_h` -- a "JPEG
        // bomb" that aborts the whole BEAM on a 512MB device (not
        // catchable by catch_unwind, since an allocator abort isn't a
        // panic). Fixture is 3000x8 -- tiny on disk (JPEG compresses a
        // solid color trivially) but wider than MAX_DIM.
        const OVERSIZED: &[u8] = include_bytes!("../tests/fixtures/oversized.jpg");
        assert_eq!(decode(OVERSIZED), Err(DecodeError::Malformed));
    }
}
