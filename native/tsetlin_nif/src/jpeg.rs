use std::panic;

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    Malformed,
}

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
        let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
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
}
