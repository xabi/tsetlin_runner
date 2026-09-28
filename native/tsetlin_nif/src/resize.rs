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
