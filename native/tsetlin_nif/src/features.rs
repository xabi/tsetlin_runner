use crate::jpeg::RgbImage;

#[allow(dead_code)] // consumed by Task 4 (cell_bits)
const GRAY_THRESHOLDS: [f32; 3] = [0.28, 0.40, 0.50];
#[allow(dead_code)] // consumed by Task 4 (cell_bits)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, data: Vec<f32>) -> RgbImage {
        RgbImage { width: w, height: h, data }
    }

    #[test]
    fn luminance_and_green_match_hand_computed_values() {
        // Single pixel, R=0.5 G=0.6 B=0.1.
        // Y = 0.299*0.5 + 0.587*0.6 + 0.114*0.1 = 0.5131
        // green = 0.6 - (0.5+0.1)/2 = 0.3
        let src = img(1, 1, vec![0.5, 0.6, 0.1]);
        let maps = compute_feature_maps(&src);
        assert!((maps.luminance[0] - 0.5131).abs() < 1e-4);
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
    fn edge_threshold_is_pinned_at_0_15() {
        // Final review (2026-09-28): nothing asserted on `edge` at a
        // magnitude close to EDGE_THRESHOLD -- e.g. mutating 0.15 to 1.5
        // left the whole suite green. A single nonzero pixel at the
        // bottom-right corner of the SOBEL_W window (kernel weight +1)
        // makes the center pixel's Sobel-W magnitude exactly that pixel's
        // luminance value, so 0.2 (> 0.15) and 0.1 (< 0.15) pin both sides.
        let above = {
            let mut data = vec![0.0f32; 3 * 3 * 3];
            let idx = 3 * ((2 * 3 + 2) as usize); // (row=2, col=2)
            data[idx] = 0.2;
            data[idx + 1] = 0.2;
            data[idx + 2] = 0.2;
            compute_feature_maps(&img(3, 3, data))
        };
        let below = {
            let mut data = vec![0.0f32; 3 * 3 * 3];
            let idx = 3 * ((2 * 3 + 2) as usize);
            data[idx] = 0.1;
            data[idx + 1] = 0.1;
            data[idx + 2] = 0.1;
            compute_feature_maps(&img(3, 3, data))
        };
        let center = (1 * 3 + 1) as usize;
        assert!(above.edge[center], "0.2 > 0.15 should be an edge");
        assert!(!below.edge[center], "0.1 < 0.15 should not be an edge");
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
        let _maps = compute_feature_maps(&src);

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
