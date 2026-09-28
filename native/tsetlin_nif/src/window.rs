use crate::features::FeatureMaps;

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
