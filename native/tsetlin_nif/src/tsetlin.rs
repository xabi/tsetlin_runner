use crate::format::{ClauseBlock, Kind, Model};

/// Mirrors `Tsetlin.jl`'s non-indexed `check_clause`: for each chunk,
/// `val` bits mark literal *mismatches* against the input, and the
/// clause's score is `max(0, LF - mismatch_count)`.
fn check_clause(chunks: &[u64], literals: &[u64], literals_inverted: &[u64], lf: i64) -> i64 {
    let mut c: i64 = 0;
    for n in 0..chunks.len() {
        let chunk = chunks[n];
        let lit = literals[n];
        let lit_inv = literals_inverted[n];
        let val = ((lit ^ lit_inv) & chunk) ^ lit;
        c += val.count_ones() as i64;
    }
    (lf - c).max(0)
}

/// Mirrors `Tsetlin.jl`'s `vote`: sums `check_clause` (not a plain 0/1)
/// across every clause column of a block, separately for the positive
/// and negative polarity.
fn vote(chunks: &[u64], block: &ClauseBlock, chunks_size: usize, ta_clauses: usize, lf: i64) -> (i64, i64) {
    let mut pos = 0i64;
    let mut neg = 0i64;
    for i in 0..ta_clauses {
        let start = i * chunks_size;
        let end = start + chunks_size;
        pos += check_clause(
            chunks,
            &block.positive_included_literals[start..end],
            &block.positive_included_literals_inverted[start..end],
            lf,
        );
        neg += check_clause(
            chunks,
            &block.negative_included_literals[start..end],
            &block.negative_included_literals_inverted[start..end],
            lf,
        );
    }
    (pos, neg)
}

/// Mirrors `Tsetlin.jl`'s two `predict` methods: the `Bool` kind compares
/// a single block's pos/neg vote; the general kind picks the class with
/// the highest `pos - neg` margin, keeping the *first* class seen on a
/// tie (`v > best_vote` is strict, matching `Tsetlin.jl`'s `is_better`).
pub fn predict(model: &Model, chunks: &[u64]) -> i64 {
    let chunks_size = model.chunks_size as usize;
    let ta_clauses = model.ta_clauses as usize;

    match model.kind {
        Kind::Bool => {
            let (pos, neg) = vote(chunks, &model.blocks[0], chunks_size, ta_clauses, model.lf);
            if pos > neg {
                model.classes[0]
            } else {
                model.classes[1]
            }
        }
        Kind::General => {
            let mut best_vote = i64::MIN;
            let mut best_class = model.classes[0];
            for (i, block) in model.blocks.iter().enumerate() {
                let (pos, neg) = vote(chunks, block, chunks_size, ta_clauses, model.lf);
                let v = pos - neg;
                if v > best_vote {
                    best_vote = v;
                    best_class = model.classes[i];
                }
            }
            best_class
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format;

    #[test]
    fn check_clause_scores_by_mismatch_count_against_lf() {
        // literals = 0b11 (positive literal on both bits), lf = 2.
        assert_eq!(check_clause(&[0b11], &[0b11], &[0], 2), 2); // 0 mismatches
        assert_eq!(check_clause(&[0b01], &[0b11], &[0], 2), 1); // 1 mismatch
        assert_eq!(check_clause(&[0b00], &[0b11], &[0], 2), 0); // 2 mismatches, floored at 0
    }

    fn predict_tiny(bits: [bool; 2]) -> i64 {
        let bytes = format::tiny_general_model_bytes();
        let model = format::parse(&bytes).unwrap();
        let chunk: u64 = (bits[0] as u64) | ((bits[1] as u64) << 1);
        predict(&model, &[chunk])
    }

    #[test]
    fn predicts_class_2_when_both_bits_false() {
        assert_eq!(predict_tiny([false, false]), 2);
    }

    #[test]
    fn predicts_class_1_when_both_bits_true() {
        assert_eq!(predict_tiny([true, true]), 1);
    }

    #[test]
    fn breaks_ties_in_favor_of_the_first_class() {
        // (true, false) and (false, true) both score pos=1/neg=2 for
        // BOTH classes (v = -1 for each) -- class 1 (index 0) must win
        // because Tsetlin.jl's `is_better = v > best_vote` is strict.
        assert_eq!(predict_tiny([true, false]), 1);
        assert_eq!(predict_tiny([false, true]), 1);
    }

    #[test]
    fn vote_sums_check_clause_across_multiple_chunks_and_clauses() {
        // clause_size=65 -> chunks_size=2 (every real model, e.g.
        // GroundTM's ~1447 bits, needs more than one chunk and more than
        // one clause; the tiny 1-chunk/1-clause fixture above can't catch
        // a bug in the `start = i * chunks_size` column slicing or in
        // looping over more than one chunk). Two positive clauses, no
        // negative literals: clause A requires bit 0 true, clause B
        // requires bit 64 true (the sole bit of the second chunk).
        // Column-major flattening: [clauseA_chunk0, clauseA_chunk1,
        // clauseB_chunk0, clauseB_chunk1].
        let block = ClauseBlock {
            positive_included_literals: vec![0b1, 0, 0, 0b1],
            positive_included_literals_inverted: vec![0, 0, 0, 0],
            negative_included_literals: vec![0, 0, 0, 0],
            negative_included_literals_inverted: vec![0, 0, 0, 0],
        };
        let lf = 3;
        let chunks_size = 2;
        let ta_clauses = 2;

        // neg is always 6: two empty (all-zero-literal) negative clause
        // columns each score LF=3 (0 mismatches against an empty mask).
        assert_eq!(vote(&[1, 0], &block, chunks_size, ta_clauses, lf), (5, 6));
        assert_eq!(vote(&[0, 1], &block, chunks_size, ta_clauses, lf), (5, 6));
        assert_eq!(vote(&[1, 1], &block, chunks_size, ta_clauses, lf), (6, 6));
        assert_eq!(vote(&[0, 0], &block, chunks_size, ta_clauses, lf), (4, 6));
    }

    #[test]
    fn bool_kind_uses_a_single_shared_block() {
        // kind=0, clause_size=2, chunks_size=1, classes_num=2 (implicit
        // true/false), ta_clauses=1, lf=2. One block: positive literals
        // require both bits true; no negative literals.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"TSTM");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(0); // kind = Bool
        bytes.extend_from_slice(&2u32.to_le_bytes()); // clause_size
        bytes.extend_from_slice(&1u32.to_le_bytes()); // chunks_size
        bytes.extend_from_slice(&2u32.to_le_bytes()); // classes_num
        bytes.extend_from_slice(&1u32.to_le_bytes()); // ta_clauses
        bytes.extend_from_slice(&2i64.to_le_bytes()); // lf
        bytes.extend_from_slice(&1i64.to_le_bytes()); // classes[0] = true
        bytes.extend_from_slice(&0i64.to_le_bytes()); // classes[1] = false
        // single block
        bytes.extend_from_slice(&3u64.to_le_bytes()); // positive literals = 0b11
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        let model = format::parse(&bytes).unwrap();
        assert_eq!(model.blocks.len(), 1);

        // both bits true -> pos=2, neg=2 -> pos > neg is false -> false
        assert_eq!(predict(&model, &[0b11]), 0);
        // both bits false -> pos=0, neg=2 -> pos > neg is false -> false
        assert_eq!(predict(&model, &[0b00]), 0);
    }
}
