use std::convert::TryInto;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Bool,
    General,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseBlock {
    pub positive_included_literals: Vec<u64>,
    pub positive_included_literals_inverted: Vec<u64>,
    pub negative_included_literals: Vec<u64>,
    pub negative_included_literals_inverted: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub kind: Kind,
    pub clause_size: u32,
    pub chunks_size: u32,
    pub classes_num: u32,
    pub ta_clauses: u32,
    pub lf: i64,
    pub classes: Vec<i64>,
    pub blocks: Vec<ClauseBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatError {
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidKind(u8),
    Truncated,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.pos + n > self.data.len() {
            return Err(FormatError::Truncated);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, FormatError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn read_matrix(cur: &mut Cursor, len: usize) -> Result<Vec<u64>, FormatError> {
    let mut v = Vec::new();
    for _ in 0..len {
        v.push(cur.u64()?);
    }
    Ok(v)
}

pub fn parse(bytes: &[u8]) -> Result<Model, FormatError> {
    let mut cur = Cursor::new(bytes);

    let magic = cur.take(4)?;
    if magic != b"TSTM" {
        return Err(FormatError::InvalidMagic);
    }

    let version = cur.u32()?;
    if version != 1 {
        return Err(FormatError::UnsupportedVersion(version));
    }

    let kind = match cur.u8()? {
        0 => Kind::Bool,
        1 => Kind::General,
        other => return Err(FormatError::InvalidKind(other)),
    };

    let clause_size = cur.u32()?;
    let chunks_size = cur.u32()?;
    let classes_num = cur.u32()?;
    let ta_clauses = cur.u32()?;
    let lf = cur.i64()?;

    let mut classes = Vec::new();
    for _ in 0..classes_num {
        classes.push(cur.i64()?);
    }

    let block_count = match kind {
        Kind::Bool => 1,
        Kind::General => classes_num as usize,
    };
    let matrix_len = (chunks_size as usize) * (ta_clauses as usize);

    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        blocks.push(ClauseBlock {
            positive_included_literals: read_matrix(&mut cur, matrix_len)?,
            positive_included_literals_inverted: read_matrix(&mut cur, matrix_len)?,
            negative_included_literals: read_matrix(&mut cur, matrix_len)?,
            negative_included_literals_inverted: read_matrix(&mut cur, matrix_len)?,
        });
    }

    Ok(Model {
        kind,
        clause_size,
        chunks_size,
        classes_num,
        ta_clauses,
        lf,
        classes,
        blocks,
    })
}

/// The tiny 2-bit, 2-class fixture also used by `tsetlin.rs`'s tests
/// and by the Elixir integration test:
/// - class 1 fires on input bits (true, true)
/// - class 2 fires on input bits (false, false)
/// - LF = 2, one clause per polarity per class, no negative literals.
#[cfg(test)]
pub(crate) fn tiny_general_model_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TSTM");
    bytes.extend_from_slice(&1u32.to_le_bytes()); // version
    bytes.push(1); // kind = General
    bytes.extend_from_slice(&2u32.to_le_bytes()); // clause_size
    bytes.extend_from_slice(&1u32.to_le_bytes()); // chunks_size
    bytes.extend_from_slice(&2u32.to_le_bytes()); // classes_num
    bytes.extend_from_slice(&1u32.to_le_bytes()); // ta_clauses
    bytes.extend_from_slice(&2i64.to_le_bytes()); // lf
    bytes.extend_from_slice(&1i64.to_le_bytes()); // classes[0] = 1
    bytes.extend_from_slice(&2i64.to_le_bytes()); // classes[1] = 2
    // class 1 block: positive literals = 0b11, rest 0
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    // class 2 block: positive_inverted literals = 0b11, rest 0
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_tiny_general_fixture() {
        let model = parse(&tiny_general_model_bytes()).unwrap();
        assert_eq!(model.kind, Kind::General);
        assert_eq!(model.clause_size, 2);
        assert_eq!(model.chunks_size, 1);
        assert_eq!(model.classes_num, 2);
        assert_eq!(model.ta_clauses, 1);
        assert_eq!(model.lf, 2);
        assert_eq!(model.classes, vec![1, 2]);
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(model.blocks[0].positive_included_literals, vec![3]);
        assert_eq!(model.blocks[1].positive_included_literals_inverted, vec![3]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = tiny_general_model_bytes();
        bytes[0] = b'X';
        assert_eq!(parse(&bytes), Err(FormatError::InvalidMagic));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = tiny_general_model_bytes();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(parse(&bytes), Err(FormatError::UnsupportedVersion(2)));
    }

    #[test]
    fn rejects_truncated_data() {
        let bytes = tiny_general_model_bytes();
        let truncated = &bytes[..bytes.len() - 4];
        assert_eq!(parse(truncated), Err(FormatError::Truncated));
    }

    #[test]
    fn rejects_oversized_chunks_size_without_panic() {
        let mut bytes = tiny_general_model_bytes();
        // Overwrite chunks_size (bytes 13..17) with 0xFFFFFFFF to trigger allocation overflow
        bytes[13..17].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        assert_eq!(parse(&bytes), Err(FormatError::Truncated));
    }
}
