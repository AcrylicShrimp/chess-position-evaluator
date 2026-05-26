use chess::{
    BitBoard, Board, Color, EMPTY, Piece, Rank, get_bishop_moves, get_king_moves, get_knight_moves,
    get_pawn_attacks, get_rook_moves,
};
use duckdb::{AccessMode, Config, Connection, Row, params};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Seek, SeekFrom, Write};
use std::{path::Path, str::FromStr};

struct ChessEvaluationRow {
    fen: String,
    cp: i32,
}

/// Converts centipawns to win probability.
///
/// It uses the ELO formula, treating the centipawns as the direct rating difference.
pub fn centipawn_to_win_prob(cp: i32) -> f32 {
    1f32 / (1f32 + 10f32.powf(-cp as f32 / 400f32))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionRejectReason {
    WhiteKingCount,
    BlackKingCount,
    WhitePieceTotal,
    BlackPieceTotal,
    WhitePawnCount,
    BlackPawnCount,
    WhitePawnBackRank,
    BlackPawnBackRank,
    OverlappingOccupancy,
}

impl fmt::Display for PositionRejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionRejectReason::WhiteKingCount => write!(f, "white_king_count"),
            PositionRejectReason::BlackKingCount => write!(f, "black_king_count"),
            PositionRejectReason::WhitePieceTotal => write!(f, "white_piece_total"),
            PositionRejectReason::BlackPieceTotal => write!(f, "black_piece_total"),
            PositionRejectReason::WhitePawnCount => write!(f, "white_pawn_count"),
            PositionRejectReason::BlackPawnCount => write!(f, "black_pawn_count"),
            PositionRejectReason::WhitePawnBackRank => write!(f, "white_pawn_back_rank"),
            PositionRejectReason::BlackPawnBackRank => write!(f, "black_pawn_back_rank"),
            PositionRejectReason::OverlappingOccupancy => write!(f, "overlapping_occupancy"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PositionValidationStats {
    pub accepted: usize,
    pub invalid_fen: usize,
    pub white_king_count: usize,
    pub black_king_count: usize,
    pub white_piece_total: usize,
    pub black_piece_total: usize,
    pub white_pawn_count: usize,
    pub black_pawn_count: usize,
    pub white_pawn_back_rank: usize,
    pub black_pawn_back_rank: usize,
    pub overlapping_occupancy: usize,
}

impl PositionValidationStats {
    pub fn rejected(&self) -> usize {
        self.invalid_fen
            + self.white_king_count
            + self.black_king_count
            + self.white_piece_total
            + self.black_piece_total
            + self.white_pawn_count
            + self.black_pawn_count
            + self.white_pawn_back_rank
            + self.black_pawn_back_rank
            + self.overlapping_occupancy
    }

    pub fn total(&self) -> usize {
        self.accepted + self.rejected()
    }

    fn add_reject_reason(&mut self, reason: PositionRejectReason) {
        match reason {
            PositionRejectReason::WhiteKingCount => self.white_king_count += 1,
            PositionRejectReason::BlackKingCount => self.black_king_count += 1,
            PositionRejectReason::WhitePieceTotal => self.white_piece_total += 1,
            PositionRejectReason::BlackPieceTotal => self.black_piece_total += 1,
            PositionRejectReason::WhitePawnCount => self.white_pawn_count += 1,
            PositionRejectReason::BlackPawnCount => self.black_pawn_count += 1,
            PositionRejectReason::WhitePawnBackRank => self.white_pawn_back_rank += 1,
            PositionRejectReason::BlackPawnBackRank => self.black_pawn_back_rank += 1,
            PositionRejectReason::OverlappingOccupancy => self.overlapping_occupancy += 1,
        }
    }

    fn merge(&mut self, other: PositionValidationStats) {
        self.accepted += other.accepted;
        self.invalid_fen += other.invalid_fen;
        self.white_king_count += other.white_king_count;
        self.black_king_count += other.black_king_count;
        self.white_piece_total += other.white_piece_total;
        self.black_piece_total += other.black_piece_total;
        self.white_pawn_count += other.white_pawn_count;
        self.black_pawn_count += other.black_pawn_count;
        self.white_pawn_back_rank += other.white_pawn_back_rank;
        self.black_pawn_back_rank += other.black_pawn_back_rank;
        self.overlapping_occupancy += other.overlapping_occupancy;
    }
}

#[derive(Clone, Debug)]
pub struct InvalidFenExample {
    pub reason: String,
    pub fen: String,
    pub cp: i32,
}

#[derive(Clone, Debug, Default)]
pub struct WriteChessEvalReport {
    pub rows_written: usize,
    pub validation: PositionValidationStats,
}

fn extract_chess_evaluation_row(row: &Row) -> Result<ChessEvaluationRow, anyhow::Error> {
    let fen = row.get::<_, String>(0)?;
    let cp = row.get::<_, i32>(1)?;
    Ok(ChessEvaluationRow { fen, cp })
}

pub async fn write_chesseval(
    duckdb_temp_path: impl AsRef<Path>,
    path: impl AsRef<Path>,
    offset: i64,
    limit: i64,
) -> Result<WriteChessEvalReport, anyhow::Error> {
    const CHUNK_SIZE: usize = 256 * 1024;

    let duckdb_temp_path = duckdb_temp_path.as_ref();
    let path = path.as_ref();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;

    let mut report = WriteChessEvalReport::default();

    // Write a placeholder length, stream the rows, then seek back to finalize it.
    file.write_all(&0u64.to_le_bytes())?;

    for chunk_offset in (offset..offset + limit).step_by(CHUNK_SIZE) {
        let chunk_size = std::cmp::min(CHUNK_SIZE, (offset + limit - chunk_offset) as usize);
        let chunk = process_chunk(chunk_size, chunk_offset, duckdb_temp_path)?;
        report.rows_written += chunk.row_count;
        report.validation.merge(chunk.validation);
        file.write_all(&chunk.bytes)?;
    }

    file.seek(SeekFrom::Start(0))?;
    file.write_all(&(report.rows_written as u64).to_le_bytes())?;
    file.flush()?;

    Ok(report)
}

struct ProcessChunkResult {
    row_count: usize,
    bytes: Vec<u8>,
    validation: PositionValidationStats,
}

fn process_chunk(
    chunk_size: usize,
    chunk_offset: i64,
    duckdb_temp_path: impl AsRef<Path>,
) -> Result<ProcessChunkResult, anyhow::Error> {
    let conn = Connection::open_with_flags(
        duckdb_temp_path,
        Config::default().access_mode(AccessMode::ReadOnly).unwrap(),
    )?;

    let mut stmt = conn.prepare("SELECT fen, cp FROM rows OFFSET ?1 LIMIT ?2")?;
    let rows = stmt.query_and_then(
        params![chunk_offset, chunk_size],
        extract_chess_evaluation_row,
    )?;

    let mut row_count = 0;
    let mut bytes = Vec::with_capacity((169 + 4) * chunk_size);
    let mut validation = PositionValidationStats::default();

    for row in rows {
        if append_row(row?, &mut bytes, &mut validation) {
            row_count += 1;
        }
    }

    Ok(ProcessChunkResult {
        row_count,
        bytes,
        validation,
    })
}

fn append_row(
    row: ChessEvaluationRow,
    bytes: &mut Vec<u8>,
    validation: &mut PositionValidationStats,
) -> bool {
    let board = match Board::from_str(&row.fen) {
        Ok(board) => board,
        Err(_) => {
            validation.invalid_fen += 1;
            return false;
        }
    };

    if let Some(reason) = validate_standard_position(&board).err() {
        validation.add_reject_reason(reason);
        return false;
    }

    validation.accepted += 1;
    let us = board.side_to_move();
    let them = !us;

    let bitflags = compute_bitflags(&board, us, them);
    let mut player_en_passant = compute_player_en_passant(&board);
    let mut pieces = compute_pieces(&board, us, them);
    let mut heatmaps = compute_heatmaps(&board, us, them);

    if us == Color::Black {
        fn flip_vertical(bitboard: BitBoard) -> BitBoard {
            let mut bytes = bitboard.0.to_ne_bytes();
            bytes.reverse();
            BitBoard(u64::from_ne_bytes(bytes))
        }

        player_en_passant = flip_vertical(player_en_passant);
        pieces = pieces.map(flip_vertical);
        heatmaps = flip_vertical_heatmap(&heatmaps);
    }

    bytes.extend(bitflags.to_le_bytes());
    bytes.extend(player_en_passant.0.to_le_bytes());

    for piece in pieces {
        bytes.extend(piece.0.to_le_bytes());
    }

    bytes.extend_from_slice(&heatmaps);

    let relative_cp = if us == Color::White { row.cp } else { -row.cp };
    let win_prob = centipawn_to_win_prob(relative_cp);
    bytes.extend(win_prob.to_le_bytes());

    true
}

pub fn validate_standard_position(board: &Board) -> Result<(), PositionRejectReason> {
    for reason in standard_position_reject_reasons(board) {
        return Err(reason);
    }

    Ok(())
}

pub fn standard_position_reject_reasons(board: &Board) -> Vec<PositionRejectReason> {
    let mut reasons = Vec::new();
    let white_occupancy = *board.color_combined(Color::White);
    let black_occupancy = *board.color_combined(Color::Black);

    if bitboard_count(white_occupancy & black_occupancy) != 0 {
        reasons.push(PositionRejectReason::OverlappingOccupancy);
    }

    validate_color_position(board, Color::White, &mut reasons);
    validate_color_position(board, Color::Black, &mut reasons);

    reasons
}

fn validate_color_position(board: &Board, color: Color, reasons: &mut Vec<PositionRejectReason>) {
    let occupancy = *board.color_combined(color);
    let kings = *board.pieces(Piece::King) & occupancy;
    let pawns = *board.pieces(Piece::Pawn) & occupancy;
    let piece_count = bitboard_count(occupancy);
    let king_count = bitboard_count(kings);
    let pawn_count = bitboard_count(pawns);
    let has_back_rank_pawn = pawns
        .into_iter()
        .any(|square| matches!(square.get_rank(), Rank::First | Rank::Eighth));

    match color {
        Color::White => {
            if king_count != 1 {
                reasons.push(PositionRejectReason::WhiteKingCount);
            }
            if piece_count > 16 {
                reasons.push(PositionRejectReason::WhitePieceTotal);
            }
            if pawn_count > 8 {
                reasons.push(PositionRejectReason::WhitePawnCount);
            }
            if has_back_rank_pawn {
                reasons.push(PositionRejectReason::WhitePawnBackRank);
            }
        }
        Color::Black => {
            if king_count != 1 {
                reasons.push(PositionRejectReason::BlackKingCount);
            }
            if piece_count > 16 {
                reasons.push(PositionRejectReason::BlackPieceTotal);
            }
            if pawn_count > 8 {
                reasons.push(PositionRejectReason::BlackPawnCount);
            }
            if has_back_rank_pawn {
                reasons.push(PositionRejectReason::BlackPawnBackRank);
            }
        }
    }
}

fn bitboard_count(bitboard: BitBoard) -> u32 {
    bitboard.0.count_ones()
}

pub fn diagnose_source_rows(
    duckdb_temp_path: impl AsRef<Path>,
    offset: i64,
    limit: i64,
    example_limit: usize,
) -> Result<(PositionValidationStats, Vec<InvalidFenExample>), anyhow::Error> {
    let conn = Connection::open_with_flags(
        duckdb_temp_path,
        Config::default().access_mode(AccessMode::ReadOnly).unwrap(),
    )?;

    let mut stmt = conn.prepare("SELECT fen, cp FROM rows OFFSET ?1 LIMIT ?2")?;
    let rows = stmt.query_and_then(params![offset, limit], extract_chess_evaluation_row)?;
    let mut validation = PositionValidationStats::default();
    let mut examples = Vec::new();
    let mut example_counts_by_reason = BTreeMap::<String, usize>::new();

    for row in rows {
        let row = row?;
        match Board::from_str(&row.fen) {
            Ok(board) => {
                let reasons = standard_position_reject_reasons(&board);
                if reasons.is_empty() {
                    validation.accepted += 1;
                } else {
                    validation.add_reject_reason(reasons[0]);
                    let reason = reasons
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    if should_store_example(&mut example_counts_by_reason, &reason, example_limit) {
                        examples.push(InvalidFenExample {
                            reason,
                            fen: row.fen,
                            cp: row.cp,
                        });
                    }
                }
            }
            Err(_) => {
                validation.invalid_fen += 1;
                if should_store_example(&mut example_counts_by_reason, "invalid_fen", example_limit)
                {
                    examples.push(InvalidFenExample {
                        reason: "invalid_fen".to_string(),
                        fen: row.fen,
                        cp: row.cp,
                    });
                }
            }
        }
    }

    Ok((validation, examples))
}

fn should_store_example(
    counts: &mut BTreeMap<String, usize>,
    reason: &str,
    limit_per_reason: usize,
) -> bool {
    if limit_per_reason == 0 {
        return false;
    }

    let count = counts.entry(reason.to_string()).or_insert(0);
    if *count >= limit_per_reason {
        return false;
    }

    *count += 1;
    true
}

fn compute_bitflags(board: &Board, us: Color, them: Color) -> u8 {
    let mut bitflags = 0;

    if us == Color::Black {
        bitflags |= 1 << 0;
    }

    let our_castle_rights = board.castle_rights(us);

    if our_castle_rights.has_kingside() {
        bitflags |= 1 << 1;
    }

    if our_castle_rights.has_queenside() {
        bitflags |= 1 << 2;
    }

    let their_castle_rights = board.castle_rights(them);

    if their_castle_rights.has_kingside() {
        bitflags |= 1 << 3;
    }

    if their_castle_rights.has_queenside() {
        bitflags |= 1 << 4;
    }

    bitflags
}

fn compute_player_en_passant(board: &Board) -> BitBoard {
    match board.en_passant() {
        Some(square) => BitBoard::from_square(square),
        None => EMPTY,
    }
}

fn compute_pieces(board: &Board, us: Color, them: Color) -> [BitBoard; 12] {
    [
        *board.pieces(Piece::Pawn) & *board.color_combined(us),
        *board.pieces(Piece::Knight) & *board.color_combined(us),
        *board.pieces(Piece::Bishop) & *board.color_combined(us),
        *board.pieces(Piece::Rook) & *board.color_combined(us),
        *board.pieces(Piece::Queen) & *board.color_combined(us),
        *board.pieces(Piece::King) & *board.color_combined(us),
        *board.pieces(Piece::Pawn) & *board.color_combined(them),
        *board.pieces(Piece::Knight) & *board.color_combined(them),
        *board.pieces(Piece::Bishop) & *board.color_combined(them),
        *board.pieces(Piece::Rook) & *board.color_combined(them),
        *board.pieces(Piece::Queen) & *board.color_combined(them),
        *board.pieces(Piece::King) & *board.color_combined(them),
    ]
}

fn compute_heatmaps(board: &Board, us: Color, them: Color) -> [u8; 64] {
    let our_attacks = compute_color_attacks(board, us);
    let their_attacks = compute_color_attacks(board, them);

    let mut packed = [0u8; 64];
    for i in 0..64 {
        let ours = our_attacks[i].min(15);
        let theirs = their_attacks[i].min(15);
        packed[i] = (theirs << 4) | ours;
    }

    packed
}

fn compute_color_attacks(board: &Board, color: Color) -> [u8; 64] {
    let occupancy = *board.combined();
    let color_bb = *board.color_combined(color);
    let mut counts = [0u8; 64];

    let mut add_attacks = |attacks: BitBoard| {
        for sq in attacks {
            let idx = sq.to_index();
            let next = counts[idx].saturating_add(1);
            counts[idx] = next.min(15);
        }
    };

    let pawns = *board.pieces(Piece::Pawn) & color_bb;
    for from in pawns {
        add_attacks(get_pawn_attacks(from, color, !EMPTY));
    }

    let knights = *board.pieces(Piece::Knight) & color_bb;
    for from in knights {
        add_attacks(get_knight_moves(from));
    }

    let bishops = *board.pieces(Piece::Bishop) & color_bb;
    for from in bishops {
        add_attacks(get_bishop_moves(from, occupancy));
    }

    let rooks = *board.pieces(Piece::Rook) & color_bb;
    for from in rooks {
        add_attacks(get_rook_moves(from, occupancy));
    }

    let queens = *board.pieces(Piece::Queen) & color_bb;
    for from in queens {
        add_attacks(get_bishop_moves(from, occupancy) | get_rook_moves(from, occupancy));
    }

    let kings = *board.pieces(Piece::King) & color_bb;
    for from in kings {
        add_attacks(get_king_moves(from));
    }

    counts
}

fn flip_vertical_heatmap(map: &[u8; 64]) -> [u8; 64] {
    let mut flipped = [0u8; 64];
    for rank in 0..8 {
        let src = rank * 8;
        let dst = (7 - rank) * 8;
        flipped[dst..dst + 8].copy_from_slice(&map[src..src + 8]);
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        Board::from_str(fen).expect("test FEN should parse")
    }

    #[test]
    fn accepts_standard_minimal_position() {
        let board = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");

        assert_eq!(validate_standard_position(&board), Ok(()));
    }

    #[test]
    fn parser_rejects_missing_king_before_validator() {
        let parsed = Board::from_str("8/8/8/8/8/8/8/4K3 w - - 0 1");

        assert!(parsed.is_err());
    }

    #[test]
    fn parser_rejects_extreme_piece_count_before_validator() {
        let parsed = Board::from_str("qqqqkqqq/qqqqqqqq/8/8/8/8/8/4K3 b - - 0 1");

        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_too_many_pawns_for_one_side() {
        let board = board("4k3/pppppppp/p7/8/8/8/8/4K3 b - - 0 1");

        let reasons = standard_position_reject_reasons(&board);

        assert!(reasons.contains(&PositionRejectReason::BlackPawnCount));
    }

    #[test]
    fn rejects_pawns_on_back_rank() {
        let board = board("4k3/8/8/8/8/8/8/P3K3 w - - 0 1");

        let reasons = standard_position_reject_reasons(&board);

        assert!(reasons.contains(&PositionRejectReason::WhitePawnBackRank));
    }

    #[test]
    fn accepts_promoted_piece_shape_when_basic_counts_are_valid() {
        let board = board("4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1");

        assert_eq!(validate_standard_position(&board), Ok(()));
    }
}
