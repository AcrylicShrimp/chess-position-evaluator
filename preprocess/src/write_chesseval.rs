use chess::{BitBoard, Board, Color, EMPTY, Piece};
use duckdb::{AccessMode, Config, Connection, Row, params};
use rayon::prelude::*;
use std::{path::Path, str::FromStr};
use tokio::fs::OpenOptions;

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
) -> Result<(), anyhow::Error> {
    use tokio::io::AsyncWriteExt;

    const CHUNK_SIZE: usize = 1024 * 1024;

    let duckdb_temp_path = duckdb_temp_path.as_ref();
    let path = path.as_ref();

    let processed = (offset..offset + limit)
        .step_by(CHUNK_SIZE)
        .par_bridge()
        .map(|chunk_offset| {
            let chunk_size = std::cmp::min(CHUNK_SIZE, (offset + limit - chunk_offset) as usize);
            process_chunk(chunk_size, chunk_offset, duckdb_temp_path)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await?;

    let total_row_count: usize = processed.iter().map(|(row_count, _)| row_count).sum();

    // metadata: length (8 bytes unsigned integer)
    file.write_u64_le(total_row_count as u64).await?;

    for (_, bytes) in processed {
        file.write_all(&bytes).await?;
    }

    Ok(())
}

fn process_chunk(
    chunk_size: usize,
    chunk_offset: i64,
    duckdb_temp_path: impl AsRef<Path>,
) -> Result<(usize, Vec<u8>), anyhow::Error> {
    let conn = Connection::open_with_flags(
        duckdb_temp_path,
        Config::default().access_mode(AccessMode::ReadOnly).unwrap(),
    )?;

    let mut stmt = conn.prepare("SELECT fen, cp FROM rows OFFSET ?1 LIMIT ?2")?;
    let chunk = stmt
        .query_and_then(
            params![chunk_offset, chunk_size],
            extract_chess_evaluation_row,
        )?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(construct_chunk(chunk))
}

fn construct_chunk(chunk: Vec<ChessEvaluationRow>) -> (usize, Vec<u8>) {
    let mut row_count = 0;
    let mut bytes = Vec::with_capacity((121 + 4) * chunk.len());

    for row in chunk {
        let board = match Board::from_str(&row.fen) {
            Ok(board) => board,
            Err(_) => {
                // ignore invalid FENs
                continue;
            }
        };

        // input: 121 bytes
        let bitflags = compute_bitflags(&board);
        let player_en_passant = compute_player_en_passant(&board);
        let white_attacks = compute_white_attacks(&board);
        let black_attacks = compute_black_attacks(&board);
        let pieces = compute_pieces(&board);

        bytes.extend(bitflags.to_le_bytes());
        bytes.extend(player_en_passant.0.to_le_bytes());
        bytes.extend(white_attacks.0.to_le_bytes());
        bytes.extend(black_attacks.0.to_le_bytes());

        for piece in pieces {
            bytes.extend(piece.0.to_le_bytes());
        }

        // label: 4 bytes
        let win_prob = centipawn_to_win_prob(row.cp);
        bytes.extend(win_prob.to_le_bytes());

        row_count += 1;
    }

    (row_count, bytes)
}

fn compute_bitflags(board: &Board) -> u8 {
    let mut bitflags = 0;

    if board.side_to_move() == Color::White {
        bitflags |= 1 << 0;
    }

    let white_castle_rights = board.castle_rights(Color::White);

    if white_castle_rights.has_kingside() {
        bitflags |= 1 << 1;
    }

    if white_castle_rights.has_queenside() {
        bitflags |= 1 << 2;
    }

    let black_castle_rights = board.castle_rights(Color::Black);

    if black_castle_rights.has_kingside() {
        bitflags |= 1 << 3;
    }

    if black_castle_rights.has_queenside() {
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

fn compute_white_attacks(board: &Board) -> BitBoard {
    compute_attack_map(board, Color::White)
}

fn compute_black_attacks(board: &Board) -> BitBoard {
    compute_attack_map(board, Color::Black)
}

fn compute_pieces(board: &Board) -> [BitBoard; 12] {
    [
        *board.pieces(Piece::Pawn) & *board.color_combined(Color::White),
        *board.pieces(Piece::Knight) & *board.color_combined(Color::White),
        *board.pieces(Piece::Bishop) & *board.color_combined(Color::White),
        *board.pieces(Piece::Rook) & *board.color_combined(Color::White),
        *board.pieces(Piece::Queen) & *board.color_combined(Color::White),
        *board.pieces(Piece::King) & *board.color_combined(Color::White),
        *board.pieces(Piece::Pawn) & *board.color_combined(Color::Black),
        *board.pieces(Piece::Knight) & *board.color_combined(Color::Black),
        *board.pieces(Piece::Bishop) & *board.color_combined(Color::Black),
        *board.pieces(Piece::Rook) & *board.color_combined(Color::Black),
        *board.pieces(Piece::Queen) & *board.color_combined(Color::Black),
        *board.pieces(Piece::King) & *board.color_combined(Color::Black),
    ]
}

fn compute_attack_map(board: &Board, color: Color) -> BitBoard {
    let mut attack_map = EMPTY;
    let occupied = *board.combined();

    for piece_square in *board.color_combined(color) {
        let piece = board.piece_on(piece_square);
        let piece = match piece {
            Some(piece) => piece,
            None => {
                continue;
            }
        };

        let attacks = match piece {
            Piece::Pawn => {
                chess::get_pawn_attacks(piece_square, color, BitBoard(0xffffffffffffffff))
            }
            Piece::Knight => chess::get_knight_moves(piece_square),
            Piece::Bishop => chess::get_bishop_moves(piece_square, occupied),
            Piece::Rook => chess::get_rook_moves(piece_square, occupied),
            Piece::Queen => {
                chess::get_bishop_moves(piece_square, occupied)
                    | chess::get_rook_moves(piece_square, occupied)
            }
            Piece::King => chess::get_king_moves(piece_square),
        };

        attack_map |= attacks;
    }

    attack_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::{ALL_FILES, ALL_RANKS, Square};

    fn visualize_bitboard(bitboard: BitBoard) {
        for rank in ALL_RANKS.iter().rev() {
            for file in ALL_FILES {
                let square = Square::make_square(*rank, file);
                if (bitboard & BitBoard::from_square(square)) != EMPTY {
                    print!("1");
                } else {
                    print!("0");
                }
            }
            println!();
        }
    }

    #[test]
    fn test_compute_attack_map() {
        let board =
            Board::from_str("rnbqkbnr/ppppp3/6pp/4p3/3P4/3B4/PPP2PPP/RNBQK1NR w KQkq - 0 5")
                .unwrap();

        let white_attacks = compute_attack_map(&board, Color::White);
        let black_attacks = compute_attack_map(&board, Color::Black);

        visualize_bitboard(white_attacks);
        println!("--------------------------------");
        visualize_bitboard(black_attacks);
    }
}
