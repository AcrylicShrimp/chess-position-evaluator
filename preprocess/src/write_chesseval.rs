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
    let mut bytes = Vec::with_capacity((105 + 4) * chunk.len());

    for row in chunk {
        let board = match Board::from_str(&row.fen) {
            Ok(board) => board,
            Err(_) => {
                // ignore invalid FENs
                continue;
            }
        };
        let us = board.side_to_move();
        let them = !us;

        // input: 105 bytes
        let bitflags = compute_bitflags(&board, us, them);
        let mut player_en_passant = compute_player_en_passant(&board);
        let mut pieces = compute_pieces(&board, us, them);

        if us == Color::Black {
            fn flip_vertical(bitboard: BitBoard) -> BitBoard {
                let mut bytes = bitboard.0.to_ne_bytes();
                bytes.reverse();
                BitBoard(u64::from_ne_bytes(bytes))
            }

            player_en_passant = flip_vertical(player_en_passant);
            pieces = pieces.map(flip_vertical);
        }

        bytes.extend(bitflags.to_le_bytes());
        bytes.extend(player_en_passant.0.to_le_bytes());

        for piece in pieces {
            bytes.extend(piece.0.to_le_bytes());
        }

        // label: 4 bytes
        let relative_cp = if us == Color::White { row.cp } else { -row.cp };
        let win_prob = centipawn_to_win_prob(relative_cp);
        bytes.extend(win_prob.to_le_bytes());

        row_count += 1;
    }

    (row_count, bytes)
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
