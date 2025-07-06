use chess::{
    ALL_SQUARES, BitBoard, Board, Color, EMPTY, Piece, Square, get_bishop_moves, get_king_moves,
    get_knight_moves, get_pawn_attacks, get_rook_moves,
};
use duckdb::{AccessMode, Config, Connection, Row, params};
use rayon::prelude::*;
use std::{path::Path, str::FromStr};
use tokio::fs::OpenOptions;

struct ChessEvaluationRow {
    fen: String,
    cp: Option<i32>,
}

fn extract_chess_evaluation_row(row: &Row) -> Result<ChessEvaluationRow, anyhow::Error> {
    let fen = row.get::<_, String>(0)?;
    let cp = row.get::<_, Option<i32>>(1)?;
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
            let chunk_size = std::cmp::min(CHUNK_SIZE, (limit - chunk_offset) as usize);
            process_chunk(chunk_size, chunk_offset, duckdb_temp_path)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await?;

    // metadata: length (8 bytes unsigned integer)
    file.write_u64_le(processed.len() as u64).await?;

    for bytes in processed {
        file.write_all(&bytes).await?;
    }

    Ok(())
}

fn process_chunk(
    chunk_size: usize,
    chunk_offset: i64,
    duckdb_temp_path: impl AsRef<Path>,
) -> Result<Vec<u8>, anyhow::Error> {
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

fn construct_chunk(chunk: Vec<ChessEvaluationRow>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((121 + 5) * chunk.len());

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

        // label: 5 bytes
        let (cp, mate) = if let Some(cp) = row.cp {
            ((cp as f32 * 0.01).clamp(-20.0, 20.0), 0u8)
        } else if board.side_to_move() == Color::White {
            (20f32, 1u8)
        } else {
            (-20f32, 2u8)
        };

        bytes.extend(cp.to_le_bytes());
        bytes.extend(mate.to_le_bytes());
    }

    bytes
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
    let mut attackers = EMPTY;

    for square in ALL_SQUARES {
        attackers |= get_attackers_to(board, square, Color::White);
    }

    attackers
}

fn compute_black_attacks(board: &Board) -> BitBoard {
    let mut attackers = EMPTY;

    for square in ALL_SQUARES {
        attackers |= get_attackers_to(board, square, Color::Black);
    }

    attackers
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

pub fn get_attackers_to(board: &Board, target_square: Square, attacking_color: Color) -> BitBoard {
    let mut attackers = EMPTY;
    let occupied = *board.combined();

    attackers |= get_pawn_attacks(target_square, !attacking_color, EMPTY)
        & board.pieces(Piece::Pawn)
        & board.color_combined(attacking_color);

    attackers |= get_knight_moves(target_square)
        & board.pieces(Piece::Knight)
        & board.color_combined(attacking_color);

    let bishop_like_attackers = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    attackers |= get_bishop_moves(target_square, occupied)
        & bishop_like_attackers
        & board.color_combined(attacking_color);

    let rook_like_attackers = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    attackers |= get_rook_moves(target_square, occupied)
        & rook_like_attackers
        & board.color_combined(attacking_color);

    attackers |= get_king_moves(target_square)
        & board.pieces(Piece::King)
        & board.color_combined(attacking_color);

    attackers
}
