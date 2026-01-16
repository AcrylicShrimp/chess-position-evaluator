use chess::{BitBoard, Board, ChessMove, Color, Piece, Rank, Square};

const PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 64000];

pub fn compute_see(board: &Board, m: ChessMove) -> i32 {
    let is_ep = board.en_passant() == Some(m.get_dest());
    let (killer, victim) = match (board.piece_on(m.get_source()), board.piece_on(m.get_dest())) {
        (Some(killer), Some(victim)) => (killer, victim),
        _ if is_ep => (Piece::Pawn, Piece::Pawn),
        _ => {
            // not a capture; no SEE
            return 0;
        }
    };

    let mut scores = Vec::with_capacity(64);
    let mut current_piece = killer;
    let mut side_to_move = board.side_to_move();
    let mut see_board = SeeBoard::new(board, m.get_dest());

    see_board.remove_piece(side_to_move, killer, m.get_source());
    see_board.remove_piece(
        !side_to_move,
        victim,
        if is_ep {
            match side_to_move {
                Color::White => m
                    .get_dest()
                    .down()
                    .expect("[INTERNAL ERROR] wrong en passant target square"),
                Color::Black => m
                    .get_dest()
                    .up()
                    .expect("[INTERNAL ERROR] wrong en passant target square"),
            }
        } else {
            m.get_dest()
        },
    );

    match m.get_promotion() {
        Some(promotion) => {
            scores.push(
                PIECE_VALUES[victim.to_index()] + PIECE_VALUES[promotion.to_index()]
                    - PIECE_VALUES[Piece::Pawn.to_index()],
            );
            see_board.place_piece(side_to_move, promotion, m.get_dest());
        }
        None => {
            scores.push(PIECE_VALUES[victim.to_index()]);
            see_board.place_piece(side_to_move, killer, m.get_dest());
        }
    }

    side_to_move = !side_to_move;

    while let Some(((piece, square), promotion)) = see_board.next_piece(side_to_move) {
        see_board.remove_piece(!side_to_move, current_piece, m.get_dest());
        see_board.remove_piece(side_to_move, piece, square);

        match promotion {
            Some(promotion) => {
                scores.push(
                    PIECE_VALUES[current_piece.to_index()] + PIECE_VALUES[promotion.to_index()]
                        - PIECE_VALUES[Piece::Pawn.to_index()],
                );
                see_board.place_piece(side_to_move, promotion, m.get_dest());
                current_piece = promotion;
            }
            None => {
                scores.push(PIECE_VALUES[current_piece.to_index()]);
                see_board.place_piece(side_to_move, piece, m.get_dest());
                current_piece = piece;
            }
        }

        side_to_move = !side_to_move;
    }

    for index in (0..scores.len()).rev() {
        let recapture_score = scores.get(index + 1).copied().unwrap_or_default().max(0);
        scores[index] -= recapture_score;
    }

    *scores.first().unwrap()
}

struct SeeBoard {
    color_occupancies: [BitBoard; 2],
    target_square: Square,
    pawns: BitBoard,
    knights: BitBoard,
    bishops: BitBoard,
    rooks: BitBoard,
    queens: BitBoard,
    kings: BitBoard,
}

impl SeeBoard {
    pub fn new(board: &Board, target_square: Square) -> Self {
        let white_occupancy = *board.color_combined(Color::White);
        let black_occupancy = *board.color_combined(Color::Black);
        let occupancy = white_occupancy | black_occupancy;
        let pawns = *board.pieces(Piece::Pawn)
            & (chess::get_pawn_attacks(target_square, Color::White, occupancy)
                | chess::get_pawn_attacks(target_square, Color::Black, occupancy));
        let knights = *board.pieces(Piece::Knight) & chess::get_knight_moves(target_square);
        let bishops = *board.pieces(Piece::Bishop);
        let rooks = *board.pieces(Piece::Rook);
        let queens = *board.pieces(Piece::Queen);
        let kings = *board.pieces(Piece::King) & chess::get_king_moves(target_square);

        Self {
            color_occupancies: [white_occupancy, black_occupancy],
            target_square,
            pawns,
            knights,
            bishops,
            rooks,
            queens,
            kings,
        }
    }

    pub fn remove_piece(&mut self, side_to_move: Color, piece: Piece, square: Square) {
        let bitboard = BitBoard::from_square(square);

        self.color_occupancies[side_to_move.to_index()] &= !bitboard;

        match piece {
            Piece::Pawn => self.pawns &= !bitboard,
            Piece::Knight => self.knights &= !bitboard,
            Piece::Bishop => self.bishops &= !bitboard,
            Piece::Rook => self.rooks &= !bitboard,
            Piece::Queen => self.queens &= !bitboard,
            Piece::King => self.kings &= !bitboard,
        }
    }

    pub fn place_piece(&mut self, side_to_move: Color, piece: Piece, square: Square) {
        let bitboard = BitBoard::from_square(square);

        self.color_occupancies[side_to_move.to_index()] |= bitboard;

        match piece {
            Piece::Pawn => self.pawns |= bitboard,
            Piece::Knight => self.knights |= bitboard,
            Piece::Bishop => self.bishops |= bitboard,
            Piece::Rook => self.rooks |= bitboard,
            Piece::Queen => self.queens |= bitboard,
            Piece::King => self.kings |= bitboard,
        }
    }

    pub fn next_piece(&mut self, side_to_move: Color) -> Option<((Piece, Square), Option<Piece>)> {
        if let Some((square, promotion)) = self.next_pawn(side_to_move) {
            return Some(((Piece::Pawn, square), promotion));
        }

        if let Some(square) = self.next_knight(side_to_move) {
            return Some(((Piece::Knight, square), None));
        }

        if let Some(square) = self.next_bishop(side_to_move, self.target_square) {
            return Some(((Piece::Bishop, square), None));
        }

        if let Some(square) = self.next_rook(side_to_move, self.target_square) {
            return Some(((Piece::Rook, square), None));
        }

        if let Some(square) = self.next_queen(side_to_move, self.target_square) {
            return Some(((Piece::Queen, square), None));
        }

        if let Some(square) = self.next_king(side_to_move) {
            return Some(((Piece::King, square), None));
        }

        None
    }

    fn combined_occupancy(&self) -> BitBoard {
        self.color_occupancies[0] | self.color_occupancies[1]
    }

    fn next_pawn(&mut self, side_to_move: Color) -> Option<(Square, Option<Piece>)> {
        let pawns = self.color_occupancies[side_to_move.to_index()] & self.pawns;
        let pawn = match Self::get_any_piece(pawns) {
            Some(pawn) => pawn,
            None => {
                return None;
            }
        };
        let pawn_bitboard = BitBoard::from_square(pawn);

        self.color_occupancies[side_to_move.to_index()] &= !pawn_bitboard;
        self.pawns &= !pawn_bitboard;

        let promotion = match (side_to_move, pawn.get_rank()) {
            (Color::White, Rank::Seventh) | (Color::Black, Rank::Second) => Some(Piece::Queen),
            _ => None,
        };

        Some((pawn, promotion))
    }

    fn next_knight(&mut self, side_to_move: Color) -> Option<Square> {
        let knights = self.color_occupancies[side_to_move.to_index()] & self.knights;
        let knight = match Self::get_any_piece(knights) {
            Some(knight) => knight,
            None => {
                return None;
            }
        };
        let knight_bitboard = BitBoard::from_square(knight);

        self.color_occupancies[side_to_move.to_index()] &= !knight_bitboard;
        self.knights &= !knight_bitboard;

        Some(knight)
    }

    fn next_bishop(&mut self, side_to_move: Color, target_square: Square) -> Option<Square> {
        let bishops = self.color_occupancies[side_to_move.to_index()]
            & self.bishops
            & chess::get_bishop_moves(target_square, self.combined_occupancy());
        let bishop = match Self::get_any_piece(bishops) {
            Some(bishop) => bishop,
            None => {
                return None;
            }
        };
        let bishop_bitboard = BitBoard::from_square(bishop);

        self.color_occupancies[side_to_move.to_index()] &= !bishop_bitboard;
        self.bishops &= !bishop_bitboard;

        Some(bishop)
    }

    fn next_rook(&mut self, side_to_move: Color, target_square: Square) -> Option<Square> {
        let rooks = self.color_occupancies[side_to_move.to_index()]
            & self.rooks
            & chess::get_rook_moves(target_square, self.combined_occupancy());
        let rook = match Self::get_any_piece(rooks) {
            Some(rook) => rook,
            None => {
                return None;
            }
        };
        let rook_bitboard = BitBoard::from_square(rook);

        self.color_occupancies[side_to_move.to_index()] &= !rook_bitboard;
        self.rooks &= !rook_bitboard;

        Some(rook)
    }

    fn next_queen(&mut self, side_to_move: Color, target_square: Square) -> Option<Square> {
        let queens = self.color_occupancies[side_to_move.to_index()]
            & self.queens
            & (chess::get_bishop_moves(target_square, self.combined_occupancy())
                | chess::get_rook_moves(target_square, self.combined_occupancy()));
        let queen = match Self::get_any_piece(queens) {
            Some(queen) => queen,
            None => {
                return None;
            }
        };
        let queen_bitboard = BitBoard::from_square(queen);

        self.color_occupancies[side_to_move.to_index()] &= !queen_bitboard;
        self.queens &= !queen_bitboard;

        Some(queen)
    }

    fn next_king(&mut self, side_to_move: Color) -> Option<Square> {
        let kings = self.color_occupancies[side_to_move.to_index()] & self.kings;
        let king = match Self::get_any_piece(kings) {
            Some(king) => king,
            None => {
                return None;
            }
        };
        let king_bitboard = BitBoard::from_square(king);

        self.color_occupancies[side_to_move.to_index()] &= !king_bitboard;
        self.kings &= !king_bitboard;

        Some(king)
    }

    fn get_any_piece(pieces: BitBoard) -> Option<Square> {
        pieces.into_iter().next()
    }
}
