use chess::{Board, BoardStatus};

pub fn is_quiescent(board: &Board) -> bool {
    if board.status() != BoardStatus::Ongoing {
        return true;
    }

    if *board.checkers() == chess::EMPTY {
        return false;
    }

    false
}
