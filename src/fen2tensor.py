import chess
import torch


TYPE = torch.float32


def shape(layers: int = 1) -> tuple[int, ...]:
    return (layers, 8, 8)


def fen2tensor_batch(fens: list[str]) -> torch.Tensor:
    return torch.vstack([fen2tensor(fen).unsqueeze(0) for fen in fens])


def fen2tensor(fen: str) -> torch.Tensor:
    game = chess.Board(fen)
    layers = [
        build_layer_turn(game),
        build_layer_white_kingside_castling_rights(game),
        build_layer_white_queenside_castling_rights(game),
        build_layer_black_kingside_castling_rights(game),
        build_layer_black_queenside_castling_rights(game),
        build_layer_player_en_passant(game),
        build_layer_white_attacks(game),
        build_layer_black_attacks(game),
        build_layer_pieces(game),
    ]

    return torch.vstack(layers)


def build_layer_turn(game: chess.Board) -> torch.Tensor:
    if game.turn == chess.WHITE:
        return torch.full(shape(), 1, dtype=TYPE)
    else:
        return torch.full(shape(), 0, dtype=TYPE)


def build_layer_white_kingside_castling_rights(game: chess.Board) -> torch.Tensor:
    if game.has_kingside_castling_rights(chess.WHITE):
        return torch.full(shape(), 1, dtype=TYPE)
    else:
        return torch.full(shape(), 0, dtype=TYPE)


def build_layer_white_queenside_castling_rights(game: chess.Board) -> torch.Tensor:
    if game.has_queenside_castling_rights(chess.WHITE):
        return torch.full(shape(), 1, dtype=TYPE)
    else:
        return torch.full(shape(), 0, dtype=TYPE)


def build_layer_black_kingside_castling_rights(game: chess.Board) -> torch.Tensor:
    if game.has_kingside_castling_rights(chess.BLACK):
        return torch.full(shape(), 1, dtype=TYPE)
    else:
        return torch.full(shape(), 0, dtype=TYPE)


def build_layer_black_queenside_castling_rights(game: chess.Board) -> torch.Tensor:
    if game.has_queenside_castling_rights(chess.BLACK):
        return torch.full(shape(), 1, dtype=TYPE)
    else:
        return torch.full(shape(), 0, dtype=TYPE)


def build_layer_player_en_passant(game: chess.Board) -> torch.Tensor:
    layer = torch.zeros(shape(), dtype=TYPE)

    for move in game.legal_moves:
        if game.is_en_passant(move):
            square = move.to_square
            row_index = chess.square_rank(square)
            column_index = chess.square_file(square)
            layer[0, row_index, column_index] = 1

    return layer


def build_layer_white_attacks(game: chess.Board) -> torch.Tensor:
    layer = torch.zeros(shape(), dtype=TYPE)

    for square in chess.SQUARES:
        if game.is_attacked_by(chess.WHITE, square):
            row_index = chess.square_rank(square)
            column_index = chess.square_file(square)
            layer[0, row_index, column_index] = 1

    return layer


def build_layer_black_attacks(game: chess.Board) -> torch.Tensor:
    layer = torch.zeros(shape(), dtype=TYPE)

    for square in chess.SQUARES:
        if game.is_attacked_by(chess.BLACK, square):
            row_index = chess.square_rank(square)
            column_index = chess.square_file(square)
            layer[0, row_index, column_index] = 1

    return layer


def build_layer_pieces(game: chess.Board) -> torch.Tensor:
    layer = torch.zeros(shape(12), dtype=TYPE)

    for square, piece in game.piece_map().items():
        if piece.color == chess.WHITE:
            piece_offset = 0
        else:
            piece_offset = 6

        piece_index = piece_offset + piece.piece_type - 1
        row_index = chess.square_rank(square)
        column_index = chess.square_file(square)
        layer[piece_index, row_index, column_index] = 1

    return layer
