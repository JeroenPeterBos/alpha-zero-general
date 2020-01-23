from abc import ABC, abstractmethod

import numpy as np

from catan.Game import CatanGame


class Player(ABC):
    def __init__(self, game: CatanGame):
        self.game = game

    @abstractmethod
    def play(self, board):
        pass


class RandomPlayer(Player):
    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1).nonzero()[0]
        return np.random.choice(valid_moves, 1).item()


class SettlementFirstPlayer(Player):
    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1).nonzero()[0]
        settlement_filter = valid_moves < self.game.width * self.game.settlement_rows
        s_options, r_options = valid_moves[settlement_filter], valid_moves[~settlement_filter]
        if len(s_options) > 0:
            return np.random.choice(s_options, 1).item()
        else:
            return np.random.choice(r_options, 1).item()


opponents = [RandomPlayer, SettlementFirstPlayer]
