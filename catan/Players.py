from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np

from catan.Game import CatanGame

INITIAL_HEX_RESOURCES = np.array([
[2, 3, 5],
[4, 1, 3, 1],
[4, 5, 0, 5, 2],
[5, 2, 4, 3],
[1, 4, 3]
])
INITIAL_HEX_NUMBERS = np.array([
    [10, 2, 9],
    [12, 6, 4, 10],
    [9, 11, 7, 3, 8],
    [8, 3, 4, 5],
    [5, 6, 11]
])
INITIAL_RESOURCES = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
INITIAL_TURN = 0
INITIAL_VP = np.array([2, 2])

class Network(nn.Module):
    """
    This class is used to handle all the communication between the AI and the game.
    """
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=176, out_features=126),
            nn.LeakyReLU(0.02),
            nn.Softmax()
        )

    def forward(self, x):
        return self.main(x)


def replace(arr):
    np.place(arr, arr == -1, [2])
    return arr


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


class CustomModelPlayer(Player):
    def __init__(self):
        self.state_dict = torch.load(f'catan/best.pth')
        self.policy = Network()
        self.policy.load_state_dict(self.state_dict)

    def play(self, board):
        s, r = self.game.splitBoard(board)

        settlements = np.array([
            replace(row[self.game.SETTLEMENT_MASK[i] == 1]).tolist()
            for i, row in enumerate(s)
        ])

        roads = np.array([
            replace(row[self.game.ROAD_MASK[i] == 1]).tolist()
            for i, row in enumerate(r)
        ])

        obs = np.array([
            INITIAL_RESOURCES,
            INITIAL_HEX_RESOURCES,
            INITIAL_HEX_NUMBERS,
            settlements,
            roads,
            INITIAL_VP
        ])

        return self.policy(obs)



opponents = [RandomPlayer, SettlementFirstPlayer]
