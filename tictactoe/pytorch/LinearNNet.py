import sys

from torch import nn
from torch.nn import functional as F

sys.path.append('..')

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super().__init__()

        # Layers
        # self.conv1 = nn.Conv2d(1, 6, 3)
        self.lin1 = nn.Linear(self.board_x * self.board_y, 18)
        self.lin2 = nn.Linear(18, 18)
        self.lin3a = nn.Linear(18, 18)
        self.lin3b = nn.Linear(18, 18)
        self.lin4a = nn.Linear(18, self.action_size)
        self.lin4b = nn.Linear(18, 1)

        # Activations
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 9)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        pi = F.relu(self.lin3a(x))
        pi = self.lin4a(pi)

        v = F.relu(self.lin3b(x))
        v = self.lin4b(v)

        return self.log_softmax(pi), self.tanh(v)
