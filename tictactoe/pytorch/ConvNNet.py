import sys

from torch import nn
from torch.nn import functional as F

sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *

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
        self.conv2d = nn.Conv2d(args.num_channels, self.board_x, self.board_y)
        self.bn2d = nn.BatchNorm2d(args.num_channels)

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
