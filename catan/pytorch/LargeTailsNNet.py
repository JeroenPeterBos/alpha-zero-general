import sys

from torch import nn
from torch.nn import functional as F

sys.path.append('..')


class CatanNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super().__init__()

        min_width = self.board_x * self.board_y

        # Layers
        # self.conv1 = nn.Conv2d(1, 6, 3)
        self.lin1 = nn.Linear(min_width, min_width * 2)
        # self.lin2 = nn.Linear(min_width * 2, min_width)
        self.lin3a = nn.Linear(min_width * 2, min_width)
        self.lin3b = nn.Linear(min_width, min_width)
        self.lin4a = nn.Linear(min_width * 2, self.action_size)
        self.lin4b = nn.Linear(min_width, 1)

        # Activations
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.board_x * self.board_y)
        x = F.relu(self.lin1(x))
        #x = F.relu(self.lin2(x))

        pi = F.relu(self.lin3a(x))
        pi = self.lin4a(pi)

        v = F.relu(self.lin3b(x))
        v = self.lin4b(v)

        return self.log_softmax(pi), self.tanh(v)
