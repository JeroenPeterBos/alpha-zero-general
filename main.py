from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn

from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.pytorch.NNet import NNetWrapper as nn
opponents = []

from catan.Game import CatanGame as Game
from catan.pytorch.NNet import NNetWrapper as nn
from catan.Players import opponents

from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.5,     # During arena playoff, new nnet will be accepted if threshold or more games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 0.5,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./catan_othello_net','checkpoint_9.pth.tar'),
    'numItersForTrainExamplesHistory': 10,
    'render': True,
})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args, opponents=opponents)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
