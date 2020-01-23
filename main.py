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

import argparse

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--numEps', type=int, default=100, help='The number of training epochs to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--numIters', type=int, default=100, help='The number of games in an epoch.')
    parser.add_argument('--tempThreshold', type=int, default=15)
    parser.add_argument('--updateThreshold', type=float, default=0.5)
    parser.add_argument('--maxlenOfQueue', type=int, default=200000)
    parser.add_argument('--numMCTSSims', type=int, default=25)
    parser.add_argument('--arenaCompare', type=int, default=40)
    parser.add_argument('--cpuct', type=float, default=0.5)
    parser.add_argument('--checkpoint', type=str, default='./temp/')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--load_folder_file', default=('./catan_othello_net','checkpoint_9.pth.tar'))
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=10)

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    c = Coach(g, nnet, args, opponents=opponents)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
