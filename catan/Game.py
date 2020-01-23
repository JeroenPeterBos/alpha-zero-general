import numpy as np
import os

from Game import Game


class CatanGame(Game):
    road_adjacency = (
        (
            np.array([
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
            ]),
            np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 1],
            ]),
        ),
        (
            np.array([
                [1, 1, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]),
            np.array([
                [1, 1, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]),
        )
    )
    settlement_adjacency = (
        np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ]),
    )
    settlement_road_adjacency = (
        np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]),
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]),
    )
    SETTLEMENT_MASK = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    ])
    ROAD_MASK = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    ])

    DEFAULT_START_BOARD = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Settlements above, roads below
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    def __init__(self):
        super().__init__()
        self.settlement_rows = 6
        self.road_rows = 11
        self.width = 11
        self.height = self.settlement_rows + self.road_rows
        self.victory_points = 10

        self._valid_cache = {
            1: {},
            -1: {}
        }

    def getInitBoard(self):
        return self.DEFAULT_START_BOARD.copy()

    def splitBoard(self, board):
        return board[:self.settlement_rows, :].copy(), board[self.settlement_rows:, :].copy()

    def combineBoard(self, settlements, roads):
        return np.concatenate((settlements, roads), axis=0)

    def road_options(self, roads):
        result = np.zeros(roads.shape)
        padded = np.pad(roads.copy(), 1, constant_values=(0, ))
        for i, j in zip(*(self.ROAD_MASK * (roads == 0)).nonzero()):
            result[i, j] = ((
                (padded[i:i+3,j:j+3] > 0) * self.road_adjacency[i % 2][((i // 2) + j) % 2]
            ).sum() > 0).astype(int)
        return result

    def settlement_options(self, settlements, roads):
        settlement = np.zeros(settlements.shape)
        s_padded = np.pad(settlements.copy(), 1, constant_values=(0,))
        r_padded = np.pad(roads.copy(), 1, constant_values=(0,))
        for i, j in zip(*(self.SETTLEMENT_MASK * (settlements == 0)).nonzero()):
            # Determine if there is enough distance between this spot and other settlements
            distance = (((s_padded[i:i + 3, j:j + 3] * self.settlement_adjacency[(i + j) % 2]) != 0).sum() == 0).astype(int)

            # Determine if the player can reach this spot with its roads
            connected = (((r_padded[(i * 2):(i * 2) + 3, j:j + 3] > 0) * self.settlement_road_adjacency[(i + j) % 2]).sum() > 0).astype(int)

            # If both accept store it in the result array
            settlement[i, j] = distance * connected
        return settlement

    def settlement_availability(self, settlements):
        settlement = np.zeros(settlements.shape)
        padded = np.pad(settlements.copy(), 1, constant_values=(0, ))
        for i, j in zip(*(self.SETTLEMENT_MASK * (settlements == 0)).nonzero()):
            settlement[i, j] = ((padded[i:i + 3, j:j + 3] * self.settlement_adjacency[(i + j) % 2]).sum() == 0)
        return (settlement != 0).any()

    def getBoardSize(self):
        return self.height, self.width

    def getActionSize(self):
        return self.height * self.width + 1

    def getNextState(self, board, player, action):
        # Check whether the action is valid
        assert action in self.getValidMoves(board, player).nonzero()[0].tolist()

        if action == self.getActionSize() - 1:
            return board.copy(), -player

        board = board.copy()
        board[action // self.width, action % self.width] = player
        return board, -player

    def getValidMoves(self, board, player):
        s = self.stringRepresentation(board)
        if s in self._valid_cache[player]:
            return self._valid_cache[player][s]

        settlements, roads = self.splitBoard(board * player)
        road_moves = self.road_options(roads)
        settlement_moves = self.settlement_options(settlements, roads)
        board_moves = self.combineBoard(settlement_moves, road_moves)
        can_pass = ((board_moves > 0).sum() == 0).astype(int).item()
        actions = np.concatenate((board_moves.flatten(), [can_pass]))

        self._valid_cache[player][s] = actions
        return actions

    def getGameEnded(self, board, player):
        #assert (self.getValidMoves(board, player) > 0).sum() > 0
        #assert (self.getValidMoves(board, -player) > 0).sum() > 0

        settlements, roads = self.splitBoard(board * player)
        if (settlements > 0).sum() >= self.victory_points:
            return 1
        elif (settlements < 0).sum() >= self.victory_points:
            return -1
        elif not self.settlement_availability(settlements):
            return 1e-4

        # Game hasn't ended yet
        return 0

    def getCanonicalForm(self, board, player):
        return board * player

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return np.array2string(board)

    def display(self, board, player1_name, player2_name, save=False, name='demo'):
        from catan.Gui import GUI
        gui = GUI()

        s, r = self.splitBoard(board)

        def replace(arr):
            np.place(arr, arr == -1, [2])
            return arr

        settlements = np.array([
            replace(row[self.SETTLEMENT_MASK[i] == 1]).tolist()
            for i, row in enumerate(s)
        ])

        roads = np.array([
            replace(row[self.ROAD_MASK[i] == 1]).tolist()
            for i, row in enumerate(r)
        ])

        gui.draw_settlements(settlements)
        gui.draw_roads(roads)
        gui.draw_text(player1_name, player2_name)
        gui.show()

        if save:
            path = f'{os.path.dirname(__file__)}/board_states/{name}'
            if (os.path.exists(path) != True):
                os.mkdir(path)
            gui.save(f'{path}.png')


class SmallGame(CatanGame):
    road_adjacency = (
        (
            np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 1],
            ]),
            np.array([
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 0],
            ]),
        ),
        (
            np.array([
                [1, 1, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]),
            np.array([
                [1, 1, 0],
                [0, 0, 0],
                [1, 1, 0],
            ]),
        )
    )
    settlement_adjacency = (
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]),
    )
    settlement_road_adjacency = (
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]),
    )
    SETTLEMENT_MASK = np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ])
    ROAD_MASK = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ])

    DEFAULT_START_BOARD = np.array([
        [0, 0, 0, 0, 0,-1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0,-1, 0, 0, 0],
        # Settlements above, roads below
        [0, 0, 0, 0,-1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0,-1, 0, 0, 0],
    ])

    def __init__(self):
        super().__init__()
        self.settlement_rows = 4
        self.road_rows = 7
        self.width = 7
        self.height = self.settlement_rows + self.road_rows
        self.victory_points = 5

    def display(self, board, player1_name, player2_name, save=False, name='demo'):
        from catan.Gui import GUI
        gui = GUI(hexes=[0, 2, 3, 2, 0])

        s, r = self.splitBoard(board)

        def replace(arr):
            np.place(arr, arr == -1, [2])
            return arr

        settlements = np.array([[]] + [
            replace(row[self.SETTLEMENT_MASK[i] == 1]).tolist()
            for i, row in enumerate(s)
        ])

        roads = np.array([[], []] + [
            replace(row[self.ROAD_MASK[i] == 1]).tolist()
            for i, row in enumerate(r)
        ])

        gui.draw_settlements(settlements)
        gui.draw_roads(roads)
        gui.draw_text(player1_name, player2_name)
        gui.show()

        if save:
            path = f'{os.path.dirname(__file__)}/board_states/{name}'
            if (os.path.exists(path) != True):
                os.mkdir(path)
            gui.save(f'{path}.png')