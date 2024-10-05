import numpy as np
import random
from random import choice
import matplotlib.pyplot as plt


def create_board():
    board = np.zeros((3, 3))
    return board


def possibilities(board):
    avail = []
    [(avail.append((x, y))) for x in range(3) for y in range(3) if board[x][y] == 0]
    return avail


def place(board, player, position):
    board[position[0]][position[1]] = player


def random_place(board, player):
    selection = possibilities(board)
    i = choice(range(len(selection)))
    # print(selection[i])
    place(board, player, selection[i])


def row_win(board, player):
    for i in range(3):
        won = True
        for j in range(3):
            if board[i][j] != player or board[i][j] == 0:
                won = False
        if won:
            # print("row")
            return True
    return False


def col_win(board, player):
    for i in range(3):
        won = True
        for j in range(3):
            if board[j][i] != player or board[j][i] == 0:
                won = False
        if won:
            # print("col")
            return True
    return False


def diag_win(board, player):
    for j in range(2):
        won = True
        if j == 0:
            for i in range(3):
                if board[i][i] != player or board[i][i] == 0:
                    won = False
            if won:
                # print("diag")
                return True
        else:
            for i in range(3):
                j = 2 - i
                if board[i][j] != player or board[i][j] == 0:
                    won = False
            if won:
                # print("diag")
                return True
    return False


def play_game():
    board = create_board()

    outcome = -1
    player = 1
    while len(possibilities(board)) != 0:
        random_place(board, player)
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            outcome = str(player)
            break
        player = 1 if player == 2 else 2
    # print(board)
    return outcome


def play_strategic_game():
    board = create_board()
    board[1][1] = 1
    outcome = -1
    player = 2
    while len(possibilities(board)) != 0:
        random_place(board, player)
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            outcome = str(player)
            break
        player = 1 if player == 2 else 2
    return outcome


result_list = []
for i in range(1000):
    result_list.append(int(play_game()))

result = [0, 0, 0]
for i in range(len(result_list)):
    if result_list[i] == -1:
        result[0] += 1
    elif result_list[i] == 1:
        result[1] += 1
    elif result_list[i] == 2:
        result[2] += 1

print(result)
plt.plot(result)
plt.show()
