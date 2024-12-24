import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.set_printoptions(sci_mode=False, linewidth=120)

def position_to_tensor(board, net):
    device = "cuda" if next(net.parameters()).is_cuda else "cpu"
    position = torch.Tensor(board.state) * board.turn
    position = position.unsqueeze(0).to(device)
    return position

"""Network"""
class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                         board_width*board_height)
        # value prediction layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.softmax(self.act_fc1(x_act), dim=1)
        x_act = x_act.view(-1, self.board_width, self.board_height)
        # value prediction layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

"""Board"""
class Board():
    def __init__(self, board_width, board_height, win_length, turn, state=None, empties=None):
        self.board_width = board_width
        self.board_height = board_height
        self.win_length = win_length
        self.turn = float(turn)
        if state == None:
            self.state = tuple([0.0 for _ in range (board_width)] for _ in range(board_height))
            self.empties = [(x, y) for y in range (board_height) for x in range(board_width)]
        else:
            self.state = tuple([x for x in y] for y in state)
            self.empties = [e for e in empties]
    def deepcopy(self):
        board = Board(self.board_width, self.board_height, self.win_length, self.turn, self.state, self.empties)
        return board
    def __str__(self):
        board_state_as_str = ""
        board_state = [row.copy() for row in self.state]
        for row in board_state:
            for i, token in enumerate(row):
                if token == 1.0:
                    row[i] = "X"
                elif token == -1.0:
                    row[i] = "O"
                elif token == 0.0:
                    row[i] = "_"
            board_state_as_str += str(row) + "\n"
        return board_state_as_str
def out_of_bounds(square, max_x, max_y):
    return (square[0] < 0 or square[1] < 0 or square[0] >= max_x or square[1] >= max_y)
def outcome(board, updated_square):
    player_num = board.state[updated_square[1]][updated_square[0]]
    vectors = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            if x == 0 and y == 0:
                continue
            # if a neighbouring square is out of bounds, skip to the next coordinate
            if updated_square[0] - 1 < 0 and x == -1:  # past left edge
                continue
            if updated_square[0] + 1 > board.board_width - 1 and x == 1:  # past right edge
                continue
            if updated_square[1] - 1 < 0 and y == -1:  # past top edge
                continue
            if updated_square[1] + 1 > board.board_height - 1 and y == 1:  # past bottom edge
                continue
            neighbour_x, neighbour_y = updated_square[0] + x, updated_square[1] + y
            if board.state[neighbour_y][neighbour_x] == player_num:
                vector = (neighbour_x - updated_square[0], neighbour_y - updated_square[1])
                if vector not in [(v[0]*-1, v[1]*-1) for v in vectors]: # if there is not already a vector in the opposite direction
                    vectors.append(vector)
    for vector in vectors:
        line_length = 1
        reversing = False
        i = 1
        # Keep checking adjacent squares in one direction and increase the row length by 1 whenever the token matches the token just placed
        # When an edge, empty space, or opponent token is encountered, count squares in the opposite direction skipping over all the ones we already counted
        while i < board.win_length:
            if not reversing:
                x = updated_square[0] + i * vector[0]
                y = updated_square[1] + i * vector[1]
                next_square = x, y
            else:
                x = updated_square[0] + i * -vector[0]
                y = updated_square[1] + i * -vector[1]
                next_square = x, y

            if out_of_bounds(next_square, board.board_width, board.board_height) \
            or board.state[next_square[1]][next_square[0]] != player_num:
                if not reversing:
                    reversing = True
                    i = 0
                else:
                    break
            else:
                line_length += 1
                if line_length >= board.win_length:
                    return player_num
            i += 1
    if len(board.empties) == 0:
        return 0.0
    return None
def make_move(board, move_coords: tuple):
    board.empties.remove(move_coords)
    board.state[move_coords[1]][move_coords[0]] = board.turn
    board.turn *= -1

"""MCTS"""
class Node:
    def __init__(self, parent, prob, board, move):
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.value = 0
        self.prob = prob
        self.board = board
        self.move = move

    def is_terminal(self):
        return outcome(self.board, self.move) != None

    def is_leaf(self):
        return self.children == []
class MCTS:
    """Search function always evaluates the best move for the 1s player"""
    def __init__(self, board, net):
        self.net = net
        self.root_node = Node(None, None, board, (-2, -2))
        self.c_puct = 1.0
        self.tau = 1.0
    def uct(self, parent, node):
        return self.c_puct * node.prob * (math.sqrt(parent.visit_count)/(1 + node.visit_count))

    def select(self, node):
        if node.is_leaf():
            return node
        else:
            max_value = -1000000
            chosen_node = None
            for child_node in node.children:
                uct_val = self.uct(node, child_node)
                q_val = child_node.value * node.board.turn
                value = q_val + uct_val
                if value > max_value:
                    max_value = value
                    chosen_node = child_node
            return self.select(chosen_node)
    def expand_to_backprop(self, node):
        position_as_tensor = position_to_tensor(node.board, self.net)
        net_output = self.net(position_as_tensor)
        legal_moves = node.board.empties
        legal_moves_prob_sum = 0.0
        for m in legal_moves:
            child_prob = net_output[0][0][m[1], m[0]].item()
            legal_moves_prob_sum += child_prob
        for m in legal_moves:
            child_board = node.board.deepcopy()
            make_move(child_board, m)
            child_prob = net_output[0][0][m[1], m[0]].item()/legal_moves_prob_sum
            child_node = Node(node, child_prob, child_board, m)
            node.children.append(child_node)
        evaluation = net_output[1][0].item() * node.board.turn
        self.backpropagate(node, evaluation)

    def backpropagate(self, node, evaluation):
        node.visit_count += 1
        node.total_value += evaluation
        node.value = node.total_value / node.visit_count
        if node.parent != None:
            self.backpropagate(node.parent, evaluation)

    def search(self, reps):
        torch.set_grad_enabled(False)
        for _ in range(reps):
            node = self.select(self.root_node)
            if node.is_terminal():
                self.backpropagate(node, outcome(node.board, node.move))
            else:
                self.expand_to_backprop(node)
        move_probs = []
        for child_node in self.root_node.children:
            prob = round(child_node.visit_count ** (1/self.tau) / (self.root_node.visit_count ** (1/self.tau)), 3)
            move_probs.append((child_node.move, prob))
        return move_probs
"""Reinforcement learning"""
class RL:
    def __init__(self, net, board_width, board_height, win_length):
        self.net = net
        self.board_width = board_width
        self.board_height = board_height
        self.win_length = win_length

    def self_play_game(self, mcts_reps):
        positions_data = []
        move_probs_data = []
        values_data = []
        game_on = True
        board = Board(self.board_width, self.board_height, self.win_length, 1)
        while game_on:
            position_as_tensor = position_to_tensor(board, self.net)
            positions_data.append(position_as_tensor)
            values_data.append(torch.Tensor([board.turn]))
            mcts_searcher = MCTS(board, self.net)
            mcts_move_probs = mcts_searcher.search(mcts_reps)
            move_probs_matrix = [[0.0] * self.board_width for y in range(self.board_height)]  # for the network
            moves = [] # for making a move
            move_probs = [] # for making a move
            for move, prob in mcts_move_probs:
                move_probs_matrix[move[1]][move[0]] = prob
                moves.append(move)
                move_probs.append(prob)
            move_probs_matrix = torch.Tensor(move_probs_matrix)
            move_probs_data.append(move_probs_matrix)
            rand_idx = np.random.multinomial(1, move_probs)
            chosen_move = moves[np.where(rand_idx==1)[0][0]]
            make_move(board, chosen_move)
            value = outcome(board, chosen_move)
            if value != None:
                for i in range(len(positions_data)):
                    values_data[i] *= value
                return (positions_data, move_probs_data, values_data)
"""Dataloader"""
class GomokuDataset(Dataset):
    def __init__(self, games):
        super().__init__()
        self.games = games
    def __len__(self):
        return len(self.games[0])
    def __getitem__(self, index):
        return self.games[0][index], self.games[1][index], self.games[2][index]

"""Play games"""
def net_choose_move(board, net):
    torch.set_grad_enabled(False)
    position_as_tensor = position_to_tensor(board, net)
    move_probs = net(position_as_tensor)[0][0]
    vals = []
    total_val = 0
    for square in board.empties:
        total_val += (move_probs[square[1], square[0]]) ** 1
    for square in board.empties:
        val = move_probs[square[1], square[0]].to("cpu")
        vals.append(math.floor(val ** 1/total_val * 10000)/10000)
    rand_idx = np.random.multinomial(1, vals)
    chosen_move = board.empties[np.where(rand_idx==1)[0][0]]
    return chosen_move

def mcts_net_vs_human(board, mcts_reps, net):
    game_on = True
    while game_on:
        if board.turn == 1:
            if mcts_reps > 0:
                mcts_searcher = MCTS(board, net)
                mcts_move_probs = mcts_searcher.search(mcts_reps)
                chosen_move = max(mcts_move_probs, key=lambda x: x[1])[0]
            else:
                chosen_move = net_choose_move(board, net)
                position_as_tensor = position_to_tensor(board, net)
                net_output = net(position_as_tensor)
                print(f"Move probabilities: \n{net_output[0][0]}\n")
                print(f"Outcome prediction: \n{net_output[1][0]}\n")
        else:
            x = input("x: ")
            y = input("y: ")
            chosen_move = (int(x), int(y))
        make_move(board, chosen_move)
        print("Board:")
        print(board)
        print("\n====================")
        value = outcome(board, chosen_move)
        if value != None:
            break


def net_vs_net(start_board, net1, net2, games):
    win_count = [0, 0]
    for game in range(games):
        if game % 2 == 0:
            nets = (net1, net2)
            first_net_idx = 0
            second_net_idx = 1
        else:
            nets = (net2, net1)
            first_net_idx = 1
            second_net_idx = 0
        board = start_board.deepcopy()
        game_on = True
        net_outputs = []
        while game_on:
            if board.turn == 1.0:
                net = nets[0]
            else:
                net = nets[1]

            chosen_move = net_choose_move(board, net)

            # mcts_searcher = MCTS(board, net)
            # mcts_move_probs = mcts_searcher.search(200)
            # chosen_move = max(mcts_move_probs, key=lambda x: x[1])[0]

            # move_probs_matrix = [[0.0] * board.board_width for y in range(board.board_height)]  # for the network
            # for move, prob in mcts_move_probs:
            #     move_probs_matrix[move[1]][move[0]] = prob
            # move_probs_matrix = torch.Tensor(move_probs_matrix)
            # position_as_tensor = position_to_tensor(board, net)
            # net_output = net(position_as_tensor)
            # net_outputs.extend((move_probs_matrix, net_output[1][0], chosen_move))

            make_move(board, chosen_move)
            value = outcome(board, chosen_move)

            if value != None:
                if value == 1.0:
                    win_count[first_net_idx] += 1
                elif value == -1.0:
                    win_count[second_net_idx] += 1
                game_on = False
    return win_count

"""Train"""
def train(main_net, opponent_net, learner, generations, games_per_generation, training_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_net = main_net.to(device)
    opponent_net = opponent_net.to(device)
    for generation in range(1, generations):
        # game generation
        all_positions = []
        all_move_probs = []
        all_values = []
        st = time.perf_counter()
        for game in range(games_per_generation):
            positions, move_probs, values = learner.self_play_game(200)
            all_positions += positions
            all_move_probs += move_probs
            all_values += values
        et = time.perf_counter()
        print(f"===== Game generation time in seconds: {et-st} ======")
        all_games = (all_positions, all_move_probs, all_values)
        # data preparation
        train_data = GomokuDataset(all_games)
        train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=False)
        # training
        torch.set_grad_enabled(True)
        policy_loss_metric = nn.BCELoss()
        value_loss_metric = nn.MSELoss()
        optimizer = torch.optim.Adam(main_net.parameters(), lr=0.01, weight_decay=1e-4)
        for epoch in range(training_epochs):
            running_loss = 0
            for position, move_probs, value in train_data_loader:
                position, move_probs, value = position.to(device), move_probs.to(device), value.to(device)
                output = main_net(position)
                policy_loss = policy_loss_metric(output[0], move_probs)
                value_loss = value_loss_metric(output[1], value)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Training loss: {running_loss/len(train_data_loader)}")
        # evaluating model
        win_counts = net_vs_net(Board(learner.board_width, learner.board_height, learner.win_length, 1), main_net, opponent_net, 1000)
        print("")
        print(f"main_net wins: {win_counts[0]}")
        print(f"main_net losses: {win_counts[1]}")
        print(f"main_net draws: {1000 - win_counts[0] - win_counts[1]}")
        if win_counts[0] - win_counts[1] > 150:
            torch.save(main_net.state_dict(), f"model_{generation}.pt")
            opponent_net.load_state_dict(torch.load(f"model_{generation}.pt"))
            print(f"Saved model {generation}")
            print(f"Updated opponent_net to copy of main_net")



main_net = Net(6, 6)
# main_net.load_state_dict(torch.load("model.pt"))
opponent_net = Net(6, 6)
learner = RL(main_net, 6, 6, 4)
train(main_net, opponent_net, learner, 10, 50, 30)

mcts_net_vs_human(Board(6, 6, 4), 0, main_net)
