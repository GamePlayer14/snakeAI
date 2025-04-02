import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.direction import turn_left, turn_right

class SnakeModel(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(SnakeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output(x), dim=1)

def build_model(input_size, output_size=3):
    return SnakeModel(input_size, output_size)

def get_features(game_state):
    height, width = game_state.board_size
    board = np.zeros((height, width), dtype=float)

    i = 1.0
    for y, x in game_state.snake:
        board[y][x] = i
        i -= 0.001

    ay, ax = game_state.apple
    board[ay][ax] = 2.0

    sy, sx = game_state.snake[0]

    flat = board.flatten()
    direction = game_state.direction
    apple_vector_x = ax - sx
    apple_vector_y = ay - sy
    return np.append(flat, [direction, apple_vector_x, apple_vector_y]).astype(np.float32)

def get_action(model, game_state):
    model.eval()
    with torch.no_grad():
        features = torch.tensor(get_features(game_state)).unsqueeze(0)
        prediction = model(features)
        move = torch.argmax(prediction).item()

    current_direction = game_state.direction
    if move == 0:
        return current_direction
    elif move == 1:
        return turn_left(current_direction)
    else:
        return turn_right(current_direction)

def mutate_weights(model, mutation_rate=0.1):
    for param in model.parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * mutation_rate
            param.data += noise

def clone_model(model):
    new_model = SnakeModel(model.fc1.in_features)
    new_model.load_state_dict(model.state_dict())
    return new_model
