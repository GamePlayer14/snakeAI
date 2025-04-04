import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from game.direction import DIRECTION_DELTAS, turn_left, turn_right

class SnakeModel(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(SnakeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.output(x), dim=1)

def build_model(input_size, output_size=3):
    return SnakeModel(input_size, output_size)

def get_features(game_state):
    from game.direction import DIRECTION_DELTAS, turn_left, turn_right
    import numpy as np

    height, width = game_state.board_size
    board = np.zeros((height, width), dtype=int)

    for y, x in game_state.snake:
        board[y][x] = 1  # snake body
    hy, hx = game_state.snake[0]
    ay, ax = game_state.apple
    board[hy][hx] = 2  # snake head

    direction = game_state.direction
    dy, dx = DIRECTION_DELTAS[direction]
    dy_l, dx_l = DIRECTION_DELTAS[turn_left(direction)]
    dy_r, dx_r = DIRECTION_DELTAS[turn_right(direction)]

    def is_wall(y, x):
        return not (0 <= y < height and 0 <= x < width)

    def is_body(y, x):
        if 0 <= y < board.shape[0] and 0 <= x < board.shape[1]:
            return board[y][x] == 1
        return False  # Out of bounds is definitely not body


    # Perception
    wall_ahead = float(is_wall(hy + dy, hx + dx))
    wall_left  = float(is_wall(hy + dy_l, hx + dx_l))
    wall_right = float(is_wall(hy + dy_r, hx + dx_r))

    body_ahead = float(is_body(hy + dy, hx + dx))
    body_left  = float(is_body(hy + dy_l, hx + dx_l))
    body_right = float(is_body(hy + dy_r, hx + dx_r))

    # Apple relative to head and direction
    ahead_vec = (dy, dx)
    left_vec  = (dy_l, dx_l)
    right_vec = (dy_r, dx_r)

    def is_apple_in_dir(vec):
        y, x = hy + vec[0], hx + vec[1]
        return float((y, x) == (ay, ax))

    apple_ahead = is_apple_in_dir(ahead_vec)
    apple_left  = is_apple_in_dir(left_vec)
    apple_right = is_apple_in_dir(right_vec)

    # Apple vector (normalized)
    apple_dx = (ax - hx) / width
    apple_dy = (ay - hy) / height

    # Direction one-hot
    dir_encoding = [0.0, 0.0, 0.0, 0.0]
    dir_encoding[direction] = 1.0  # 0=up, 1=right, 2=down, 3=left

    return np.array([
        wall_ahead, wall_left, wall_right,
        body_ahead, body_left, body_right,
        apple_ahead, apple_left, apple_right,
        apple_dx, apple_dy,
        *dir_encoding
    ], dtype=np.float32)


def is_apple_in_direction(game_state, pos):
    return pos == game_state.apple

def get_action(model, game_state, epsilon = 0.0):
    model.eval()

    if random.random() < epsilon:
        move = random.randint(0,2)
    else:
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
