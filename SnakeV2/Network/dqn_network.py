import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random
from Game.direction import DIRECTION_DELTAS, turn_left, turn_right

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes, activation_fn=nn.ReLU, softmax_output=False):
        super().__init__()
        self.softmax_output = softmax_output
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.activation_fn = activation_fn()

        # Apply Kaiming initialization
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.softmax_output:
            x = F.softmax(x, dim=1)
        return x

    def clone(self):
        cloned = CustomNetwork([layer.in_features for layer in self.layers] + [self.layers[-1].out_features], 
                               activation_fn=self.activation_fn.__class__, 
                               softmax_output=self.softmax_output)
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned.to(device)

    def mutate(self, mutation_rate=0.1):
        for param in self.parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param) * mutation_rate

def get_features(game_state):
    height, width = game_state.board_size
    board = np.zeros((height, width), dtype=int)

    for y, x in game_state.snake:
        board[y][x] = 1
    ay, ax = game_state.apple
    board[ay][ax] = 2
    hy, hx = game_state.snake[0]

    direction = game_state.direction
    dy, dx = DIRECTION_DELTAS[direction]
    dy_l, dx_l = DIRECTION_DELTAS[turn_left(direction)]
    dy_r, dx_r = DIRECTION_DELTAS[turn_right(direction)]

    def is_wall(y, x):
        return not (0 <= y < height and 0 <= x < width)

    def is_body(y, x):
        return 0 <= y < height and 0 <= x < width and board[y][x] == 1

    wall = [float(is_wall(hy + dy, hx + dx)), float(is_wall(hy + dy_l, hx + dx_l)), float(is_wall(hy + dy_r, hx + dx_r))]
    body = [float(is_body(hy + dy, hx + dx)), float(is_body(hy + dy_l, hx + dx_l)), float(is_body(hy + dy_r, hx + dx_r))]
    apple = [float((hy + dy, hx + dx) == (ay, ax)), float((hy + dy_l, hx + dx_l) == (ay, ax)), float((hy + dy_r, hx + dx_r) == (ay, ax))]

    apple_dx = (ax - hx) / width
    apple_dy = (ay - hy) / height

    dir_encoding = [0.0] * 4
    dir_encoding[direction] = 1.0

    def dist(y, x, dy, dx, max_d=10):
        for i in range(1, max_d + 1):
            ny, nx = y + i * dy, x + i * dx
            if is_wall(ny, nx) or is_body(ny, nx):
                return i
        return max_d

    danger = [dist(hy, hx, dy, dx)/10, dist(hy, hx, dy_l, dx_l)/10, dist(hy, hx, dy_r, dx_r)/10]
    length_norm = len(game_state.snake) / (height * width)

    grid = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = hy + dy, hx + dx
            val = 3 if is_wall(ny, nx) else board[ny][nx]
            grid.append(val / 3.0)

    return np.array(wall + body + apple + [apple_dx, apple_dy] + dir_encoding + danger + grid + [length_norm], dtype=np.float32)

def get_action(model, game_state, epsilon=0.0):
    model.eval()
    if random.random() < epsilon:
        move = random.randint(0, 2)
    else:
        with torch.no_grad():
            features = torch.tensor(model.get_features(game_state), dtype=torch.float32).unsqueeze(0).to(device)
            prediction = model(features)
            move = torch.argmax(prediction).item()

    direction = game_state.direction
    return [direction, turn_left(direction), turn_right(direction)][move]


