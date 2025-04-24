import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Main.Config import BOARD_SIZE
from Network.pathfinding_heatmap import compute_path_distances

class ConvNetwork(nn.Module):
    def __init__(self, board_size, output_size):
        super().__init__()
        self.in_channels = 3
        flat_size = self.in_channels * board_size[0] * board_size[1]

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, flat_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        board = x[:, :-7].reshape(x.size(0), self.in_channels, *BOARD_SIZE)
        extras = x[:, -7:]
        encoded = self.encoder(board)
        q_values = self.fc(torch.cat([encoded, extras], dim=1))
        recon = self.decoder(encoded).reshape(x.size(0), self.in_channels, *BOARD_SIZE)
        return q_values, recon

def get_features(game_state):
    height, width = BOARD_SIZE

    def is_blocked_snake(y, x):
        return False

    snake_dist = compute_path_distances(game_state.snake[0], is_blocked_snake, BOARD_SIZE)
    snake_layer = 1.0 / (snake_dist + 1)
    snake_layer[np.isinf(snake_layer)] = 0.0

    def is_snake(y, x):
        return (y, x) in game_state.snake

    apple_dist = compute_path_distances(game_state.apple, is_snake, BOARD_SIZE)
    apple_layer = 1.0 / (apple_dist + 1)
    apple_layer[np.isinf(apple_layer)] = 0.0

    trail_layer = game_state.trail_map.copy()

    board_stack = np.stack([snake_layer, apple_layer, trail_layer], axis=0)
    flat_board = board_stack.flatten()

    direction = game_state.direction
    dir_encoding = [0.0] * 4
    dir_encoding[direction] = 1.0

    hx, hy = game_state.snake[0][1], game_state.snake[0][0]
    ax, ay = game_state.apple[1], game_state.apple[0]
    apple_dx = (ax - hx) / width
    apple_dy = (ay - hy) / height
    length_norm = len(game_state.snake) / (height * width)

    extras = dir_encoding + [length_norm, apple_dx, apple_dy]
    return np.array(list(flat_board) + extras, dtype=np.float32)
