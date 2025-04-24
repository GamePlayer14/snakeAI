import sys
import os

# Add SnakeV2 (the actual root where Game/, Main/, Network/ live) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tkinter as tk

from Game.snake_game import SnakeGame
from Game.controls import bind_controls
from Game.screen_builder import buildScreen
from Network.dqn_network import get_features
from Main.Config import BOARD_SIZE, USE_FULL_BOARD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OneLayerAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        flat_size = 3 * BOARD_SIZE[0] * BOARD_SIZE[1]  # 300 for 10x10

        self.encoder = nn.Sequential(
            nn.Flatten(),                # [B, 3, 10, 10] → [B, 300]
            nn.Linear(flat_size, 256),  # [B, 300] → [B, 256]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, flat_size),               # [B, 256] → [B, 300]
            nn.Sigmoid(),                            # keep output in [0, 1]
            nn.Unflatten(1, (3, *BOARD_SIZE))        # [B, 300] → [B, 3, 10, 10]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


    def extract_spatial(self, game):
        full = get_features(game.state())
        if USE_FULL_BOARD:
            spatial = full[:-7].reshape(1, 3, *BOARD_SIZE)
        else:
            raise ValueError("USE_FULL_BOARD must be True for autoencoder training.")
        return torch.tensor(spatial, dtype=torch.float32)


