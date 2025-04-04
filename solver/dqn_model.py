import torch
import torch.nn as nn
import random
from snake_ai_model import get_features
from game.direction import turn_left, turn_right

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.output
        )
        for layer in [self.fc1, self.fc2, self.output]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

def get_action(model, game_state, epsilon=0.0):
    model.eval()

    if random.random() < epsilon:
        move = random.randint(0, 2)
    else:
        with torch.no_grad():
            features = torch.tensor(get_features(game_state), dtype=torch.float32).unsqueeze(0)
            prediction = model(features)
            print("Model Q-values (in get_action):", prediction.detach().numpy())   
            move = torch.argmax(prediction).item()
            print("Selected move:", move)

    current_direction = game_state.direction
    if move == 0:
        return current_direction
    elif move == 1:
        return turn_left(current_direction)
    else:
        return turn_right(current_direction)
