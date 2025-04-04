import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from game.direction import turn_left, turn_right
from dqn_model import DQN, get_action
from snake_ai_model import get_features
from game.game_state import GameState
from game.sprite_loader import load_sprites
from reward_manager import RewardManager

# --- Hyperparameters ---
BOARD_SIZE = (40, 40)
INPUT_SIZE = 15   # or 15 if you're using the simplified get_features
OUTPUT_SIZE = 3

GAMMA = 0.9
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MAX_EPISODES = 500
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.00


# --- Replay Buffer ---
memory = deque(maxlen=MEMORY_SIZE)

def get_reward(prev_len, new_len, alive):
    if not alive:
        return -10
    if new_len > prev_len:
        return 50  # Ate an apple
    return -0.1   # Time cost

def train():
    model = DQN(INPUT_SIZE, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    epsilon = EPSILON_START
    episode_scores = []
    
    
    for episode in range(1, MAX_EPISODES + 1):

        RM = RewardManager(generation = episode)

        gs = GameState(None, BOARD_SIZE, load_sprites(False), render=False)
        gs.reset()

        RM.reset_distance(gs.snake[0], gs.apple)

        total_reward = 0
        steps = 0

        while gs.alive:
            state = get_features(gs)
            features = torch.tensor(get_features(gs), dtype=torch.float32).unsqueeze(0)

            if random.random() < epsilon:
                move_idx = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = model(features)
                    move_idx = torch.argmax(q_values).item()

            direction = get_direction_from_idx(move_idx, gs.direction)

            prev_len = len(gs.snake)
            prev_head = gs.snake[0]

            RM.update_distance(prev_head, gs.apple)

            gs.set_direction(direction)
            gs.step()
            
            # if random.random() < epsilon:
            #     move_idx = random.randint(0, 2)
            # else:
            #     with torch.no_grad():
            #         state_tensor = torch.tensor(get_features(gs), dtype=torch.float32).unsqueeze(0)
            #         q_values = model(state_tensor)
            #         move_idx = torch.argmax(q_values).item()

            # # Apply the actual direction using move_idx
            # direction = get_direction_from_idx(move_idx, gs.direction)  # <- you'll define this function
            # gs.set_direction(direction)
            # gs.step()
            if len(gs.snake) > prev_len:
                RM.ate_apple()

            next_state = get_features(gs)
            reward = RM.get_total()
            RM.total_reward = 0
            done = not gs.alive

            memory.append((state, move_idx, reward, next_state, done))
            total_reward += reward
            steps += 1

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                s, a, r, s2, d = zip(*batch)

                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                q_values = model(s).gather(1, a)
                with torch.no_grad():
                    q_next = model(s2).max(1)[0].unsqueeze(1)
                    q_target = r + GAMMA * q_next * (1 - d)

                loss = loss_fn(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_scores.append(len(gs.snake))

        print(f"Ep {episode} | Score: {len(gs.snake)} | Steps: {steps} | Eps: {epsilon:.3f}")

    torch.save(model.state_dict(), "dqn_snake.pth")
    print("✅ Training complete. Model saved to dqn_snake.pth")

def get_direction_from_idx(idx, current_direction):
    if idx == 0:
        return current_direction
    elif idx == 1:
        return turn_left(current_direction)
    else:
        return turn_right(current_direction)
    
if __name__ == "__main__":
    train()
