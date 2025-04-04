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
import os
from config import INPUT_SIZE, OUTPUT_SIZE

# --- Hyperparameters ---
BOARD_SIZE = (40, 40)

GAMMA = 0.9
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MAX_EPISODES = 50000
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.00

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay Buffer ---
memory = deque(maxlen=MEMORY_SIZE)

def get_reward(prev_len, new_len, alive):
    if not alive:
        return -10
    if new_len > prev_len:
        return 50  # Ate an apple
    return -0.1   # Time cost

def train():
    MODEL_DIR = "saved models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    TOP_N = 10
    best_score = 0
    top_models = []
    print("Running on:", device)

    model = DQN(INPUT_SIZE, OUTPUT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    epsilon = EPSILON_START
    episode_scores = []
    
    episodes_no_improve = 0
    PATIENCE = 500  
    mx = MAX_EPISODES
    for episode in range(1, MAX_EPISODES + 1):

        RM = RewardManager(generation = episode)

        gs = GameState(None, BOARD_SIZE, load_sprites(False), render=False)
        gs.reset()

        RM.reset_distance(gs.snake[0], gs.apple)

        total_reward = 0
        steps = 0

        while gs.alive:
            state = get_features(gs)
            features = torch.tensor(get_features(gs), dtype=torch.float32).unsqueeze(0).to(device)

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

                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

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

        score = len(gs.snake)

        if score > best_score:
            model_filename = f"model_score{score}_ep{episode}.pth"
            model_path = os.path.join(MODEL_DIR, model_filename)
            torch.save(model.cpu().state_dict(), model_path)

            model.to(device)

            top_models.append((score, model_path))
            top_models.sort(reverse=True)  # highest score first

            # Keep only top N
            if len(top_models) > TOP_N:
                _, path_to_remove = top_models.pop()  # remove the worst
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    print(f"🗑️ Removed old model: {path_to_remove}")

            print(f"💾 Saved new top model: {model_path}")




        score = len(gs.snake)
        if score > best_score:
            best_score = score
            episodes_no_improve = 0
            print('interest reset')
        else:
            episodes_no_improve += 1

        if episodes_no_improve >= PATIENCE:
            print(f"🛑 Early stopping: No improvement for {PATIENCE} episodes.")
            episode = MAX_EPISODES


        print(f"Ep {episode} | Score: {len(gs.snake)} | Steps: {steps} | Eps: {epsilon:.3f}")
    
    with open(os.path.join(MODEL_DIR, "top_models.txt"), "w") as f:
        for score, m in top_models:
            f.write(f"{score},{m}\n")


    torch.save(top_models[0].cpu().state_dict(), "dqn_snake.pth")
    model.to(device)
    print("✅ Training complete. Model saved to dqn_snake.pth")

def get_direction_from_idx(idx, current_direction):
    if idx == 0:
        return current_direction
    elif idx == 1:
        return turn_left(current_direction)
    else:
        return turn_right(current_direction)
    
def get_best_model_path(model_dir="saved_models/top_models.txt"):
    if not os.path.exists(model_dir):
        return "dqn_snake.pth"  # fallback

    with open(model_dir, "r") as f:
        lines = f.readlines()

    if not lines:
        return "dqn_snake.pth"

    # Line format: score, path
    best_path = lines[0].strip().split(",")[1] if "," in lines[0] else lines[0].strip()
    return best_path

    
if __name__ == "__main__":
    train()
