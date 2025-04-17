import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Network.dqn_network import ConvNetwork, get_features
from Network.reward_manager import RewardManager
from Main.Config import INPUT_SIZE, OUTPUT_SIZE, BOARD_SIZE
from Game.snake_game import SnakeGame
from Game.direction import turn_left, turn_right


class DQNTrainer:
    def __init__(self, board_size=BOARD_SIZE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_size = board_size
        self.model = ConvNetwork([BOARD_SIZE, OUTPUT_SIZE]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)

        self.gamma = 0.9
        self.batch_size = 128
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.max_episodes = 50000
        self.patience = 500

    def train(self):
        os.makedirs("saved models", exist_ok=True)
        best_score = 0
        top_models = []
        recent_lengths = deque(maxlen=50)
        starting_length = 3
        episodes_no_improve = 0

        for episode in range(self.max_episodes + 1):
            score, steps, improved = self._run_episode(episode, starting_length, best_score)

            if improved:
                best_score = score
                episodes_no_improve = 0
                path = f"saved models/model_score{score}_ep{episode}.pth"
                torch.save(self.model.cpu().state_dict(), path)
                self.model.to(self.device)
                top_models.append((score, path))
                top_models.sort(reverse=True)
                if len(top_models) > 10:
                    _, old_path = top_models.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)
                print(f"\U0001F4BE Saved new top model: {path}")
            else:
                episodes_no_improve += 1

            if episodes_no_improve >= self.patience:
                print(f"\U0001F6D1 Early stopping at episode {episode}.")
                break

            recent_lengths.append(score)
            if episode % 20 == 0 and recent_lengths:
                avg_len = sum(recent_lengths) / len(recent_lengths)
                starting_length = int(avg_len)

            print(f"Ep {episode} | Score: {score} | Steps: {steps} | Eps: {self.epsilon:.3f}")

        torch.save(self.model.cpu().state_dict(), "dqn_snake.pth")
        with open("saved models/top_models.txt", "w") as f:
            for score, path in top_models:
                f.write(f"{score},{path}\n")
        print("\u2705 Training complete. Model saved to dqn_snake.pth")

    def _run_episode(self, episode, starting_length, top_score):
        RM = RewardManager(generation=episode)
        game = SnakeGame(self.board_size, render=False)
        game.reset(length=starting_length)

        RM.reset_distance(game.head(), game.apple())
        steps = 0
        total_reward = 0
        best_score = top_score
        alive = True

        while alive:
            move_idx = self._select_action(game)
            direction = [game.state().direction,
                         turn_left(game.state().direction),
                         turn_right(game.state().direction)][move_idx]

            prev_len = game.length()
            RM.update_distance(game.head(), game.apple())
            game.step(direction)

            if game.length() > prev_len:
                RM.ate_apple()

            if not game.alive():
                RM.death_penalty(steps, game.length())
                alive = False

            # self._store_experience(game, move_idx, RM)
            total_reward += RM.get_total()
            RM.total_reward = 0
            steps += 1

            if len(self.memory) >= self.batch_size:
                self._optimize_model()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        score = game.length()
        return score, steps, score > best_score

    def _select_action(self, game):
        state = get_features(game.state())
        features = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            return torch.argmax(self.model(features)).item()

    def _store_experience(self, game, move_idx, RM):
        state = get_features(game.state())
        next_state = get_features(game.state())
        reward = RM.get_total()
        done = not game.alive()
        self.memory.append((state, move_idx, reward, next_state, done))

    def _optimize_model(self):
        s, a, r, s2, d = zip(*random.sample(self.memory, self.batch_size))
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(s).gather(1, a)
        with torch.no_grad():
            q_target = r + self.gamma * self.model(s2).max(1)[0].unsqueeze(1) * (1 - d)

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_best_model_path(model_file="saved models/top_models.txt"):
    if not os.path.exists(model_file):
        return "dqn_snake.pth"
    with open(model_file, "r") as f:
        lines = f.readlines()
    return lines[0].strip().split(",")[1] if lines else "dqn_snake.pth"