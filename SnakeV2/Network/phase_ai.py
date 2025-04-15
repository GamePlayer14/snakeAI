import torch
import torch.nn.functional as F
import numpy as np
import random
from Network.dqn_network import ConvNetwork
from Network.dqn_network import get_features
from Game.direction import turn_left, turn_right
from Main.Config import INPUT_SIZE, OUTPUT_SIZE, BOARD_SIZE

class PhaseAI:
    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, model_paths=None):
        self.hunger = 0
        self.hunger_limit = 200
        self.recent_rewards = []  # for smoothing live reward
        self.reward_window = 10
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.replay_buffer = []
        self.buffer_limit = 10000
        self.batch_size = 64
        self.recent_rewards = []  # for smoothing live reward
        self.reward_window = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size

        self.models = {
            "early": ConvNetwork(board_size=BOARD_SIZE, output_size=output_size).to(self.device),
            "mid": ConvNetwork(board_size=BOARD_SIZE, output_size=output_size).to(self.device),
            "late": ConvNetwork(board_size=BOARD_SIZE, output_size=output_size).to(self.device)
        }

        self.optimizers = {
            phase: torch.optim.Adam(model.parameters(), lr=0.001)
            for phase, model in self.models.items()
        }

        if model_paths:
            self.load_models(model_paths)

    def load_models(self, paths_or_file):
        if isinstance(paths_or_file, dict):
            for phase, path in paths_or_file.items():
                self.models[phase].load_state_dict(torch.load(path, map_location=self.device))
        elif isinstance(paths_or_file, str):
            self.load_all(paths_or_file)

    def load_all(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        for phase, state in checkpoint.items():
            self.models[phase].load_state_dict(state)

    def save_all(self, filepath):
        torch.save({phase: model.state_dict() for phase, model in self.models.items()}, filepath)

    def get_action(self, game_state, epsilon=0.1):
        if random.random() < epsilon:
            move = random.randint(0, 2)
        else:
            features = torch.tensor(get_features(game_state), dtype=torch.float32).unsqueeze(0).to(self.device)
            progress = len(game_state.snake) / (game_state.board_size[0] * game_state.board_size[1])

            weights = {
                "early": 1 - progress,
                "mid": 1 - abs(0.5 - progress) * 2,
                "late": progress
            }

            q_sum = torch.zeros(1, self.output_size).to(self.device)
            for phase, model in self.models.items():
                q_sum += model(features) * weights[phase]

            move = torch.argmax(q_sum).item()

        # Prevent repeating the same move too often (spin detection)
        if hasattr(self, 'prev_moves'):
            self.prev_moves.append(move)
            if len(self.prev_moves) > 8:
                self.prev_moves.pop(0)
        else:
            self.prev_moves = [move]

        # If it's repeating the same move too much, force a random one
        if self.prev_moves.count(self.prev_moves[-1]) > 6:
            move = random.choice([m for m in [0, 1, 2] if m != move])

        direction = game_state.direction
        return [direction, turn_left(direction), turn_right(direction)][move], move

    def train_step(self, gamma=0.9):
        if len(self.replay_buffer) < self.batch_size:
            return

        recent_data = self.replay_buffer[-128:]
        batch = random.sample(recent_data, min(self.batch_size, len(recent_data)))
        for state, action_idx, reward, next_state, done, progress in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([[action_idx]], dtype=torch.int64).to(self.device)
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor([[done]], dtype=torch.float32).to(self.device)

            weights = {
                "early": 1 - progress,
                "mid": 1 - abs(0.5 - progress) * 2,
                "late": progress
            }

            for phase, model in self.models.items():
                model.train()
                optimizer = self.optimizers[phase]

                q_pred = model(state_tensor).gather(1, action_tensor)
                with torch.no_grad():
                    q_next = model(next_tensor).max(1)[0].unsqueeze(1)
                    q_target = reward_tensor + gamma * q_next * (1 - done_tensor)

                loss = F.mse_loss(q_pred, q_target) * weights[phase]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def step_and_train(self, game, rm):
        prev_state = game.state()
        prev_len = rm.apple_eaten
        direction, move_idx = self.get_action(prev_state, self.epsilon)
        self.hunger += 1
        game.step(direction)
        next_state = game.state()
        reward = 0

        rm.update_distance(game.head(), game.apple())
        if rm.apple_eaten > prev_len:
            rm.ate_apple()
            self.hunger = 0  # reset hunger when eating

        done = not game.alive()
        # if self.hunger >= self.hunger_limit:
        #     done = True
        #     reward -= 25  # penalty for starving

        if done:
            rm.death_penalty(len(prev_state.snake), game.length())

        reward += rm.get_total()
        rm.total_reward = 0

        state_array = get_features(prev_state)
        next_array = get_features(next_state)
        progress = len(prev_state.snake) / (prev_state.board_size[0] * prev_state.board_size[1])

        self.replay_buffer.append((state_array, move_idx, reward, next_array, done, progress))
        if len(self.replay_buffer) > self.buffer_limit:
            self.replay_buffer.pop(0)

        if not hasattr(self, 'train_counter'):
            self.train_counter = 0
        self.train_counter += 1
        if self.train_counter % 128 == 0:
            self.train_step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window:
            self.recent_rewards.pop(0)
        return sum(self.recent_rewards) / len(self.recent_rewards)
