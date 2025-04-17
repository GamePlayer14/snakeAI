import torch
import torch.nn.functional as F
import random
from Network.dqn_network import ConvNetwork, get_features
from Game.direction import turn_left, turn_right
from Main.Config import BOARD_SIZE, OUTPUT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleAI:
    def __init__(self):
        self.model = ConvNetwork(board_size=BOARD_SIZE, output_size=OUTPUT_SIZE, in_channels=3).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = []
        self.buffer_limit = 10000
        self.batch_size = 128
        self.gamma = 0.9
        self.epsilon = 0
        self.epsilon_min = 0
        self.epsilon_decay = 0.9995
        self.train_counter = 0

    def get_action(self, game_state):
        

        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            features = torch.tensor(get_features(game_state), dtype=torch.float32).unsqueeze(0).to(device)
            return torch.argmax(self.model(features)).item()

    def step_and_train(self, game, reward_manager):
        state = game.state()
        action_idx = self.get_action(state)
        direction = [state.direction, turn_left(state.direction), turn_right(state.direction)][action_idx]

        prev_len = game.length()
        reward_manager.update_distance(game.head(), game.apple())

        game.step(direction)

        if game.length() > prev_len:
            reward_manager.ate_apple()

        done = not game.alive()
        if done:
            reward_manager.death_penalty(len(state.snake), game.length())

        reward = reward_manager.get_total()
        reward_manager.total_reward = 0

        next_state = game.state()
        self.replay_buffer.append((
            get_features(state),
            action_idx,
            reward,
            get_features(next_state),
            done
        ))

        if len(self.replay_buffer) > self.buffer_limit:
            self.replay_buffer.pop(0)

        self.train_step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return reward

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 🔄 Sample from full buffer for better training diversity
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

        q_pred = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.model(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

