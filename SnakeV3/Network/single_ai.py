import torch
import torch.nn.functional as F
import random
from Network.dqn_network import ConvNetwork, get_features
from Game.direction import turn_left, turn_right
from Main.Config import BOARD_SIZE, OUTPUT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleAI:
    def __init__(self):
        self.model = ConvNetwork(BOARD_SIZE, OUTPUT_SIZE).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = []
        self.buffer_limit = 5000
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.steps = 0

    def get_action(self, game_state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state = torch.tensor(get_features(game_state), dtype=torch.float32).unsqueeze(0).to(device)
            q_values, _ = self.model(state)
            return torch.argmax(q_values).item()

    def step_and_train(self, game, reward_manager):
        state = game.state()
        action_idx = self.get_action(state)
        direction = [state.direction, turn_left(state.direction), turn_right(state.direction)][action_idx]

        prev_len = game.length()
        reward_manager.update_distance(game.head(), game.apple())
        game.step(direction)

        if game.length() > prev_len:
            reward_manager.ate_apple()

        reward = reward_manager.get_total()
        reward_manager.total_reward = 0
        done = not game.alive()
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

        if len(self.replay_buffer) >= self.batch_size:
            self.train_step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.steps += 1
        return reward

    def train_step(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

        q_pred, recon = self.model(s)
        q_next, _ = self.model(s2)
        q_target = r + self.gamma * q_next.max(1)[0].unsqueeze(1) * (1 - d)

        recon_target = s[:, :-7].reshape(self.batch_size, 3, *BOARD_SIZE)
        recon_loss = F.mse_loss(recon, recon_target)
        q_loss = F.mse_loss(q_pred, q_target)

        loss = q_loss + 0.01 * recon_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
