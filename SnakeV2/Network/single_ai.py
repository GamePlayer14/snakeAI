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
        self.buffer_limit = 5000
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.01
        self.epsilon_min = 0
        self.epsilon_decay = 0.9995
        self.train_counter = 0
        self.steps = 0
        self.steps_since_apple = 0
        self.last_idx = 1
        self.last_game_experiences = []
        self.game_reward = 0
        

    def get_action(self, game_state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            features = torch.tensor(get_features(game_state), dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.model(features)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

            # 🌀 Add noise if too confident
            if entropy.item() < 0.1:
                logits += torch.randn_like(logits) * 0.2

            move = torch.argmax(logits).item()
            return move



    def step_and_train(self, game, reward_manager):
        state = game.state()
        action_idx = self.get_action(state)
        direction = [state.direction, turn_left(state.direction), turn_right(state.direction)][action_idx]
        reward_manager.line_reward(action_idx, self.last_idx)
        prev_len = game.length()
        reward_manager.update_distance(game.head(), game.apple())

        game.step(direction)
        self.steps += 1
        self.steps_since_apple += 1

        if game.head() in reward_manager.seen_squares:
            game.state().alive = False

        if game.length() > prev_len:
            reward_manager.ate_apple()
            self.steps_since_apple = 0
        
        reward_manager.survival_bonus(self.steps)
        reward_manager.hunger_penalty(self.steps_since_apple)

        done = not game.alive()
        steps_survived = self.steps
        importance = max(0, -30 / steps_survived + 3)
        
        if done:
            reward_manager.death_penalty()
            self.steps = 0
            self.steps_since_apple = 0
            reward = reward_manager.get_total()
            self.game_reward += reward
            print(f"[Episode] Steps: {steps_survived} | Total Reward: {self.game_reward}")
            self.game_reward = 0
        else:
            reward = reward_manager.get_total()
            self.game_reward += reward

        reward = reward_manager.get_total()
        
        # Optional: stronger penalty for confident wrong decisions
        with torch.no_grad():
            features = torch.tensor(get_features(state), dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.model(features)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

        if done and reward < -5 and entropy.item() < 0.2:
            reward -= 5
            for _ in range(2):
                self.replay_buffer.append((
                    get_features(state),
                    action_idx,
                    reward,
                    get_features(game.state()),
                    done,
                    importance
                ))

        reward_manager.total_reward = 0

        next_state = game.state()
        importance = 2.0 if reward >= 100 or done else 1.0  # prioritize apples & deaths

        move = (
            get_features(state),
            action_idx,
            reward,
            get_features(next_state),
            done,
            importance
        )
        self.replay_buffer.append(move)
        self.last_game_experiences.append(move)


        if len(self.replay_buffer) > self.buffer_limit:
            self.replay_buffer.pop(0)
        
        if done:
            for x in range(3):
                self.train_on_last_game()


        if self.steps % 10 == 0:
            self.train_step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.last_idx = direction
        return reward

    def train_step(self):
        if len(self.replay_buffer) < self.buffer_limit / 10:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        for i, (s, a, r, s2, d, imp) in enumerate(batch):
            # Use imp during loss weighting...

            # Subtract 1 from importance
            imp -= 1

            # Update buffer if importance > 0, otherwise remove
            if imp > 0:
                self.replay_buffer[i] = (s, a, r, s2, d, imp)
            else:
                self.replay_buffer.pop(i)
        s, a, r, s2, d, w = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)
        w = torch.tensor(w, dtype=torch.float32).unsqueeze(1).to(device)

        # Forward Q-values and targets
        q_pred = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.model(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        # 🔥 Entropy bonus to encourage diverse action selection
        with torch.no_grad():
            q_all = self.model(s)
            probs = F.softmax(q_all, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # shape: [batch, 1]
        
        # Scale and add entropy to target reward (alternatively add to q_target directly)
        q_target += 0.01 * entropy  # small bonus for uncertainty

        # Loss with weights
        loss = F.mse_loss(q_pred, q_target, reduction='none')
        loss = (loss * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train_on_last_game(self):
        if len(self.last_game_experiences) < 10:
            return  # not enough data

        batch = self.last_game_experiences

        s, a, r, s2, d, w = zip(*batch)
        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)
        w = torch.tensor(w, dtype=torch.float32).unsqueeze(1).to(device)

        q_pred = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.model(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        loss = F.mse_loss(q_pred, q_target, reduction='none')
        loss = (loss * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_game_experiences.clear()


