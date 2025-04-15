import copy
import torch
import numpy as np

from Network.dqn_network import CustomNetwork
from Network.reward_manager import RewardManager
from Main.Config import INPUT_SIZE, OUTPUT_SIZE
from Game.snake_game import SnakeGame
from Game.screen_builder import create_monitor, update_monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvolutionEngine:
    def __init__(self, board_size=(40, 40), pop_size=100, generations=100, top_k=20):
        self.board_size = board_size
        self.pop_size = pop_size
        self.generations = generations
        self.top_k = top_k
        self.population = [CustomNetwork([INPUT_SIZE, 128, 64, OUTPUT_SIZE], softmax_output=True).to(device) for _ in range(pop_size)]
        self.starting_length = 3
        self.render = True
        self.step_limit = 40
        self.step_bonus = 80

    def evolve(self):
        monitor, label = create_monitor()
        scores = []

        for gen in range(self.generations):
            if gen % 5 == 0 and len(scores) >= self.pop_size:
                avg_len = sum(scores[-self.pop_size:]) / self.pop_size
                self.starting_length = int(min(avg_len, self.board_size[1] - 2))

            gen_scores = [self._evaluate_model(model, gen) for model in self.population]
            scores.extend(gen_scores)
            top_models = [m for _, m in sorted(zip(gen_scores, self.population), key=lambda x: x[0], reverse=True)[:self.top_k]]

            self.population = self._reproduce(top_models)
            best_score = max(gen_scores)
            update_monitor(gen + 1, best_score, label, monitor)
            print(f"Generation {gen + 1} | Best Score: {best_score}")

    def _evaluate_model(self, model, generation):
        game = SnakeGame(self.board_size, render=self.render)
        game.reset(length=self.starting_length)
        RM = RewardManager(generation)
        RM.reset_distance(game.head(), game.apple())

        steps = 0
        visited = set()
        recent_heads = []
        steps_since_apple = 0
        max_steps = self.step_limit

        while game.alive() and steps < max_steps:
            prev_len = game.length()
            direction = model.get_action(model, game.state(), epsilon=0.0)
            game.step(direction)

            visited.add(game.head())
            recent_heads.append(game.head())
            if len(recent_heads) > 20:
                recent_heads.pop(0)

            if game.length() > prev_len:
                steps_since_apple = 0
                max_steps += self.step_bonus
            else:
                steps_since_apple += 1

            steps += 1
            if self.render:
                monitor = game.master or game.tiles[0][0].master
                monitor.update()

            if steps_since_apple > 100:
                break

        return self._calculate_score(game, RM, steps, visited, recent_heads)

    def _calculate_score(self, game, RM, steps, visited, recent_heads):
        apples = game.length() - 3
        RM.apple_eaten = apples
        RM.update_distance(game.head(), game.apple())
        RM.total_reward += apples * len(visited) * 0.05
        RM.survival_bonus(steps)

        if len(recent_heads) > 0:
            repeats = len(recent_heads) - len(set(recent_heads))
            RM.loop_penalty(repeats)

        if not game.alive():
            RM.total_reward -= 10

        return RM.get_total()

    def _reproduce(self, top_models):
        new_population = top_models[:]
        while len(new_population) < self.pop_size - 10:
            p1, p2 = np.random.choice(top_models, 2, replace=False)
            child = p1.clone()
            with torch.no_grad():
                for param, w1, w2 in zip(child.parameters(), p1.parameters(), p2.parameters()):
                    mask = torch.rand_like(w1) < 0.5
                    param.copy_(torch.where(mask, w1, w2))
            child.mutate(0.3)
            new_population.append(child.to(device))

        while len(new_population) < self.pop_size:
            self.population = [CustomNetwork([INPUT_SIZE, 128, 64, OUTPUT_SIZE], softmax_output=True).to(device) for _ in range(self.pop_size)]

        return new_population
