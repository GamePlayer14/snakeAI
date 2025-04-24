import tkinter as tk
import torch
from Game import screen_builder as sb
from Game.snake_game import SnakeGame
from Game.direction import turn_left, turn_right
from Network.reward_manager import RewardManager
from Network.single_ai import SingleAI
from Main.Config import BOARD_SIZE

class SnakeGameRunner:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.root = None
        self.tiles = None
        self.game = None
        self.reward_manager = None
        self.model = None
        self.step_counter = 0
        self.total_reward = 0
        self.total_apples = 0
        self.games_played = 0

    def setup_gui(self):
        self.root, self.tiles = sb.buildScreen(self.board_size)
        self.game = SnakeGame(self.board_size, render=True, tiles=self.tiles, master=self.root)

    def run_model(self):
        self.model = SingleAI()
        self.reward_manager = RewardManager(generation=0)
        self.game.reset()
        self.reward_manager.reset_distance(self.game.head(), self.game.apple())

        def loop():
            reward = self.model.step_and_train(self.game, self.reward_manager)

            if not self.game.alive():
                self.total_apples += self.game.length() - 3
                self.games_played += 1
                self.game.reset()
                self.reward_manager.reset_distance(self.game.head(), self.game.apple())

            self.step_counter += 1
            self.total_reward += reward

            if self.step_counter % 500 == 0:
                avg = self.total_apples / max(1, self.games_played)
                print(f"Step {self.step_counter} | Avg Apples: {avg:.2f} | Epsilon: {self.model.epsilon:.3f}")
                self.total_reward = 0
                self.total_apples = 0
                self.games_played = 0

            self.root.after(1, loop)

        tk.Button(self.root, text="Reset", command=self.game.reset).pack()
        loop()
        self.root.mainloop()

    def run(self):
        self.setup_gui()
        self.run_model()