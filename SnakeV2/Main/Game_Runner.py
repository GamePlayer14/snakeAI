import os
import tkinter as tk
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Game import screen_builder as sb
from Network.reward_manager import RewardManager
from Network.dqn_network import CustomNetwork, get_features
from Network.config import INPUT_SIZE, OUTPUT_SIZE
from Game.snake_game import SnakeGame
from Network.evolution_engine import EvolutionEngine
from Network.dqn_trainer import DQNTrainer, get_best_model_path
from Network.dqn_network import CustomNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LivePlotter:
    def __init__(self, root):
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, pady=10)
        self.steps = []
        self.rewards = []

    def update(self, step, reward, window=100):
        self.steps.append(step)
        self.rewards.append(reward)

        self.ax.cla()
        self.ax.plot(self.steps[-window:], self.rewards[-window:], label="Reward", color='green')
        self.ax.set_xlim(self.steps[-window] if len(self.steps) > window else 0, self.steps[-1])
        self.ax.set_title("Live Reward")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


class SnakeGameRunner:
    def __init__(self, board_size=(40, 40), mode='manual', model_path='dqn_snake.pth'):
        self.board_size = board_size
        self.mode = mode
        self.model_path = model_path

        self.root = None
        self.tiles = None
        self.game = None
        self.model = None
        self.reward_manager = None
        self.plotter = None
        self.step_counter = 0

    def setup_gui(self):
        self.root, self.tiles = sb.buildScreen(self.board_size)
        self.game = SnakeGame(self.board_size, render=True, tiles=self.tiles, master=self.root)

    def load_model(self):
        if not os.path.exists(self.model_path):
            print("Model not found. Training from scratch...")
            DQNTrainer().train()

        self.model = CustomNetwork([INPUT_SIZE, 512, 256, 128, OUTPUT_SIZE]).to(device)
        best_path = get_best_model_path()
        self.model.load_state_dict(torch.load(best_path, map_location=device))
        self.model.eval()

    def run_manual(self):
        self.game.bind_controls(self.root)

        def loop():
            try:
                self.game.step()
                self.root.after(100, loop)
            except tk.TclError:
                print("Window closed. Manual game ended.")

        self.game.reset()
        tk.Button(self.root, text="Reset", command=self.game.reset).pack()
        loop()
        self.root.mainloop()

    def run_model(self):
        self.load_model()
        self.reward_manager = RewardManager(generation=0)
        self.reward_manager.reset_distance(self.game.head(), self.game.apple())
        self.plotter = LivePlotter(self.root)

        def loop():
            try:
                direction = self.get_action(self.model, self.game.state(), epsilon=0.0)
                self.game.step(direction)

                self.step_counter += 1
                self.reward_manager.update_distance(self.game.head(), self.game.apple())

                if not self.game.alive():
                    self.reward_manager.loop_penalty(10)

                reward = self.reward_manager.get_total()
                self.reward_manager.total_reward = 0

                if self.step_counter % 10 == 0:
                    self.plotter.update(self.step_counter, reward)

                self.root.after(100, loop)
            except tk.TclError:
                print("Window closed. Model game ended.")

        self.game.reset()
        tk.Button(self.root, text="Reset", command=self.game.reset).pack()
        loop()
        self.root.mainloop()

    def run(self):
        if self.mode == 'evolve':
            EvolutionEngine(self.board_size).evolve()
        else:
            self.setup_gui()
            if self.mode == 'model':
                self.run_model()
            else:
                self.run_manual()
