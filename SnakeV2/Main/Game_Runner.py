import numpy as np
import tkinter as tk
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Game import screen_builder as sb
from Network.reward_manager import RewardManager
from Network.phase_ai import PhaseAI
from Main.Config import INPUT_SIZE, OUTPUT_SIZE, SQUARE_COUNT
from Game.snake_game import SnakeGame
from Network.evolution_engine import EvolutionEngine
from Network.dqn_network import get_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LivePlotter:
    def __init__(self, root):
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.ax2 = self.ax.twinx()  # second Y-axis
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, pady=10)

        self.steps = []
        self.rewards = []
        self.avg_apples = []


    def update(self, step, reward, avg_apples, window=100):
        self.steps.append(step)
        self.rewards.append(reward)
        self.avg_apples.append(avg_apples)

        x = np.array(self.steps[-window:])
        y1 = np.array(self.rewards[-window:])
        y2 = np.array(self.avg_apples[-window:])

        self.ax.cla()
        self.ax2.cla()

        # Plot raw data (faint lines)
        self.ax.plot(x, y1, label="Reward", color='green', alpha=0.3)
        self.ax2.plot(x, y2, label="Avg Apples", color='orange', alpha=0.3)

        # Plot trend lines (polynomial regression)
        if len(x) >= 5:
            reward_trend = np.poly1d(np.polyfit(x, y1, 2))
            apple_trend = np.poly1d(np.polyfit(x, y2, 2))
            self.ax.plot(x, reward_trend(x), label="Reward Trend", color='green', linewidth=2)
            self.ax2.plot(x, apple_trend(x), label="Apple Trend", color='orange', linewidth=2)

        self.ax.set_title("Live Reward")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.ax.legend(loc="upper left")

        self.ax2.set_ylabel("Avg Apples")
        self.ax2.legend(loc="upper right")

        self.canvas.draw()



class SnakeGameRunner:
    
    def __init__(self, board_size=(40, 40), mode='manual', model_path='dqn_snake.pth'):
        self.board_size = board_size
        self.mode = mode
        self.model_path = model_path

        self.root = None
        self.tiles = None
        self.game = None
        self.reward_manager = None
        self.plotter = None
        self.step_counter = 0
        self.total_reward = 0
        self.total_apples = 0
        self.games_played = 0

    def setup_gui(self):
        self.root, self.tiles = sb.buildScreen(self.board_size)
        self.game = SnakeGame(self.board_size, render=True, tiles=self.tiles, master=self.root)

    def load_model(self):
        self.model = PhaseAI()

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
            while True:
                reward = self.model.step_and_train(self.game, self.reward_manager)

                # handle reset logic, counters, plotting...
                if not self.game.alive():
                    self.total_apples += self.game.length() - 3  # apples eaten
                    self.games_played += 1
                    self.reward_manager.loop_penalty(10)
                    self.game.reset(length = SQUARE_COUNT//2)
                    self.reward_manager.reset_distance(self.game.head(), self.game.apple())
                    

                self.step_counter += 1
                self.reward_manager.update_distance(self.game.head(), self.game.apple())

                self.total_reward += reward
                
                if self.step_counter % 1000 == 0:
                    avg = self.total_apples / min(1,self.games_played)
                    print(self.model.epsilon)
                    self.plotter.update(self.step_counter, self.total_reward/1000, avg)
                    self.total_reward = 0
                    self.total_apples = 0
                    self.games_played = 0


                self.root.update()  # FAST manual screen refresh


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
