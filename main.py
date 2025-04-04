import game.screen_builder as sb
from game.game_state import GameState
from game.direction import DIRECTION_DELTAS, EAST, NORTH, SOUTH, WEST
from game.controls import bind_controls
from game.sprite_loader import load_sprites
import tkinter as tk
import threading
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'solver')))
import solver.snake_ai_model as ss
import solver.evolve as evo
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
import matplotlib.pyplot as plt # type: ignore
from solver.reward_manager import RewardManager
import torch
from solver.train_dqn import get_best_model_path
from solver.config import INPUT_SIZE, OUTPUT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_snake_game(mode: str, board_size, model_path = 'dqn_snake.pth'):
    if mode == "evolve":
        root = tk.Tk()
        root.title("Evolution Monitor")
        info_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
        info_label.pack()
        evo.evolve_population_with_monitor(info_label, root, board_size)
        return

    # GUI + Model / Manual mode
    tiles = None
    root, tiles = sb.buildScreen(board_size)
    sprites = load_sprites(True, master=root)
    gs = GameState(tiles, board_size, sprites, True)

    if mode == "model":
        from solver.dqn_model import DQN, get_action
        import torch

        RM = RewardManager(generation=0)
        RM.reset_distance(gs.snake[0], gs.apple)

        #  Train if model is missing
        if not os.path.exists(model_path):
            print("Model not found. Training from scratch...")
            from solver.train_dqn import train
            train()  # this should save dqn_snake.pth

        model = DQN(INPUT_SIZE, OUTPUT_SIZE).to(device)
        best_model_path = get_best_model_path()
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        root = tk.Tk()
        root.title("Snake Game with Live Reward")

        # Reward figure and canvas
        reward_fig, reward_ax = plt.subplots(figsize=(4, 3))
        reward_canvas = FigureCanvasTkAgg(reward_fig, master=root)
        reward_canvas_widget = reward_canvas.get_tk_widget()
        reward_canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, pady=10)

        reward_steps = []
        reward_values = []

        def update_reward_plot(step, reward, window = 100):
            reward_steps.append(step)
            reward_values.append(reward)

            reward_ax.cla()  # faster clear
            reward_ax.plot(reward_steps[-window:], reward_values[-window:], label="Reward", color='green')


            # Limit the x-axis to a rolling window
            if len(reward_steps) > window:
                reward_ax.set_xlim(reward_steps[-window], reward_steps[-1])
            else:
                reward_ax.set_xlim(0, window)

            reward_ax.set_title("Live Reward")
            reward_ax.set_xlabel("Step")
            reward_ax.set_ylabel("Reward")
            reward_ax.legend()
            reward_ax.grid(True)
            reward_canvas.draw()

        # Optional: Game canvas or board view goes on the LEFT side (your tile canvas)
        # tile_canvas.pack(side=tk.LEFT)
        step_counter = 0
        def game_loop():
            nonlocal step_counter
            try:
                direction = get_action(model, gs, epsilon=0.0)
                gs.set_direction(direction)
                prev_len = len(gs.snake)
                gs.step()

                step_counter += 1

                head = gs.snake[0]
                apple = gs.apple

                RM.update_distance(head, apple)

                if len(gs.snake) > prev_len:
                    RM.ate_apple()

                if not gs.alive:
                    RM.loop_penalty(10)

                current_reward = RM.get_total()
                RM.total_reward = 0
                if step_counter %10 == 0:
                    update_reward_plot(step_counter, current_reward)
                    
                root.after(100, game_loop)
            except tk.TclError:
                print("Window closed. Stopping game.")
    else:  # manual play
        bind_controls(root, gs)
        def game_loop():
            try:
                gs.step()
                root.after(100, game_loop)
            except tk.TclError:
                print("Window closed. Stopping game.")

    gs.reset()
    reset_button = tk.Button(root, text="Reset", command=gs.reset)
    reset_button.pack()
    game_loop()
    root.mainloop()
