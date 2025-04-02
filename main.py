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

def run_snake_game(play_game: bool):
    # Headless Evolution Monitor
    if not play_game:
        root = tk.Tk()
        root.title("Evolution Monitor")
        info_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
        info_label.pack()
        evo.evolve_population_with_monitor(info_label, root)
        return

    # Game Mode (with GUI)
    tiles = None
    board_size = (40, 40)

    if play_game:
        root, tiles = sb.buildScreen(*board_size)
        sprites = load_sprites(play_game, master=root)
    else:
        root = tk.Tk()
        root.withdraw()
        sprites = load_sprites(play_game, master=root)

    gs = GameState(tiles, board_size, sprites, play_game)

    # def headless_loop():
    #     generation = 1
    #     best_score = 0
    #     gs.reset()

    #     while gs.alive:
    #         gs.step()
    #         score = len(gs.snake)
    #         if score > best_score:
    #             best_score = score
    #         sb.update_monitor(generation, best_score, info_label, root)

    def game_loop():
        try:
            gs.step()
            root.after(100, game_loop)
        except tk.TclError:
            print("Window closed. Stopping game.")

    if play_game:
        bind_controls(root, gs)
        gs.reset()
        reset_button = tk.Button(root, text="Reset", command=gs.reset)
        reset_button.pack()
        game_loop()

    root.mainloop()