gs = None
vis_root = None
tiles = None
sprites = None

import tkinter as tk
import threading
import numpy as np
import tensorflow as tf
import random

from game.game_state import GameState
from game.sprite_loader import load_sprites
from game.screen_builder import buildScreen
import snake_ai_model as ss

# Toggle rendering ON/OFF
RENDER_MODE = True
last_best = None

def mutate_weights(weights, mutation_rate=0.1):
    new_weights = []
    for w in weights:
        noise = np.random.randn(*w.shape) * mutation_rate
        new_weights.append(w + noise)
    return new_weights

def evaluate_game(gs, steps, recent_heads=None, visited_tiles=None, turn_history=None):
    visited = visited_tiles if visited_tiles else set(gs.snake)
    score = 20
    apples_eaten = len(gs.snake) - 3
    if not gs.alive:
        score -= 5
    score += apples_eaten * 30
    head_y, head_x = gs.snake[0]
    apple_y, apple_x = gs.apple
    distance = abs(head_y - apple_y) + abs(head_x - apple_x)
    score += (20-distance) * 0.2
    score += len(visited) * 0.4
    if recent_heads:
        unique_recent = len(set(recent_heads))
        loop_penalty = (len(recent_heads) - unique_recent) * 1
        score -= loop_penalty
    return score

def run_model(model, max_steps=0, generation=1):
    global gs, vis_root, sprites, tiles

    if RENDER_MODE:
        if gs is None:
            vis_root = tk.Tk()
            vis_root.title("Snake Viewer")
            vis_root.geometry("800x800")
            vis_root.lift()
            vis_root.attributes('-topmost', True)
            vis_root.after_idle(vis_root.attributes, '-topmost', False)
            vis_root, tiles = buildScreen(40, 40)
            sprites = load_sprites(True, master=vis_root)
            gs = GameState(tiles, (40, 40), sprites, render=True)
        else:
            gs.reset()
    else:
        sprites = load_sprites(False)
        gs = GameState(None, (40, 40), sprites, render=False)
        gs.reset()

    steps = 0
    visited_tiles = set()
    recent_heads = []
    if max_steps is None or max_steps <= 0:
        max_steps = min(500, 40 + (20 * generation))

    while gs.alive and steps < max_steps:
        direction = ss.get_action(model, gs, (40,40))
        gs.set_direction(direction)
        gs.step()
        recent_heads.append(gs.snake[0])
        if len(recent_heads) > 20:
            recent_heads.pop(0)
        visited_tiles.add(gs.snake[0])
        steps += 1
        if RENDER_MODE and vis_root:
            vis_root.update()

    final_score = evaluate_game(gs, steps, recent_heads, visited_tiles, gs.turn_history if hasattr(gs, 'turn_history') else None)
    print("[DEBUG] run_model done with score:", final_score)
    return final_score

def evolve_with_monitor():
    root = tk.Tk()
    root.title("Evolution Monitor")
    root.geometry("400x200")

    info_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
    info_label.pack()

    canvas = tk.Canvas(root, width=300, height=50, bg='white')
    canvas.pack(pady=10)
    bar = canvas.create_rectangle(0, 0, 0, 50, fill='green')

    def background():
        print("[DEBUG] Background thread started")
        try:
            population = []
            for gen in range(100):
                pop_size = 50
                step_limit = min(40 + 20 * gen, 500)
                print(f"[DEBUG] Generation {gen+1} using pop_size {pop_size}")
                population = [ss.build_model(1602, 3) for _ in range(pop_size)]
                print(f"[DEBUG] Generation {gen+1} starting with step_limit {step_limit}...")
                scores = [run_model(model, max_steps=step_limit, generation=gen+1) for model in population]
                keep_top_n = min(5 + gen, 20)
                top_scores_models = [(s, m) for s, m in zip(scores, population)]
                top_models = [model for _, model in sorted(top_scores_models, key=lambda x: x[0], reverse=True)[:keep_top_n]]

                new_population = top_models[:]
                while len(new_population) < 50:
                    parent = random.choice(top_models)
                    child = tf.keras.models.clone_model(parent)
                    mutation_rate = max(0.02, 0.05 - gen * 0.003)
                    child.set_weights(mutate_weights(parent.get_weights(), mutation_rate=mutation_rate))
                    new_population.append(child)

                population = new_population
                best_score = max(scores)

                def update_ui():
                    info_label.config(text=f"Generation: {gen+1} | Best Score: {best_score:.2f}")
                    canvas.coords(bar, 0, 0, min(best_score * 5, 300), 50)

                root.after(0, update_ui)
                print(f"[DEBUG] Generation {gen+1} | Best Score: {best_score:.2f}")
        except Exception as e:
            print(f"[ERROR] Exception in evolution thread: {e}")

    threading.Thread(target=background, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    evolve_with_monitor()
