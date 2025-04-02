import torch
import numpy as np
import tkinter as tk
from snake_ai_model import build_model, get_action, mutate_weights, clone_model
from game.game_state import GameState
from game.sprite_loader import load_sprites
from game.screen_builder import buildScreen

POP_SIZE = 50
INPUT_SIZE = 1603  # 40x40 board + direction
OUTPUT_SIZE = 3
GENERATIONS = 20
TOP_K = 10

RENDER = True  # Toggle this to False to disable visualization
INITIAL_STEP_LIMIT = 40  # Starting step limit per snake
STEP_BONUS_PER_APPLE = 80  # Steps added per apple eaten

def evaluate_score(game_state, steps, visited_tiles, recent_heads=None):
    score = 0
    apples_eaten = len(game_state.snake) - 3

    if not game_state.alive:
        score -= 10

    score += apples_eaten * 50

    head_y, head_x = game_state.snake[0]
    apple_y, apple_x = game_state.apple
    distance = abs(head_y - apple_y) + abs(head_x - apple_x)
    score += max(0, 20 - distance)

    score += len(visited_tiles) * 0.5

    score -= steps * 0.1

    if recent_heads:
        unique_heads = len(set(recent_heads))
        score -= (len(recent_heads) - unique_heads)*2

    return score


def run_model(model, base_step_limit=INITIAL_STEP_LIMIT, gs_cache={}):
    if RENDER:
        if 'gs' not in gs_cache:
            root, tiles = buildScreen(40, 40)
            sprites = load_sprites(True, master=root)
            gs = GameState(tiles, (40, 40), sprites, render=True)
            gs_cache['gs'] = gs
            gs_cache['root'] = root
        else:
            gs = gs_cache['gs']
            root = gs_cache['root']
    else:
        if 'gs' not in gs_cache:
            sprites = load_sprites(False)
            gs = GameState(None, (40, 40), sprites, render=False)
            gs_cache['gs'] = gs
        else:
            gs = gs_cache['gs']

    gs.reset()
    steps = 0
    step_limit = base_step_limit
    initial_length = len(gs.snake)
    visited_tiles = set()
    recent_heads = []

    while gs.alive and steps < step_limit:
        prev_length = len(gs.snake)

        direction = get_action(model, gs)
        gs.set_direction(direction)
        gs.step()
        visited_tiles.add(gs.snake[0])
        recent_heads.append(gs.snake[0])
        if len(recent_heads) > 20:
            recent_heads.pop(0)

        steps += 1

        if len(gs.snake) > prev_length:
            step_limit += STEP_BONUS_PER_APPLE

        if RENDER:
            root.update()

        print('ran model')
    return evaluate_score(gs, steps, visited_tiles, recent_heads)

def evolve_population_with_monitor(info_label, root):
    population = [build_model(INPUT_SIZE, OUTPUT_SIZE) for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        step_limit = INITIAL_STEP_LIMIT
        scores = []
        for model in population:
            score = run_model(model, base_step_limit=step_limit)
            step_limit = max(step_limit, INITIAL_STEP_LIMIT + (score - 3) * STEP_BONUS_PER_APPLE)
            scores.append(score)

        top_models = [model for _, model in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[:TOP_K]]

        new_population = top_models[:]
        while len(new_population) < POP_SIZE:
            parent = np.random.choice(top_models)
            child = clone_model(parent)
            mutate_weights(child)
            new_population.append(child)

        population = new_population
        best_score = max(scores)

        def update_label():
            info_label.config(text=f"Generation: {gen+1} | Best Score: {best_score}")
        root.after(0, update_label)

        print(f"Generation {gen+1} | Best Score: {best_score}")
        
