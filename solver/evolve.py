import torch
import numpy as np
import tkinter as tk
from snake_ai_model import build_model, get_action, mutate_weights, clone_model
from game.game_state import GameState
from game.sprite_loader import load_sprites
from game.screen_builder import buildScreen
from reward_manager import RewardManager
import copy

POP_SIZE = 100
INPUT = 15  # 40x40 board + direction
OUTPUT_SIZE = 3
GENERATIONS = 100
TOP_K = 20

RENDER = True  # Toggle this to False to disable visualization
INITIAL_STEP_LIMIT = 40  # Starting step limit per snake
STEP_BONUS_PER_APPLE = 80  # Steps added per apple eaten

def evaluate_score(game_state, steps, visited_tiles, recent_heads=None, generation=0):
    reward = RewardManager(generation)

    apples_eaten = len(game_state.snake) - 3
    head_y, head_x = game_state.snake[0]
    apple_y, apple_x = game_state.apple

    reward.reset_distance((head_y, head_x), (apple_y, apple_x))

    if not game_state.alive:
        reward.total_reward -= 10

    # Apple rewards
    reward.apple_eaten = apples_eaten  # count total
    if generation > 5:
        reward.total_reward += (apples_eaten ** 2) * 75

    # Distance improvement bonus
    reward.update_distance((head_y, head_x), (apple_y, apple_x))

    # Exploration reward
    reward.total_reward += apples_eaten * len(visited_tiles) * 0.05

    # Step penalty
    reward.survival_bonus(steps)

    # Loop penalty
    if recent_heads:
        unique_heads = len(set(recent_heads))
        repeats = len(recent_heads) - unique_heads
        reward.loop_penalty(repeats)

    return reward.get_total()



def run_model(model, board_size, generation, epsilon = 0.0, base_step_limit=INITIAL_STEP_LIMIT, gs_cache={}):
    if RENDER:
        if 'gs' not in gs_cache:
            root, tiles = buildScreen(board_size)
            sprites = load_sprites(True, master=root)
            gs = GameState(tiles, board_size, sprites, render=True)
            gs_cache['gs'] = gs
            gs_cache['root'] = root
        else:
            gs = gs_cache['gs']
            root = gs_cache['root']
    else:
        if 'gs' not in gs_cache:
            sprites = load_sprites(False)
            gs = GameState(None, board_size, sprites, render=False)
            gs_cache['gs'] = gs
        else:
            gs = gs_cache['gs']

    gs.reset()
    steps = 0
    step_limit = base_step_limit
    initial_length = len(gs.snake)
    visited_tiles = set()
    recent_heads = []

    steps_since_apple = 0
    max_idle_steps = 100

    while gs.alive and steps < step_limit:
        score = 0
        prev_length = len(gs.snake)

        direction = get_action(model, gs, epsilon = epsilon)
        gs.set_direction(direction)
        prev_score = len(gs.snake)
        gs.step()

        prev_dist = manhattan(gs.snake[0], gs.apple)
        gs.step()
        new_dist = manhattan(gs.snake[0], gs.apple)
        score += (prev_dist - new_dist) * 5

        if len(gs.snake) == prev_score:
            steps_since_apple += 1
        else:
            steps_since_apple = 0

        visited_tiles.add(gs.snake[0])
        recent_heads.append(gs.snake[0])
        if len(recent_heads) > 20:
            recent_heads.pop(0)

        steps += 1

        if len(gs.snake) > prev_length:
            step_limit += STEP_BONUS_PER_APPLE

        if RENDER:
            root.update()

        if steps_since_apple > max_idle_steps:
            gs.alive = False
    score += evaluate_score(gs, steps, visited_tiles, recent_heads, generation)
    print(f"gen: {generation} | steps: {steps} | score: {score}")
    return score

def evolve_population_with_monitor(info_label, root, board_size):
    lx, ly = board_size
    total_size = lx*ly
    population = [build_model(INPUT, OUTPUT_SIZE) for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        epsilon = max(0, .5-(gen/(GENERATIONS*0.5)))
        step_limit = INITIAL_STEP_LIMIT
        scores = []
        for model in population:
            score = run_model(model, board_size, gen, epsilon, base_step_limit=step_limit)
            step_limit = max(step_limit, INITIAL_STEP_LIMIT + (score - 3) * STEP_BONUS_PER_APPLE)
            scores.append(score)

        top_models = [model for _, model in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[:TOP_K]]

        new_population = top_models[:]
        while len(new_population) < POP_SIZE-10:
            parent1, parent2 = np.random.choice(top_models, size=2, replace=False)
            child = copy.deepcopy(parent1)
            weights1 = [param.data.clone() for param in parent1.parameters()]
            weights2 = [param.data.clone() for param in parent2.parameters()]
            new_weights = []
            for w1, w2 in zip(weights1, weights2):
                mask = torch.rand_like(w1) < 0.5
                mixed = torch.where(mask, w1, w2)
                new_weights.append(mixed)
            with torch.no_grad():
                for param, new_param in zip(child.parameters(), new_weights):
                    param.copy_(new_param)
            mutate_weights(child, 0.3)
            new_population.append(child)

        while len(new_population) < POP_SIZE:
            new_population.append(build_model(INPUT, OUTPUT_SIZE))

        population = new_population
        best_score = max(scores)

        def update_label():
            info_label.config(text=f"Generation: {gen+1} | Best Score: {best_score}")
        root.after(0, update_label)

        print(f"Generation {gen+1} | Best Score: {best_score}")

def manhattan(a, b):
    return (abs(b[0])-abs(a[0])) + (abs(b[1])-abs(a[1]))
        
