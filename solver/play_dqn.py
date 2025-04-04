import torch
from dqn_model import DQN, get_action
from snake_ai_model import get_features
from game.game_state import GameState
from game.sprite_loader import load_sprites

# --- Settings ---
BOARD_SIZE = (40, 40)
INPUT_SIZE = 15   # or 15 if using simplified features
OUTPUT_SIZE = 3
MODEL_PATH = "dqn_snake.pth"

def play_model():
    # Load model
    model = DQN(INPUT_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Setup game with rendering
    sprites = load_sprites(render=True)
    gs = GameState(None, BOARD_SIZE, sprites, render=True)
    gs.reset()

    while gs.alive:
        action = get_action(model, gs, epsilon=0.0)  # no exploration
        gs.set_direction(action)
        gs.step()

    print("Game over. Final length:", len(gs.snake))

if __name__ == "__main__":
    play_model()
