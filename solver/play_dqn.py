import torch
from dqn_model import DQN, get_action
from snake_ai_model import get_features
from game.game_state import GameState
from game.sprite_loader import load_sprites
from train_dqn import get_best_model_path
from config import INPUT_SIZE, OUTPUT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Settings ---
BOARD_SIZE = (40, 40)
MODEL_PATH = "dqn_snake.pth"

def play_model():
    # Load model
    model = DQN(INPUT_SIZE, OUTPUT_SIZE).to(device)
    best_model_path = get_best_model_path()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
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
