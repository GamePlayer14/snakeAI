import random
from Game.game_state import GameState
from Game.sprite_loader import load_sprites
from Game.direction import EAST
from Game.controls import bind_controls

class SnakeGame:
    def __init__(self, board_size=(40, 40), render=True, tiles=None, master=None):
        self.board_size = board_size
        self.render = render
        self.tiles = tiles
        self.master = master
        self.sprites = load_sprites(render=render, master=master)
        self.gs = GameState(tiles, board_size, self.sprites, render=render)

    def reset(self, length=3):
        self.gs.reset(length)

    def step(self, direction=None):
        if direction is not None:
            self.gs.set_direction(direction)
        self.gs.step()

    def alive(self):
        return self.gs.alive

    def head(self):
        return self.gs.snake[0] if self.gs.snake else None

    def apple(self):
        return self.gs.apple

    def length(self):
        return len(self.gs.snake)

    def set_dir(self, direction):
        self.gs.set_direction(direction)

    def state(self):
        return self.gs

    def bind_controls(self, root):
        bind_controls(root, self.gs)