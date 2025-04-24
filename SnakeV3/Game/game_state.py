import random
import numpy as np
from .direction import DIRECTION_DELTAS, EAST
from Main.Config import FIXED_LENGTH
class GameState:
    def __init__(self, tiles, board_size, sprite_images, render):
        self.render = render
        self.tiles = tiles
        self.board_size = board_size
        self.sprite_images = sprite_images
        self.snake = []
        self.direction = EAST
        self.next_directions = []
        self.apple = None
        self.alive = True
        self.trail_map = np.zeros(self.board_size, dtype=np.float32)
        self.reset()

    def reset(self, starting_length=3):
        self.trail_map = np.zeros(self.board_size, dtype=np.float32)
        self.clear_tiles()
        self.snake.clear()
        self.next_directions.clear()
        self.direction = EAST
        self.alive = True

        mid_y, mid_x = self.board_size[0] // 2, self.board_size[1] // 2
        dy, dx = DIRECTION_DELTAS[(self.direction + 2) % 4]

        for i in range(starting_length):
            y = mid_y + dy
            x = mid_x + dx - 1
            if not (0 <= y < self.board_size[0] and 0 <= x < self.board_size[1]):
                break
            self.snake.append((y, x))
            self.draw_tile(y, x, self.sprite_images['snakeHorz'])

        self.spawn_apple()

    def clear_tiles(self):
        if self.render:
            for row in self.tiles:
                for tile in row:
                    tile.delete("all")
                    tile.config(bg='black')

    def spawn_apple(self):
        open_tiles = [(y, x) for y in range(self.board_size[0]) for x in range(self.board_size[1]) if (y, x) not in self.snake]
        if not open_tiles:
            return
        self.apple = random.choice(open_tiles)
        self.draw_tile(*self.apple, self.sprite_images['apple'])
        self.trail_map = np.zeros(self.board_size, dtype=np.float32)

    def draw_tile(self, y, x, image):
        if self.render and self.tiles is not None:
            tile = self.tiles[y][x]
            tile.delete("all")
            tile.create_image(0, 0, anchor='nw', image=image)

    def step(self):
        if not self.alive:
            return
            
        head_y, head_x = self.snake[0]
        self.trail_map[head_y][head_x] += 1.0

        if self.next_directions:
            new_dir = self.next_directions.pop(0)
            if (new_dir + 2) % 4 != self.direction:
                self.direction = new_dir

        dy, dx = DIRECTION_DELTAS[self.direction]
        new_head = (head_y + dy, head_x + dx)

        if (not (0 <= new_head[0] < self.board_size[0]) or
            not (0 <= new_head[1] < self.board_size[1]) or
            new_head in self.snake):
            self.alive = False
            return

        self.snake.insert(0, new_head)

        if not FIXED_LENGTH:
            if new_head == self.apple:
                self.spawn_apple()
            else:
                tail_y, tail_x = self.snake.pop()
                if self.render:
                    self.tiles[tail_y][tail_x].delete("all")
        else:
            if new_head == self.apple:
                self.spawn_apple()
            tail_y, tail_x = self.snake.pop()
            if self.render:
                self.tiles[tail_y][tail_x].delete("all")

        self.redraw_head_and_neck()

    def redraw_head_and_neck(self):
        if len(self.snake) < 2:
            return

        y, x = self.snake[0]
        head_sprite = self.sprite_images[f'snakeHead{self.direction}']
        self.draw_tile(y, x, head_sprite)

        y, x = self.snake[1]
        y1, x1 = self.snake[0]
        y2, x2 = self.snake[2] if len(self.snake) > 2 else self.snake[0]

        if x1 == x2:
            sprite = self.sprite_images['snakeVert']
        elif y1 == y2:
            sprite = self.sprite_images['snakeHorz']
        elif ((x1 < x and y2 < y) or (x2 < x and y1 < y)):
            sprite = self.sprite_images['snakeTurn23']
        elif ((x1 < x and y2 > y) or (x2 < x and y1 > y)):
            sprite = self.sprite_images['snakeTurn12']
        elif ((x1 > x and y2 < y) or (x2 > x and y1 < y)):
            sprite = self.sprite_images['snakeTurn30']
        else:
            sprite = self.sprite_images['snakeTurn01']

        self.draw_tile(y, x, sprite)

    def set_direction(self, new_direction):
        if not self.next_directions or self.next_directions[-1] != new_direction:
            self.next_directions.append(new_direction)

    def ai_step(self, direction):
        self.set_direction(direction)
        self.step()

    def get_state(self):
        return {
            'snake': self.snake.copy(),
            'direction': self.direction,
            'next_directions': self.next_directions.copy(),
            'apple': self.apple,
            'alive': self.alive,
            'trail_map': self.trail_map.copy(),
        }

    def set_state(self, state):
        self.clear_tiles()
        self.snake = state['snake'].copy()
        self.direction = state['direction']
        self.next_directions = state['next_directions'].copy()
        self.apple = state['apple']
        self.alive = state['alive']
        self.trail_map = state['trail_map'].copy()

        # Redraw tiles for snake
        for y, x in self.snake:
            self.draw_tile(y, x, self.sprite_images['snakeHorz'])

        # Redraw apple
        if self.apple:
            self.draw_tile(*self.apple, self.sprite_images['apple'])

        # Redraw head and neck (for sprite accuracy)
        self.redraw_head_and_neck()