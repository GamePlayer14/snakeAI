import heapq
import numpy as np

def compute_path_distances(start, is_blocked, board_size):
    height, width = board_size
    visited = set()
    distances = np.full((height, width), np.inf)
    queue = [(0, start)]

    while queue:
        cost, (y, x) = heapq.heappop(queue)
        if (y, x) in visited or not (0 <= y < height and 0 <= x < width):
            continue
        visited.add((y, x))
        distances[y][x] = cost

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not is_blocked(ny, nx):
                heapq.heappush(queue, (cost + 1, (ny, nx)))

    return distances

# Example usage in get_features (snake proximity map):
#
# def is_snake(y, x):
#     return (y, x) in game_state.snake
#
# snake_distances = compute_path_distances(game_state.snake[0], is_snake, game_state.board_size)
# snake_layer = 1.0 / (snake_distances + 1)

# For apple proximity:
# def is_wall_or_snake(y, x):
#     return (y, x) in game_state.snake
#
# apple_distances = compute_path_distances(game_state.apple, is_wall_or_snake, game_state.board_size)
# apple_layer = 1.0 / (apple_distances + 1)
# apple_layer[np.isinf(apple_layer)] = 0.0  # Unreachable tiles

# Replace current layer building with these matrices instead.
