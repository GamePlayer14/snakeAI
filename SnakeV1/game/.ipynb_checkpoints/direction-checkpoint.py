EAST = 0
SOUTH = 1
WEST = 2
NORTH = 3

DIRECTION_DELTAS = {
    EAST:  (0, 1),
    SOUTH: (1, 0),
    WEST:  (0, -1),
    NORTH: (-1, 0),
}

def turn_left(direction):
    return (direction - 1) % 4

def turn_right(direction):
    return (direction + 1) % 4
