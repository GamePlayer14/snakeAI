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

def relative_to_absolute(current_direction, relative_action):
    directions = [NORTH, EAST, SOUTH, WEST]  # The possible directions in order
    current_idx = directions.index(current_direction)  # Get the current direction index

    if relative_action == 0:  # turn left
        return directions[(current_idx - 1) % 4]  # Turn left (counterclockwise)
    elif relative_action == 2:  # turn right
        return directions[(current_idx + 1) % 4]  # Turn right (clockwise)
    else:  # forward
        return current_direction  # Keep going forward

