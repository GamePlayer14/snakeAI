BOARD_SIZE = (10, 10)
USE_FULL_BOARD = True

SQUARE_COUNT = BOARD_SIZE[0] * BOARD_SIZE[1]
if USE_FULL_BOARD:
    INPUT_SIZE = SQUARE_COUNT * 2 + 7
else:
    INPUT_SIZE = 44 

OUTPUT_SIZE = 3
MAX_DANGER_DISTANCE = 10
FIXED_LENGTH = True
