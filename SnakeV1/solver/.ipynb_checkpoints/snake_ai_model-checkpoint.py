import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def build_model(input_size, output_size):
    model = Sequential([
        Input(shape=(input_size,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def get_features(game_state):
    height, width = game_state.board_size
    board = np.zeros((height, width), dtype=float)

    # Mark snake body
    for y, x in game_state.snake:
        board[y][x] = 0.5  # Snake body

    # Mark apple
    ay, ax = game_state.apple
    board[ay][ax] = 1.0  # Apple

    # Mark head separately
    head_y, head_x = game_state.snake[0]
    board[head_y][head_x] = 0.75

    # Flatten and append direction
    flat = board.flatten()
    direction = game_state.direction
    return np.append(flat, direction).astype(np.float32)


def get_action(model, game_state):
    features = get_features(game_state).reshape(1, -1)
    prediction = model.predict(features, verbose=0)
    return np.argmax(prediction)


# Mapping output index to directions (match with your direction constants)
OUTPUT_TO_DIRECTION = [EAST, NORTH, WEST, SOUTH]
