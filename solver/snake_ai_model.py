import numpy as np
import tensorflow as tf

def build_model(input_size, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model

def get_features(game_state):
    # Step 1: Flatten the board state to 1601 elements
    board_vector = np.zeros((1601,), dtype=np.float32)  # TODO: Replace with actual encoding

    # Step 2: Append apple distance
    head_y, head_x = game_state.snake[0]
    apple_y, apple_x = game_state.apple
    apple_distance = abs(head_y - apple_y) + abs(head_x - apple_x)
    normalized_distance = apple_distance / sum(game_state.board_size)

    # Step 3: Append current direction (0–3)
    direction = game_state.direction

    # Step 4: Combine all features into one array
    features = np.append(board_vector, [normalized_distance, direction])

    return features.reshape(1, -1)


from game.direction import relative_to_absolute

def get_action(model, game_state):
    features = get_features(game_state)
    prediction = model.predict(features.reshape(1, -1), verbose=0)
    relative_action = np.argmax(prediction)
    return relative_to_absolute(game_state.direction, relative_action)
