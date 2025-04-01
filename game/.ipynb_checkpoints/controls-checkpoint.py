from .direction import NORTH, SOUTH, EAST, WEST

def bind_controls(root, game):
    def on_key(event):
        key = event.keysym.lower()
        if key == 'w':
            game.set_direction(NORTH)
        elif key == 'a':
            game.set_direction(WEST)
        elif key == 's':
            game.set_direction(SOUTH)
        elif key == 'd':
            game.set_direction(EAST)

    root.bind('<KeyPress>', on_key)
