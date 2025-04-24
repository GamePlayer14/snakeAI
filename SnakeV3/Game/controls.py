from .direction import NORTH, SOUTH, EAST, WEST

def bind_controls(root, game):
    def on_key(event):
        key = event.keysym.lower()
        if key == 'w':
            game.set_dir(NORTH)
        elif key == 'a':
            game.set_dir(WEST)
        elif key == 's':
            game.set_dir(SOUTH)
        elif key == 'd':
            game.set_dir(EAST)

    root.bind('<KeyPress>', on_key)
