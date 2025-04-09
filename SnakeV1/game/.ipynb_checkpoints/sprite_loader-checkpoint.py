
def load_sprites(render, sprite_dir="sprites", master = None):
    sprites = {}
    if not render:
        return {
            'snakeHead0': '',
            'snakeHead1': '',
            'snakeHead2': '',
            'snakeHead3': '',
            'snakeHorz': '',
            'snakeVert': '',
            'snakeTurn01': '',
            'snakeTurn12': '',
            'snakeTurn23': '',
            'snakeTurn30': '',
            'apple': ''
        }
    else:
        from tkinter import PhotoImage
        import os

        filenames = {
            'snakeHead0': "snakeHead0.png",
            'snakeHead1': "snakeHead1.png",
            'snakeHead2': "snakeHead2.png",
            'snakeHead3': "snakeHead3.png",
            'snakeHorz': "snakeHorz.png",
            'snakeVert': "snakeVert.png",
            'apple': "apple.png",
            'snakeTurn01': "snakeTurn01.png",
            'snakeTurn12': "snakeTurn12.png",
            'snakeTurn23': "snakeTurn23.png",
            'snakeTurn30': "snakeTurn30.png"
        }

    for key, filename in filenames.items():
        path = os.path.join(sprite_dir, filename)
        sprites[key] = PhotoImage(file=path, master = master)

    return sprites
