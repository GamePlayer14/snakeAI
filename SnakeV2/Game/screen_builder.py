import tkinter as tk

def buildScreen(board_size):
    tile_size = 20
    width, height = board_size
    screenW, screenH = width * tile_size, height * tile_size

    root = tk.Tk()
    screen = tk.Canvas(root, width=screenW, height=screenH)
    screen.pack()

    grid_frame = tk.Frame(screen)
    screen.create_window((0, 0), window=grid_frame, anchor="nw")

    tiles = [[tk.Canvas(grid_frame, width=tile_size, height=tile_size, bg='black', highlightthickness=0)
              for x in range(width)] for y in range(height)]

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            tile.grid(row=y, column=x)

    return root, tiles

def create_monitor():
    monitor = tk.Tk()
    monitor.title("AI Monitor")
    info_label = tk.Label(monitor, text="Generation: 0 | Best Score: 0")
    info_label.pack()
    return monitor, info_label

def update_monitor(generation, score, info_label, root):
    root.after(0, lambda: info_label.config(text=f"Generation: {generation} | Best Score: {score}"))