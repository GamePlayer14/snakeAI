import tkinter as tk

def buildScreen(x, y):
    tile_size = 20
    XRange = x
    YRange = y
    screenW = XRange * tile_size
    screenH = YRange * tile_size

    root = tk.Tk()
    screen = tk.Canvas(root, width=screenW, height=screenH)
    screen.pack()

    grid_frame = tk.Frame(screen)
    screen.create_window((0, 0), window=grid_frame, anchor="nw")

    tiles = []
    for y in range(YRange):
        row = []
        for x in range(XRange):
            tile = tk.Canvas(
                grid_frame,
                width=tile_size,
                height=tile_size,
                bg='black',
                highlightthickness=0
            )
            tile.grid(row=y, column=x)
            row.append(tile)
        tiles.append(row)

    return root, tiles

def create_monitor():
    monitor = tk.Tk()
    monitor.title("AI Monitor")

    info_label = tk.Label(monitor, text="Generation: 0 | Best Score: 0")
    info_label.pack()

    return monitor, info_label

def update_monitor(generation, score, info_label, root):
    def update():
        info_label.config(text=f"Generation: {generation} | Best Score: {score}")
    root.after(0, update)  # ✅ safely run this on the main thread
