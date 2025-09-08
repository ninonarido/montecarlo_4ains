import pygame as pg
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 10

GRAPH_HEIGHT = 120
NUM_Y_MARKERS = 5
NUM_X_MARKERS = 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRAPH_BG = (220, 220, 220)
LINE_COLOR = (0, 100, 255)
GRID_COLOR = (180, 180, 180)

def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT + GRAPH_HEIGHT))
    pg.display.set_caption("Conway's Game of Life")
    clock = pg.time.Clock()

    grid = np.random.choice([0, 1], size=(GRID_WIDTH, GRID_HEIGHT))
    alive_history = []

    paused = False
    font = pg.font.SysFont(None, 18)

    def count_neighbors(x, y):
        total = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < GRID_WIDTH and 0 <= y + j < GRID_HEIGHT:
                    total += grid[x + i, y + j]
        return total

    def update_grid():
        new_grid = grid.copy()
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                neighbors = count_neighbors(x, y)
                if grid[x, y] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_grid[x, y] = 0
                else:
                    if neighbors == 3:
                        new_grid[x, y] = 1
        return new_grid

    def draw_grid():
        screen.fill(BLACK)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if grid[x, y] == 1:
                    rect = pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pg.draw.rect(screen, GREEN, rect)

    def draw_graph():
        # Graph background
        graph_rect = pg.Rect(0, HEIGHT, WIDTH, GRAPH_HEIGHT)
        pg.draw.rect(screen, GRAPH_BG, graph_rect)

        # Draw horizontal grid lines
        if alive_history:
            max_alive = max(max(alive_history), 1)
        else:
            max_alive = 1
        for i in range(NUM_Y_MARKERS + 1):
            y = HEIGHT + i * (GRAPH_HEIGHT / NUM_Y_MARKERS)
            pg.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)
            # Label
            value = int(max_alive * (NUM_Y_MARKERS - i) / NUM_Y_MARKERS)
            label = font.render(str(value), True, BLACK)
            screen.blit(label, (5, y - 10))

        # Draw vertical tick marks
        for i in range(NUM_X_MARKERS + 1):
            x = i * (WIDTH / NUM_X_MARKERS)
            pg.draw.line(screen, GRID_COLOR, (x, HEIGHT), (x, HEIGHT + GRAPH_HEIGHT), 1)

        # Draw line
        if alive_history:
            scale_y = GRAPH_HEIGHT / max_alive
            scale_x = WIDTH / max(len(alive_history), 1)
            for i in range(1, len(alive_history)):
                x1 = (i - 1) * scale_x
                y1 = HEIGHT + GRAPH_HEIGHT - alive_history[i - 1] * scale_y
                x2 = i * scale_x
                y2 = HEIGHT + GRAPH_HEIGHT - alive_history[i] * scale_y
                pg.draw.line(screen, LINE_COLOR, (x1, y1), (x2, y2), 2)

            # Draw markers (small circles) for each point
            for i, val in enumerate(alive_history):
                x = i * scale_x
                y = HEIGHT + GRAPH_HEIGHT - val * scale_y
                pg.draw.circle(screen, LINE_COLOR, (int(x), int(y)), 2)

        # Label
        label = font.render("Alive Cells Over Time", True, BLACK)
        screen.blit(label, (10, HEIGHT + 5))

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == [pg].K_SPACE:
                paused = not paused
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                if x < WIDTH and y < HEIGHT:
                    grid[x // CELL_SIZE, y // CELL_SIZE] ^= 1

        if not paused:
            grid = update_grid()
            alive_count = np.sum(grid)
            alive_history.append(alive_count)
            if len(alive_history) > WIDTH:  # Limit history
                alive_history.pop(0)

        draw_grid()
        draw_graph()
        pg.display.flip()
        clock.tick(FPS)

    pg.quit()

if __name__ == "__main__":
    main()
