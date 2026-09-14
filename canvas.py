import pygame
import pygame.freetype
import numpy as np
import trainer

pygame.init()
pygame.freetype.init()

SKIP_TRAINING = False

GRID_SIZE = 28
CELL_SIZE = 20

PANEL_WIDTH = 280

GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE

WIDTH = GRID_WIDTH + PANEL_WIDTH
HEIGHT = GRID_HEIGHT

# Colors (RGB)
PANEL_COLOR = (20, 40, 70)
GRID_COLOR = (60, 60, 60)
WHITE = (255, 255, 255)
LGRAY = (200, 200, 200)
BLACK = (0, 0, 0)


if not SKIP_TRAINING:
    trainer.init()


# Fonts
font = pygame.freetype.SysFont("Sans Serif", 16)
small_font = pygame.freetype.SysFont("Sans Serif", 12)
title_font = pygame.freetype.SysFont("Sans Serif", 24, True)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Drawing")

clock = pygame.time.Clock()

grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

brush_size = 4


def draw_at_mouse(mouse_x, mouse_y, erase = False):

    center_x = mouse_x / CELL_SIZE - 0.5
    center_y = mouse_y / CELL_SIZE - 0.5

    radius = (brush_size + 1) / 2

    for y in range(max(0, int(center_y - radius)), min(GRID_SIZE, int(center_y + radius + 1))):
        for x in range(max(0, int(center_x - radius)), min(GRID_SIZE, int(center_x + radius + 1))):
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance > radius:
                continue

            strength = 1 - distance / radius
            strength /= 0.5

            if erasing:
                grid[y, x] = max(min(grid[y, x], 1-strength), 0)
            else:
                grid[y, x] = min(max(grid[y, x], strength), 1)

def clear_grid():
    grid.fill(0)

def draw_grid():
    # Draw pixels
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = int(grid[y, x] * 255)

            pygame.draw.rect(
                screen,
                (value, value, value),
                (
                    x * CELL_SIZE, y * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
            )

            # Pixel grid
            pygame.draw.rect(
                screen,
                GRID_COLOR,
                (
                    x * CELL_SIZE, y * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                ),
                1
            )

def draw_indicator():
    pygame.draw.circle(
        screen,
        BLACK,
        (mouse_x + 1, mouse_y + 1),
        CELL_SIZE * brush_size / 2,
        1
    )
    pygame.draw.circle(
        screen,
        WHITE,
        (mouse_x - 1, mouse_y - 1),
        CELL_SIZE * brush_size / 2,
        1
    )

panel_rect = pygame.Rect(GRID_WIDTH, 0, PANEL_WIDTH, HEIGHT)

def draw_panel():
    panel = pygame.Surface(panel_rect.size)
    panel.fill(PANEL_COLOR)
    title_font.render_to(panel, (30, 40), "AI Handwriting", WHITE)
    title_font.render_to(panel, (30, 70), "Recognition", WHITE)
    small_font.render_to(panel, (30, 110), "[LMB] : Draw", LGRAY)
    small_font.render_to(panel, (128, 110), "[RMB] : Erase", LGRAY)
    small_font.render_to(panel, (30, 130), "[C] : Clear", LGRAY)
    small_font.render_to(panel, (128, 130), "[Scroll] : Brush Size", LGRAY)

    labels = [
        "ZERO",        "ONE",        "TWO",        "THREE",        "FOUR",
        "FIVE",        "SIX",        "SEVEN",        "EIGHT",        "NINE"
    ]

    outputs = trainer.AI.calculate_outputs(grid.ravel())
    # outputs = np.array((0, 0, .1, .2, 0, 1, 0, 0, 0, 0), dtype=np.float32)
    outputs /= np.sum(outputs, dtype=np.float32)

    indices = outputs.argsort()

    for i in range(10):
        index = indices[9-i]
        font.render_to(panel, (40, 180+36*i), labels[index], LGRAY if i > 0 else WHITE)
        font.render_to(panel, (160, 180+36*i), f"{outputs[index]*100:.2f}%", LGRAY if i > 0 else WHITE)

    screen.blit(panel, (GRID_WIDTH, 0))


drawing = False
erasing = False

running = True

if __name__ == "__main__":

    while running:

        mouse_x, mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_c:
                    clear_grid()

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:
                    drawing = True

                elif event.button == 3:
                    erasing = True

            elif event.type == pygame.MOUSEBUTTONUP:

                if event.button == 1:
                    drawing = False

                elif event.button == 3:
                    erasing = False

            elif event.type == pygame.MOUSEWHEEL:
                brush_size = min(max(1, brush_size + event.y), GRID_SIZE)

        if drawing or erasing:
            draw_at_mouse(mouse_x, mouse_y, erasing)

        draw_grid()

        if pygame.mouse.get_focused():
            draw_indicator()

        draw_panel()

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()