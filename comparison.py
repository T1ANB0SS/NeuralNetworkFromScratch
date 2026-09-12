"""import sys
import pygame
import pygame.surfarray as surfarray
import pygame.freetype
import numpy as np
import neural_network as nn

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
FPS = 60

# Colors (RGB)
BG_COLOR = (10, 20, 30)
WHITE = (255, 255, 255)

pygame.freetype.init()
FONT = pygame.freetype.SysFont('Sans Serif', 16)

AI = nn.NeuralNetwork((2, 5, 2), "Tanh")
AI.randomize()

pixel_buffer = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)

# PRE-CALCULATE COORDINATE GRID ONCE
# Mappings for coordinates mapped normalized from 0.0 to 1.0
x_coords = np.linspace(0, 1, SCREEN_WIDTH)
y_coords = np.linspace(0, 1, SCREEN_HEIGHT)

# indexing='ij' lines up the matrix coordinates perfectly with Pygame's [x, y] surface layout
# meshgrid is similar to linspace but u get all the coordinates for a 2d matrix
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
# ravel flattens an array into a 1D array
# stack combines arrays into one array
# using axis = -1 combines X and Y into [[x1, y1]...] instead of [[x1, x2 ...],[y1, y2 ...]]
grid_inputs = np.stack([X.ravel(), Y.ravel()], axis=-1)

SAMPLE_SIZE = 250
LEARN_RATE = .2
BATCH_SIZE = 250
MOMENTUM = 0.9

def get_inputs():
    inputs = np.random.uniform(0, 1, size=(SAMPLE_SIZE, 2))
    return inputs

def get_expected(inputs):
    x = inputs[:, 0]
    y = inputs[:, 1]

    # EXPECTED FUNCTION
    #labels = (y<(0.4+2*(x-0.5)**2)).astype(int)
    labels = (y < (0.5 + 6 * x * (x - 0.7) ** 2)).astype(int)
    #labels = ((x-.5)**2+(y-.5)**2 < .1).astype(int)
    #labels = (x**2+y**2<.5).astype(int)
    #labels = (y>x**3).astype(int)
    #labels = (y<.2*x*np.sin(20*x)+0.5).astype(int)

    expected_outputs = np.ones((SAMPLE_SIZE, 2))
    expected_outputs[np.arange(SAMPLE_SIZE), labels] = 0

    return expected_outputs

inputs = get_inputs()
expected = get_expected(inputs)

def update(screen, dt):

    for _ in range(20):
        AI.slow_learn(inputs, expected, LEARN_RATE, BATCH_SIZE, MOMENTUM)

    # Pass the entire screen input to the network at once
    # outputs shape: (10000, 2)
    outputs = AI.calculate_outputs(grid_inputs)

    # Separate the results and reshape back into the 100x100 frame grids
    out_0 = outputs[:, 0].reshape(SCREEN_WIDTH, SCREEN_HEIGHT)
    out_1 = outputs[:, 1].reshape(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Using [..., i] is the same as using [:, :, i]
    '''
    pixel_buffer[..., 0] = 255 * out_1  # Red channel
    pixel_buffer[..., 1] = 0            # Green channel
    pixel_buffer[..., 2] = 255 * out_0  # Blue channel
    '''
    pixel_buffer[..., 0] = 255 * (out_1 > out_0)  # Red channel
    pixel_buffer[..., 1] = 0                      # Green channel
    pixel_buffer[..., 2] = 255 * (out_0 > out_1)  # Blue channel

    surfarray.blit_array(screen, pixel_buffer)

    for i in range(len(inputs)):
        pygame.draw.circle(screen, (expected[i, 1]*255, 150, expected[i, 0]*255), (int(inputs[i, 0]*SCREEN_WIDTH), int(inputs[i, 1]*SCREEN_HEIGHT)), 3)

    FONT.render_to(screen, (10, 10), f"FPS: {round(1/dt)}", WHITE)
    FONT.render_to(screen, (10, 30), f"Cost: {round(AI.cost(inputs, expected), 4)}", WHITE)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simple Neural Network")
    clock = pygame.time.Clock()
    dt = 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        update(screen, dt)
        pygame.display.flip()
        dt = clock.tick(FPS)/1000

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
"""

# Code below (visualization) used ChatGPT cuz I'm too lazy

import sys
import time
import pygame
import pygame.surfarray as surfarray
import pygame.freetype
import numpy as np
import neural_network as nn

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DISPLAY_WIDTH = SCREEN_WIDTH // 2
FPS = 60

WHITE = (255, 255, 255)

pygame.freetype.init()
FONT = pygame.freetype.SysFont("Sans Serif", 16)

# ============================================================
# AI
# ============================================================

AI_SLOW = nn.NeuralNetwork((2, 5, 2), "Tanh")
AI_FAST = nn.NeuralNetwork((2, 5, 2), "Tanh")

AI_SLOW.randomize()

# Give both networks identical starting parameters
for slow, fast in zip(AI_SLOW.layers, AI_FAST.layers):
    fast.weights = slow.weights.copy()
    fast.biases = slow.biases.copy()


# ============================================================
# TRAINING SETTINGS
# ============================================================

SAMPLE_SIZE = 320
LEARN_RATE = 0.9
BATCH_SIZE = 320
MOMENTUM = 0.9

# Maximum amount of time each network is allowed
# to train per frame.
TRAIN_TIME = 0.005


# ============================================================
# DATASET
# ============================================================

def get_inputs():
    return np.random.uniform(
        0,
        1,
        size=(SAMPLE_SIZE, 2)
    )


def get_expected(inputs):
    x = inputs[:, 0]
    y = inputs[:, 1]

    # labels = (y<(0.4+2*(x-0.5)**2)).astype(int)
    # labels = (y < (0.5 + 6 * x * (x - 0.7) ** 2)).astype(int)
    # labels = ((x-.5)**2+(y-.5)**2 < .1).astype(int)
    # labels = (x**2+y**2<.5).astype(int)
    # labels = (y>x**3).astype(int)
    labels = (y<.2*x*np.sin(15*x)+0.5).astype(int)

    expected_outputs = np.ones((SAMPLE_SIZE, 2))

    expected_outputs[
        np.arange(SAMPLE_SIZE),
        labels
    ] = 0

    return expected_outputs


inputs = get_inputs()
expected = get_expected(inputs)


# ============================================================
# GRID
# ============================================================

x_coords = np.linspace(0, 1, DISPLAY_WIDTH)
y_coords = np.linspace(0, 1, SCREEN_HEIGHT)

X, Y = np.meshgrid(
    x_coords,
    y_coords,
    indexing="ij"
)

grid_inputs = np.stack(
    [X.ravel(), Y.ravel()],
    axis=-1
)


# ============================================================
# PIXEL BUFFERS
# ============================================================

slow_buffer = np.zeros(
    (DISPLAY_WIDTH, SCREEN_HEIGHT, 3),
    dtype=np.uint8
)

fast_buffer = np.zeros(
    (DISPLAY_WIDTH, SCREEN_HEIGHT, 3),
    dtype=np.uint8
)


# ============================================================
# TRAINING STATS
# ============================================================

slow_steps = 0
fast_steps = 0

slow_time = 0
fast_time = 0


# ============================================================
# TRAIN
# ============================================================

def train():

    global slow_steps
    global fast_steps
    global slow_time
    global fast_time

    # --------------------------------------------------------
    # SLOW
    # --------------------------------------------------------

    start = time.perf_counter()
    steps = 0

    while time.perf_counter() - start < TRAIN_TIME:

        AI_SLOW.slow_learn(
            inputs,
            expected,
            LEARN_RATE,
            BATCH_SIZE,
        )

        steps += 1

    slow_time = time.perf_counter() - start
    slow_steps = steps


    # --------------------------------------------------------
    # FAST
    # --------------------------------------------------------

    start = time.perf_counter()
    steps = 0

    while time.perf_counter() - start < TRAIN_TIME:

        AI_FAST.learn(
            inputs,
            expected,
            LEARN_RATE,
            BATCH_SIZE,
            MOMENTUM
        )

        steps += 1

    fast_time = time.perf_counter() - start
    fast_steps = steps


# ============================================================
# DRAW NETWORK OUTPUT
# ============================================================

def draw_networks():

    slow_outputs = AI_SLOW.calculate_outputs(
        grid_inputs
    )

    fast_outputs = AI_FAST.calculate_outputs(
        grid_inputs
    )


    # --------------------------------------------------------
    # SLOW
    # --------------------------------------------------------

    slow_0 = slow_outputs[:, 0].reshape(
        DISPLAY_WIDTH,
        SCREEN_HEIGHT
    )

    slow_1 = slow_outputs[:, 1].reshape(
        DISPLAY_WIDTH,
        SCREEN_HEIGHT
    )

    slow_buffer[..., 0] = (
        255 * (slow_1 > slow_0)
    )

    slow_buffer[..., 1] = 0

    slow_buffer[..., 2] = (
        255 * (slow_0 > slow_1)
    )


    # --------------------------------------------------------
    # FAST
    # --------------------------------------------------------

    fast_0 = fast_outputs[:, 0].reshape(
        DISPLAY_WIDTH,
        SCREEN_HEIGHT
    )

    fast_1 = fast_outputs[:, 1].reshape(
        DISPLAY_WIDTH,
        SCREEN_HEIGHT
    )

    fast_buffer[..., 0] = (
        255 * (fast_1 > fast_0)
    )

    fast_buffer[..., 1] = 0

    fast_buffer[..., 2] = (
        255 * (fast_0 > fast_1)
    )


# ============================================================
# DRAW
# ============================================================

def update(screen, dt):

    train()

    draw_networks()


    # --------------------------------------------------------
    # Draw networks
    # --------------------------------------------------------

    slow_surface = screen.subsurface(
        (0, 0, DISPLAY_WIDTH, SCREEN_HEIGHT)
    )

    fast_surface = screen.subsurface(
        (DISPLAY_WIDTH, 0, DISPLAY_WIDTH, SCREEN_HEIGHT)
    )

    surfarray.blit_array(
        slow_surface,
        slow_buffer
    )

    surfarray.blit_array(
        fast_surface,
        fast_buffer
    )


    # --------------------------------------------------------
    # Training points
    # --------------------------------------------------------

    for i in range(len(inputs)):

        color = (
            int(expected[i, 1] * 255),
            150,
            int(expected[i, 0] * 255)
        )

        x = int(
            inputs[i, 0] * DISPLAY_WIDTH
        )

        y = int(
            inputs[i, 1] * SCREEN_HEIGHT
        )

        pygame.draw.circle(
            screen,
            color,
            (x, y),
            3
        )

        pygame.draw.circle(
            screen,
            color,
            (DISPLAY_WIDTH + x, y),
            3
        )


    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------

    slow_cost = AI_SLOW.cost(
        inputs,
        expected
    )

    fast_cost = AI_FAST.cost(
        inputs,
        expected
    )


    # --------------------------------------------------------
    # Slow text
    # --------------------------------------------------------

    FONT.render_to(
        screen,
        (10, 10),
        "SLOW LEARN (Manual Gradient Descent)",
        WHITE
    )

    FONT.render_to(
        screen,
        (10, 30),
        f"Cost: {slow_cost:.4f}",
        WHITE
    )

    FONT.render_to(
        screen,
        (10, 50),
        f"Steps: {slow_steps}",
        WHITE
    )

    FONT.render_to(
        screen,
        (10, 70),
        f"Process Time: {slow_time * 1000:.2f} ms",
        WHITE
    )


    # --------------------------------------------------------
    # Fast text
    # --------------------------------------------------------

    FONT.render_to(
        screen,
        (DISPLAY_WIDTH + 10, 10),
        "FAST LEARN (Back Propagation + Momentum)",
        WHITE
    )

    FONT.render_to(
        screen,
        (DISPLAY_WIDTH + 10, 30),
        f"Cost: {fast_cost:.4f}",
        WHITE
    )

    FONT.render_to(
        screen,
        (DISPLAY_WIDTH + 10, 50),
        f"Steps: {fast_steps}",
        WHITE
    )

    FONT.render_to(
        screen,
        (DISPLAY_WIDTH + 10, 70),
        f"Process Time: {fast_time * 1000:.2f} ms",
        WHITE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    pygame.display.set_caption(
        "Neural Network - Slow vs Fast"
    )

    clock = pygame.time.Clock()

    dt = 1

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False


        update(screen, dt)

        pygame.display.flip()

        dt = clock.tick(FPS) / 1000


    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()