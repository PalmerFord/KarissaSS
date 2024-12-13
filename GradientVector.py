import math
import noise
import matplotlib.pyplot as plt

def perlin_descent_direction(x, y, epsilon=1e-3):
    # Compute partial derivatives
    d_noise_dx = (noise.pnoise2(x + epsilon, y) - noise.pnoise2(x - epsilon, y)) / (2 * epsilon)
    d_noise_dy = (noise.pnoise2(x, y + epsilon) - noise.pnoise2(x, y - epsilon)) / (2 * epsilon)

    return -d_noise_dx, -d_noise_dy

def draw_arrows(grid_size=64, scale=0.1, tangent=False):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    for i in range(grid_size):
        for j in range(grid_size):
            x, y = i * scale, j * scale
            dx, dy = perlin_descent_direction(x, y)

            magnitude = math.sqrt(dx**2 + dy**2)
            if magnitude > 0:
                dx /= magnitude
                dy /= magnitude

            if tangent:
                dx, dy = -dy, dx

            ax.arrow(i, j, dx * 0.5, dy * 0.5, head_width=0.3, head_length=0.4, fc='blue', ec='blue')

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

draw_arrows(grid_size=64, scale=0.1, tangent=True)