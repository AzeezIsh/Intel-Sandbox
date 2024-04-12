import numpy as np
import matplotlib.pyplot as plt

# Define the complex function and its derivative manually.
def f(z):
    return z**3 - 1

def df(z):
    return 3*z**2

# Iteration function using NumPy operations.
def iterate(z):
    num = 0
    for _ in range(50):  # Max iterations
        # Avoid division by zero
        dz = df(z)
        with np.errstate(all='ignore'):  # Ignore warnings from divisions etc.
            z_new = z - f(z) / dz
            # Break if change is below threshold
            if np.all(np.abs(z_new - z) < 1e-4):
                break
            z = z_new
        num += np.exp(-1 / np.abs(z_new - z))
    return num

def render(imgsize):
    # Create a complex grid
    y, x = np.ogrid[1 : -1 : imgsize * 2j, -1 : 1 : imgsize * 2j]
    z = x + y * 1j

    # Initialize an empty image
    img = np.zeros(z.shape, dtype=float)

    # Apply the iterate function to each element
    vectorized_iterate = np.vectorize(iterate)
    img = vectorized_iterate(z)

    # Plotting
    fig = plt.figure(figsize=(imgsize / 100.0, imgsize / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], aspect=1)
    ax.axis("off")
    ax.imshow(img, cmap="hot")
    plt.show()  # Show the image

if __name__ == "__main__":
    render(imgsize=600)
