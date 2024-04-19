# %%
# import numpy as np
import array_api_strict as np
import matplotlib.pyplot as plt
def render(imgsize):
    # Generate num samples
    y_vals = np.linspace(1, -1, imgsize * 2)
    x_vals = np.linspace(-1, 1, imgsize * 2)

    # Manually broadcast y_vals and x_vals to form a grid
    y = y_vals[:, np.newaxis]
    x = x_vals[np.newaxis, :]
    y = np.asarray(y, dtype=np.complex64)
    x = np.asarray(x, dtype=np.complex64)
    z = x + y * 1j
    img = iterate_vectorized(z)

    # Normalize the image data to be between 0 and 1
    if np.any(np.isnan(img)):
        print("NaN values detected in img, which will affect normalization.")
        img = np.where(np.isnan(img), 0, img)  # Replace NaN with 0
        img = np.where(np.isinf(img), np.finfo(img.dtype).max, img)  # Replace inf with max
        img = np.where(img == -np.inf, np.finfo(img.dtype).min, img)  # Replace -inf with min

    if np.max(img) != np.min(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        img = np.zeros(img.shape)

    colormaps = ['hot', 'plasma', 'inferno', 'twilight']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, cmap in zip(axes.flatten(), colormaps):
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        ax.set_title(cmap)

    plt.tight_layout()
    plt.show()

# Define the complex function and its derivative manually.
def f(z):
    return z**3 - 1

def df(z):
    return 3*z**2

# Vectorized iterate function using NumPy operations.
def iterate_vectorized(z):
    max_iterations = 50 
    tolerance = 1e-4 
    iterations = np.zeros(z.shape, dtype=np.float32)  # Number of iterations for convergence
    for _ in range(max_iterations):
        dz = df(z)
        z_new = z - f(z) / dz  # Newton's update formula: z_new = z - f(z) / f'(z)
        converged = np.abs(z_new - z) < tolerance
        z[~converged] = z_new[~converged]  # Update z only where it has not yet converged
        iterations[~converged] += 1  # Increment iteration count where not yet converged
        if np.all(converged):
            break  # Exit loop if all values have converged

    return iterations  # Num iterations for each element in z

# %%
render(imgsize=600)

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

def find_roots(degree):
    """Find the roots of unity for a given degree. These are complex numbers 
    that, when raised to the 'degree', equal 1."""
    # Super awkward manipulation for strict compliance
    roots = [np.e ** np.asarray(2j * np.pi * k / degree, dtype=np.complex64) for k in range(degree)]
    return np.asarray(roots, dtype=np.complex128)

def f(z, degree):
    """Computes the polynomial \( z^degree - 1 \)."""
    return z**degree - 1

def df(z, degree):
    """Derivative of the polynomial \( z^degree - 1 \)."""
    return degree * z**(degree - 1)

def iterate_vectorized(z, roots, max_iterations=50, tolerance=1e-4):
    iterations = np.zeros(z.shape, dtype=np.float32)  # Store the index of closest root for each point
    for _ in range(max_iterations):
        length = roots.shape[0] # Strict API doesn't do length
        dz = df(z, length)  # Calculate the derivative for the current z
        z_new = z - f(z, length) / dz  # Newton's method update
        converged = np.abs(z_new - z) < tolerance  # Check for convergence
        z[~converged] = z_new[~converged]  # Update only non-converged entries
        if np.all(converged):
            break

    # Calculate the index of the closest root for each point using a manual tolerance check
    for i, root in enumerate(roots):
        mask = np.abs(z - root) < tolerance
        iterations[mask] = i

    return iterations

def render(imgsize, degree):
    # Target points for convergence based on roots
    roots = find_roots(degree)
    # Linear space for x-axis, [-1,1]. "Real" part of our complex numbers.
    x = np.linspace(-1, 1, imgsize)
    # Linear space for  y-axis, [1, -1] (inverse for correct orientation)
    # Imaginary part of our complex numbers.
    y = np.linspace(1, -1, imgsize)
    # Meshgrid which constructs a 2D grid from the 1D arrays of x and y values.
    # Represent the complex plane, where each point (pixel) has a complex number.
    X, Y = np.meshgrid(x, y)
    # Real parts from X and imaginary parts from Y, combined to form complex numbers z = x + yi.
    z = np.asarray(X,dtype=np.complex64) + 1j * np.asarray(Y,dtype=np.complex64)

    img = iterate_vectorized(z, roots)

    # Customize the color mapping for visualization
    hex_colors = ['#15b8fc', '#6ccffc', '#0773c4']  # Intel color palette
    cmap = ListedColormap(hex_colors)

    plt.imshow(img, cmap=cmap)
    plt.title(f"Newton Fractal for Degree {degree} Roots")
    plt.axis("off")
    plt.show()

# Example usage
render(imgsize=1000, degree=5)


# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

def find_roots(degree):
    """Find the roots of unity for a given degree. These are complex numbers 
    that, when raised to the 'degree', equal 1."""
    # Super awkward manipulation for strict compliance
    roots = [np.e ** np.asarray(2j * np.pi * k / degree, dtype=np.complex64) for k in range(degree)]
    return np.asarray(roots, dtype=np.complex128)

def f(z, degree):
    """Computes the polynomial \( z^degree - 1 \)."""
    return z**degree - 1

def df(z, degree):
    """Derivative of the polynomial \( z^degree - 1 \)."""
    return degree * z**(degree - 1)

def iterate_vectorized(z, roots, max_iterations=50, tolerance=1e-4):
    iterations = np.zeros(z.shape, dtype=np.float32)  # Store the index of closest root for each point
    for _ in range(max_iterations):
        length = roots.shape[0] # Strict API doesn't do length
        dz = df(z, length)  # Calculate the derivative for the current z
        z_new = z - f(z, length) / dz  # Newton's method update
        converged = np.abs(z_new - z) < tolerance  # Check for convergence
        z[~converged] = z_new[~converged]  # Update only non-converged entries
        if np.all(converged):
            break

    # Calculate the index of the closest root for each point using a manual tolerance check
    for i, root in enumerate(roots):
        mask = np.abs(z - root) < tolerance
        iterations[mask] = i

    return iterations

def render(imgsize, degree):
    # Target points for convergence based on roots
    roots = find_roots(degree)
    # Linear space for x-axis, [-1,1]. "Real" part of our complex numbers.
    x = np.linspace(-1, 1, imgsize)
    # Linear space for  y-axis, [1, -1] (inverse for correct orientation)
    # Imaginary part of our complex numbers.
    y = np.linspace(1, -1, imgsize)
    # Meshgrid which constructs a 2D grid from the 1D arrays of x and y values.
    # Represent the complex plane, where each point (pixel) has a complex number.
    X, Y = np.meshgrid(x, y)
    # Real parts from X and imaginary parts from Y, combined to form complex numbers z = x + yi.
    z = np.asarray(X,dtype=np.complex64) + 1j * np.asarray(Y,dtype=np.complex64)

    img = iterate_vectorized(z, roots)

    # Customize the color mapping for visualization
    hex_colors = ['#FFFFFF', '#000000', '#777777']  # Black and gray color palette
    cmap = ListedColormap(hex_colors)

    plt.imshow(img, cmap=cmap)
    plt.title(f"Newton Fractal for Degree {degree} Roots")
    plt.axis("off")
    plt.show()

# Example usage
render(imgsize=1000, degree=3)



