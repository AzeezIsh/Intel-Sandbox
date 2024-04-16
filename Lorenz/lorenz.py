# import numpy as np
import array_api_strict as np
import matplotlib.pyplot as plt
# chaotic solution 
# σ is the Prandtl number representing the ratio of the fluid viscosity to its thermal conductivity
# ρ represents the difference in temperature between the top and bottom of the system
# β is the ratio of the width to height of the box used to hold the system
σ, ρ, β = 10, 28, 8 / 3


dt = 0.01  # sample rate in seconds.
x = 20000

dxdt = np.empty(x + 1)
dydt = np.empty(x + 1)
dzdt = np.empty(x + 1)

# Initial values
dxdt[0], dydt[0], dzdt[0] = (0.0, 1.0, 1.05)

for i in range(x):
    dxdt[i + 1] = dxdt[i] + σ * (dydt[i] - dxdt[i]) * dt # P(y-x)
    dydt[i + 1] = dydt[i] + (dxdt[i] * (ρ - dzdt[i]) - dydt[i]) * dt # Rx - y - xz
    dzdt[i + 1] = dzdt[i] + (dxdt[i] * dydt[i] - β * dzdt[i]) * dt # xy - Bz

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a color gradient based on the iteration
colors = plt.cm.jet(np.linspace(0, 1, x + 1))

# Plot with color gradient
for i in range(x):
    xG = list(dxdt[i:i+2])
    yG = list(dydt[i:i+2])
    zG = list(dzdt[i:i+2])
    colorsG = list(colors[i])
    ax.plot(xG, yG, zG, color=colorsG, alpha=0.6)

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor with Color Gradient")

plt.show()


