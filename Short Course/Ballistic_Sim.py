import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm  # ✅ Add this line

# Defining constants and initial conditions
mass = 2000  # kg
g = 9.81  # m/s²
CdA = 15.3 + 7  # Effective drag area (Cd*A), m²
r0 = 1.225  # Air density at sea level, kg/m³
H = 7200  # Scale height for atmosphere, m

Vxi = 70  # Initial horizontal velocity, m/s
Vyi = 0   # Initial vertical velocity, m/s
thetai = 0  # Initial angle, degrees
Hi = 6100  # Initial altitude, m
xi = 0     # Initial horizontal position, m
Dt = 0.25  # Time step, s

# Initialize lists for storing results
index = [0]
timelist = [0]
Vxy_list = [np.sqrt(Vxi**2 + Vyi**2)]
thetalist = [thetai]
y_list = [Hi]
x_list = [xi]
density_list = []
axy_list = []
angular_acceleration_list = []
vy_list = [Vyi]
vx_list = [Vxi]

# Initialize variables
t = 0
x = xi
y = Hi
vx = Vxi
vy = Vyi

# Estimate a safe upper bound for steps
max_steps = int((Hi / g) / Dt * 2)

# Simulation loop with tqdm progress bar
for _ in tqdm(range(max_steps), desc="Simulating Drogue Fall"):
    if y <= 0:
        break  # Exit if drogue hits the ground

    # Compute air density at current altitude
    density = r0 * np.exp(-y / H)

    # Compute velocity magnitude
    V = np.sqrt(vx**2 + vy**2)

    # Compute drag force magnitude
    Fd = 0.5 * density * CdA * V**2

    # Compute acceleration components
    ax = -Fd * (vx / V) / mass  # Drag opposes motion
    ay = -Fd * (vy / V) / mass - g  # Drag + gravity

    # Update velocities
    vx += ax * Dt
    vy += ay * Dt

    # Update positions
    x += vx * Dt
    y += vy * Dt

    # Update time
    t += Dt

    # Compute descent angle (from horizontal)
    theta = np.degrees(np.arctan2(-vy, vx))

    # Append results
    index.append(index[-1] + 1)
    timelist.append(t)
    Vxy_list.append(V)
    thetalist.append(theta)
    y_list.append(y)
    x_list.append(x)
    density_list.append(density)
    axy_list.append(np.sqrt(ax**2 + ay**2))
    angular_acceleration_list.append(0)  # Placeholder
    vy_list.append(vy)
    vx_list.append(vx)

# Plot the trajectory
fig1, axs1 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axs1[0].plot(timelist, y_list, label="Altitude", color='tab:blue')
axs1[0].set_ylabel("Altitude (m)")
axs1[0].set_title("Altitude vs Time")
axs1[0].grid(True)

axs1[1].plot(timelist, x_list, label="X Position", color='tab:purple')
axs1[1].set_ylabel("X Position (m)")
axs1[1].set_title("Horizontal Position vs Time")
axs1[1].grid(True)

axs1[2].plot(timelist, vx_list, label="Vx", color='tab:orange')
axs1[2].set_ylabel("Vx (m/s)")
axs1[2].set_title("Horizontal Velocity vs Time")
axs1[2].grid(True)

axs1[3].plot(timelist, vy_list, label="Vy", color='tab:green')
axs1[3].set_ylabel("Vy (m/s)")
axs1[3].set_title("Vertical Velocity vs Time")
axs1[3].set_xlabel("Time (s)")
axs1[3].grid(True)

plt.tight_layout()
plt.show()
