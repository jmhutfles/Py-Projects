import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from tqdm import tqdm

# --------------------------
# Constants and Initial Conditions
# --------------------------
mass = 2000                  # kg
g = 9.81                     # m/s²
CdA_drogue = 8.2 + 7.5       # Drogue drag area (m²)
CdA_main = 312.7 + 7.5       # Main parachute drag area (m²)
r0 = 1.225                   # Sea-level air density (kg/m³)
H = 7200                     # Scale height (m)
tau = 7                      # Inflation time constant (s)

Vxi = 70                     # Initial horizontal velocity (m/s)
Vyi = 0                      # Initial vertical velocity (m/s)
Hi = 6100                    # Initial altitude (m)
xi = 0                       # Initial horizontal position (m)
Dt = 0.25                    # Time step (s)
deploy_altitude = 500        # Main parachute deployment altitude (m)

# --------------------------
# Initialize state variables
# --------------------------
CdA = CdA_drogue
t = 0
x = xi
y = Hi
vx = Vxi
vy = Vyi
deploy_time = None
inflating = False

# --------------------------
# Data storage
# --------------------------
timelist = [0]
x_list = [xi]
y_list = [Hi]
vx_list = [Vxi]
vy_list = [Vyi]

# --------------------------
# Simulation Loop
# --------------------------
max_steps = int((Hi / g) / Dt * 2)

for _ in tqdm(range(max_steps), desc="Simulating Drogue Fall"):
    if y <= 0:
        break

    # Begin inflation
    if y <= deploy_altitude and not inflating:
        deploy_time = t
        inflating = True
        print(f"Main parachute deployment started at t={t:.2f}s, altitude={y:.2f}m")

    # Smooth inflation
    if inflating:
        CdA = CdA_main - (CdA_main - CdA_drogue) * np.exp(-(t - deploy_time) / tau)

    # Air density
    density = r0 * np.exp(-y / H)

    # Velocity magnitude
    V = np.sqrt(vx**2 + vy**2)

    # Drag force
    Fd = 0.5 * density * CdA * V**2

    # Accelerations
    ax = -Fd * (vx / V) / mass
    ay = -Fd * (vy / V) / mass - g

    # Velocity update
    vx += ax * Dt
    vy += ay * Dt

    # Position update
    x += vx * Dt
    y += vy * Dt

    # Time update
    t += Dt

    # Append data
    timelist.append(t)
    x_list.append(x)
    y_list.append(max(0, y))  # Clamp to ground
    vx_list.append(vx)
    vy_list.append(vy)


# --------------------------
# Extend animation to hold last frame
# --------------------------
hold_frames = int(10 / Dt)  # Hold for 10 seconds

for _ in range(hold_frames):
    timelist.append(t)
    x_list.append(x_list[-1])
    y_list.append(0)
    vx_list.append(0)
    vy_list.append(0)


# --------------------------
# Animation
# --------------------------
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(min(x_list), max(x_list) + 10000)
ax.set_ylim(0, Hi * 1.1)
ax.set_xlabel("Horizontal Position (m)")
ax.set_ylabel("Altitude (m)")
ax.set_title("Parachute Descent Animation")

dot, = ax.plot([], [], 'o', color='blue', markersize=6)
trail, = ax.plot([], [], '-', color='gray', linewidth=1)
parachute_patch = plt.Circle((0, 0), radius=0, color='red', alpha=0.4)
ax.add_patch(parachute_patch)

# Speed, Altitude, Time text (upper-left corner)
info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')

# Deployment message text (upper-center)
deploy_text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center', fontsize=14, color='green')

# Touchdown congratulations message (centered)
touchdown_text = ax.text(0.5, 0.5, '', transform=ax.transAxes, ha='center', fontsize=18, color='blue', weight='bold')

def update(frame):
    if frame >= len(x_list):
        return dot, trail, parachute_patch, info_text, deploy_text, touchdown_text

    x = x_list[frame]
    y = y_list[frame]

    dot.set_data([x], [y])
    trail.set_data(x_list[:frame], y_list[:frame])

    # Deployment effects
    if deploy_time is not None and timelist[frame] >= deploy_time:
        time_since_deploy = timelist[frame] - deploy_time
        CdA_current = CdA_main - (CdA_main - CdA_drogue) * np.exp(-time_since_deploy / tau)
        parachute_patch.set_color('green')  # color changes on deployment
        deploy_text.set_text("Main Parachute Deployed!")
    else:
        CdA_current = CdA_drogue
        parachute_patch.set_color('red')
        deploy_text.set_text('')

    radius = math.sqrt(CdA_current / math.pi)
    parachute_patch.center = (x, y + 100)
    parachute_patch.set_radius(radius * 0.2)  # scale for visibility

    # Display speed, altitude, time
    vx = vx_list[frame]
    vy = vy_list[frame]
    speed = np.sqrt(vx**2 + vy**2)
    altitude = y_list[frame]
    current_time = timelist[frame]
    info_text.set_text(f"Speed: {speed:.1f} m/s\nAltitude MSL: {altitude:.1f} m\nTime: {current_time:.1f} s")

    # Touchdown message
    if altitude <= 0:
        touchdown_text.set_text("Congratulations! Touchdown!")
    else:
        touchdown_text.set_text('')

    return dot, trail, parachute_patch, info_text, deploy_text, touchdown_text

ani = animation.FuncAnimation(fig, update, frames=len(x_list), interval=Dt*500, blit=True)
plt.show()
