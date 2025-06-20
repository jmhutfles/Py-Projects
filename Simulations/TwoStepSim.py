import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

while True:

    # --------------------------
    # Constants and Initial Conditions
    # --------------------------
    mass = 2000  # kg
    g = 9.81     # m/s²
    CdA_drogue = 8.2 + 7.5           # Initial drogue drag area (m²)
    CdA_main = 312.7 + 7.5          # Main parachute drag area (m²)
    r0 = 1.225                      # Sea-level air density (kg/m³)
    H = 7200                        # Scale height (m)
    tau = 7                       # Inflation time constant (s)

    Vxi = 70      # Initial horizontal velocity (m/s)
    Vyi = 0       # Initial vertical velocity (m/s)
    Hi = 6100     # Initial altitude (m)
    xi = 0        # Initial horizontal position (m)
    Dt = 0.25     # Time step (s)
    deploy_altitude = 500  # Main parachute deployment altitude (m)

    # --------------------------
    # Initialize state variables
    # --------------------------
    CdA = CdA_drogue  # Start with drogue
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
    index = [0]
    timelist = [0]
    Vxy_list = [np.sqrt(Vxi**2 + Vyi**2)]
    thetalist = [0]
    y_list = [Hi]
    x_list = [xi]
    density_list = []
    axy_list = []
    angular_acceleration_list = []
    vy_list = [Vyi]
    vx_list = [Vxi]
    force_list = []
    gforce_list = []

    # Estimate a safe upper limit
    max_steps = int((Hi / g) / Dt * 2)

    # --------------------------
    # Simulation Loop
    # --------------------------
    for _ in tqdm(range(max_steps), desc="Simulating Drogue Fall"):
        if y <= 0:
            break

        # Begin inflation if below deploy altitude
        if y <= deploy_altitude and not inflating:
            deploy_time = t
            inflating = True
            print(f"Main parachute deployment started at t={t:.2f}s, altitude={y:.2f}m")

        # Smooth inflation: exponential increase of CdA
        if inflating:
            CdA = CdA_main - (CdA_main - CdA_drogue) * np.exp(-(t - deploy_time) / tau)

        # Air density at current altitude
        density = r0 * np.exp(-y / H)

        # Total velocity magnitude
        V = np.sqrt(vx**2 + vy**2)

        # Drag force
        Fd = 0.5 * density * CdA * V**2

        # Accelerations
        ax = -Fd * (vx / V) / mass
        ay = -Fd * (vy / V) / mass - g
        a_total = np.sqrt(ax**2 + ay**2)
        g_force = a_total / g

        # Velocity update
        vx += ax * Dt
        vy += ay * Dt

        # Position update
        x += vx * Dt
        y += vy * Dt

        # Time update
        t += Dt

        # Angle from horizontal
        theta = np.degrees(np.arctan2(-vy, vx))

        # Append results
        index.append(index[-1] + 1)
        timelist.append(t)
        Vxy_list.append(V)
        thetalist.append(theta)
        y_list.append(y)
        x_list.append(x)
        density_list.append(density)
        axy_list.append(a_total)
        angular_acceleration_list.append(0)
        vy_list.append(vy)
        vx_list.append(vx)
        force_list.append(Fd)
        gforce_list.append(g_force)

    # --------------------------
    # Plotting in separate windows
    # --------------------------

    # Plot: Altitude vs Time
    fig_alt = plt.figure()
    plt.plot(timelist, y_list, label="Altitude", color='tab:blue')
    plt.title("Altitude vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Plot: X Position vs Time
    fig_x = plt.figure()
    plt.plot(timelist, x_list, label="X Position", color='tab:purple')
    plt.title("Horizontal Position vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Plot: Vx vs Time
    fig_vx = plt.figure()
    plt.plot(timelist, vx_list, label="Vx", color='tab:orange')
    plt.title("Horizontal Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Vx (m/s)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Plot: Vy vs Time
    fig_vy = plt.figure()
    plt.plot(timelist, vy_list, label="Vy", color='tab:green')
    plt.title("Vertical Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Vy (m/s)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Plot: Drag Force vs Time
    fig_force = plt.figure()
    plt.plot(timelist[1:], force_list, label="Drag Force", color='tab:red')
    plt.title("Drag Force vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Plot: g-Force vs Time
    fig_gforce = plt.figure()
    plt.plot(timelist[1:], gforce_list, label="g-Force", color='tab:cyan')
    plt.title("g-Force vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("g-Force (g)")
    plt.grid(True)
    if deploy_time:
        plt.axvline(deploy_time, color='gray', linestyle='--', label='Main Deploy')
        plt.legend()

    # Show all plots
    plt.show()
    if input("Run again? (y/n): ").lower() != 'y':
        break
    plt.close('all')  # Close all figures before the next run
