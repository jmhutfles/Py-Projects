# Two adiabatic pressure vessels in series with two orifices (compressible flow)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# Constants and Parameters
# ---------------------------
R = 259                    # Specific gas constant for oxygen [J/kg.K]
gamma = 1.4                # Ratio of specific heats for oxygen
cp = gamma * R / (gamma - 1)
cv = R / (gamma - 1)

# Orifice and upstream conditions
Cd = 0.65
orifice_diameter1 = 0.0007874  # Orifice 1 (upstream to vessel 1)
orifice_diameter2 = 0.0007874  # Orifice 2 (vessel 1 to vessel 2)
A_orifice1 = np.pi * (orifice_diameter1 / 2)**2
A_orifice2 = np.pi * (orifice_diameter2 / 2)**2
P_upstream = 3.1026e7
T_upstream = 300.0

# Pressure vessels
V_vessel1 = 4.83e-5 # Vessel 1 volume [m^3]
V_vessel2 = .442 # Vessel 2 volume [m^3]
P0_1 = 1e5 # Initial pressure for vessel 1 [Pa]
T0_1 = 300.0 # Initial temperature for vessel 1 [K]
P0_2 = 1e5 # Initial pressure for vessel 2 [Pa]
T0_2 = 300.0 # Initial temperature for vessel 2 [K]

# Simulation settings
dt = 0.01 # Time step [s]
t_max = 10000 # Max simulation time [s]
time = np.arange(0, t_max, dt)

# Conversion factors
PA_TO_PSI = 0.000145038
K_TO_F = lambda K: (K - 273.15) * 9/5 + 32

# ---------------------------
# Initialization
# ---------------------------
P1 = [P0_1]
T1 = [T0_1]
m1 = [P0_1 * V_vessel1 / (R * T0_1)]

P2 = [P0_2]
T2 = [T0_2]
m2 = [P0_2 * V_vessel2 / (R * T0_2)]

# Critical pressure ratio for choking
P_ratio_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

# ---------------------------
# Simulation Loop
# ---------------------------
for t in tqdm(time[1:], desc="Simulating"):
    # --- Vessel 1 ---
    P1_d = P1[-1]
    T1_d = T1[-1]
    m1_d = m1[-1]

    # Flow from upstream to vessel 1
    P_ratio1 = P1_d / P_upstream
    if P_ratio1 > P_ratio_crit:
        sqrt_arg1 = (2 * gamma / (R * T_upstream * (gamma - 1))) * (
            (P_ratio1)**(2/gamma) - (P_ratio1)**((gamma + 1)/gamma)
        )
        mdot1 = 0.0
        if sqrt_arg1 > 0:
            mdot1 = Cd * A_orifice1 * P_upstream * np.sqrt(sqrt_arg1)
    else:
        mdot1 = Cd * A_orifice1 * P_upstream * np.sqrt(
            gamma / (R * T_upstream) *
            (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        )

    # Flow from vessel 1 to vessel 2
    P2_d = P2[-1]
    T2_d = T2[-1]
    m2_d = m2[-1]

    P_ratio2 = P2_d / P1_d if P1_d > 0 else 0
    if P_ratio2 > P_ratio_crit:
        sqrt_arg2 = (2 * gamma / (R * T1_d * (gamma - 1))) * (
            (P_ratio2)**(2/gamma) - (P_ratio2)**((gamma + 1)/gamma)
        )
        mdot2 = 0.0
        if sqrt_arg2 > 0:
            mdot2 = Cd * A_orifice2 * P1_d * np.sqrt(sqrt_arg2)
    else:
        mdot2 = Cd * A_orifice2 * P1_d * np.sqrt(
            gamma / (R * T1_d) *
            (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        )

    # Ensure no negative mass flow if P2 > P1
    if P2_d >= P1_d:
        mdot2 = 0.0

    # --- Update vessel 1 ---
    m1_new = m1_d + (mdot1 - mdot2) * dt
    if m1_new > 0:
        T1_new = (m1_d * T1_d + mdot1 * dt * (cp / cv) * T_upstream - mdot2 * dt * (cp / cv) * T1_d) / m1_new
    else:
        T1_new = T1_d
    P1_new = m1_new * R * T1_new / V_vessel1

    # --- Update vessel 2 ---
    m2_new = m2_d + mdot2 * dt
    if m2_new > 0:
        T2_new = (m2_d * T2_d + mdot2 * dt * (cp / cv) * T1_d) / m2_new
    else:
        T2_new = T2_d
    P2_new = m2_new * R * T2_new / V_vessel2

    # Store results
    P1.append(P1_new)
    T1.append(T1_new)
    m1.append(m1_new)
    P2.append(P2_new)
    T2.append(T2_new)
    m2.append(m2_new)

# ---------------------------
# Plotting Results
# ---------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, np.array(P1) * PA_TO_PSI, label="Vessel 1 Pressure [psi]")
plt.plot(time, np.array(P2) * PA_TO_PSI, label="Vessel 2 Pressure [psi]")
plt.ylabel("Pressure [psi]")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, K_TO_F(np.array(T1)), label="Vessel 1 Temperature [°F]", color='orange')
plt.plot(time, K_TO_F(np.array(T2)), label="Vessel 2 Temperature [°F]", color='red')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°F]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
