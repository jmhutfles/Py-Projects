# Adiabatic pressure vessel filling model using compressible flow through a calibrated orifice
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Add this import

# ---------------------------
# Constants and Parameters
# ---------------------------
R = 259                    # Specific gas constant for oxygen [J/kg.K]
gamma = 1.4                 # Ratio of specific heats for oxygen
cp = gamma * R / (gamma - 1)  # Specific heat at constant pressure [J/kg.K]
cv = R / (gamma - 1)          # Specific heat at constant volume [J/kg.K]

# Orifice and upstream conditions
Cd = 0.65                    # Discharge coefficient
orifice_diameter = 0.0007874     # meters
A_orifice = np.pi * (orifice_diameter / 2)**2
P_upstream = 3.1026e7             # Upstream pressure [Pa]
T_upstream = 300.0           # Upstream temperature [K]

# Pressure vessel
V_vessel = .442              # Vessel volume [m^3]
P0 = 1e5                     # Initial vessel pressure [Pa]
T0 = 300.0                   # Initial vessel temperature [K]

# Simulation settings
dt = 0.000001                    # Time step [s]
t_max = 5                  # Max simulation time [s]
time = np.arange(0, t_max, dt)

# ---------------------------
# Initialization
# ---------------------------
P = [P0]                     # Vessel pressure history
T = [T0]                     # Vessel temperature history
m = [P0 * V_vessel / (R * T0)]  # Initial gas mass in vessel [kg]

# Critical pressure ratio for choking
P_ratio_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

# ---------------------------
# Simulation Loop
# ---------------------------
for t in tqdm(time[1:], desc="Simulating"):  # Wrap the loop with tqdm
    P_d = P[-1]              # Downstream (vessel) pressure
    T_d = T[-1]              # Vessel temperature
    m_d = m[-1]              # Mass in vessel

    # Determine flow regime
    P_ratio = P_d / P_upstream
    if P_ratio > P_ratio_crit:
        # Subsonic (unchoked) flow
        sqrt_arg = (2 * gamma / (R * T_upstream * (gamma - 1))) * (
            (P_ratio)**(2/gamma) - (P_ratio)**((gamma + 1)/gamma)
        )
        mdot = 0.0
        if sqrt_arg > 0:
            mdot = Cd * A_orifice * P_upstream * np.sqrt(sqrt_arg)
    else:
        # Choked flow
        mdot = Cd * A_orifice * P_upstream * np.sqrt(
            gamma / (R * T_upstream) *
            (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        )

    # Energy balance: update temperature (more accurate)
    m_new = m_d + mdot * dt
    if m_new > 0:
        T_new = (m_d * T_d + mdot * dt * (cp / cv) * T_upstream) / m_new
    else:
        T_new = T_d

    # Update mass and pressure
    P_new = m_new * R * T_new / V_vessel

    # Store results
    T.append(T_new)
    P.append(P_new)
    m.append(m_new)
#
# ---------------------------
# Conversion factors
PA_TO_PSI = 0.000145038
K_TO_F = lambda K: (K - 273.15) * 9/5 + 32

# ---------------------------
# Plotting Results
# ---------------------------
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, np.array(P) * PA_TO_PSI, label="Pressure [psi]")
plt.ylabel("Pressure [psi]")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, K_TO_F(np.array(T)), label="Temperature [°F]", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°F]")
plt.grid()

plt.tight_layout()
plt.show()
