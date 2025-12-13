import numpy as np
import matplotlib.pyplot as plt
import csv

# Initialize Parameters
altitude = 1000  # Initial altitude in meters
mass = 80       # Mass of the object in kg
dragArea = 0.5  # Drag area in m²
ViVertical = 100    # Initial vertical velocity in m/s
ViHorizontal = 50  # Initial horizontal velocity in m/s
timeStep = 0.1   # Time step in seconds
g = 9.81        # Acceleration due to gravity in m/s²

# Additional physics constants
airDensity = 1.225  # Air density at sea level in kg/m³
dragCoefficient = 0.47  # Drag coefficient (sphere approximation)

def calculate_air_density(altitude):
    """Calculate air density at given altitude using barometric formula"""
    return 1.225 * np.exp(-altitude / 8400)  # Approximate formula

def calculate_drag_force(velocity, drag_area, altitude):
    """Calculate drag force given velocity, drag area, and altitude"""
    rho = calculate_air_density(altitude)
    v_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
    if v_magnitude == 0:
        return np.array([0.0, 0.0])
    
    drag_magnitude = 0.5 * dragCoefficient * rho * drag_area * v_magnitude**2
    # Drag opposes velocity direction
    drag_force = -drag_magnitude * velocity / v_magnitude
    return drag_force

def simulate_trajectory():
    """Simulate trajectory for the object"""
    # Initialize arrays for storing trajectory data
    time = [0]
    x_pos = [0]
    y_pos = [altitude]
    vx = [ViHorizontal]
    vy = [ViVertical]
    
    # Current state
    current_time = 0
    current_x = 0
    current_y = altitude
    current_vx = ViHorizontal
    current_vy = ViVertical
    
    # Simulation loop
    while current_y > 0:
        # Calculate drag force
        velocity = np.array([current_vx, current_vy])
        drag_force = calculate_drag_force(velocity, dragArea, current_y)
        
        # Calculate accelerations
        ax = drag_force[0] / mass
        ay = -g + drag_force[1] / mass
        
        # Update velocity (Euler integration)
        current_vx += ax * timeStep
        current_vy += ay * timeStep
        
        # Update position
        current_x += current_vx * timeStep
        current_y += current_vy * timeStep
        
        # Update time
        current_time += timeStep
        
        # Store data
        time.append(current_time)
        x_pos.append(current_x)
        y_pos.append(current_y)
        vx.append(current_vx)
        vy.append(current_vy)
        
        # Safety check for extremely long simulations
        if current_time > 300:  # 5 minutes max
            break
    
    return {
        'time': np.array(time),
        'x': np.array(x_pos),
        'y': np.array(y_pos),
        'vx': np.array(vx),
        'vy': np.array(vy)
    }

# Run simulation
trajectory = simulate_trajectory()

# Calculate key results
def analyze_trajectory(traj):
    """Analyze trajectory results"""
    flight_time = traj['time'][-1]
    max_range = traj['x'][-1]
    final_velocity = np.sqrt(traj['vx'][-1]**2 + traj['vy'][-1]**2)
    max_altitude = np.max(traj['y'])
    
    return {
        'flight_time': flight_time,
        'range': max_range,
        'impact_velocity': final_velocity,
        'max_altitude': max_altitude
    }

results = analyze_trajectory(trajectory)

# Print results
print("DART TRAJECTORY SIMULATION RESULTS")
print("=" * 50)
print(f"Initial Conditions:")
print(f"  Altitude: {altitude} m")
print(f"  Mass: {mass} kg")
print(f"  Drag Area: {dragArea} m²")
print(f"  Initial Horizontal Velocity: {ViHorizontal} m/s")
print(f"  Initial Vertical Velocity: {ViVertical} m/s")
print()

print("Results:")
print(f"  Flight Time: {results['flight_time']:.2f} s")
print(f"  Range: {results['range']:.2f} m")
print(f"  Impact Velocity: {results['impact_velocity']:.2f} m/s")
print(f"  Maximum Altitude: {results['max_altitude']:.2f} m")

# Create visualization
plt.figure(figsize=(12, 8))

# Trajectory plot
plt.subplot(2, 2, 1)
plt.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2)
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Altitude (m)')
plt.title('Trajectory')
plt.grid(True, alpha=0.3)

# Velocity vs time
plt.subplot(2, 2, 2)
speed = np.sqrt(trajectory['vx']**2 + trajectory['vy']**2)
plt.plot(trajectory['time'], speed, 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed vs Time')
plt.grid(True, alpha=0.3)

# Horizontal velocity vs time
plt.subplot(2, 2, 3)
plt.plot(trajectory['time'], trajectory['vx'], 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Horizontal Velocity (m/s)')
plt.title('Horizontal Velocity vs Time')
plt.grid(True, alpha=0.3)

# Vertical velocity vs time
plt.subplot(2, 2, 4)
plt.plot(trajectory['time'], trajectory['vy'], 'b-')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Velocity (m/s)')
plt.title('Vertical Velocity vs Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()







