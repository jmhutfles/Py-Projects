import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

class DARTSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DART Trajectory Simulator")
        self.root.geometry("1200x800")
        
        # Default parameters
        self.params = {
            'altitude': 1000,
            'mass': 80,
            'dragArea': 0.5,
            'ViVertical': 100,
            'ViHorizontal': 50,
            'timeStep': 0.1,
            'g': 9.81,
            'dragCoefficient': 0.47
        }
        
        self.trajectory_history = []
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_parameter_controls()
        self.setup_plot_area()
        self.setup_results_area()
        
    def setup_parameter_controls(self):
        # Parameter input frame
        param_frame = ttk.LabelFrame(self.control_frame, text="Simulation Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.param_vars = {}
        param_labels = {
            'altitude': 'Initial Altitude (m)',
            'mass': 'Mass (kg)',
            'dragArea': 'Drag Area (m²)',
            'ViVertical': 'Initial Vertical Velocity (m/s)',
            'ViHorizontal': 'Initial Horizontal Velocity (m/s)',
            'timeStep': 'Time Step (s)',
            'g': 'Gravity (m/s²)',
            'dragCoefficient': 'Drag Coefficient'
        }
        
        for i, (param, label) in enumerate(param_labels.items()):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            
            var = tk.DoubleVar(value=self.params[param])
            entry = ttk.Entry(param_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky='ew', pady=2, padx=(10, 0))
            
            self.param_vars[param] = var
        
        param_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear History", command=self.clear_history).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters).pack(fill=tk.X, pady=2)
        
    def setup_plot_area(self):
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        self.fig.tight_layout()
        
    def setup_results_area(self):
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=15, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def calculate_air_density(self, altitude):
        """Calculate air density at given altitude using barometric formula"""
        return 1.225 * np.exp(-altitude / 8400)
    
    def calculate_drag_force(self, velocity, drag_area, altitude, drag_coeff):
        """Calculate drag force given velocity, drag area, and altitude"""
        rho = self.calculate_air_density(altitude)
        v_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if v_magnitude == 0:
            return np.array([0.0, 0.0])
        
        drag_magnitude = 0.5 * drag_coeff * rho * drag_area * v_magnitude**2
        drag_force = -drag_magnitude * velocity / v_magnitude
        return drag_force
    
    def simulate_trajectory(self, params):
        """Simulate trajectory for the object"""
        # Initialize arrays
        time = [0]
        x_pos = [0]
        y_pos = [params['altitude']]
        vx = [params['ViHorizontal']]
        vy = [params['ViVertical']]
        
        # Current state
        current_time = 0
        current_x = 0
        current_y = params['altitude']
        current_vx = params['ViHorizontal']
        current_vy = params['ViVertical']
        
        # Simulation loop
        while current_y > 0 and current_time < 300:
            velocity = np.array([current_vx, current_vy])
            drag_force = self.calculate_drag_force(velocity, params['dragArea'], 
                                                 current_y, params['dragCoefficient'])
            
            # Calculate accelerations
            ax = drag_force[0] / params['mass']
            ay = -params['g'] + drag_force[1] / params['mass']
            
            # Update velocity and position
            current_vx += ax * params['timeStep']
            current_vy += ay * params['timeStep']
            current_x += current_vx * params['timeStep']
            current_y += current_vy * params['timeStep']
            current_time += params['timeStep']
            
            # Store data
            time.append(current_time)
            x_pos.append(current_x)
            y_pos.append(current_y)
            vx.append(current_vx)
            vy.append(current_vy)
        
        return {
            'time': np.array(time),
            'x': np.array(x_pos),
            'y': np.array(y_pos),
            'vx': np.array(vx),
            'vy': np.array(vy),
            'params': params.copy()
        }
    
    def analyze_trajectory(self, traj):
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
    
    def run_simulation(self):
        try:
            # Update parameters from GUI
            for param, var in self.param_vars.items():
                self.params[param] = var.get()
            
            # Run simulation in separate thread to prevent GUI freezing
            def simulate():
                trajectory = self.simulate_trajectory(self.params)
                results = self.analyze_trajectory(trajectory)
                
                # Store trajectory
                trajectory['results'] = results
                self.trajectory_history.append(trajectory)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.update_display(trajectory, results))
            
            threading.Thread(target=simulate, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")
    
    def update_display(self, trajectory, results):
        """Update plots and results display"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot all trajectories in history
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectory_history)))
        
        for i, traj in enumerate(self.trajectory_history):
            alpha = 0.3 if i < len(self.trajectory_history) - 1 else 1.0
            linewidth = 1 if i < len(self.trajectory_history) - 1 else 2
            color = colors[i]
            
            # Trajectory plot
            self.ax1.plot(traj['x'], traj['y'], color=color, alpha=alpha, linewidth=linewidth)
            
            # Speed vs time
            speed = np.sqrt(traj['vx']**2 + traj['vy']**2)
            self.ax2.plot(traj['time'], speed, color=color, alpha=alpha, linewidth=linewidth)
            
            # Horizontal velocity vs time
            self.ax3.plot(traj['time'], traj['vx'], color=color, alpha=alpha, linewidth=linewidth)
            
            # Vertical velocity vs time
            self.ax4.plot(traj['time'], traj['vy'], color=color, alpha=alpha, linewidth=linewidth)
        
        # Set labels and titles
        self.ax1.set_xlabel('Horizontal Distance (m)')
        self.ax1.set_ylabel('Altitude (m)')
        self.ax1.set_title('Trajectory')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed (m/s)')
        self.ax2.set_title('Speed vs Time')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Horizontal Velocity (m/s)')
        self.ax3.set_title('Horizontal Velocity vs Time')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Vertical Velocity (m/s)')
        self.ax4.set_title('Vertical Velocity vs Time')
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update results text
        self.update_results_display(results)
    
    def update_results_display(self, results):
        """Update the results text display"""
        self.results_text.insert(tk.END, f"\n{'='*30}\n")
        self.results_text.insert(tk.END, f"Simulation #{len(self.trajectory_history)}\n")
        self.results_text.insert(tk.END, f"Flight Time: {results['flight_time']:.2f} s\n")
        self.results_text.insert(tk.END, f"Range: {results['range']:.2f} m\n")
        self.results_text.insert(tk.END, f"Impact Velocity: {results['impact_velocity']:.2f} m/s\n")
        self.results_text.insert(tk.END, f"Max Altitude: {results['max_altitude']:.2f} m\n")
        
        # Scroll to bottom
        self.results_text.see(tk.END)
    
    def clear_history(self):
        """Clear trajectory history and plots"""
        self.trajectory_history.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.canvas.draw()
        self.results_text.delete(1.0, tk.END)
    
    def reset_parameters(self):
        """Reset parameters to default values"""
        defaults = {
            'altitude': 1000,
            'mass': 80,
            'dragArea': 0.5,
            'ViVertical': 100,
            'ViHorizontal': 50,
            'timeStep': 0.1,
            'g': 9.81,
            'dragCoefficient': 0.47
        }
        
        for param, value in defaults.items():
            self.param_vars[param].set(value)

if __name__ == "__main__":
    root = tk.Tk()
    app = DARTSimulatorGUI(root)
    root.mainloop()
