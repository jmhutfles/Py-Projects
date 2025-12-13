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
        self.root.geometry("1400x800")
        
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
        
        # Unit conversion factors to metric
        self.unit_conversions = {
            'altitude': {'m': 1.0, 'ft': 0.3048, 'km': 1000.0},
            'mass': {'kg': 1.0, 'lb': 0.453592, 'g': 0.001},
            'dragArea': {'m²': 1.0, 'ft²': 0.092903, 'cm²': 0.0001},
            'ViVertical': {'m/s': 1.0, 'ft/s': 0.3048, 'km/h': 0.277778, 'mph': 0.44704},
            'ViHorizontal': {'m/s': 1.0, 'ft/s': 0.3048, 'km/h': 0.277778, 'mph': 0.44704},
            'timeStep': {'s': 1.0, 'ms': 0.001},
            'g': {'m/s²': 1.0, 'ft/s²': 0.3048}
        }
        
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
        self.unit_vars = {}
        
        param_labels = {
            'altitude': 'Initial Altitude',
            'mass': 'Mass',
            'dragArea': 'Drag Area',
            'ViVertical': 'Initial Vertical Velocity',
            'ViHorizontal': 'Initial Horizontal Velocity',
            'timeStep': 'Time Step',
            'g': 'Gravity',
            'dragCoefficient': 'Drag Coefficient'
        }
        
        for i, (param, label) in enumerate(param_labels.items()):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            
            # Value entry
            var = tk.DoubleVar(value=self.params[param])
            entry = ttk.Entry(param_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky='ew', pady=2, padx=(10, 5))
            self.param_vars[param] = var
            
            # Unit dropdown
            if param in self.unit_conversions:
                unit_var = tk.StringVar()
                unit_options = list(self.unit_conversions[param].keys())
                unit_var.set(unit_options[0])  # Set default to first option (metric)
                
                unit_combo = ttk.Combobox(param_frame, textvariable=unit_var, 
                                        values=unit_options, width=8, state='readonly')
                unit_combo.grid(row=i, column=2, sticky='w', pady=2, padx=(0, 10))
                self.unit_vars[param] = unit_var
            else:
                # For dragCoefficient (unitless)
                ttk.Label(param_frame, text="(unitless)").grid(row=i, column=2, sticky='w', pady=2, padx=(0, 10))
        
        param_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters).pack(fill=tk.X, pady=2)
    
    def convert_to_metric(self, param, value, unit):
        """Convert parameter value to metric units"""
        if param in self.unit_conversions and unit in self.unit_conversions[param]:
            return value * self.unit_conversions[param][unit]
        return value
    
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
            # Update parameters from GUI with unit conversion
            for param, var in self.param_vars.items():
                value = var.get()
                if param in self.unit_vars:
                    unit = self.unit_vars[param].get()
                    self.params[param] = self.convert_to_metric(param, value, unit)
                else:
                    self.params[param] = value
            
            # Run simulation in separate thread to prevent GUI freezing
            def simulate():
                trajectory = self.simulate_trajectory(self.params)
                results = self.analyze_trajectory(trajectory)
                
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
        
        # Plot trajectory
        self.ax1.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2)
        
        # Speed vs time
        speed = np.sqrt(trajectory['vx']**2 + trajectory['vy']**2)
        self.ax2.plot(trajectory['time'], speed, 'b-', linewidth=2)
        
        # Horizontal velocity vs time
        self.ax3.plot(trajectory['time'], trajectory['vx'], 'b-', linewidth=2)
        
        # Vertical velocity vs time
        self.ax4.plot(trajectory['time'], trajectory['vy'], 'b-', linewidth=2)
        
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
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "DART Trajectory Results\n")
        self.results_text.insert(tk.END, "="*25 + "\n\n")
        self.results_text.insert(tk.END, f"Flight Time: {results['flight_time']:.2f} s\n")
        self.results_text.insert(tk.END, f"Range: {results['range']:.2f} m\n")
        self.results_text.insert(tk.END, f"         {results['range']*3.28084:.2f} ft\n")
        self.results_text.insert(tk.END, f"         {results['range']/1000:.3f} km\n")
        self.results_text.insert(tk.END, f"Impact Velocity: {results['impact_velocity']:.2f} m/s\n")
        self.results_text.insert(tk.END, f"                 {results['impact_velocity']*3.28084:.2f} ft/s\n")
        self.results_text.insert(tk.END, f"                 {results['impact_velocity']*2.23694:.2f} mph\n")
        self.results_text.insert(tk.END, f"Max Altitude: {results['max_altitude']:.2f} m\n")
        self.results_text.insert(tk.END, f"              {results['max_altitude']*3.28084:.2f} ft\n")
    
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
            
        # Reset units to metric defaults
        for param, unit_var in self.unit_vars.items():
            unit_options = list(self.unit_conversions[param].keys())
            unit_var.set(unit_options[0])

if __name__ == "__main__":
    root = tk.Tk()
    app = DARTSimulatorGUI(root)
    root.mainloop()
