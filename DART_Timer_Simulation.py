import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import math

class DARTTimerSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DART Parachute Test Timer Calculator")
        self.root.geometry("1600x1000")
        
        # Default DART test parameters
        self.params = {
            # Basic vehicle parameters
            'altitude_msl': 15000,           # Drop altitude MSL (ft)
            'dart_weight': 150,              # DART weight (lbs)
            'freefall_drag_area': 0.125,    # DART freefall drag area x Cd (sq ft)
            'drogue_drag_area': 10.4,       # Drogue parachute drag area x Cd (sq ft)
            
            # Aircraft parameters
            'aircraft_horizontal_speed': 120, # Aircraft speed (KIAS)
            
            # Test parameters
            'desired_deployment_speed': 150,  # Target deployment speed (kts SDSL)
            
            # Simulation parameters
            'time_step': 0.1,                # Simulation time step (seconds)
            'max_time': 300                   # Maximum simulation time (seconds)
        }
        
        # Unit conversion factors
        self.conversions = {
            'ft_to_m': 0.3048,
            'lbs_to_kg': 0.453592,
            'kts_to_mps': 0.514444,
            'sqft_to_sqm': 0.092903,
            'fpm_to_mps': 0.00508
        }
        
        # Standard sea level conditions for SDSL correction
        self.sdsl_pressure = 29.92          # inches Hg
        self.sdsl_temperature = 59          # degrees F
        self.sdsl_density = 0.002378        # slugs/ft³
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_input_controls()
        self.setup_timer_controls()
        self.setup_plot_area()
        self.setup_results_area()
        
    def setup_input_controls(self):
        # DART Vehicle Parameters
        vehicle_frame = ttk.LabelFrame(self.control_frame, text="DART Vehicle Parameters", padding="10")
        vehicle_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.param_vars = {}
        
        # Vehicle parameters with units
        vehicle_params = [
            ('altitude_msl', 'Drop Altitude MSL', 'ft'),
            ('dart_weight', 'DART Weight', 'lbs'),
            ('freefall_drag_area', 'Freefall Drag Area x Cd', 'sq ft'),
            ('drogue_drag_area', 'Drogue Drag Area x Cd', 'sq ft')
        ]
        
        for i, (param, label, unit) in enumerate(vehicle_params):
            ttk.Label(vehicle_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', pady=2)
            
            var = tk.DoubleVar(value=self.params[param])
            entry = ttk.Entry(vehicle_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky='ew', pady=2, padx=(10, 5))
            self.param_vars[param] = var
            
            ttk.Label(vehicle_frame, text=unit).grid(row=i, column=2, sticky='w', pady=2)
        
        vehicle_frame.columnconfigure(1, weight=1)
        
        # Aircraft Parameters
        aircraft_frame = ttk.LabelFrame(self.control_frame, text="Aircraft Parameters", padding="10")
        aircraft_frame.pack(fill=tk.X, pady=(0, 10))
        
        aircraft_params = [
            ('aircraft_horizontal_speed', 'Aircraft Speed', 'KIAS')
        ]
        
        for i, (param, label, unit) in enumerate(aircraft_params):
            ttk.Label(aircraft_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', pady=2)
            
            var = tk.DoubleVar(value=self.params[param])
            entry = ttk.Entry(aircraft_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky='ew', pady=2, padx=(10, 5))
            self.param_vars[param] = var
            
            ttk.Label(aircraft_frame, text=unit).grid(row=i, column=2, sticky='w', pady=2)
        
        aircraft_frame.columnconfigure(1, weight=1)
    
    def setup_timer_controls(self):
        # Test Parameters
        test_frame = ttk.LabelFrame(self.control_frame, text="Test Parameters", padding="10")
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        test_params = [
            ('desired_deployment_speed', 'Target Deployment Speed', 'kts SDSL')
        ]
        
        for i, (param, label, unit) in enumerate(test_params):
            ttk.Label(test_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', pady=2)
            
            var = tk.DoubleVar(value=self.params[param])
            entry = ttk.Entry(test_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky='ew', pady=2, padx=(10, 5))
            self.param_vars[param] = var
            
            ttk.Label(test_frame, text=unit).grid(row=i, column=2, sticky='w', pady=2)
        
        test_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Calculate Timer Setting", command=self.run_simulation, 
                  style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config).pack(fill=tk.X, pady=2)
    
    def setup_plot_area(self):
        # Create matplotlib figure with multiple subplots
        self.fig = Figure(figsize=(14, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots for comprehensive analysis
        self.ax1 = self.fig.add_subplot(231)  # Altitude vs Time
        self.ax2 = self.fig.add_subplot(232)  # Speed vs Time (with SDSL correction)
        self.ax3 = self.fig.add_subplot(233)  # Trajectory view
        self.ax4 = self.fig.add_subplot(234)  # Drag forces
        self.ax5 = self.fig.add_subplot(235)  # Air density vs altitude
        self.ax6 = self.fig.add_subplot(236)  # Timer countdown visualization
        
        self.fig.tight_layout(pad=3.0)
        
    def setup_results_area(self):
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Timer Calculation Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=25, width=50, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def calculate_air_density(self, altitude_ft):
        """Calculate air density at given altitude using standard atmosphere model"""
        # Convert altitude to meters for calculation
        altitude_m = altitude_ft * self.conversions['ft_to_m']
        
        # Standard atmosphere model
        if altitude_m <= 11000:  # Troposphere
            temperature = 288.15 - 0.0065 * altitude_m  # K
            pressure = 101325 * (temperature / 288.15) ** 5.2561  # Pa
        else:  # Stratosphere approximation
            temperature = 216.65  # K (isothermal)
            pressure = 22632 * np.exp(-0.0001577 * (altitude_m - 11000))  # Pa
        
        # Air density (kg/m³)
        density = pressure / (287.05 * temperature)
        
        # Convert to slugs/ft³ for consistency with Imperial units
        density_imperial = density / 515.379  # slugs/ft³
        
        return density_imperial
    
    def calculate_sdsl_speed_correction(self, indicated_speed_kts, altitude_ft):
        """
        Calculate Standard Day Sea Level (SDSL) equivalent speed
        This corrects for air density changes with altitude
        """
        current_density = self.calculate_air_density(altitude_ft)
        density_ratio = current_density / self.sdsl_density
        
        # SDSL speed = indicated speed * sqrt(density ratio)
        sdsl_speed = indicated_speed_kts * np.sqrt(density_ratio)
        
        return sdsl_speed
    
    def calculate_drag_force(self, velocity_vector, drag_area_cd, altitude_ft):
        """Calculate drag force vector given velocity, drag area x Cd, and altitude"""
        density = self.calculate_air_density(altitude_ft)
        
        # Convert velocity to ft/s if needed and calculate magnitude
        v_magnitude = np.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2)
        
        if v_magnitude == 0:
            return np.array([0.0, 0.0])
        
        # Drag force magnitude: F = 0.5 * rho * (Cd * A) * V²
        drag_magnitude = 0.5 * density * drag_area_cd * v_magnitude**2
        
        # Drag force opposes velocity direction
        drag_force = -drag_magnitude * velocity_vector / v_magnitude
        
        return drag_force
    
    def simulate_dart_descent(self, params):
        dt = params['time_step']
        max_time = params['max_time']
        
        # Initial conditions
        time = [0]
        altitude = [params['altitude_msl']]
        
        # Convert aircraft speeds to ft/s
        vx_initial = params['aircraft_horizontal_speed'] * self.conversions['kts_to_mps'] / self.conversions['ft_to_m']  # ft/s
        vy_initial = 0.0  # ft/s (aircraft has no initial vertical speed)
        
        vx = [vx_initial]  # Horizontal velocity (ft/s)
        vy = [vy_initial]  # Vertical velocity (ft/s, negative = down)
        
        x_pos = [0]  # Horizontal position (ft)
        y_pos = [params['altitude_msl']]  # Vertical position (ft MSL)
        
        # Convert DART weight to mass (slugs)
        dart_mass = params['dart_weight'] / 32.174  # slugs (lbs / g in ft/s²)
        
        # Speed tracking
        indicated_speeds = []
        sdsl_speeds = []
        drag_forces = []
        air_densities = []
        
        # Simulation state
        current_time = 0
        parachute_deployed = False
        deployment_time = None
        deployment_altitude = None
        deployment_speed_sdsl = None
        
        # Target speed in ft/s
        target_speed_kts = params['desired_deployment_speed']
        
        # Simulation loop
        while current_time < max_time and y_pos[-1] > 0:
            current_alt = y_pos[-1]
            current_vx = vx[-1]
            current_vy = vy[-1]
            
            # Calculate current indicated speed (kts)
            velocity_fts = np.sqrt(current_vx**2 + current_vy**2)
            velocity_kts = velocity_fts * self.conversions['ft_to_m'] / self.conversions['kts_to_mps']
            
            # Calculate SDSL corrected speed
            sdsl_speed_kts = self.calculate_sdsl_speed_correction(velocity_kts, current_alt)
            
            # Check for parachute deployment
            if not parachute_deployed and sdsl_speed_kts >= target_speed_kts:
                parachute_deployed = True
                deployment_time = current_time
                deployment_altitude = current_alt
                deployment_speed_sdsl = sdsl_speed_kts
            
            # Calculate drag force
            velocity_vector = np.array([current_vx, current_vy])
            
            if parachute_deployed:
                # Use drogue parachute characteristics
                drag_force = self.calculate_drag_force(velocity_vector, 
                                                     params['drogue_drag_area'], 
                                                     current_alt)
            else:
                # Use freefall DART characteristics
                drag_force = self.calculate_drag_force(velocity_vector, 
                                                     params['freefall_drag_area'], 
                                                     current_alt)
            
            # Calculate accelerations (ft/s²)
            ax = drag_force[0] / dart_mass
            ay = -32.174 + drag_force[1] / dart_mass  # Include gravity (32.174 ft/s²)
            
            # Update velocities
            new_vx = current_vx + ax * dt
            new_vy = current_vy + ay * dt
            
            # Update positions
            new_x = x_pos[-1] + current_vx * dt
            new_y = y_pos[-1] + current_vy * dt
            
            # Update time
            current_time += dt
            
            # Store results
            time.append(current_time)
            altitude.append(current_alt)
            vx.append(new_vx)
            vy.append(new_vy)
            x_pos.append(new_x)
            y_pos.append(new_y)
            indicated_speeds.append(velocity_kts)
            sdsl_speeds.append(sdsl_speed_kts)
            drag_forces.append(np.linalg.norm(drag_force))
            air_densities.append(self.calculate_air_density(current_alt))
        
        # Calculate timer setting
        if deployment_time is not None:
            timer_setting = deployment_time
        else:
            timer_setting = None
        
        return {
            'time': np.array(time),
            'altitude': np.array(y_pos),
            'x_position': np.array(x_pos),
            'velocity_x': np.array(vx),
            'velocity_y': np.array(vy),
            'indicated_speeds': np.array(indicated_speeds),
            'sdsl_speeds': np.array(sdsl_speeds),
            'drag_forces': np.array(drag_forces),
            'air_densities': np.array(air_densities),
            'deployment_time': deployment_time,
            'deployment_altitude': deployment_altitude,
            'deployment_speed_sdsl': deployment_speed_sdsl,
            'timer_setting': timer_setting,
            'target_speed': target_speed_kts,
            'params': params.copy()
        }
    
    def run_simulation(self):
        try:
            # Update parameters from GUI with unit conversion
            params = {}
            for param, var in self.param_vars.items():
                params[param] = var.get()
            
            # Add simulation parameters that aren't in GUI
            params['time_step'] = self.params['time_step']
            params['max_time'] = self.params['max_time']
            
            def simulate():
                results = self.simulate_dart_descent(params)
                self.root.after(0, lambda: self.update_display(results))
            
            threading.Thread(target=simulate, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Simulation failed: {str(e)}")
    
    def update_display(self, results):
        """Update all plots and results display"""
        # Clear all plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()
        
        time = results['time']
        
        # 1. Altitude vs Time
        self.ax1.plot(time, results['altitude'], 'b-', linewidth=2, label='Altitude MSL')
        if results['deployment_time']:
            self.ax1.axvline(x=results['deployment_time'], color='r', linestyle='--', 
                           label=f'Deployment ({results["deployment_time"]:.1f}s)')
            if results['timer_setting']:
                self.ax1.axvline(x=results['timer_setting'], color='g', linestyle=':', 
                               label=f'Timer Set ({results["timer_setting"]:.1f}s)')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Altitude (ft MSL)')
        self.ax1.set_title('DART Altitude Profile')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # 2. Speed vs Time (both indicated and SDSL)
        self.ax2.plot(time[:-1], results['indicated_speeds'], 'b-', linewidth=2, label='Indicated Speed')
        self.ax2.plot(time[:-1], results['sdsl_speeds'], 'r-', linewidth=2, label='SDSL Speed')
        self.ax2.axhline(y=results['target_speed'], color='g', linestyle='--', 
                        label=f'Target ({results["target_speed"]:.0f} kts SDSL)')
        if results['deployment_time']:
            self.ax2.axvline(x=results['deployment_time'], color='r', linestyle='--', alpha=0.7)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed (kts)')
        self.ax2.set_title('DART Speed Profile')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # 3. Trajectory View (Ground Track)
        self.ax3.plot(results['x_position'], results['altitude'], 'b-', linewidth=2)
        if results['deployment_time']:
            dep_idx = np.where(time <= results['deployment_time'])[0][-1]
            self.ax3.plot(results['x_position'][dep_idx], results['altitude'][dep_idx], 
                         'ro', markersize=8, label='Deployment Point')
        self.ax3.set_xlabel('Horizontal Distance (ft)')
        self.ax3.set_ylabel('Altitude (ft MSL)')
        self.ax3.set_title('DART Trajectory')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        # 4. Drag Forces
        self.ax4.plot(time[:-1], results['drag_forces'], 'g-', linewidth=2)
        if results['deployment_time']:
            self.ax4.axvline(x=results['deployment_time'], color='r', linestyle='--', alpha=0.7)
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Drag Force (lbf)')
        self.ax4.set_title('Drag Force vs Time')
        self.ax4.grid(True, alpha=0.3)
        
        # 5. Air Density vs Altitude
        self.ax5.plot(results['air_densities'], results['altitude'][:-1], 'm-', linewidth=2)
        self.ax5.set_xlabel('Air Density (slugs/ft³)')
        self.ax5.set_ylabel('Altitude (ft MSL)')
        self.ax5.set_title('Atmospheric Density Profile')
        self.ax5.grid(True, alpha=0.3)
        
        # 6. Timer Countdown Visualization
        if results['timer_setting'] and results['deployment_time']:
            countdown_time = np.linspace(0, results['deployment_time'], 100)
            timer_countdown = results['deployment_time'] - countdown_time
            timer_countdown = np.maximum(timer_countdown, 0)
            
            self.ax6.plot(countdown_time, timer_countdown, 'r-', linewidth=3, label='Timer Countdown')
            self.ax6.axhline(y=0, color='g', linestyle='--', label='Deployment Command')
            self.ax6.axvline(x=results['timer_setting'], color='g', linestyle=':', label='Timer Expires')
            self.ax6.set_xlabel('Time (s)')
            self.ax6.set_ylabel('Timer Remaining (s)')
            self.ax6.set_title('Timer Countdown')
            self.ax6.grid(True, alpha=0.3)
            self.ax6.legend()
        else:
            self.ax6.text(0.5, 0.5, 'Timer calculation failed\nTarget speed not reached', 
                         ha='center', va='center', transform=self.ax6.transAxes,
                         fontsize=12, color='red')
            self.ax6.set_title('Timer Calculation Status')
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
        
        self.update_results_display(results)
    
    def update_results_display(self, results):
        """Update the results text display with timer calculation details"""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "DART PARACHUTE TIMER CALCULATION\n")
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        # Timer Setting Result
        if results['timer_setting'] is not None:
            self.results_text.insert(tk.END, "🎯 TIMER SETTING RESULT:\n")
            self.results_text.insert(tk.END, f"   SET TIMER TO: {results['timer_setting']:.1f} SECONDS\n\n")
            
            self.results_text.insert(tk.END, "📊 DEPLOYMENT ANALYSIS:\n")
            self.results_text.insert(tk.END, f"   • Deployment Time: {results['deployment_time']:.1f} s\n")
            self.results_text.insert(tk.END, f"   • Deployment Altitude: {results['deployment_altitude']:.0f} ft MSL\n")
            self.results_text.insert(tk.END, f"   • Deployment Speed (SDSL): {results['deployment_speed_sdsl']:.1f} kts\n")
            self.results_text.insert(tk.END, f"   • Target Speed (SDSL): {results['target_speed']:.0f} kts\n\n")
        else:
            self.results_text.insert(tk.END, "❌ TIMER CALCULATION FAILED!\n")
            self.results_text.insert(tk.END, "   Target deployment speed not reached.\n")
            max_sdsl = np.max(results['sdsl_speeds']) if len(results['sdsl_speeds']) > 0 else 0
            self.results_text.insert(tk.END, f"   Maximum SDSL speed achieved: {max_sdsl:.1f} kts\n")
            self.results_text.insert(tk.END, f"   Target speed required: {results['target_speed']:.0f} kts\n\n")
        
        # Flight Performance Summary
        self.results_text.insert(tk.END, "🚀 FLIGHT PERFORMANCE:\n")
        final_time = results['time'][-1]
        final_alt = results['altitude'][-1]
        max_range = results['x_position'][-1]
        max_indicated_speed = np.max(results['indicated_speeds']) if len(results['indicated_speeds']) > 0 else 0
        max_sdsl_speed = np.max(results['sdsl_speeds']) if len(results['sdsl_speeds']) > 0 else 0
        
        self.results_text.insert(tk.END, f"   • Total Flight Time: {final_time:.1f} s\n")
        self.results_text.insert(tk.END, f"   • Final Altitude: {final_alt:.0f} ft MSL\n")
        self.results_text.insert(tk.END, f"   • Horizontal Range: {max_range:.0f} ft ({max_range/5280:.1f} miles)\n")
        self.results_text.insert(tk.END, f"   • Max Indicated Speed: {max_indicated_speed:.1f} kts\n")
        self.results_text.insert(tk.END, f"   • Max SDSL Speed: {max_sdsl_speed:.1f} kts\n\n")
        
        # Test Configuration Summary
        self.results_text.insert(tk.END, "⚙️ TEST CONFIGURATION:\n")
        params = results['params']
        self.results_text.insert(tk.END, f"   • DART Weight: {params['dart_weight']:.0f} lbs\n")
        self.results_text.insert(tk.END, f"   • Freefall Drag Area x Cd: {params['freefall_drag_area']:.3f} sq ft\n")
        self.results_text.insert(tk.END, f"   • Drogue Drag Area x Cd: {params['drogue_drag_area']:.1f} sq ft\n")
        self.results_text.insert(tk.END, f"   • Aircraft Speed: {params['aircraft_horizontal_speed']:.0f} KIAS\n")
        self.results_text.insert(tk.END, f"   • Drop Altitude: {params['altitude_msl']:.0f} ft MSL\n\n")
        
        # Safety Notes
        self.results_text.insert(tk.END, "⚠️ SAFETY NOTES:\n")
        self.results_text.insert(tk.END, "   • Verify timer setting before flight\n")
        self.results_text.insert(tk.END, "   • Account for actual weather conditions\n")
        self.results_text.insert(tk.END, "   • Check DART weight and balance\n")
        self.results_text.insert(tk.END, "   • Confirm drogue packing and rigging\n")
        self.results_text.insert(tk.END, "   • Brief aircraft crew on timing\n\n")
        
        if results['timer_setting'] is not None:
            self.results_text.insert(tk.END, "✅ READY FOR TEST DISPATCH!")
        else:
            self.results_text.insert(tk.END, "❌ ADJUST PARAMETERS AND RECALCULATE")
    
    def reset_parameters(self):
        """Reset all parameters to default values"""
        defaults = {
            'altitude_msl': 15000,
            'dart_weight': 150,
            'freefall_drag_area': 0.125,
            'drogue_drag_area': 10.4,
            'aircraft_horizontal_speed': 120,
            'desired_deployment_speed': 150,
            'time_step': 0.1,
            'max_time': 300
        }
        
        for param, value in defaults.items():
            if param in self.param_vars:
                self.param_vars[param].set(value)
    
    def save_config(self):
        """Save current configuration to a file"""
        try:
            import json
            from tkinter import filedialog
            
            config = {}
            for param, var in self.param_vars.items():
                config[param] = var.get()
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save DART Configuration"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
                messagebox.showinfo("Success", "Configuration saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DARTTimerSimulationGUI(root)
    root.mainloop()