import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import math
import time

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
            
            # Aircraft parameters
            'aircraft_horizontal_speed': 120, # Aircraft speed (KIAS)
            
            # Test parameters
            'desired_deployment_speed': 150,  # Target deployment speed (KIAS)
            
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
            ('freefall_drag_area', 'Freefall Drag Area x Cd', 'sq ft')
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
            ('desired_deployment_speed', 'Target Deployment Speed', 'KIAS'),
            ('time_step', 'Simulation Time Step', 'sec')
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
        
        # Progress bar
        self.progress_frame = ttk.Frame(self.control_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.progress_frame, text="Calculation Progress:").pack()
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack()
    
    def setup_plot_area(self):
        # Create tabbed interface for plots
        self.notebook = ttk.Notebook(self.plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Altitude Profile
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Altitude Profile")
        
        self.fig1 = Figure(figsize=(12, 8))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax1 = self.fig1.add_subplot(111)
        
        # Add navigation toolbar for zoom/pan
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.tab1)
        self.toolbar1.update()
        
        # Tab 2: Speed Profile
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Speed Profile")
        
        self.fig2 = Figure(figsize=(12, 8))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax2 = self.fig2.add_subplot(111)
        
        # Add navigation toolbar for zoom/pan
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.tab2)
        self.toolbar2.update()
        
        # Tab 3: Trajectory
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Trajectory")
        
        self.fig3 = Figure(figsize=(12, 8))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax3 = self.fig3.add_subplot(111)
        
        # Add navigation toolbar for zoom/pan
        self.toolbar3 = NavigationToolbar2Tk(self.canvas3, self.tab3)
        self.toolbar3.update()
        
        # Tab 4: Drag Forces
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Drag Forces")
        
        self.fig4 = Figure(figsize=(12, 8))
        self.canvas4 = FigureCanvasTkAgg(self.fig4, self.tab4)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax4 = self.fig4.add_subplot(111)
        
        # Add navigation toolbar for zoom/pan
        self.toolbar4 = NavigationToolbar2Tk(self.canvas4, self.tab4)
        self.toolbar4.update()
        
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
    
    def calculate_kias_to_sdsl_at_altitude(self, kias_speed, altitude_ft):
        """
        Convert KIAS to SDSL equivalent at given altitude
        KIAS represents what the airspeed indicator shows
        SDSL is the equivalent speed at standard day sea level conditions
        """
        # First convert KIAS to true airspeed at current altitude
        current_density = self.calculate_air_density(altitude_ft)
        density_ratio = current_density / self.sdsl_density
        true_airspeed = kias_speed / np.sqrt(density_ratio)
        
        # SDSL equivalent is the true airspeed (since SDSL is true airspeed at sea level standard)
        return true_airspeed
    
    def calculate_sdsl_speed_correction(self, indicated_speed_kts, altitude_ft):
        """
        Calculate Standard Day Sea Level (SDSL) equivalent speed
        Input: Indicated airspeed (KIAS)
        Output: SDSL equivalent speed (what the speed would be at sea level standard conditions)
        """
        # Convert KIAS to true airspeed at current altitude
        current_density = self.calculate_air_density(altitude_ft)
        density_ratio = current_density / self.sdsl_density
        true_airspeed = indicated_speed_kts / np.sqrt(density_ratio)
        
        # SDSL speed is the true airspeed (since SDSL assumes sea level standard conditions)
        return true_airspeed
    
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
    
    def update_progress(self, percent, status_text):
        """Update progress bar and status text"""
        self.progress_var.set(percent)
        self.progress_label.config(text=status_text)
        self.root.update_idletasks()
    
    def simulate_dart_descent(self, params):
        dt = params['time_step']
        max_time = params['max_time']
        
        print(f"Debug: Using time step = {dt} seconds")  # Debug output
        
        # Initial conditions
        time_array = [0]
        altitude = [params['altitude_msl']]
        
        # Convert aircraft speeds to ft/s
        # Aircraft speed input is KIAS, convert to true airspeed at drop altitude
        aircraft_kias = params['aircraft_horizontal_speed']
        drop_altitude_density = self.calculate_air_density(params['altitude_msl'])
        density_ratio_drop = drop_altitude_density / self.sdsl_density
        aircraft_true_airspeed = aircraft_kias / np.sqrt(density_ratio_drop)  # Convert KIAS to TAS
        
        vx_initial = aircraft_true_airspeed * self.conversions['kts_to_mps'] / self.conversions['ft_to_m']  # ft/s
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
        true_airspeeds = []
        drag_forces = []
        air_densities = []
        
        # Simulation state
        current_time = 0
        timer_triggered = False  # Track when timer would trigger
        deployment_time = None
        deployment_altitude = None
        deployment_speed_sdsl = None
        deployment_speed_kias = None
        
        # Track speed trend to handle both acceleration and deceleration
        previous_speed = 0
        speed_trend_samples = []
        target_crossed = False
        
        # Convert target KIAS to equivalent SDSL speed for comparison
        target_speed_kias = params['desired_deployment_speed']
        
        # We'll compare against SDSL speed during simulation
        # KIAS at altitude will be converted to SDSL for comparison
        
        # Simulation loop
        iteration_count = 0
        max_iterations = int(max_time / dt)  # Estimate total iterations
        start_calc_time = time.time()  # Track calculation time
        
        print(f"Debug: Estimated iterations = {max_iterations}")  # Debug output
        
        while current_time < max_time and y_pos[-1] > 0:
            current_alt = y_pos[-1]
            current_vx = vx[-1]
            current_vy = vy[-1]
            
            # Update progress every 100 iterations to avoid GUI slowdown
            iteration_count += 1
            if iteration_count % 100 == 0:
                progress_percent = min(95, (current_time / max_time) * 100)  # Cap at 95% until complete
                
                # Calculate estimated time remaining
                elapsed_time = time.time() - start_calc_time
                if progress_percent > 5:  # Only show estimate after some progress
                    estimated_total = elapsed_time * 100 / progress_percent
                    remaining_time = estimated_total - elapsed_time
                    status_text = f"Time: {current_time:.1f}s | ETA: {remaining_time:.1f}s"
                else:
                    status_text = f"Time: {current_time:.1f}s | Calculating..."
                
                self.root.after(0, lambda p=progress_percent, t=status_text: self.update_progress(p, t))
            current_alt = y_pos[-1]
            current_vx = vx[-1]
            current_vy = vy[-1]
            
            # Calculate current true airspeed (TAS) first
            velocity_fts = np.sqrt(current_vx**2 + current_vy**2)
            true_airspeed_kts = velocity_fts * self.conversions['ft_to_m'] / self.conversions['kts_to_mps']
            
            # Calculate indicated airspeed (KIAS) from true airspeed
            # KIAS = TAS * sqrt(density_ratio)
            current_density = self.calculate_air_density(current_alt)
            density_ratio = current_density / self.sdsl_density
            velocity_kts = true_airspeed_kts * np.sqrt(density_ratio)  # This is KIAS
            
            # Calculate SDSL corrected speed
            sdsl_speed_kts = self.calculate_sdsl_speed_correction(velocity_kts, current_alt)
            
            # Check for timer trigger point (handle both acceleration and deceleration)
            if not timer_triggered:
                # Track speed trend over last few samples
                speed_trend_samples.append(velocity_kts)
                if len(speed_trend_samples) > 10:  # Keep last 10 samples
                    speed_trend_samples.pop(0)
                
                # Determine if we're accelerating or decelerating
                if len(speed_trend_samples) >= 5:
                    recent_trend = speed_trend_samples[-1] - speed_trend_samples[-5]
                    is_accelerating = recent_trend > 0.1  # Accelerating
                    is_decelerating = recent_trend < -0.1  # Decelerating
                    
                    # Check for target speed crossing
                    if is_accelerating and velocity_kts >= target_speed_kias and previous_speed < target_speed_kias:
                        # Accelerating and just crossed target speed upward
                        timer_triggered = True
                        deployment_time = current_time
                        deployment_altitude = current_alt
                        deployment_speed_sdsl = sdsl_speed_kts
                        deployment_speed_kias = velocity_kts
                        target_crossed = True
                    elif is_decelerating and velocity_kts <= target_speed_kias and previous_speed > target_speed_kias:
                        # Decelerating and just crossed target speed downward
                        timer_triggered = True
                        deployment_time = current_time
                        deployment_altitude = current_alt
                        deployment_speed_sdsl = sdsl_speed_kts
                        deployment_speed_kias = velocity_kts
                        target_crossed = True
                    elif not is_accelerating and not is_decelerating and abs(velocity_kts - target_speed_kias) < 1.0:
                        # Near steady state at target speed
                        timer_triggered = True
                        deployment_time = current_time
                        deployment_altitude = current_alt
                        deployment_speed_sdsl = sdsl_speed_kts
                        deployment_speed_kias = velocity_kts
                        target_crossed = True
                
                previous_speed = velocity_kts
            
            # Calculate drag force - ALWAYS use freefall characteristics
            # (no parachute deployment, just mark the timer point)
            velocity_vector = np.array([current_vx, current_vy])
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
            time_array.append(current_time)
            altitude.append(current_alt)
            vx.append(new_vx)
            vy.append(new_vy)
            x_pos.append(new_x)
            y_pos.append(new_y)
            indicated_speeds.append(velocity_kts)
            sdsl_speeds.append(sdsl_speed_kts)
            true_airspeeds.append(true_airspeed_kts)
            drag_forces.append(np.linalg.norm(drag_force))
            air_densities.append(self.calculate_air_density(current_alt))
        
        # Calculate timer setting
        if deployment_time is not None:
            timer_setting = deployment_time
        else:
            timer_setting = None
        
        # Final progress update
        self.root.after(0, lambda: self.update_progress(100, "Processing results..."))
        
        return {
            'time': np.array(time_array),
            'altitude': np.array(y_pos),
            'x_position': np.array(x_pos),
            'velocity_x': np.array(vx),
            'velocity_y': np.array(vy),
            'indicated_speeds': np.array(indicated_speeds),
            'sdsl_speeds': np.array(sdsl_speeds),
            'true_airspeeds': np.array(true_airspeeds),
            'drag_forces': np.array(drag_forces),
            'air_densities': np.array(air_densities),
            'deployment_time': deployment_time,
            'deployment_altitude': deployment_altitude,
            'deployment_speed_sdsl': deployment_speed_sdsl if 'deployment_speed_sdsl' in locals() else None,
            'deployment_speed_kias': deployment_speed_kias if 'deployment_speed_kias' in locals() else None,
            'target_crossed': target_crossed if 'target_crossed' in locals() else False,
            'timer_setting': timer_setting,
            'target_speed_kias': target_speed_kias,
            'params': params.copy()
        }
    
    def run_simulation(self):
        try:
            # Update parameters from GUI with unit conversion
            params = {}
            for param, var in self.param_vars.items():
                params[param] = var.get()
            
            # Add simulation parameters that aren't in GUI (but time_step IS in GUI now)
            params['max_time'] = self.params['max_time']
            # time_step is now coming from the GUI input, don't override it
            
            # Reset progress bar and disable button
            self.progress_var.set(0)
            self.progress_label.config(text="Starting calculation...")
            self.root.update_idletasks()
            
            # Find calculate button and disable it
            for child in self.control_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    for button in child.winfo_children():
                        if isinstance(button, ttk.Button) and "Calculate" in button.cget('text'):
                            button.config(state='disabled')
            
            def simulate():
                try:
                    results = self.simulate_dart_descent(params)
                    self.root.after(0, lambda: self.simulation_complete(results))
                except Exception as sim_error:
                    self.root.after(0, lambda: self.simulation_error(str(sim_error)))
            
            threading.Thread(target=simulate, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Simulation failed: {str(e)}")
            self.reset_progress_bar()
    
    def simulation_complete(self, results):
        """Called when simulation completes successfully"""
        self.update_display(results)
        self.reset_progress_bar()
        self.progress_label.config(text="Calculation complete!")
    
    def simulation_error(self, error_msg):
        """Called when simulation encounters an error"""
        messagebox.showerror("Simulation Error", f"Simulation failed: {error_msg}")
        self.reset_progress_bar()
        self.progress_label.config(text="Calculation failed")
    
    def reset_progress_bar(self):
        """Reset progress bar and re-enable calculate button"""
        self.progress_var.set(0)
        # Re-enable calculate button
        for child in self.control_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for button in child.winfo_children():
                    if isinstance(button, ttk.Button) and "Calculate" in button.cget('text'):
                        button.config(state='normal')
    
    def update_display(self, results):
        """Update all plots and results display"""
        # Clear all plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        time_data = results['time']
        
        # Tab 1: Altitude Profile
        self.ax1.plot(time_data, results['altitude'], 'b-', linewidth=2, label='Altitude MSL')
        if results['deployment_time']:
            self.ax1.axvline(x=results['deployment_time'], color='r', linestyle='--', 
                           label=f'Timer Setting ({results["deployment_time"]:.1f}s)', linewidth=2)
            # Mark deployment point
            dep_idx = np.where(time_data <= results['deployment_time'])[0][-1]
            self.ax1.plot(time_data[dep_idx], results['altitude'][dep_idx], 'ro', markersize=10, 
                         label=f'Deploy Alt: {results["altitude"][dep_idx]:.0f} ft')
        self.ax1.set_xlabel('Time (s)', fontsize=12)
        self.ax1.set_ylabel('Altitude (ft MSL)', fontsize=12)
        self.ax1.set_title('DART Altitude Profile', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(fontsize=10)
        
        # Tab 2: Speed Profile
        self.ax2.plot(time_data[:-1], results['indicated_speeds'], 'b-', linewidth=2, label='Indicated Speed (KIAS)')
        self.ax2.plot(time_data[:-1], results['true_airspeeds'], 'g-', linewidth=2, label='True Airspeed (KTAS)')
        self.ax2.axhline(y=results['target_speed_kias'], color='orange', linestyle=':', linewidth=3, 
                        label=f'Target: {results["target_speed_kias"]:.0f} KIAS')
        if results['deployment_time']:
            self.ax2.axvline(x=results['deployment_time'], color='r', linestyle='--', linewidth=2, 
                           label=f'Timer Setting ({results["deployment_time"]:.1f}s)')
            # Mark deployment speed
            dep_idx = np.where(time_data[:-1] <= results['deployment_time'])[0][-1]
            self.ax2.plot(time_data[dep_idx], results['indicated_speeds'][dep_idx], 'ro', markersize=10, 
                         label=f'Deploy Speed: {results["indicated_speeds"][dep_idx]:.1f} KIAS')
        self.ax2.set_xlabel('Time (s)', fontsize=12)
        self.ax2.set_ylabel('Speed (knots)', fontsize=12)
        self.ax2.set_title('DART Speed Profile', fontsize=14, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(fontsize=10)
        
        # Tab 3: Trajectory View
        self.ax3.plot(results['x_position'], results['altitude'], 'purple', linewidth=2, label='Flight Path')
        if results['deployment_time']:
            dep_idx = np.where(time_data <= results['deployment_time'])[0][-1]
            self.ax3.plot(results['x_position'][dep_idx], results['altitude'][dep_idx], 
                         'ro', markersize=10, label=f'Timer Setting Point')
        self.ax3.set_xlabel('Horizontal Distance (ft)', fontsize=12)
        self.ax3.set_ylabel('Altitude (ft MSL)', fontsize=12)
        self.ax3.set_title('DART Flight Trajectory', fontsize=14, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(fontsize=10)
        
        # Tab 4: Drag Forces
        self.ax4.plot(time_data[:-1], results['drag_forces'], 'orange', linewidth=2, label='Drag Force')
        weight = float(self.param_vars['dart_weight'].get())
        self.ax4.axhline(y=weight, color='k', linestyle='--', linewidth=2, label=f'Weight: {weight:.0f} lbf')
        if results['deployment_time']:
            self.ax4.axvline(x=results['deployment_time'], color='r', linestyle='--', linewidth=2, 
                           label=f'Timer Setting ({results["deployment_time"]:.1f}s)')
        self.ax4.set_xlabel('Time (s)', fontsize=12)
        self.ax4.set_ylabel('Force (lbf)', fontsize=12)
        self.ax4.set_title('DART Drag Forces vs Time', fontsize=14, fontweight='bold')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(fontsize=10)
        
        # Update all canvases
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()
        
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
            
            self.results_text.insert(tk.END, "📊 TIMER TRIGGER ANALYSIS:\n")
            self.results_text.insert(tk.END, f"   • Timer Trigger Time: {results['deployment_time']:.1f} s\n")
            self.results_text.insert(tk.END, f"   • Trigger Altitude: {results['deployment_altitude']:.0f} ft MSL\n")
            self.results_text.insert(tk.END, f"   • Trigger Speed: {results.get('deployment_speed_kias', 0):.1f} KIAS\n")
            self.results_text.insert(tk.END, f"   • Target Speed: {results['target_speed_kias']:.0f} KIAS\n\n")
        else:
            self.results_text.insert(tk.END, "❌ TIMER CALCULATION FAILED!\n")
            self.results_text.insert(tk.END, "   Target deployment speed not reached or crossed.\n")
            max_kias = np.max(results['indicated_speeds']) if len(results['indicated_speeds']) > 0 else 0
            min_kias = np.min(results['indicated_speeds']) if len(results['indicated_speeds']) > 0 else 0
            self.results_text.insert(tk.END, f"   Speed range: {min_kias:.1f} - {max_kias:.1f} KIAS\n")
            self.results_text.insert(tk.END, f"   Target speed required: {results['target_speed_kias']:.0f} KIAS\n\n")
        
        # Flight Performance Summary
        self.results_text.insert(tk.END, f"🚀 FREE FALL IMPACT ANALYSIS:\n")
        final_time = results['time'][-1]
        final_alt = results['altitude'][-1]
        max_range = results['x_position'][-1]
        final_speed = results['indicated_speeds'][-1] if len(results['indicated_speeds']) > 0 else 0
        max_indicated_speed = np.max(results['indicated_speeds']) if len(results['indicated_speeds']) > 0 else 0
        max_true_airspeed = np.max(results['true_airspeeds']) if len(results['true_airspeeds']) > 0 else 0
        
        self.results_text.insert(tk.END, f"   • Total Fall Time: {final_time:.1f} s\n")
        self.results_text.insert(tk.END, f"   • Impact Speed: {final_speed:.1f} KIAS (NO PARACHUTE!)\n")
        self.results_text.insert(tk.END, f"   • Horizontal Range: {max_range:.0f} ft ({max_range/5280:.1f} miles)\n")
        self.results_text.insert(tk.END, f"   • Max Indicated Speed: {max_indicated_speed:.1f} KIAS\n")
        self.results_text.insert(tk.END, f"   • Max True Airspeed: {max_true_airspeed:.1f} KTAS\n\n")
        
        # Test Configuration Summary
        self.results_text.insert(tk.END, "⚙️ TEST CONFIGURATION:\n")
        params = results['params']
        self.results_text.insert(tk.END, f"   • DART Weight: {params['dart_weight']:.0f} lbs\n")
        self.results_text.insert(tk.END, f"   • Freefall Drag Area x Cd: {params['freefall_drag_area']:.3f} sq ft\n")
        self.results_text.insert(tk.END, f"   • Aircraft Speed: {params['aircraft_horizontal_speed']:.0f} KIAS\n")
        self.results_text.insert(tk.END, f"   • Drop Altitude: {params['altitude_msl']:.0f} ft MSL\n\n")

        
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
            'aircraft_horizontal_speed': 120,
            'desired_deployment_speed': 150,
            'time_step': 0.1,
            'max_time': 300
        }
        
        for param, value in defaults.items():
            if param in self.param_vars:
                self.param_vars[param].set(value)
    


if __name__ == "__main__":
    root = tk.Tk()
    app = DARTTimerSimulationGUI(root)
    root.mainloop()