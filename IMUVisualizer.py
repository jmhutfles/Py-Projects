import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import messagebox
from ReadRawData import ReadIMU
import pandas as pd
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
except ImportError:
    # Fallback if scipy is not available
    from numpy.fft import fft, fftfreq
    signal = None

class IMU3DVisualizer:
    def __init__(self):
        self.data = None
        self.fig = None
        self.ax = None
        self.animation_obj = None
        self.current_frame = 0
        self.is_playing = False
        self.frame_step = 1
        
    def load_data(self):
        """Load IMU data using ReadRawData functions"""
        try:
            data, paths = ReadIMU("Select IMU Data Files")
            if data is not None and not data.empty:
                # Validate required columns exist - using XYZ gyroscope data
                required_cols = ['Gx', 'Gy', 'Gz']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    messagebox.showerror("Error", f"Missing required XYZ gyroscope columns: {', '.join(missing_cols)}")
                    return False
                
                # Check data types
                for col in required_cols:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        except Exception as e:
                            messagebox.showerror("Error", f"Cannot convert column {col} to numeric: {str(e)}")
                            return False
                
                # Clean the data - fill NaN values in gyroscope columns with 0
                initial_len = len(data)
                
                # Fill NaN values in gyroscope columns with 0 instead of dropping rows
                for col in required_cols:
                    if col in data.columns:
                        nan_count = data[col].isna().sum()
                        if nan_count > 0:
                            print(f"Warning: Filling {nan_count} NaN values in {col} with 0")
                            data[col] = data[col].fillna(0.0)
                
                # Only drop rows if ALL gyroscope columns are NaN
                data = data.dropna(subset=required_cols, how='all')
                if data.empty:
                    messagebox.showerror("Error", "No valid XYZ gyroscope data found in the files")
                    return False
                
                if len(data) < initial_len:
                    print(f"Warning: Removed {initial_len - len(data)} rows where all XYZ gyroscope data was missing")
                
                # Validate gyroscope data is reasonable (typical range: +/- 2000 deg/s)
                for col in required_cols:
                    if (np.abs(data[col]) > 5000).any():
                        print(f"Warning: Some {col} values seem unusually large (>5000 deg/s)")
                
                self.data = data
                # Convert gyroscope data from milli-degrees/sec to degrees/sec  
                self.convert_gyro_units()
                # Calculate additional metrics using XYZ data
                try:
                    self.calculate_xyz_metrics()
                except Exception as e:
                    print(f"Warning: Error calculating XYZ metrics: {str(e)}")
                    # Continue anyway with basic functionality
                
                print(f"Loaded {len(data)} data points from {len(paths)} file(s)")
                return True
            else:
                messagebox.showerror("Error", "No data loaded or files not selected")
                return False
        except ImportError as e:
            messagebox.showerror("Error", f"Cannot import ReadIMU function: {str(e)}")
            return False
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File not found: {str(e)}")
            return False
        except PermissionError as e:
            messagebox.showerror("Error", f"Permission denied accessing file: {str(e)}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            return False
    
    def convert_gyro_units(self):
        """Convert gyroscope data from milli-degrees/sec to degrees/sec"""
        # Check if gyroscope columns exist
        if not all(col in self.data.columns for col in ['Gx', 'Gy', 'Gz']):
            print("Warning: Gyroscope columns (Gx, Gy, Gz) not found")
            return
            
        # Check typical magnitudes to determine if conversion is needed
        sample_size = min(100, len(self.data))
        sample_data = self.data[['Gx', 'Gy', 'Gz']].head(sample_size)
        
        # Calculate typical magnitudes
        max_vals = sample_data.abs().max()
        max_rate = max_vals.max()
        
        print(f"Raw gyroscope data sample ranges:")
        print(f"  Gx: {sample_data['Gx'].min():.1f} to {sample_data['Gx'].max():.1f}")
        print(f"  Gy: {sample_data['Gy'].min():.1f} to {sample_data['Gy'].max():.1f}")
        print(f"  Gz: {sample_data['Gz'].min():.1f} to {sample_data['Gz'].max():.1f}")
        print(f"Maximum magnitude: {max_rate:.1f}")
        
        # Convert from milli-degrees/sec to degrees/sec
        if max_rate > 10:  # Values suggest they're in milli units
            print("Converting from milli-degrees/sec to degrees/sec (dividing by 1000)")
            self.data['Gx'] = self.data['Gx'] / 1000.0
            self.data['Gy'] = self.data['Gy'] / 1000.0  
            self.data['Gz'] = self.data['Gz'] / 1000.0
            
            # Show converted values
            print(f"Converted gyroscope data ranges:")
            print(f"  Gx: {self.data['Gx'].min():.3f} to {self.data['Gx'].max():.3f} deg/s")
            print(f"  Gy: {self.data['Gy'].min():.3f} to {self.data['Gy'].max():.3f} deg/s")
            print(f"  Gz: {self.data['Gz'].min():.3f} to {self.data['Gz'].max():.3f} deg/s")
        else:
            print("Gyroscope data appears to already be in degrees/sec, no conversion needed")
    
    def calculate_xyz_metrics(self):
        """Calculate metrics using XYZ gyroscope data directly"""
        if self.data is None:
            raise ValueError("No data available for metric calculation")
        
        if len(self.data) == 0:
            raise ValueError("Data is empty")
        
        # Use XYZ gyroscope data directly (now in deg/s after conversion)
        xyz_orientations = []
        
        try:
            for idx, row in self.data.iterrows():
                # Use XYZ gyroscope data directly (angular velocities in deg/s)
                if all(col in self.data.columns for col in ['Gx', 'Gy', 'Gz']):
                    try:
                        gx, gy, gz = row['Gx'], row['Gy'], row['Gz']
                        # Convert to float to avoid array issues
                        gx, gy, gz = float(gx), float(gy), float(gz)
                        gyro_array = np.array([gx, gy, gz])
                        if np.any(np.isnan(gyro_array)) or np.any(np.isinf(gyro_array)):
                            gx, gy, gz = 0, 0, 0
                        xyz_orientations.append([gx, gy, gz])
                    except Exception:
                        # Silently handle gyro errors to avoid spam
                        xyz_orientations.append([0, 0, 0])
                else:
                    xyz_orientations.append([0, 0, 0])
            
            if not xyz_orientations:
                raise ValueError("No valid XYZ orientation data calculated")
            
            xyz_orientations = np.array(xyz_orientations)
            
            # Validate array shape
            if xyz_orientations.shape[1] != 3:
                raise ValueError("Invalid array shape for XYZ orientations")
            
            # Store XYZ data (already available as Gx, Gy, Gz but ensure clean copies)
            self.data['X_Rate'] = xyz_orientations[:, 0]  # Gx
            self.data['Y_Rate'] = xyz_orientations[:, 1]  # Gy
            self.data['Z_Rate'] = xyz_orientations[:, 2]  # Gz
            
            # Calculate oscillation amplitude and frequency using XYZ data
            self.calculate_oscillation_metrics()
            
        except Exception as e:
            print(f"Error in XYZ metrics calculation: {str(e)}")
            # Initialize with zeros to prevent crashes
            n_points = len(self.data)
            self.data['X_Rate'] = np.zeros(n_points)
            self.data['Y_Rate'] = np.zeros(n_points)
            self.data['Z_Rate'] = np.zeros(n_points)
            raise
    
    def calculate_parachute_metrics(self):
        """Calculate parachute-specific oscillation metrics"""
        if self.data is None:
            raise ValueError("No data available for metric calculation")
        
        if len(self.data) == 0:
            raise ValueError("Data is empty")
        
        # Convert quaternions to Euler angles for easier interpretation
        euler_angles = []
        angular_velocities = []
        
        try:
            for idx, row in self.data.iterrows():
                # Convert quaternion to Euler angles (roll, pitch, yaw)
                try:
                    roll, pitch, yaw = self.quaternion_to_euler(row['Qw'], row['Qx'], row['Qy'], row['Qz'])
                    # Check for valid Euler angles
                    euler_array = np.array([roll, pitch, yaw])
                    if np.any(np.isnan(euler_array)) or np.any(np.isinf(euler_array)):
                        roll, pitch, yaw = 0, 0, 0
                    euler_angles.append([roll, pitch, yaw])
                except Exception:
                    # Silently handle conversion errors to avoid spam
                    euler_angles.append([0, 0, 0])
                
                # Get angular velocities (already in deg/s)
                if all(col in self.data.columns for col in ['Gx', 'Gy', 'Gz']):
                    try:
                        gx, gy, gz = row['Gx'], row['Gy'], row['Gz']
                        # Convert to float to avoid array issues
                        gx, gy, gz = float(gx), float(gy), float(gz)
                        gyro_array = np.array([gx, gy, gz])
                        if np.any(np.isnan(gyro_array)) or np.any(np.isinf(gyro_array)):
                            gx, gy, gz = 0, 0, 0
                        angular_velocities.append([gx, gy, gz])
                    except Exception:
                        # Silently handle gyro errors to avoid spam
                        angular_velocities.append([0, 0, 0])
                else:
                    angular_velocities.append([0, 0, 0])
            
            if not euler_angles:
                raise ValueError("No valid Euler angles calculated")
            
            euler_angles = np.array(euler_angles)
            angular_velocities = np.array(angular_velocities)
            
            # Validate array shapes
            if euler_angles.shape[1] != 3 or angular_velocities.shape[1] != 3:
                raise ValueError("Invalid array shapes for Euler angles or angular velocities")
            
            # Add to dataframe
            self.data['Roll'] = euler_angles[:, 0]
            self.data['Pitch'] = euler_angles[:, 1] 
            self.data['Yaw'] = euler_angles[:, 2]
            self.data['Roll_Rate'] = angular_velocities[:, 0]
            self.data['Pitch_Rate'] = angular_velocities[:, 1]
            self.data['Yaw_Rate'] = angular_velocities[:, 2]
            
            # Calculate oscillation amplitude and frequency
            self.calculate_oscillation_metrics()
            
        except Exception as e:
            print(f"Error in parachute metrics calculation: {str(e)}")
            # Initialize with zeros to prevent crashes
            n_points = len(self.data)
            self.data['Roll'] = np.zeros(n_points)
            self.data['Pitch'] = np.zeros(n_points)
            self.data['Yaw'] = np.zeros(n_points)
            self.data['Roll_Rate'] = np.zeros(n_points)
            self.data['Pitch_Rate'] = np.zeros(n_points)
            self.data['Yaw_Rate'] = np.zeros(n_points)
            raise
    
    def quaternion_to_euler(self, qw, qx, qy, qz):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        try:
            # Check for NaN or inf values
            quat_array = np.array([qw, qx, qy, qz])
            if np.any(np.isnan(quat_array)) or np.any(np.isinf(quat_array)):
                print("Warning: NaN or inf values in quaternion, returning zeros")
                return 0, 0, 0
            
            # Normalize quaternion
            norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            if norm < 1e-10:  # More robust check for near-zero norm
                print("Warning: Quaternion norm too small, returning zeros")
                return 0, 0, 0
            
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            # Clamp sinp to valid range to prevent numerical issues
            sinp = np.clip(sinp, -1.0, 1.0)
            if np.abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
            else:
                pitch = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Check results for validity
            euler_array = np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
            if np.any(np.isnan(euler_array)) or np.any(np.isinf(euler_array)):
                print("Warning: Invalid Euler angles computed, returning zeros")
                return 0, 0, 0
            
            return float(euler_array[0]), float(euler_array[1]), float(euler_array[2])
            
        except Exception as e:
            print(f"Error in quaternion_to_euler conversion: {str(e)}")
            return 0, 0, 0
    
    def calculate_oscillation_metrics(self):
        """Calculate parachute oscillation characteristics"""
        if self.data is None:
            print("Warning: No data available for oscillation metrics")
            self._initialize_default_metrics()
            return
            
        if len(self.data) < 10:
            print("Warning: Insufficient data for oscillation metrics (need at least 10 points)")
            self._initialize_default_metrics()
            return
        
        try:
            # Calculate sampling rate
            if 'Time' in self.data.columns:
                time_diff = self.data['Time'].diff().mean()
                if pd.isna(time_diff) or time_diff <= 0:
                    print("Warning: Invalid time data, using default sampling rate")
                    fs = 100  # Default to 100Hz
                else:
                    fs = 1.0 / time_diff
                    if fs > 10000:  # Sanity check for unrealistic sampling rates
                        print(f"Warning: Sampling rate {fs:.1f} Hz seems too high, capping at 1000 Hz")
                        fs = 1000
            else:
                print("Warning: No time column found, using default sampling rate")
                fs = 100
            
            # Calculate RMS oscillation for roll and pitch
            if 'Roll' in self.data.columns and 'Pitch' in self.data.columns:
                roll_data = self.data['Roll'].dropna()
                pitch_data = self.data['Pitch'].dropna()
                
                if len(roll_data) > 0 and len(pitch_data) > 0:
                    self.roll_rms = np.sqrt(np.mean(roll_data**2))
                    self.pitch_rms = np.sqrt(np.mean(pitch_data**2))
                    
                    # Calculate peak-to-peak oscillations
                    self.roll_pk2pk = np.max(roll_data) - np.min(roll_data)
                    self.pitch_pk2pk = np.max(pitch_data) - np.min(pitch_data)
                else:
                    print("Warning: No valid roll/pitch data")
                    self.roll_rms = self.pitch_rms = 0
                    self.roll_pk2pk = self.pitch_pk2pk = 0
            else:
                print("Warning: Roll/Pitch columns not found")
                self.roll_rms = self.pitch_rms = 0
                self.roll_pk2pk = self.pitch_pk2pk = 0
            
            # Find dominant oscillation frequency using FFT
            try:
                self.analyze_oscillation_frequency(fs)
            except Exception as e:
                print(f"Warning: Error in frequency analysis: {str(e)}")
                self.freqs = np.array([])
                self.roll_psd = np.array([])
                self.pitch_psd = np.array([])
                self.roll_dominant_freq = 0
                self.pitch_dominant_freq = 0
            
            # Calculate stability metrics
            try:
                self.calculate_stability_metrics()
            except Exception as e:
                print(f"Warning: Error in stability metrics: {str(e)}")
                self._initialize_default_stability_metrics()
                
        except Exception as e:
            print(f"Error calculating oscillation metrics: {str(e)}")
            self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize metrics with default values"""
        self.roll_rms = 0
        self.pitch_rms = 0
        self.roll_pk2pk = 0
        self.pitch_pk2pk = 0
        self.freqs = np.array([])
        self.roll_psd = np.array([])
        self.pitch_psd = np.array([])
        self.roll_dominant_freq = 0
        self.pitch_dominant_freq = 0
        self._initialize_default_stability_metrics()
    
    def _initialize_default_stability_metrics(self):
        """Initialize stability metrics with default values"""
        self.roll_stability = 0
        self.pitch_stability = 0
        self.coning_angle = np.array([0])
        self.max_coning_angle = 0
        self.avg_coning_angle = 0
    
    def create_hann_window(self, n):
        """Create Hann window manually if scipy is not available"""
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    
    def analyze_oscillation_frequency(self, fs):
        """Analyze dominant oscillation frequencies"""
        if len(self.data) < 100:
            self.freqs = np.array([])
            self.roll_psd = np.array([])
            self.pitch_psd = np.array([])
            self.roll_dominant_freq = 0
            self.pitch_dominant_freq = 0
            return
        
        n = len(self.data)
        # Apply window to reduce spectral leakage
        if signal is not None and hasattr(signal, 'hann'):
            window = signal.hann(n)
        elif signal is not None and hasattr(signal, 'windows') and hasattr(signal.windows, 'hann'):
            window = signal.windows.hann(n)
        else:
            # Create Hann window manually
            window = self.create_hann_window(n)
        
        # FFT for roll and pitch
        roll_fft = fft(self.data['Roll'] * window)
        pitch_fft = fft(self.data['Pitch'] * window)
        freqs = fftfreq(n, 1/fs)
        
        # Only look at positive frequencies up to reasonable parachute oscillation range (0-5 Hz)
        mask = (freqs > 0) & (freqs < 5)
        freqs_pos = freqs[mask]
        
        roll_power = np.abs(roll_fft[mask])**2
        pitch_power = np.abs(pitch_fft[mask])**2
        
        # Find dominant frequencies
        self.roll_dominant_freq = freqs_pos[np.argmax(roll_power)] if len(roll_power) > 0 else 0
        self.pitch_dominant_freq = freqs_pos[np.argmax(pitch_power)] if len(pitch_power) > 0 else 0
        
        # Store for plotting
        self.freqs = freqs_pos
        self.roll_psd = roll_power
        self.pitch_psd = pitch_power
    
    def calculate_stability_metrics(self):
        """Calculate parachute stability metrics"""
        # Standard deviation as measure of stability
        self.roll_stability = np.std(self.data['Roll'])
        self.pitch_stability = np.std(self.data['Pitch'])
        
        # Calculate coning angle (combination of roll and pitch)
        self.coning_angle = np.sqrt(self.data['Roll']**2 + self.data['Pitch']**2)
        self.max_coning_angle = np.max(self.coning_angle)
        self.avg_coning_angle = np.mean(self.coning_angle)

    def quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """Convert quaternion to rotation matrix"""
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm == 0:
            return np.eye(3)
        
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def create_sensor_axes(self, rotation_matrix):
        """Create 3D axes representing the sensor orientation"""
        # Define unit vectors for X, Y, Z axes
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Apply rotation
        rotated_axes = rotation_matrix @ axes.T
        
        return rotated_axes.T
    
    def setup_plot(self):
        """Setup the 3D plot for XYZ gyroscope rate visualization"""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(221, projection='3d')
        
        # Set equal aspect ratio and limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        
        # Labels for XYZ gyroscope rates
        self.ax.set_xlabel('X Axis (deg/s)')
        self.ax.set_ylabel('Y Axis (deg/s)')
        self.ax.set_zlabel('Z Axis (deg/s)')
        self.ax.set_title('XYZ Gyroscope Angular Rates')
        
        # Create initial axes showing XYZ rotation rates
        self.x_line, = self.ax.plot([0, 1], [0, 0], [0, 0], 'r-', linewidth=4, label='X Rate (Roll)')
        self.y_line, = self.ax.plot([0, 0], [0, 1], [0, 0], 'g-', linewidth=4, label='Y Rate (Pitch)')
        self.z_line, = self.ax.plot([0, 0], [0, 0], [0, 1], 'b-', linewidth=4, label='Z Rate (Yaw)')
        
        # Add sensor representation at origin
        self.ax.scatter([0], [0], [0], color='black', s=100, marker='s', label='IMU')
        
        # Add legend
        self.ax.legend()
        
        # Add text for frame and XYZ rate info
        self.frame_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10)
        
        return self.fig, self.ax
    
    def update_frame(self, frame_num):
        """Update the plot for animation with parachute metrics"""
        try:
            if self.data is None:
                print("Warning: No data available for frame update")
                return self.x_line, self.y_line, self.z_line, self.frame_text
            
            if frame_num < 0 or frame_num >= len(self.data):
                print(f"Warning: Frame {frame_num} out of range (0-{len(self.data)-1})")
                return self.x_line, self.y_line, self.z_line, self.frame_text
            
            # Get XYZ gyroscope data for current frame
            try:
                row = self.data.iloc[frame_num]
            except Exception as e:
                print(f"Error accessing data at frame {frame_num}: {str(e)}")
                return self.x_line, self.y_line, self.z_line, self.frame_text
            
            # Validate gyroscope data exists and handle missing values
            required_cols = ['Gx', 'Gy', 'Gz']
            missing_cols = [col for col in required_cols if col not in row]
            if missing_cols:
                print(f"Warning: Missing gyroscope columns at frame {frame_num}: {missing_cols}")
                return self.x_line, self.y_line, self.z_line, self.frame_text
            
            # Get gyroscope values, using 0 for NaN values
            gx = row['Gx'] if not pd.isna(row['Gx']) else 0.0
            gy = row['Gy'] if not pd.isna(row['Gy']) else 0.0
            gz = row['Gz'] if not pd.isna(row['Gz']) else 0.0
            
            # Scale gyroscope values for visualization
            # Since values are now in deg/s, scale them appropriately for display
            max_display_rate = 10.0  # deg/s - reasonable max for clear visualization
            scale_factor = 1.0
            
            # Calculate total rotation rate to determine scaling
            total_rate = np.sqrt(gx**2 + gy**2 + gz**2)
            if total_rate > max_display_rate:
                scale_factor = max_display_rate / total_rate
            elif total_rate > 0 and total_rate < 0.1:
                scale_factor = 10.0  # Scale up very small values
            
            # Apply scaling for visualization
            gx_scaled = gx * scale_factor  
            gy_scaled = gy * scale_factor
            gz_scaled = gz * scale_factor
            
            # Update line data - showing XYZ rotation rates directly
            try:
                # X axis shows rotation rate around X (red) - length represents magnitude
                self.x_line.set_data([0, gx_scaled], [0, 0])
                self.x_line.set_3d_properties([0, 0])
                
                # Y axis shows rotation rate around Y (green)
                self.y_line.set_data([0, 0], [0, gy_scaled])
                self.y_line.set_3d_properties([0, 0])
                
                # Z axis shows rotation rate around Z (blue)
                self.z_line.set_data([0, 0], [0, 0])
                self.z_line.set_3d_properties([0, gz_scaled])
            except Exception as e:
                print(f"Error updating axis lines at frame {frame_num}: {str(e)}")
            
            # Update frame info with XYZ gyroscope data
            try:
                time_val = row.get('Time', frame_num) if 'Time' in row else frame_num
                
                # Calculate total rotation rate magnitude
                try:
                    total_rate = np.sqrt(gx**2 + gy**2 + gz**2) if not (np.isnan(gx) or np.isnan(gy) or np.isnan(gz)) else 0
                except:
                    total_rate = 0
                
                info_text = f'Frame: {frame_num}/{len(self.data)-1}\n'
                info_text += f'Time: {time_val:.2f}s\n'
                info_text += f'X Rate: {gx:.2f}°/s\n'
                info_text += f'Y Rate: {gy:.2f}°/s\n'
                info_text += f'Z Rate: {gz:.2f}°/s\n'
                info_text += f'Total: {total_rate:.2f}°/s'
                
                self.frame_text.set_text(info_text)
            except Exception as e:
                print(f"Error updating frame text at frame {frame_num}: {str(e)}")
                self.frame_text.set_text(f'Frame: {frame_num}\nError displaying info')
            
            self.current_frame = frame_num
            
            return self.x_line, self.y_line, self.z_line, self.frame_text
            
        except Exception as e:
            print(f"Critical error in update_frame: {str(e)}")
            return self.x_line, self.y_line, self.z_line, self.frame_text
    
    def parachute_analysis_dashboard(self):
        """Create comprehensive parachute analysis dashboard"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D Orientation (top left)
        ax1 = fig.add_subplot(331, projection='3d')
        self.setup_3d_subplot(ax1)
        
        # 2. Roll/Pitch time series (top center)
        ax2 = fig.add_subplot(332)
        time_data = self.data['Time'] if 'Time' in self.data.columns else range(len(self.data))
        ax2.plot(time_data, self.data['Roll'], 'r-', label='Roll', alpha=0.7)
        ax2.plot(time_data, self.data['Pitch'], 'b-', label='Pitch', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (degrees)')
        ax2.set_title('Roll & Pitch Oscillations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Coning angle (top right)
        ax3 = fig.add_subplot(333)
        coning_angle = np.sqrt(self.data['Roll']**2 + self.data['Pitch']**2)
        ax3.plot(time_data, coning_angle, 'g-', linewidth=2)
        ax3.axhline(y=self.avg_coning_angle, color='orange', linestyle='--', label=f'Avg: {self.avg_coning_angle:.1f}°')
        ax3.axhline(y=self.max_coning_angle, color='red', linestyle='--', label=f'Max: {self.max_coning_angle:.1f}°')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Coning Angle (degrees)')
        ax3.set_title('Parachute Coning Angle')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Angular velocities (middle left)
        ax4 = fig.add_subplot(334)
        if 'Roll_Rate' in self.data.columns:
            ax4.plot(time_data, self.data['Roll_Rate'], 'r-', label='Roll Rate', alpha=0.7)
            ax4.plot(time_data, self.data['Pitch_Rate'], 'b-', label='Pitch Rate', alpha=0.7)
            ax4.plot(time_data, self.data['Yaw_Rate'], 'g-', label='Yaw Rate', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Velocity (deg/s)')
        ax4.set_title('Angular Velocities')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Frequency analysis (middle center)
        ax5 = fig.add_subplot(335)
        if hasattr(self, 'freqs') and len(self.freqs) > 0:
            ax5.semilogy(self.freqs, self.roll_psd, 'r-', label=f'Roll (peak: {self.roll_dominant_freq:.2f} Hz)', alpha=0.7)
            ax5.semilogy(self.freqs, self.pitch_psd, 'b-', label=f'Pitch (peak: {self.pitch_dominant_freq:.2f} Hz)', alpha=0.7)
            ax5.set_xlabel('Frequency (Hz)')
            ax5.set_ylabel('Power Spectral Density')
            ax5.set_title('Oscillation Frequency Analysis')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            ax5.set_xlim(0, 5)  # Focus on parachute-relevant frequencies
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for frequency analysis\n(need >100 points)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Frequency Analysis - Insufficient Data')
        
        # 6. Oscillation statistics (middle right)
        ax6 = fig.add_subplot(336)
        metrics = ['Roll RMS', 'Pitch RMS', 'Roll Pk-Pk', 'Pitch Pk-Pk', 'Avg Coning', 'Max Coning']
        values = [self.roll_rms, self.pitch_rms, self.roll_pk2pk, self.pitch_pk2pk, 
                 self.avg_coning_angle, self.max_coning_angle]
        colors = ['red', 'blue', 'darkred', 'darkblue', 'green', 'orange']
        bars = ax6.bar(range(len(metrics)), values, color=colors, alpha=0.7)
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(metrics, rotation=45, ha='right')
        ax6.set_ylabel('Angle (degrees)')
        ax6.set_title('Oscillation Metrics Summary')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}°', ha='center', va='bottom', fontsize=8)
        
        # 7. Phase plot (bottom left)
        ax7 = fig.add_subplot(337)
        ax7.plot(self.data['Roll'], self.data['Pitch'], 'purple', alpha=0.6, linewidth=0.5)
        ax7.scatter(self.data['Roll'].iloc[0], self.data['Pitch'].iloc[0], color='green', s=50, label='Start', zorder=5)
        ax7.scatter(self.data['Roll'].iloc[-1], self.data['Pitch'].iloc[-1], color='red', s=50, label='End', zorder=5)
        ax7.set_xlabel('Roll (degrees)')
        ax7.set_ylabel('Pitch (degrees)')
        ax7.set_title('Roll-Pitch Phase Plot')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        ax7.axis('equal')
        
        # 8. Stability timeline (bottom center)
        ax8 = fig.add_subplot(338)
        window_size = max(50, len(self.data) // 20)  # Rolling window
        roll_std = self.data['Roll'].rolling(window=window_size).std()
        pitch_std = self.data['Pitch'].rolling(window=window_size).std()
        ax8.plot(time_data, roll_std, 'r-', label='Roll Stability', alpha=0.7)
        ax8.plot(time_data, pitch_std, 'b-', label='Pitch Stability', alpha=0.7)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Rolling Std Dev (degrees)')
        ax8.set_title('Stability Over Time')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # 9. Summary text (bottom right)
        ax9 = fig.add_subplot(339)
        ax9.axis('off')
        summary_text = f"""PARACHUTE TEST SUMMARY
        
Data Duration: {time_data.iloc[-1] - time_data.iloc[0]:.1f} seconds
Sample Count: {len(self.data)} points
        
OSCILLATION ANALYSIS:
Roll RMS: {self.roll_rms:.2f}°
Pitch RMS: {self.pitch_rms:.2f}°
Peak Coning Angle: {self.max_coning_angle:.2f}°
Average Coning: {self.avg_coning_angle:.2f}°

FREQUENCY ANALYSIS:
Roll Dominant Freq: {self.roll_dominant_freq:.3f} Hz
Pitch Dominant Freq: {self.pitch_dominant_freq:.3f} Hz

STABILITY METRICS:
Roll Stability: {self.roll_stability:.2f}°
Pitch Stability: {self.pitch_stability:.2f}°

ASSESSMENT:
Stability: {'EXCELLENT' if max(self.roll_rms, self.pitch_rms) < 5 else 'GOOD' if max(self.roll_rms, self.pitch_rms) < 15 else 'POOR'}
"""
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def setup_3d_subplot(self, ax):
        """Setup a 3D subplot for the dashboard using current XYZ data"""
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X (deg/s)')
        ax.set_ylabel('Y (deg/s)')
        ax.set_zlabel('Z (deg/s)')
        ax.set_title('Current XYZ Rates')
        
        # Show current frame XYZ rates with proper scaling
        if self.data is not None and len(self.data) > 0:
            frame_num = min(self.current_frame, len(self.data) - 1)
            row = self.data.iloc[frame_num]
            
            # Get actual gyroscope values (in deg/s)
            gx = row.get('Gx', 0)
            gy = row.get('Gy', 0) 
            gz = row.get('Gz', 0)
            
            # Scale for visualization 
            max_display = 1.5  # Max length for display vectors
            max_val = max(abs(gx), abs(gy), abs(gz))
            if max_val > 0:
                scale = min(max_display, max_display / max_val * 10)  # Scale to show direction and magnitude
            else:
                scale = 1.0
            
            gx_vis = gx * scale
            gy_vis = gy * scale
            gz_vis = gz * scale
            
            # Plot XYZ rate vectors
            ax.plot([0, gx_vis], [0, 0], [0, 0], 'r-', linewidth=3, label=f'X: {gx:.2f}°/s')
            ax.plot([0, 0], [0, gy_vis], [0, 0], 'g-', linewidth=3, label=f'Y: {gy:.2f}°/s')
            ax.plot([0, 0], [0, 0], [0, gz_vis], 'b-', linewidth=3, label=f'Z: {gz:.2f}°/s')
            ax.scatter([0], [0], [0], color='black', s=50)
            ax.legend()

    def animate(self):
        """Start the animation with enhanced parachute visualization"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        self.setup_plot()
        
        # Create animation controls
        self.create_controls()
        
        # Create animation
        self.animation_obj = animation.FuncAnimation(
            self.fig, self.update_frame, frames=len(self.data),
            interval=50, blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
    
    def create_controls(self):
        """Create playback controls"""
        # Make room for controls
        plt.subplots_adjust(bottom=0.2)
        
        # Frame slider
        ax_frame = plt.axes([0.1, 0.05, 0.6, 0.03])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, len(self.data)-1, 
                                  valinit=0, valfmt='%d')
        self.frame_slider.on_changed(self.on_frame_change)
        
        # Play/Pause button
        ax_play = plt.axes([0.75, 0.05, 0.1, 0.04])
        self.play_button = Button(ax_play, 'Play')
        self.play_button.on_clicked(self.toggle_play)
        
        # Speed control
        ax_speed = plt.axes([0.1, 0.01, 0.3, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        self.speed_slider.on_changed(self.on_speed_change)
    
    def on_frame_change(self, val):
        """Handle frame slider change"""
        frame = int(self.frame_slider.val)
        self.update_frame(frame)
        self.fig.canvas.draw()
    
    def toggle_play(self, event):
        """Toggle play/pause"""
        if self.animation_obj:
            if self.is_playing:
                self.animation_obj.pause()
                self.play_button.label.set_text('Play')
                self.is_playing = False
            else:
                self.animation_obj.resume()
                self.play_button.label.set_text('Pause')
                self.is_playing = True
    
    def on_speed_change(self, val):
        """Handle speed change"""
        if self.animation_obj:
            new_interval = max(10, int(50 / val))  # Adjust interval based on speed
            self.animation_obj.event_source.interval = new_interval
    
    def static_visualization(self, frame_index=0):
        """Create a static visualization of a single frame"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if frame_index >= len(self.data):
            frame_index = len(self.data) - 1
        
        self.setup_plot()
        self.update_frame(frame_index)
        
        plt.tight_layout()
        plt.show()
    
    def plot_quaternion_components(self):
        """Plot quaternion components over time"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        time_data = self.data['Time'] if 'Time' in self.data.columns else range(len(self.data))
        
        # Plot each quaternion component
        axes[0, 0].plot(time_data, self.data['Qw'], 'r-', label='Qw')
        axes[0, 0].set_title('Quaternion W Component')
        axes[0, 0].set_ylabel('Qw')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_data, self.data['Qx'], 'g-', label='Qx')
        axes[0, 1].set_title('Quaternion X Component')
        axes[0, 1].set_ylabel('Qx')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time_data, self.data['Qy'], 'b-', label='Qy')
        axes[1, 0].set_title('Quaternion Y Component')
        axes[1, 0].set_ylabel('Qy')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time_data, self.data['Qz'], 'm-', label='Qz')
        axes[1, 1].set_title('Quaternion Z Component')
        axes[1, 1].set_ylabel('Qz')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the IMU visualizer"""
    print("IMU XYZ Gyroscope Analysis Tool")
    print("===============================")
    
    try:
        visualizer = IMU3DVisualizer()
    except Exception as e:
        print(f"Error initializing visualizer: {str(e)}")
        return
    
    # Load data with retry option
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if visualizer.load_data():
                break
            else:
                if attempt < max_retries - 1:
                    retry = input(f"Failed to load data (attempt {attempt + 1}/{max_retries}). Try again? (y/n): ")
                    if retry.lower() != 'y':
                        return
                else:
                    print("Failed to load data after multiple attempts. Exiting.")
                    return
        except Exception as e:
            print(f"Error during data loading attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                retry = input(f"Try again? (y/n): ")
                if retry.lower() != 'y':
                    return
            else:
                print("Failed to load data after multiple attempts. Exiting.")
                return
    
    while True:
        try:
            print("\nSelect analysis option:")
            print("1. Animated 3D XYZ visualization")
            print("2. XYZ orientation rates vs time plot")
            print("3. Static 3D visualization (single frame)")
            print("4. Comprehensive XYZ analysis dashboard")
            print("5. Exit")
            
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == '1':
                try:
                    visualizer.animate()
                except Exception as e:
                    print(f"Error in animation: {str(e)}")
            elif choice == '2':
                try:
                    visualizer.plot_xyz_vs_time()
                except Exception as e:
                    print(f"Error in XYZ vs time plot: {str(e)}")
            elif choice == '3':
                try:
                    if visualizer.data is None or len(visualizer.data) == 0:
                        print("No data available for visualization")
                        continue
                    max_frame = len(visualizer.data) - 1
                    frame = input(f"Enter frame number (0 to {max_frame}): ")
                    try:
                        frame_num = int(frame)
                        if 0 <= frame_num <= max_frame:
                            visualizer.static_visualization(frame_num)
                        else:
                            print(f"Frame number must be between 0 and {max_frame}")
                    except ValueError:
                        print("Invalid frame number. Please enter an integer.")
                except Exception as e:
                    print(f"Error in static visualization: {str(e)}")
            elif choice == '4':
                try:
                    visualizer.xyz_analysis_dashboard()
                except Exception as e:
                    print(f"Error in dashboard: {str(e)}")
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
            break
        except EOFError:
            print("\nInput stream ended. Exiting.")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue

if __name__ == "__main__":
    main()