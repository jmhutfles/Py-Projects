import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox
from ReadRawData import ReadIMU


def load_imu_data(prompt="Select one or more IMU CSV files"):
	"""Load IMU data using the existing ReadRawData.ReadIMU helper."""
	data, paths = ReadIMU(prompt)
	if data is None or data.empty:
		return None, None

	for col in data.columns:
		data[col] = pd.to_numeric(data[col], errors="coerce")

	required_cols = ["Time", "Gx", "Gy", "Gz"]
	missing = [col for col in required_cols if col not in data.columns]
	if missing:
		raise ValueError(f"Missing required IMU columns: {', '.join(missing)}")

	data = data.sort_values("Time").reset_index(drop=True)
	return data, paths


def convert_gyro_mdps_to_dps(data):
	"""Convert gyro columns from mdps to dps if magnitudes indicate mdps input."""
	for col in ["Gx", "Gy", "Gz"]:
		if col not in data.columns:
			raise ValueError(f"Required gyro column missing: {col}")

	max_abs_rate = data[["Gx", "Gy", "Gz"]].abs().max().max()
	if pd.notna(max_abs_rate) and max_abs_rate > 100:
		data[["Gx", "Gy", "Gz"]] = data[["Gx", "Gy", "Gz"]] / 1000.0

	return data


def compute_orientation_and_change(data):
	"""Compute orientation (deg) and orientation change rate (deg/s) for X/Y/Z."""
	work = data.copy()

	work = work.dropna(subset=["Time"]).copy()
	work[["Gx", "Gy", "Gz"]] = work[["Gx", "Gy", "Gz"]].fillna(0.0)
	work = work.sort_values("Time").reset_index(drop=True)

	time = work["Time"].to_numpy(dtype=float)
	gx = work["Gx"].to_numpy(dtype=float)
	gy = work["Gy"].to_numpy(dtype=float)
	gz = work["Gz"].to_numpy(dtype=float)

	dt = np.diff(time, prepend=time[0])
	dt[0] = 0.0
	dt = np.where(dt < 0, 0.0, dt)

	orientation_x = np.cumsum(gx * dt)
	orientation_y = np.cumsum(gy * dt)
	orientation_z = np.cumsum(gz * dt)

	unique_times = np.unique(time)
	if unique_times.size > 1:
		change_x = np.gradient(orientation_x, time)
		change_y = np.gradient(orientation_y, time)
		change_z = np.gradient(orientation_z, time)
	else:
		change_x = np.zeros_like(orientation_x)
		change_y = np.zeros_like(orientation_y)
		change_z = np.zeros_like(orientation_z)

	result = pd.DataFrame(
		{
			"Time": time,
			"Orientation_X_deg": orientation_x,
			"Orientation_Y_deg": orientation_y,
			"Orientation_Z_deg": orientation_z,
			"Change_X_deg_per_s": change_x,
			"Change_Y_deg_per_s": change_y,
			"Change_Z_deg_per_s": change_z,
		}
	)

	return result


def plot_orientation_quick_view(result):
	"""Plot 3 orientation plots and 3 orientation-change plots."""
	time = result["Time"].to_numpy()

	fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
	fig.suptitle("IMU Orientation Quick View", fontsize=14)

	top_plots = [
		("Orientation_X_deg", "X Orientation (deg)", "tab:red"),
		("Orientation_Y_deg", "Y Orientation (deg)", "tab:green"),
		("Orientation_Z_deg", "Z Orientation (deg)", "tab:blue"),
	]

	bottom_plots = [
		("Change_X_deg_per_s", "dX/dt (deg/s)", "tab:red"),
		("Change_Y_deg_per_s", "dY/dt (deg/s)", "tab:green"),
		("Change_Z_deg_per_s", "dZ/dt (deg/s)", "tab:blue"),
	]

	for idx, (column, ylabel, color) in enumerate(top_plots):
		axes[0, idx].plot(time, result[column], color=color, linewidth=1.2)
		axes[0, idx].set_title(ylabel)
		axes[0, idx].set_ylabel(ylabel)
		axes[0, idx].grid(True, alpha=0.3)

	for idx, (column, ylabel, color) in enumerate(bottom_plots):
		axes[1, idx].plot(time, result[column], color=color, linewidth=1.2)
		axes[1, idx].set_title(ylabel)
		axes[1, idx].set_xlabel("Time (s)")
		axes[1, idx].set_ylabel(ylabel)
		axes[1, idx].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def main():
	try:
		data, paths = load_imu_data("Select IMU Data File(s) for Orientation Quick View")
		if data is None or data.empty:
			print("No IMU files selected.")
			return

		data = convert_gyro_mdps_to_dps(data)
		result = compute_orientation_and_change(data)

		print(f"Loaded {len(result)} IMU samples from {len(paths)} file(s).")
		plot_orientation_quick_view(result)

	except Exception as error:
		messagebox.showerror("Orientation Quick View Error", str(error))
		print(f"Error: {error}")


if __name__ == "__main__":
	main()
