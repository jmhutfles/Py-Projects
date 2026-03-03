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


def _estimate_gyro_bias(gx, gy, gz):
	"""Estimate gyro bias from lowest-motion samples."""
	if len(gx) == 0:
		return 0.0, 0.0, 0.0

	mag = np.sqrt(gx**2 + gy**2 + gz**2)
	threshold = min(float(np.quantile(mag, 0.2)), 5.0)
	mask = mag <= threshold

	if np.count_nonzero(mask) < 10:
		return 0.0, 0.0, 0.0

	return (
		float(np.median(gx[mask])),
		float(np.median(gy[mask])),
		float(np.median(gz[mask])),
	)


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

	bias_x, bias_y, bias_z = _estimate_gyro_bias(gx, gy, gz)
	gx = gx - bias_x
	gy = gy - bias_y
	gz = gz - bias_z
	work["Gx"] = gx
	work["Gy"] = gy
	work["Gz"] = gz

	dt = np.diff(time, prepend=time[0])
	dt[0] = 0.0
	dt = np.where(dt < 0, 0.0, dt)

	orientation_x = np.cumsum(gx * dt)
	orientation_y = np.cumsum(gy * dt)
	orientation_z = np.cumsum(gz * dt)
	abs_orientation_x = np.cumsum(np.abs(gx) * dt)
	abs_orientation_y = np.cumsum(np.abs(gy) * dt)
	abs_orientation_z = np.cumsum(np.abs(gz) * dt)

	turns_x = orientation_x / 360.0
	turns_y = orientation_y / 360.0
	turns_z = orientation_z / 360.0
	eps = 1e-9
	turns_count_x = np.where(turns_x >= 0, np.floor(turns_x + eps), np.ceil(turns_x - eps))
	turns_count_y = np.where(turns_y >= 0, np.floor(turns_y + eps), np.ceil(turns_y - eps))
	turns_count_z = np.where(turns_z >= 0, np.floor(turns_z + eps), np.ceil(turns_z - eps))
	abs_turns_x = abs_orientation_x / 360.0
	abs_turns_y = abs_orientation_y / 360.0
	abs_turns_z = abs_orientation_z / 360.0

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
			"AbsOrientation_X_deg": abs_orientation_x,
			"AbsOrientation_Y_deg": abs_orientation_y,
			"AbsOrientation_Z_deg": abs_orientation_z,
			"Turns_X": turns_x,
			"Turns_Y": turns_y,
			"Turns_Z": turns_z,
			"TurnsCount_X": turns_count_x,
			"TurnsCount_Y": turns_count_y,
			"TurnsCount_Z": turns_count_z,
			"AbsTurns_X": abs_turns_x,
			"AbsTurns_Y": abs_turns_y,
			"AbsTurns_Z": abs_turns_z,
			"Change_X_deg_per_s": change_x,
			"Change_Y_deg_per_s": change_y,
			"Change_Z_deg_per_s": change_z,
			"GyroBias_X_deg_per_s": np.full_like(time, bias_x, dtype=float),
			"GyroBias_Y_deg_per_s": np.full_like(time, bias_y, dtype=float),
			"GyroBias_Z_deg_per_s": np.full_like(time, bias_z, dtype=float),
		}
	)

	rotation_matrices = _accumulate_rotation_matrices(work)
	z_align = np.array([rot[2, 2] for rot in rotation_matrices], dtype=float)
	z_align = np.clip(z_align, -1.0, 1.0)
	result["ConeAngle_deg"] = np.degrees(np.arccos(z_align))

	return result


def _accumulate_rotation_matrices(work):
	"""Build cumulative rotation matrices from angular velocities (no wrapping)."""
	gx = work["Gx"].to_numpy(dtype=float)
	gy = work["Gy"].to_numpy(dtype=float)
	gz = work["Gz"].to_numpy(dtype=float)
	time = work["Time"].to_numpy(dtype=float)

	dt = np.diff(time, prepend=time[0])
	dt[0] = 0.0
	dt = np.where(dt < 0, 0.0, dt)

	rot = np.eye(3)
	rotation_list = []

	for idx in range(len(work)):
		rotation_list.append(rot.copy())

		if idx < len(gx) - 1:
			omega = np.array([gx[idx], gy[idx], gz[idx]]) * np.pi / 180.0
			theta = np.linalg.norm(omega)

			if theta > 1e-6:
				axis = omega / theta
				K = np.array([
					[0, -axis[2], axis[1]],
					[axis[2], 0, -axis[0]],
					[-axis[1], axis[0], 0],
				])
				drot = np.eye(3) + np.sin(theta * dt[idx]) * K + (1 - np.cos(theta * dt[idx])) * (K @ K)
				rot = drot @ rot
				u, _, vh = np.linalg.svd(rot)
				rot = u @ vh
				if np.linalg.det(rot) < 0:
					u[:, -1] *= -1
					rot = u @ vh

	return rotation_list


def plot_orientation_quick_view(result):
	"""Plot 6 core graphs plus cone-angle summary graph."""
	time = result["Time"].to_numpy()

	fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
	axes = axes.flatten()
	fig.suptitle("IMU Orientation Quick View", fontsize=14)

	top_plots = [
		("Orientation_X_deg", "AbsTurns_X", "X Orientation (deg)", "tab:red"),
		("Orientation_Y_deg", "AbsTurns_Y", "Y Orientation (deg)", "tab:green"),
		("Orientation_Z_deg", "AbsTurns_Z", "Z Orientation (deg)", "tab:blue"),
	]

	bottom_plots = [
		("Change_X_deg_per_s", "dX/dt (deg/s)", "tab:red"),
		("Change_Y_deg_per_s", "dY/dt (deg/s)", "tab:green"),
		("Change_Z_deg_per_s", "dZ/dt (deg/s)", "tab:blue"),
	]

	for idx, (column, total_turns_column, ylabel, color) in enumerate(top_plots):
		ax = axes[idx]
		ax.plot(time, result[column], color=color, linewidth=1.2)
		ax.set_title(ylabel)
		ax.set_ylabel(ylabel)
		ax.grid(True, alpha=0.3)

		turns_ax = ax.twinx()
		net_turns_line, = turns_ax.plot(
			time,
			result[total_turns_column],
			color="tab:purple",
			linestyle=":",
			linewidth=1.4,
			alpha=0.95,
			label="Total Turns",
		)
		turns_ax.set_ylabel("Turns", color="tab:purple")
		turns_ax.tick_params(axis="y", colors="tab:purple")
		turns_ax.legend(handles=[net_turns_line], loc="upper right", fontsize=8)

	for idx, (column, ylabel, color) in enumerate(bottom_plots):
		ax = axes[3 + idx]
		ax.plot(time, result[column], color=color, linewidth=1.2)
		ax.set_title(ylabel)
		ax.set_xlabel("Time (s)")
		ax.set_ylabel(ylabel)
		ax.grid(True, alpha=0.3)

	ax_summary = axes[6]
	ax_summary.plot(time, result["ConeAngle_deg"], color="tab:orange", linewidth=1.4, label="Cone Angle [deg]")
	ax_summary.set_title("Cone Angle Summary")
	ax_summary.set_xlabel("Time (s)")
	ax_summary.set_ylabel("Angle (deg)")
	ax_summary.grid(True, alpha=0.3)
	ax_summary.legend(loc="upper right", fontsize=8)

	axes[7].axis("off")
	axes[8].axis("off")

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
