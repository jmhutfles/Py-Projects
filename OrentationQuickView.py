import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
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
	result["RotationMatrix"] = pd.Series(rotation_matrices, index=result.index, dtype=object)
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
		("Orientation_X_deg", "Turns_X", "X Orientation (deg)", "tab:red"),
		("Orientation_Y_deg", "Turns_Y", "Y Orientation (deg)", "tab:green"),
		("Orientation_Z_deg", "Turns_Z", "Z Orientation (deg)", "tab:blue"),
	]

	bottom_plots = [
		("Change_X_deg_per_s", "dX/dt (deg/s)", "tab:red"),
		("Change_Y_deg_per_s", "dY/dt (deg/s)", "tab:green"),
		("Change_Z_deg_per_s", "dZ/dt (deg/s)", "tab:blue"),
	]

	for idx, (column, turns_column, ylabel, color) in enumerate(top_plots):
		ax = axes[idx]
		ax.plot(time, result[column], color=color, linewidth=1.2)
		ax.set_title(ylabel)
		ax.set_ylabel(ylabel)
		ax.grid(True, alpha=0.3)

		turns_ax = ax.twinx()
		net_turns_line, = turns_ax.plot(
			time,
			result[turns_column],
			color="tab:purple",
			linestyle="--",
			linewidth=1.0,
			alpha=0.8,
			label="Net Turns",
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


def _rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
	rx = np.deg2rad(rx_deg)
	ry_ = np.deg2rad(ry_deg)
	rz = np.deg2rad(rz_deg)

	rx_matrix = np.array([
		[1, 0, 0],
		[0, np.cos(rx), -np.sin(rx)],
		[0, np.sin(rx), np.cos(rx)],
	])

	ry_matrix = np.array([
		[np.cos(ry_), 0, np.sin(ry_)],
		[0, 1, 0],
		[-np.sin(ry_), 0, np.cos(ry_)],
	])

	rz_matrix = np.array([
		[np.cos(rz), -np.sin(rz), 0],
		[np.sin(rz), np.cos(rz), 0],
		[0, 0, 1],
	])

	return rz_matrix @ ry_matrix @ rx_matrix


def animate_sensor_orientation(result, max_frames=1200):
	"""Animate a physical-looking IMU sensor body rotating over time."""
	time = result["Time"].to_numpy(dtype=float)
	orientation_x = result["Orientation_X_deg"].to_numpy(dtype=float)
	orientation_y = result["Orientation_Y_deg"].to_numpy(dtype=float)
	orientation_z = result["Orientation_Z_deg"].to_numpy(dtype=float)
	rotation_matrices = result["RotationMatrix"].to_numpy()

	n_samples = len(result)
	if n_samples == 0:
		return

	step = max(1, int(np.ceil(n_samples / max_frames)))
	frame_indices = np.arange(0, n_samples, step)

	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title("IMU Sensor Body Rotation")

	ax.set_xlim(-0.06, 0.06)
	ax.set_ylim(-0.06, 0.06)
	ax.set_zlim(-0.06, 0.06)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_box_aspect([1, 1, 1])
	ax.grid(True, alpha=0.25)

	axis_len = 0.05
	ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color="tab:red", alpha=0.25, linewidth=1)
	ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color="tab:green", alpha=0.25, linewidth=1)
	ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color="tab:blue", alpha=0.25, linewidth=1)

	half_l, half_w, half_h = 0.02, 0.012, 0.003
	local_vertices = np.array([
		[-half_l, -half_w, -half_h],
		[ half_l, -half_w, -half_h],
		[ half_l,  half_w, -half_h],
		[-half_l,  half_w, -half_h],
		[-half_l, -half_w,  half_h],
		[ half_l, -half_w,  half_h],
		[ half_l,  half_w,  half_h],
		[-half_l,  half_w,  half_h],
	])

	faces_idx = [
		[0, 1, 2, 3],
		[4, 5, 6, 7],
		[0, 1, 5, 4],
		[1, 2, 6, 5],
		[2, 3, 7, 6],
		[3, 0, 4, 7],
	]

	face_colors = [
		"#D8D8D8",
		"#FFCC66",
		"#C0C0C0",
		"#A9A9A9",
		"#B8B8B8",
		"#B0B0B0",
	]

	front_local = np.array([half_l, 0.0, 0.0])
	front_line, = ax.plot([0, front_local[0]], [0, front_local[1]], [0, front_local[2]], color="tab:red", linewidth=2.5, label="Sensor Front")
	center_pt = ax.scatter([0], [0], [0], color="k", s=18)
	text_info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
	ax.legend(loc="upper right")
	sensor_poly = None

	if len(time) > 1:
		dt = np.median(np.diff(time))
		interval_s = max(0.01, float(dt * step))
	else:
		interval_s = 0.05

	plt.tight_layout()
	plt.show(block=False)

	for idx in frame_indices:
		if not plt.fignum_exists(fig.number):
			break

		rot = rotation_matrices[idx]

		rot_vertices = (rot @ local_vertices.T).T
		face_vertices = [[rot_vertices[i] for i in face] for face in faces_idx]

		if sensor_poly is not None:
			sensor_poly.remove()

		sensor_poly = Poly3DCollection(
			face_vertices,
			facecolors=face_colors,
			edgecolors="k",
			linewidths=0.8,
			alpha=0.95,
		)
		ax.add_collection3d(sensor_poly)

		front_vec = rot @ front_local
		front_line.set_data_3d([0, front_vec[0]], [0, front_vec[1]], [0, front_vec[2]])

		text_info.set_text(
			f"t={time[idx]:.2f}s\n"
			f"X={orientation_x[idx]:.1f}°\n"
			f"Y={orientation_y[idx]:.1f}°\n"
			f"Z={orientation_z[idx]:.1f}°"
		)

		fig.canvas.draw_idle()
		plt.pause(interval_s)

	if plt.fignum_exists(fig.number):
		plt.show()


def interactive_scrubber_view(result):
	"""Show 2D orientation plots + 3D sensor model in one pane with time scrubber."""
	time = result["Time"].to_numpy(dtype=float)
	if len(time) == 0:
		return

	series_info = [
		("Orientation_X_deg", "X Orientation (deg)", "tab:red"),
		("Orientation_Y_deg", "Y Orientation (deg)", "tab:green"),
		("Orientation_Z_deg", "Z Orientation (deg)", "tab:blue"),
		("Change_X_deg_per_s", "dX/dt (deg/s)", "tab:red"),
		("Change_Y_deg_per_s", "dY/dt (deg/s)", "tab:green"),
		("Change_Z_deg_per_s", "dZ/dt (deg/s)", "tab:blue"),
	]

	fig = plt.figure(figsize=(18, 9))
	gs = fig.add_gridspec(
		2,
		4,
		left=0.05,
		right=0.98,
		top=0.93,
		bottom=0.16,
		wspace=0.3,
		hspace=0.3,
	)
	fig.suptitle("IMU One-Pane Orientation Scrubber", fontsize=14)

	plot_axes = [
		fig.add_subplot(gs[0, 0]),
		fig.add_subplot(gs[0, 1]),
		fig.add_subplot(gs[0, 2]),
		fig.add_subplot(gs[1, 0]),
		fig.add_subplot(gs[1, 1]),
		fig.add_subplot(gs[1, 2]),
	]
	ax3d = fig.add_subplot(gs[:, 3], projection="3d")

	vlines = []
	markers = []
	series_values = []

	for idx, (col, title, color) in enumerate(series_info):
		ax = plot_axes[idx]
		values = result[col].to_numpy(dtype=float)
		series_values.append(values)

		ax.plot(time, values, color=color, linewidth=1.2)
		ax.set_title(title)
		ax.set_ylabel(title)
		if idx >= 3:
			ax.set_xlabel("Time (s)")
		ax.grid(True, alpha=0.3)

		vline = ax.axvline(time[0], color="k", linestyle="--", linewidth=1.2, alpha=0.75)
		marker, = ax.plot([time[0]], [values[0]], "o", color="k", markersize=5)
		vlines.append(vline)
		markers.append(marker)

		if idx < 3:
			turn_col = ["Turns_X", "Turns_Y", "Turns_Z"][idx]
			abs_turn_col = ["AbsTurns_X", "AbsTurns_Y", "AbsTurns_Z"][idx]
			turns_values = result[turn_col].to_numpy(dtype=float)
			abs_turns_values = result[abs_turn_col].to_numpy(dtype=float)
			turns_ax = ax.twinx()
			turns_ax.plot(
				time,
				turns_values,
				color="tab:purple",
				linestyle="--",
				linewidth=1.0,
				alpha=0.8,
				label="Net Turns",
			)
			turns_ax.plot(
				time,
				abs_turns_values,
				color="tab:orange",
				linestyle=":",
				linewidth=1.1,
				alpha=0.9,
				label="Abs Turns",
			)
			turns_ax.set_ylabel("Turns", color="tab:purple")
			turns_ax.tick_params(axis="y", colors="tab:purple")
			turns_ax.legend(loc="upper right", fontsize=8)

	ax3d.set_title("3D Sensor Body")
	ax3d.set_xlim(-0.06, 0.06)
	ax3d.set_ylim(-0.06, 0.06)
	ax3d.set_zlim(-0.06, 0.06)
	ax3d.set_xlabel("X")
	ax3d.set_ylabel("Y")
	ax3d.set_zlabel("Z")
	ax3d.set_box_aspect([1, 1, 1])
	ax3d.grid(True, alpha=0.25)

	axis_len = 0.05
	ax3d.plot([-axis_len, axis_len], [0, 0], [0, 0], color="tab:red", alpha=0.25, linewidth=1)
	ax3d.plot([0, 0], [-axis_len, axis_len], [0, 0], color="tab:green", alpha=0.25, linewidth=1)
	ax3d.plot([0, 0], [0, 0], [-axis_len, axis_len], color="tab:blue", alpha=0.25, linewidth=1)
	ax3d.scatter([0], [0], [0], color="k", s=18)

	half_l, half_w, half_h = 0.02, 0.012, 0.003
	local_vertices = np.array([
		[-half_l, -half_w, -half_h],
		[half_l, -half_w, -half_h],
		[half_l, half_w, -half_h],
		[-half_l, half_w, -half_h],
		[-half_l, -half_w, half_h],
		[half_l, -half_w, half_h],
		[half_l, half_w, half_h],
		[-half_l, half_w, half_h],
	])

	faces_idx = [
		[0, 1, 2, 3],
		[4, 5, 6, 7],
		[0, 1, 5, 4],
		[1, 2, 6, 5],
		[2, 3, 7, 6],
		[3, 0, 4, 7],
	]
	face_colors = ["#D8D8D8", "#FFCC66", "#C0C0C0", "#A9A9A9", "#B8B8B8", "#B0B0B0"]

	front_local = np.array([half_l, 0.0, 0.0])
	front_line, = ax3d.plot(
		[0, front_local[0]],
		[0, front_local[1]],
		[0, front_local[2]],
		color="tab:red",
		linewidth=2.5,
		label="Sensor Front",
	)
	text_info = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes)
	ax3d.legend(loc="upper right")

	sensor_poly_ref = {"poly": None}

	orientation_x = result["Orientation_X_deg"].to_numpy(dtype=float)
	orientation_y = result["Orientation_Y_deg"].to_numpy(dtype=float)
	orientation_z = result["Orientation_Z_deg"].to_numpy(dtype=float)
	turns_x = result["Turns_X"].to_numpy(dtype=float)
	turns_y = result["Turns_Y"].to_numpy(dtype=float)
	turns_z = result["Turns_Z"].to_numpy(dtype=float)
	abs_turns_x = result["AbsTurns_X"].to_numpy(dtype=float)
	abs_turns_y = result["AbsTurns_Y"].to_numpy(dtype=float)
	abs_turns_z = result["AbsTurns_Z"].to_numpy(dtype=float)
	rotation_matrices = result["RotationMatrix"].to_numpy()

	def update_at_index(index):
		index = int(np.clip(index, 0, len(time) - 1))
		t_now = time[index]

		for j in range(6):
			vlines[j].set_xdata([t_now, t_now])
			markers[j].set_data([t_now], [series_values[j][index]])

		rot = rotation_matrices[index]
		rot_vertices = (rot @ local_vertices.T).T
		face_vertices = [[rot_vertices[i] for i in face] for face in faces_idx]

		if sensor_poly_ref["poly"] is not None:
			sensor_poly_ref["poly"].remove()

		sensor_poly_ref["poly"] = Poly3DCollection(
			face_vertices,
			facecolors=face_colors,
			edgecolors="k",
			linewidths=0.8,
			alpha=0.95,
		)
		ax3d.add_collection3d(sensor_poly_ref["poly"])

		front_vec = rot @ front_local
		front_line.set_data_3d([0, front_vec[0]], [0, front_vec[1]], [0, front_vec[2]])

		text_info.set_text(
			f"t={t_now:.2f}s\n"
			f"X={orientation_x[index]:.1f}°\n"
			f"Y={orientation_y[index]:.1f}°\n"
			f"Z={orientation_z[index]:.1f}°\n"
			f"Tn: X={turns_x[index]:.2f} Y={turns_y[index]:.2f} Z={turns_z[index]:.2f}\n"
			f"Ta: X={abs_turns_x[index]:.2f} Y={abs_turns_y[index]:.2f} Z={abs_turns_z[index]:.2f}"
		)

		fig.canvas.draw_idle()

	slider_ax = fig.add_axes([0.12, 0.07, 0.76, 0.03])
	time_slider = Slider(
		ax=slider_ax,
		label="Time (s)",
		valmin=float(time[0]),
		valmax=float(time[-1]),
		valinit=float(time[0]),
	)

	def on_slider_change(val):
		idx = np.searchsorted(time, val, side="left")
		if idx >= len(time):
			idx = len(time) - 1
		update_at_index(idx)

	time_slider.on_changed(on_slider_change)
	update_at_index(0)
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
