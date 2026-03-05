import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, simpledialog, messagebox
from tqdm import tqdm
import matplotlib.pyplot as plt

import ReadRawData


def _prompt_float(root, title, prompt, min_value=None):
    while True:
        value = simpledialog.askfloat(title, prompt, parent=root)
        if value is None:
            return None
        if min_value is not None and value < min_value:
            messagebox.showerror("Invalid Value", f"Value must be >= {min_value}.", parent=root)
            continue
        return float(value)


def _prepare_abt_dataframe(raw_df):
    data = raw_df.copy()
    data = data.dropna(subset=["Time", "P"])
    data = data.sort_values("Time")
    data = data.drop_duplicates(subset=["Time"], keep="first")

    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data["P"] = pd.to_numeric(data["P"], errors="coerce")
    data = data.dropna(subset=["Time", "P"])

    data["Altitude MSL (m)"] = 44330 * (1 - (data["P"] / 101325) ** (1 / 5.255))
    data["Altitude MSL (ft)"] = data["Altitude MSL (m)"] * 3.28084

    data["Altitude MSL (ft)"] = data["Altitude MSL (ft)"].rolling(window=21, center=True, min_periods=1).mean()

    return data.reset_index(drop=True)


def _format_timer(seconds_elapsed):
    minutes = int(seconds_elapsed // 60)
    seconds = seconds_elapsed - (minutes * 60)
    return f"{minutes:02d}:{seconds:05.2f}"


def _pick_landing_time_from_plot(df):
    clicked_time = []

    def on_click(event):
        if plt.get_current_fig_manager().toolbar.mode != "":
            return
        if event.xdata is None:
            return
        clicked_time.append(float(event.xdata))
        plt.close()

    fig, ax = plt.subplots()
    ax.plot(df["Time"], df["Altitude MSL (ft)"], color="deepskyblue", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude MSL (ft)")
    ax.set_title("Zoom/pan as needed, then click LANDING on ABT data")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if not clicked_time:
        return None
    return clicked_time[0]


def run_special_request_abt_video():
    root = Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select the video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        parent=root,
    )
    if not video_path:
        return

    output_path = filedialog.asksaveasfilename(
        title="Save output video as",
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        parent=root,
    )
    if not output_path:
        return

    raw_df, _ = ReadRawData.ReadABT("Select ABT file(s)")
    if raw_df is None or raw_df.empty:
        messagebox.showerror("No Data", "No ABT data loaded.", parent=root)
        return

    df = _prepare_abt_dataframe(raw_df)
    if df.empty:
        messagebox.showerror("No Data", "ABT data is empty after cleaning.", parent=root)
        return

    min_time = float(df["Time"].min())
    max_time = float(df["Time"].max())

    exit_time_abt = _prompt_float(
        root,
        "Exit Time",
        f"Enter exit time from ABT file (s)\nValid range: {min_time:.2f} to {max_time:.2f}",
    )
    if exit_time_abt is None:
        return

    full_open_time_abt = _prompt_float(
        root,
        "Full Open Time",
        f"Enter full-open time from ABT file (s)\nValid range: {min_time:.2f} to {max_time:.2f}",
    )
    if full_open_time_abt is None:
        return

    landing_data_time = _pick_landing_time_from_plot(df)
    if landing_data_time is None:
        messagebox.showerror("No Selection", "No landing point selected on ABT graph.", parent=root)
        return

    video_landing_time = _prompt_float(
        root,
        "Video Landing Time",
        "Enter landing time in VIDEO (s).",
        min_value=0,
    )
    if video_landing_time is None:
        return

    if not (min_time <= exit_time_abt <= max_time):
        messagebox.showerror("Invalid Time", "Exit ABT time is out of data range.", parent=root)
        return

    if not (min_time <= full_open_time_abt <= max_time):
        messagebox.showerror("Invalid Time", "Full-open ABT time is out of data range.", parent=root)
        return

    if full_open_time_abt < exit_time_abt:
        messagebox.showerror("Invalid Time", "Full-open time must be >= exit time.", parent=root)
        return

    if not (min_time <= landing_data_time <= max_time):
        messagebox.showerror("Invalid Time", "Selected landing time is out of ABT data range.", parent=root)
        return

    time_values = df["Time"].to_numpy(dtype=float)
    alt_msl_values = df["Altitude MSL (ft)"].to_numpy(dtype=float)

    landing_idx = int(np.abs(time_values - landing_data_time).argmin())
    ground_elevation_msl_ft = float(alt_msl_values[landing_idx])
    exit_idx = int(np.abs(time_values - exit_time_abt).argmin())
    exit_alt_agl_ft = float(alt_msl_values[exit_idx] - ground_elevation_msl_ft)

    sync_offset = video_landing_time - landing_data_time

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        messagebox.showerror("Video Error", "Failed to read video metadata.", parent=root)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    max_clock = full_open_time_abt - exit_time_abt
    font = cv2.FONT_HERSHEY_SIMPLEX
    displayed_rod_ft_s = None
    last_rod_update_data_time = None

    frame_idx = 0
    with tqdm(total=total_frames, desc="Rendering Special Request ABT Overlay") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps
            data_time = video_time - sync_offset

            nearest_idx = int(np.abs(time_values - data_time).argmin())
            current_alt_msl_ft = float(alt_msl_values[nearest_idx])
            current_alt_agl_ft = current_alt_msl_ft - ground_elevation_msl_ft

            elapsed_since_exit = max(0.0, data_time - exit_time_abt)
            display_clock = min(elapsed_since_exit, max_clock)
            frozen_time_for_loss = min(max(data_time, exit_time_abt), full_open_time_abt)
            loss_idx = int(np.abs(time_values - frozen_time_for_loss).argmin())
            loss_alt_agl_ft = float(alt_msl_values[loss_idx] - ground_elevation_msl_ft)
            altitude_loss_ft = max(0.0, exit_alt_agl_ft - loss_alt_agl_ft)
            parachute_open = data_time >= full_open_time_abt

            current_rod_ft_s = None
            if parachute_open and len(time_values) > 1:
                should_update_rod = (
                    displayed_rod_ft_s is None
                    or last_rod_update_data_time is None
                    or (data_time - last_rod_update_data_time) >= 0.5
                )

                if should_update_rod:
                    window_start = data_time
                    window_end = data_time + 1.0
                    idx0 = int(np.searchsorted(time_values, window_start, side="left"))
                    idx1 = int(np.searchsorted(time_values, window_end, side="right")) - 1

                    if idx0 >= len(time_values):
                        idx0 = len(time_values) - 1
                    if idx1 < 0:
                        idx1 = 0

                    if idx1 <= idx0:
                        if nearest_idx <= 0:
                            idx0, idx1 = 0, 1
                        elif nearest_idx >= len(time_values) - 1:
                            idx0, idx1 = len(time_values) - 2, len(time_values) - 1
                        else:
                            idx0, idx1 = nearest_idx - 1, nearest_idx + 1

                    dt = float(time_values[idx1] - time_values[idx0])
                    if dt > 0:
                        d_alt = float(alt_msl_values[idx1] - alt_msl_values[idx0])
                        vertical_speed_ft_s = d_alt / dt
                        displayed_rod_ft_s = max(0.0, -vertical_speed_ft_s)
                    last_rod_update_data_time = data_time

                current_rod_ft_s = displayed_rod_ft_s
            else:
                displayed_rod_ft_s = None
                last_rod_update_data_time = None

            info_lines = [
                f"Altitude AGL: {current_alt_agl_ft:,.0f} ft",
                f"Time Since Exit: {_format_timer(display_clock)}",
                f"Altitude Loss: {altitude_loss_ft:,.0f} ft",
            ]

            if current_rod_ft_s is not None:
                info_lines.append(f"Current ROD: {current_rod_ft_s:.0f} ft/s")

            panel_x = 30
            panel_y = 30
            panel_w = 520
            panel_h = 65 + (len(info_lines) * 40)

            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            for i, text in enumerate(info_lines):
                cv2.putText(
                    frame,
                    text,
                    (panel_x + 20, panel_y + 40 + i * 40),
                    font,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    messagebox.showinfo("Complete", f"Overlay video saved:\n{output_path}", parent=root)


if __name__ == "__main__":
    run_special_request_abt_video()
