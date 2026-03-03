import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, simpledialog, messagebox
from tqdm import tqdm

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

    exit_alt_agl = _prompt_float(root, "Exit Altitude", "Enter exit altitude AGL (ft):", min_value=0)
    if exit_alt_agl is None:
        return

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

    video_exit_time = _prompt_float(
        root,
        "Video Exit Time",
        "Enter exit time in VIDEO (s).\nUse 0 if video starts at exit.",
        min_value=0,
    )
    if video_exit_time is None:
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

    time_values = df["Time"].to_numpy(dtype=float)
    alt_msl_values = df["Altitude MSL (ft)"].to_numpy(dtype=float)

    exit_idx = int(np.abs(time_values - exit_time_abt).argmin())
    exit_alt_msl_ft = float(alt_msl_values[exit_idx])

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

    frame_idx = 0
    with tqdm(total=total_frames, desc="Rendering Special Request ABT Overlay") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps
            data_time = exit_time_abt + (video_time - video_exit_time)

            nearest_idx = int(np.abs(time_values - data_time).argmin())
            current_alt_msl_ft = float(alt_msl_values[nearest_idx])
            current_alt_agl_ft = exit_alt_agl + (current_alt_msl_ft - exit_alt_msl_ft)

            elapsed_since_exit = max(0.0, data_time - exit_time_abt)
            display_clock = min(elapsed_since_exit, max_clock)

            info_lines = [
                f"Altitude AGL: {current_alt_agl_ft:,.0f} ft",
                f"Time Since Exit: {_format_timer(display_clock)}",
            ]

            panel_x = 30
            panel_y = 30
            panel_w = 520
            panel_h = 105

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
