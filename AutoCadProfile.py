import ezdxf
import tkinter as tk
from tkinter import filedialog

def extract_spline_points(dxf_file, num_segments=100):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    spline_points = []

    for spline in msp.query("SPLINE"):
        points = list(spline.flattening(distance=0.01))
        spline_points.append(points)

    return spline_points

def main():
    # Set up file dialog
    root = tk.Tk()
    root.withdraw()

    # Open DXF file
    dxf_path = filedialog.askopenfilename(
        title="Select DXF File",
        filetypes=[("DXF files", "*.dxf")]
    )
    if not dxf_path:
        print("No DXF file selected.")
        return

    # Extract points from splines
    points_list = extract_spline_points(dxf_path)

    # Prompt to save .gnu file
    save_path = filedialog.asksaveasfilename(
        title="Save GNUPLOT File",
        defaultextension=".gnu",
        filetypes=[("Gnuplot files", "*.gnu"), ("All files", "*.*")]
    )
    if not save_path:
        print("Save cancelled.")
        return

    try:
        with open(save_path, 'w') as f:
            for i, points in enumerate(points_list):
                for x, y, _ in points:
                    f.write(f"{x} {y}\n")
                if i < len(points_list) - 1:
                    f.write("\n")  # Only add newline if not the last spline

            print(f"Data saved to {save_path}")
    except Exception as e:
        print(f"Failed to write file: {e}")

if __name__ == "__main__":
    main()
