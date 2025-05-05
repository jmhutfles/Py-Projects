import ezdxf
import tkinter as tk
from tkinter import filedialog

def extract_spline_points(dxf_file, num_segments=100):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    spline_points = []

    for spline in msp.query("SPLINE"):
        points = list(spline.approximate(segments=num_segments))
        spline_points.append(points)

    return spline_points

def main():
    # Set up the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    dxf_path = filedialog.askopenfilename(
        title="Select DXF File",
        filetypes=[("DXF files", "*.dxf")]
    )

    if not dxf_path:
        print("No file selected.")
        return

    points_list = extract_spline_points(dxf_path, num_segments=50)

    # Display the results
    for idx, points in enumerate(points_list):
        print(f"Spline {idx + 1}:")
        for pt in points:
            print(f"{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}")

if __name__ == "__main__":
    main()