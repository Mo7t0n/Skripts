import numpy as np
import csv
import os
from scipy.spatial.transform import Rotation as R


def compute_bed_offset_from_tool_and_platform(
        tool_x, tool_y, tool_z, tool_a, tool_b, tool_c,
        platform_x, platform_y, platform_z, platform_a, platform_b, platform_c
):
    epsilon = 1e-5  # Werte kleiner als das werden als 0 behandelt

    # Düsenrotation und Position
    R_tool = R.from_euler('ZYX', [tool_c, tool_b, tool_a], degrees=True).as_matrix()
    tool_tip = np.array([tool_x, tool_y, tool_z])

    # Plattformmatrix
    R_platform = R.from_euler('ZYX', [platform_c, platform_b, platform_a], degrees=True).as_matrix()
    T_platform = np.eye(4)
    T_platform[:3, :3] = R_platform
    T_platform[:3, 3] = [platform_x, platform_y, platform_z]

    # Düsenmatrix (invers der Plattform)
    T_nozzle = np.linalg.inv(T_platform)
    nozzle_pos = T_nozzle[:3, 3]
    R_nozzle = T_nozzle[:3, :3]

    # Löse (R - I) * offset = new_nozzle_pos - tool_tip
    A = R_tool - np.eye(3)

    # Werte nahe 0 setzen
    A[np.abs(A) < epsilon] = 0.0

    b = nozzle_pos - tool_tip

    try:
        bed_offset_xyz, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return bed_offset_xyz
    except np.linalg.LinAlgError:
        return None

def parse_float(value):
    try:
        return float(value.replace(",", "."))
    except:
        return 0.0

offsets = []

# CSV-Datei einlesen
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "BedOffset_2.CSV")

with open(csv_file, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    for row in reader:
        # Eingabe: Düsenpose (commanded)
        tool_x = parse_float(row["soll_X"])
        tool_y = parse_float(row["soll_Y"])
        tool_z = parse_float(row["soll_Z"])
        tool_a = parse_float(row["soll_A"])
        tool_b = parse_float(row["soll_B"])
        tool_c = parse_float(row["soll_C"])

        # Gemessene Plattformpose (real)
        platform_x = parse_float(row["korr_X"])
        platform_y = parse_float(row["korr_Y"])
        platform_z = parse_float(row["korr_Z"])
        platform_a = parse_float(row["korr_A"])
        platform_b = parse_float(row["korr_B"])
        platform_c = parse_float(row["korr_C"])
        print(
            tool_x, tool_y, tool_z, tool_a, tool_b, tool_c,
            platform_x, platform_y, platform_z, platform_a, platform_b, platform_c
        )
        offset = compute_bed_offset_from_tool_and_platform(
            tool_x, tool_y, tool_z, tool_a, tool_b, tool_c,
            platform_x, platform_y, platform_z, platform_a, platform_b, platform_c
        )

        if offset is not None:
            offsets.append(offset)

# Mittelwert berechnen
if offsets:
    offsets_np = np.array(offsets)
    positive_mask = offsets_np != 0
    filtered_offsets = np.where(positive_mask, offsets_np, np.nan)
    mean_offset = np.nanmean(filtered_offsets, axis=0)
    print("Gemittelter bed_offset_xyz:", mean_offset)
else:
    print("Keine gültigen Offsets berechnet.")
