import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Ein-/Ausgabe
INPUT_PATH_OR_DIR = r"C:\Users\admin\source\repos\S3_DeformFDM_2\DataSet\TOOL_PATH"            # Datei ODER Ordner
GLOB_PATTERN      = "*.txt"            # Wenn Ordner:  welche Dateien?
INPUT_ORDER       = "x z y i k j"      # Reihenfolge der 6 Spalten IN DER DATEI

# Visualisierung
PLOT_AXES         = "x y z"            # Reihenfolge der Raumachsen im Plot (nur x/y/z)
COLOR_BY_FILE     = True               # True → pro Datei eigene Farbe/Legende

# Vektoren (A,B,C als Richtungspfeile anzeigen)
DRAW_VECTORS      = False
VECTOR_EVERY      = 5
VECTOR_NORMALIZE  = True
VECTOR_SCALE      = 2.0
ARROW_LENGTH_RATIO = 0.25

# Export
EXPORT_TRAJEKTORIE            = True
OUTPUT_TRAJEKTORIE_PATH       = "output_trajektorie/Kegel_v3.txt"
WRITE_A_B_C             = True           # A/B/C schreiben (als Euler in Grad)
DECIMALS_XYZ            = 5              # Nachkommastellen für X/Y/Z
DECIMALS_ABC            = 6              # Nachkommastellen für A/B/C
FLIP_A, FLIP_B, FLIP_C  = False, False, False
DROP_DUPLICATES         = True
DUP_TOL                 = 1e-9
Z_LIFT                  = 7.0           # Clearance-Aufschlag über max(Z) im Layer
LONG_MOVE_MM            = 10.0           # Kollisionsschutz innerhalb der Datei: ab dieser Länge mit Hub

# Setzt M101 (Extruder an --> Feed) vor Arbeitsfahrten und M103 (Extruder aus --> Eilfahrten)

VALID_NAMES = ["x","y","z","i","j","k"]

def _parse_order(spec, expected_len, allow_subset=False):
    """Parst/prüft Spalten- oder Achsenspezifikation"""
    if isinstance(spec, str):
        tokens = spec.replace(",", " ").lower().split()
    else:
        tokens = [str(s).lower() for s in spec]
    if len(tokens) != expected_len:
        raise ValueError(f"Erwarte {expected_len} Einträge, bekommen: {tokens}")
    if any(t not in VALID_NAMES for t in tokens):
        illegals = [t for t in tokens if t not in VALID_NAMES]
        raise ValueError(f"Ungültige Namen: {illegals} (erlaubt: {VALID_NAMES})")
    if not allow_subset:
        if set(tokens) != set(VALID_NAMES):
            raise ValueError(f"INPUT_ORDER muss eine Permutation von {VALID_NAMES} sein. Bekommen: {tokens}")
    else:
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"PLOT_AXES enthält doppelte Einträge: {tokens}")
    return tokens

def _natural_key(s):
    """Sortierschlüssel, der Zahlenfolgen numerisch berücksichtigt."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_xyzijk_file(filename, input_order="x y z i j k"):
    """Lädt einzelne Datei oder bündelt mehrere Dateien aus einem Ordner"""
    order = _parse_order(input_order, expected_len=6, allow_subset=False)
    raw = np.loadtxt(filename, dtype=float, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 6:
        raise ValueError(f"{filename}: erwarte mindestens 6 Spalten, gefunden: {raw.shape[1]}")
    name_to_idx = {name: i for i, name in enumerate(order)}
    std_cols = [name_to_idx[n] for n in VALID_NAMES]  # [x,y,z,a,b,c]
    data_std = raw[:, std_cols]
    data_dict = {n: data_std[:, i] for i, n in enumerate(VALID_NAMES)}
    return data_std, data_dict

def load_xyzijk_path(path, input_order="x y z i j k", glob_pattern="*.txt"):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, glob_pattern)), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"Keine Dateien in '{path}' für Muster '{glob_pattern}' gefunden.")
        arrays, segments = [], []
        start = 0
        for fp in files:
            arr, _ = load_xyzijk_file(fp, input_order)
            arrays.append(arr)
            end = start + arr.shape[0]
            segments.append((fp, slice(start, end)))
            start = end
        data_std = np.vstack(arrays)
        data_dict = {n: data_std[:, i] for i, n in enumerate(VALID_NAMES)}
        meta = {"files": files, "segments": segments, "counts": [sl.stop - sl.start for _, sl in segments]}
        return data_std, data_dict, meta
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pfad nicht gefunden: {path}")
        data_std, data_dict = load_xyzijk_file(path, input_order)
        meta = {"files": [path], "segments": [(path, slice(0, data_std.shape[0]))], "counts": [data_std.shape[0]]}
        return data_std, data_dict, meta

# Orientierung: IJK nach oben erzwingen + Euler(Z-Y-X) (Grad)
def _force_upward(ijk):
    """Dreht Vektoren, deren Z < 0 ist, um: alle zeigen nach +Z."""
    ijk = ijk.copy()
    mask = ijk[:, 2] < 0.0
    ijk[mask] *= -1.0
    return ijk


def euler_from_vector_deg(ijk):
    """
    IJK-Richtungen (N×3) → ZYX-Euler [A,B,C] in Grad.
    Roll-Achse wird so gewählt, dass die Projektion von Welt-Z auf die
    senkrechte Ebene die lokale X-Achse definiert (Fallback: Welt-X).
    """
    # 1) Normieren
    v = ijk / np.maximum(np.linalg.norm(ijk, axis=1, keepdims=True), 1e-12)

    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    ex = np.array([1.0, 0.0, 0.0], dtype=float)

    X_list, Y_list, Z_list = [], [], []
    for vi in v:
        xi = ez - np.dot(ez, vi) * vi
        n = np.linalg.norm(xi)
        if n < 1e-8:
            xi = ex - np.dot(ex, vi) * vi
            n = np.linalg.norm(xi)
        xi = xi / n

        yi = np.cross(vi, xi)
        X_list.append(xi)
        Y_list.append(yi)
        Z_list.append(vi)

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    Z = np.stack(Z_list)

    # 4) Rotationsmatrix
    R = np.stack([X, Y, Z], axis=2)  # (N,3,3)

    # 5) ZYX-Euler aus R extrahieren
    Zx = R[:, 0, 2]  # 3. Spalte von R → Tool-Z.x
    Zy = R[:, 1, 2]  # Tool-Z.y
    Zz = R[:, 2, 2]  # Tool-Z.z

    r = np.sqrt(Zx * Zx + Zz * Zz)

    A = np.degrees(np.arctan2(-Zy, r))
    B = np.degrees(np.arctan2(Zx, Zz))
    mask = r < 1e-8
    if np.any(mask):
        B = np.where(mask, 0.0, B)
    C = np.zeros_like(A)

    def _clean(a):
        a = a.copy()
        a[np.isclose(a, 0.0, atol=1e-12)] = 0.0
        return (a + 180.0) % 360.0 - 180.0

    A = _clean(A)
    B = _clean(B)
    C = _clean(C)

    return np.column_stack([A, B, C])

# Writer
def _fmt_axis(letter, value, dec):
    return f"{letter}{value:.{dec}f}"

def _apply_flip_abc_deg(p):
    """Optionales Vorzeichenflippen für A/B/C"""
    if FLIP_A: p[0] = -p[0]
    if FLIP_B: p[1] = -p[1]
    if FLIP_C: p[2] = -p[2]
    return p

def write_trajektorie_from_xyzijk(out_path, data_std, meta,
                            write_abc=True,
                            dec_xyz=5, dec_abc=6,
                            drop_duplicates=True, dup_tol=1e-9,
                            z_lift=20.0,
                            long_move_mm=2.0):
    """
    Schreibt eine Trajektorie mit sicheren An-/Abfahrten.
    - Sicherheitsfahrten auf Z_clear = max(Z im Layer) + z_lift
    - Lange Moves (> long_move_mm) werden oben verlagert
    - Arbeitsfahrten mit F=feed, Sicherheitsfahrten mit F=safety_feed
    """
    # Orientierung (IJK) -> alle nach oben → Euler Grad
    ijk_up = _force_upward(data_std[:, 3:6])
    euler_deg_all = euler_from_vector_deg(ijk_up)

    with open(out_path, "w", encoding="utf-8") as f:
        current_xyz = None
        prev_z_clear = None

        for seg_idx, (fp, sl) in enumerate(meta["segments"]):
            seg = data_std[sl]
            eul = euler_deg_all[sl]
            if seg.size == 0:
                continue

            # Feste Sicherheits-Höhe für diesen Layer:
            z_clear = float(np.max(seg[:, 2]) + z_lift)

            p0 = seg[0].copy()
            abc0 = _apply_flip_abc_deg(eul[0].copy())

            f.write(f"(Begin {os.path.basename(fp)})\n")

            # von vorherigem Layer weg: auf dessen Z_clear (falls vorhanden)
            if prev_z_clear is not None:
                f.write("M103\n")
                f.write(f"G01 {_fmt_axis('Z', prev_z_clear, dec_xyz)}\n")

            # Anfahrt: auf Z_clear dieses Layers, dann XY(+ABC) oben, dann absenken
            if current_xyz is None or not np.isclose(current_xyz[2], z_clear, atol=1e-12):
                f.write("M103\n")
                f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")

            words = ["G01", _fmt_axis("X", p0[0], dec_xyz), _fmt_axis("Y", p0[1], dec_xyz)]
            if write_abc:
                words += [_fmt_axis("A", abc0[0], dec_abc),
                          _fmt_axis("B", abc0[1], dec_abc),
                          _fmt_axis("C", abc0[2], dec_abc)]
            f.write("M103\n")
            f.write(" ".join(words) + "\n")
            f.write("M103\n")
            f.write(f"G01 {_fmt_axis('Z', p0[2], dec_xyz)}\n")

            current_xyz = p0.copy()
            prev_xyz = p0.copy()
            need_feed = True  # nächster Arbeitszug setzt F=feed

            # Pfad mit Kollisions-Schutz für lange Moves (> long_move_mm)
            for k in range(1, seg.shape[0]):
                p = seg[k].copy()
                if drop_duplicates and np.allclose(p, prev_xyz, atol=dup_tol, rtol=0):
                    continue

                abc = _apply_flip_abc_deg(eul[k].copy())
                move_len = float(np.linalg.norm(p - prev_xyz))

                if move_len > long_move_mm:
                    # Sicherheits-Sequenz IMMER auf z_clear
                    f.write("M103\n")
                    f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")
                    words = ["G01",
                             _fmt_axis("X", p[0], dec_xyz),
                             _fmt_axis("Y", p[1], dec_xyz)]
                    if write_abc:
                        words += [_fmt_axis("A", abc[0], dec_abc),
                                  _fmt_axis("B", abc[1], dec_abc),
                                  _fmt_axis("C", abc[2], dec_abc)]
                    f.write(" ".join(words) + "\n")
                    f.write(f"G01 {_fmt_axis('Z', p[2], dec_xyz)}\n")
                    need_feed = True
                else:
                    # normaler Arbeitszug
                    words = ["G01",
                             _fmt_axis("X", p[0], dec_xyz),
                             _fmt_axis("Y", p[1], dec_xyz),
                             _fmt_axis("Z", p[2], dec_xyz)]
                    if write_abc:
                        words += [_fmt_axis("A", abc[0], dec_abc),
                                  _fmt_axis("B", abc[1], dec_abc),
                                  _fmt_axis("C", abc[2], dec_abc)]
                    if need_feed:
                        f.write("M101\n")
                        need_feed = False
                    f.write(" ".join(words) + "\n")

                prev_xyz = p
                current_xyz = p

            # Layer-Ende: hoch auf festes z_clear
            f.write("M103\n")
            f.write(f"(Clearance after {os.path.basename(fp)})\n")
            f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")
            current_xyz[2] = z_clear
            prev_z_clear = z_clear

            f.write(f"(End {os.path.basename(fp)})\n\n")


# Daten laden
data_std, data, meta = load_xyzijk_path(INPUT_PATH_OR_DIR, INPUT_ORDER, GLOB_PATTERN)

# Visualisierung
axes = _parse_order(PLOT_AXES, expected_len=3, allow_subset=True)
idx_map = {n: i for i, n in enumerate(VALID_NAMES)}
i0, i1, i2 = (idx_map[axes[0]], idx_map[axes[1]], idx_map[axes[2]])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
if COLOR_BY_FILE and len(meta["files"]) > 1:
    for layer_idx, (fp, sl) in enumerate(meta["segments"]):
        ax.plot(
            data_std[sl, i0], data_std[sl, i1], data_std[sl, i2],
            marker=".", linewidth=0.3, markersize=0.8, alpha=0.8
        )
else:
    ax.plot(
        data_std[:, i0], data_std[:, i1], data_std[:, i2],
        marker=".", linewidth=0.3, markersize=0.8, alpha=0.8
    )

# Pfeile aus IJK zeichnen
if DRAW_VECTORS and set(axes).issubset({"x","y","z"}):
    idx = np.arange(0, data_std.shape[0], max(1, int(VECTOR_EVERY)))
    ijk_all_up = _force_upward(data_std[:, 3:6])
    X = data_std[idx, i0]; Y = data_std[idx, i1]; Z = data_std[idx, i2]
    U = ijk_all_up[idx, 0]; Vv = ijk_all_up[idx, 1]; W = ijk_all_up[idx, 2]
    norms = np.linalg.norm(np.column_stack([U, Vv, W]), axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    U, Vv, W = U/norms[:,0], Vv/norms[:,0], W/norms[:,0]
    if VECTOR_NORMALIZE:
        ax.quiver(X, Y, Z, U, Vv, W, length=VECTOR_SCALE, normalize=True,
                  arrow_length_ratio=ARROW_LENGTH_RATIO, linewidths=0.5)
    else:
        ax.quiver(X, Y, Z, VECTOR_SCALE*U, VECTOR_SCALE*Vv, VECTOR_SCALE*W,
                  length=1.0, normalize=False,
                  arrow_length_ratio=ARROW_LENGTH_RATIO, linewidths=0.5)

ax.set_xlabel(f"{axes[0].upper()} in mm");
ax.set_ylabel(f"{axes[1].upper()} in mm");
ax.set_zlabel(f"{axes[2].upper()} in mm")
ax.set_box_aspect((
    np.ptp(data_std[:, i0]) if np.ptp(data_std[:, i0]) > 0 else 1.0,
    np.ptp(data_std[:, i1]) if np.ptp(data_std[:, i1]) > 0 else 1.0,
    np.ptp(data_std[:, i2]) if np.ptp(data_std[:, i2]) > 0 else 1.0
))
plt.title("Visualisierung Trajektorie")
plt.tight_layout()
plt.show()

# Trajektorie exportieren
if EXPORT_TRAJEKTORIE:
    write_trajektorie_from_xyzijk(OUTPUT_TRAJEKTORIE_PATH, data_std, meta, write_abc=WRITE_A_B_C,
                            dec_xyz=DECIMALS_XYZ, dec_abc=DECIMALS_ABC, drop_duplicates=DROP_DUPLICATES,
                            dup_tol=DUP_TOL, z_lift=Z_LIFT, long_move_mm=LONG_MOVE_MM)
    print(f"Trajektorie geschrieben nach: {OUTPUT_TRAJEKTORIE_PATH}")
