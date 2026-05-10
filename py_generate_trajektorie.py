import os
import glob
import re
import shutil
import filecmp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Ein-/Ausgabe
INPUT_PATH_OR_DIR = r"C:\Users\admin\source\repos\S3_DeformFDM_2\DataSet\TOOL_PATH"            # Datei ODER Ordner
INPUT_FALLBACK_DIR = "input_path"       # Fallback-Ordner (relativ zum Skript)
GLOB_PATTERN      = "*.txt"            # Wenn Ordner:  welche Dateien?
INPUT_ORDER       = "x z y i k j"      # Reihenfolge der 6 Spalten IN DER DATEI

# Visualisierung
PLOT_AXES         = "x y z"            # Reihenfolge der Raumachsen im Plot (nur x/y/z)
COLOR_BY_FILE     = True               # True → pro Datei eigene Farbe/Legende
PLOT_SLIDER       = True               # True → mit Slider (Dateiweise einblenden)

# Vektoren (A,B,C als Richtungspfeile anzeigen)
DRAW_VECTORS      = False
VECTOR_EVERY      = 5
VECTOR_NORMALIZE  = True
VECTOR_SCALE      = 2.0
ARROW_LENGTH_RATIO = 0.25

# Export
EXPORT_TRAJEKTORIE            = True
OUTPUT_TRAJEKTORIE_PATH       = "output_trajektorie/Kegel_v6_5x2.txt"
WRITE_A_B_C             = True           # A/B/C schreiben (als Euler in Grad)
DECIMALS_XYZ            = 5              # Nachkommastellen für X/Y/Z
DECIMALS_ABC            = 6              # Nachkommastellen für A/B/C
FLIP_A, FLIP_B, FLIP_C  = False, False, False
DROP_DUPLICATES         = True
DUP_TOL                 = 1e-9
Z_LIFT                  = 20.0           # Clearance-Aufschlag über max(Z) im Layer
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

def _backup_existing_tool_path_data(target_dir, glob_pattern="*.txt"):
    """Verschiebt vorhandene Tool-Path-Dateien in einen eigenen Zeitstempel-Ordner."""
    existing_files = sorted(glob.glob(os.path.join(target_dir, glob_pattern)), key=_natural_key)
    if not existing_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(target_dir, f"backup_tool_path_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    for fp in existing_files:
        shutil.move(fp, os.path.join(backup_dir, os.path.basename(fp)))

    return backup_dir

def _list_source_path_files(path, glob_pattern="*.txt"):
    """Liefert Quelldateien, die in den Fallback kopiert werden würden."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, glob_pattern)), key=_natural_key)
    return [path]

def _target_needs_update(source_files, target_dir, glob_pattern="*.txt"):
    """Prüft, ob sich Ziel-Dateien in Namen oder Inhalt von den Quell-Dateien unterscheiden."""
    target_files = sorted(glob.glob(os.path.join(target_dir, glob_pattern)), key=_natural_key)
    target_by_name = {os.path.basename(fp): fp for fp in target_files}

    source_names = [os.path.basename(fp) for fp in source_files]
    target_names = [os.path.basename(fp) for fp in target_files]
    if source_names != target_names:
        return True

    for src in source_files:
        dst = target_by_name.get(os.path.basename(src))
        if dst is None:
            return True
        if not filecmp.cmp(src, dst, shallow=False):
            return True

    return False

def _output_named_input_subdir(base_dir, output_path):
    """Erstellt den Ziel-Unterordner auf Basis des Output-Dateinamens."""
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    if not output_name:
        output_name = "default"
    return os.path.join(base_dir, output_name)

def load_xyzijk_path(path, input_order="x y z i j k", glob_pattern="*.txt", output_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_base_dir = os.path.join(script_dir, INPUT_FALLBACK_DIR)
    fallback_dir = fallback_base_dir
    if output_path:
        fallback_dir = _output_named_input_subdir(fallback_base_dir, output_path)

    if os.path.exists(path):
        # Primärpfad existiert: Dateien laden und in Fallback-Ordner kopieren
        use_path = path
        os.makedirs(fallback_dir, exist_ok=True)
        same_location = os.path.abspath(path) == os.path.abspath(fallback_dir)

        if not same_location:
            source_files = _list_source_path_files(path, glob_pattern)
            needs_update = _target_needs_update(source_files, fallback_dir, glob_pattern)

            if needs_update:
                backup_dir = _backup_existing_tool_path_data(fallback_dir, glob_pattern)
                if backup_dir:
                    print(f"Vorherige Tool-Path-Daten nach '{backup_dir}' verschoben.")

                for fp in source_files:
                    shutil.copy2(fp, os.path.join(fallback_dir, os.path.basename(fp)))
                print(f"Dateien von '{path}' nach '{fallback_dir}' kopiert.")
            else:
                print(f"Tool-Path-Dateien unverändert - kein Backup und kein Kopieren nach '{fallback_dir}'.")
        else:
            print(f"Primärpfad und Fallback sind identisch ('{fallback_dir}') - kein Kopieren erforderlich.")
    else:
        # Primärpfad nicht gefunden: Fallback verwenden
        if not os.path.exists(fallback_dir):
            raise FileNotFoundError(
                f"Primärpfad nicht gefunden: '{path}'\n"
                f"Fallback-Ordner existiert ebenfalls nicht: '{fallback_dir}'"
            )
        print(f"Primärpfad '{path}' nicht gefunden – verwende Fallback '{fallback_dir}'.")
        use_path = fallback_dir

    if os.path.isdir(use_path):
        files = sorted(glob.glob(os.path.join(use_path, glob_pattern)), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"Keine Dateien in '{use_path}' für Muster '{glob_pattern}' gefunden.")
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
        data_std, data_dict = load_xyzijk_file(use_path, input_order)
        meta = {"files": [use_path], "segments": [(use_path, slice(0, data_std.shape[0]))], "counts": [data_std.shape[0]]}
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
    - Sicherheitsfahrten auf Z_clear = max(Z aller Punkte) + z_lift
    - Lange Moves (> long_move_mm) werden über Z_clear umgeleitet, um Kollisionen zu vermeiden.

    Parameter:
        out_path       : Ausgabepfad der Textdatei
        data_std       : (N×6) Array [X,Y,Z,I,J,K] aller Punkte
        meta           : Dict mit 'segments' (Liste von (Dateiname, slice))
        write_abc      : A/B/C-Winkel (Euler) in jede Zeile schreiben
        dec_xyz        : Nachkommastellen für X/Y/Z
        dec_abc        : Nachkommastellen für A/B/C
        drop_duplicates: Aufeinanderfolgende identische Punkte überspringen
        dup_tol        : Toleranz für Duplikat-Erkennung (in mm)
        z_lift         : Sicherheits-Aufschlag über dem höchsten Z-Punkt
        long_move_mm   : Schwelle für Kollisionsschutz-Umfahrung (in mm)
    """
    # IJK-Vektoren: alle in +Z-Richtung drehen (Werkzeug zeigt immer nach oben)
    ijk_up = _force_upward(data_std[:, 3:6])
    # Euler-Winkel (A/B/C in Grad) für alle Punkte berechnen
    euler_deg_all = euler_from_vector_deg(ijk_up)

    # Einmalige globale Sicherheitshöhe: höchster Z aller Segmente + Aufschlag
    # Alle Verfahrwege zwischen Segmenten laufen auf dieser Höhe ab
    z_clear = float(np.max(data_std[:, 2]) + z_lift)

    with open(out_path, "w", encoding="utf-8") as f:
        current_xyz = None   # letzte bekannte Maschinenposition (XYZ)
        prev_z_clear = None  # Z_clear des vorangegangenen Segments (für Abfahrt)

        for seg_idx, (fp, sl) in enumerate(meta["segments"]):
            seg = data_std[sl]          # XYZ+IJK-Punkte dieses Segments
            eul = euler_deg_all[sl]     # zugehörige Euler-Winkel
            if seg.size == 0:
                continue

            # Ersten Punkt und seine Orientierung lesen
            p0 = seg[0].copy()
            abc0 = _apply_flip_abc_deg(eul[0].copy())

            # Segment-Kommentar in die Ausgabe schreiben (Dateiname als Label)
            f.write(f"(Begin {os.path.basename(fp)})\n")
            f.write("M103\n")   # Extruder aus (Eilfahrt)
            
            # --- Abfahrt vom vorherigen Segment ---
            # Nach dem letzten Segment wurde bereits auf z_clear gefahren;
            # dieser Block stellt sicher, dass die Achse dort bleibt.
            if prev_z_clear is not None:
                f.write(f"G01 {_fmt_axis('Z', prev_z_clear, dec_xyz)}\n")

            # --- Anfahrt auf Sicherheitshöhe ---
            # Nur schreiben, wenn die Achse noch nicht auf z_clear steht
            if current_xyz is None or not np.isclose(current_xyz[2], z_clear, atol=1e-12):
                f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")

            # --- XY-Positionierung auf Sicherheitshöhe (+ optionale ABC-Orientierung) ---
            # Werkzeug wird zuerst in XY zum Startpunkt gefahren, bevor es absenkt
            words = ["G01", _fmt_axis("X", p0[0], dec_xyz), _fmt_axis("Y", p0[1], dec_xyz)]
            if write_abc:
                words += [_fmt_axis("A", abc0[0], dec_abc),
                          _fmt_axis("B", abc0[1], dec_abc),
                          _fmt_axis("C", abc0[2], dec_abc)]
            f.write(" ".join(words) + "\n")
            f.write(f"G01 {_fmt_axis('Z', p0[2], dec_xyz)}\n")  # auf Arbeits-Z absenken

            current_xyz = p0.copy()
            prev_xyz = p0.copy()
            need_feed = True  # beim nächsten Arbeitszug M101 (Extruder an) setzen

            # --- Pfad abfahren: Punkt für Punkt ---
            for k in range(1, seg.shape[0]):
                p = seg[k].copy()

                # Doppelten Punkt überspringen (identische Position zum Vorgänger)
                if drop_duplicates and np.allclose(p, prev_xyz, atol=dup_tol, rtol=0):
                    continue

                abc = _apply_flip_abc_deg(eul[k].copy())
                move_len = float(np.linalg.norm(p - prev_xyz))  # Distanz zum Vorpunkt

                if move_len > long_move_mm:
                    # --- Langer Sprung: Kollisionsschutz-Umfahrung über z_clear ---
                    # Ablauf: Z hoch → XY(+ABC) fahren → Z auf Zieltiefe absenken
                    f.write("M103\n")   # Extruder aus
                    f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")   # Z hochfahren
                    words = ["G01",
                             _fmt_axis("X", p[0], dec_xyz),
                             _fmt_axis("Y", p[1], dec_xyz)]
                    if write_abc:
                        words += [_fmt_axis("A", abc[0], dec_abc),
                                  _fmt_axis("B", abc[1], dec_abc),
                                  _fmt_axis("C", abc[2], dec_abc)]
                    f.write(" ".join(words) + "\n")                         # XY positionieren
                    f.write(f"G01 {_fmt_axis('Z', p[2], dec_xyz)}\n")      # Z absenken
                    need_feed = True  # nach Umfahrung wieder Extruder einschalten
                else:
                    # --- Normaler Arbeitszug: direkter Move mit X/Y/Z (+ABC) ---
                    words = ["G01",
                             _fmt_axis("X", p[0], dec_xyz),
                             _fmt_axis("Y", p[1], dec_xyz),
                             _fmt_axis("Z", p[2], dec_xyz)]
                    if write_abc:
                        words += [_fmt_axis("A", abc[0], dec_abc),
                                  _fmt_axis("B", abc[1], dec_abc),
                                  _fmt_axis("C", abc[2], dec_abc)]
                    if need_feed:
                        f.write("M101\n")   # Extruder an (Feed-Modus)
                        need_feed = False
                    f.write(" ".join(words) + "\n")

                prev_xyz = p
                current_xyz = p

            # --- Segment-Ende: Werkzeug auf Sicherheitshöhe fahren ---
            f.write("M103\n")   # Extruder aus
            f.write(f"(Clearance after {os.path.basename(fp)})\n")
            f.write(f"G01 {_fmt_axis('Z', z_clear, dec_xyz)}\n")
            current_xyz[2] = z_clear
            prev_z_clear = z_clear  # merken für Abfahrt beim nächsten Segment

            f.write(f"(End {os.path.basename(fp)})\n\n")


def visualize_trajektorie(data_std, meta):
    axes = _parse_order(PLOT_AXES, expected_len=3, allow_subset=True)
    idx_map = {n: i for i, n in enumerate(VALID_NAMES)}
    i0, i1, i2 = (idx_map[axes[0]], idx_map[axes[1]], idx_map[axes[2]])

    fig = plt.figure(figsize=(10, 11))
    ax = fig.add_subplot(111, projection='3d')
    if COLOR_BY_FILE and len(meta["files"]) > 1:
        if PLOT_SLIDER:
            lines = []
            for layer_idx, (fp, sl) in enumerate(meta["segments"]):
                ln, = ax.plot(
                    data_std[sl, i0], data_std[sl, i1], data_std[sl, i2],
                    marker=".", linewidth=1.0, markersize=2.0, alpha=0.8, visible=False
                )
                lines.append(ln)

            # Anfangszustand: erste Datei sichtbar
            if lines:
                lines[0].set_visible(True)

            # State für Vektoren
            state = {'show_vectors': False, 'quiver_objs': [], 'ijk_all_up': None}
            if set(axes).issubset({"x","y","z"}):
                state['ijk_all_up'] = _force_upward(data_std[:, 3:6])

            slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
            file_slider = Slider(slider_ax, 'Datei', 1, len(lines), valinit=1, valstep=1)

            button_ax = fig.add_axes([0.82, 0.08, 0.15, 0.03])
            vectors_button = Button(button_ax, 'Vektoren AN')

            def slider_update(val):
                idx = int(val) - 1
                for j, ln in enumerate(lines):
                    ln.set_visible(j <= idx)

                # Vektoren aktualisieren wenn aktiviert
                if state['show_vectors'] and set(axes).issubset({"x","y","z"}):
                    # Alte Vektoren löschen
                    for qobj in state['quiver_objs']:
                        qobj.remove()
                    state['quiver_objs'].clear()

                    # Zeichne Vektoren pro Schicht mit entsprechender Farbe
                    for j in range(idx + 1):
                        sl = meta["segments"][j][1]
                        indices = np.arange(sl.start, sl.stop)
                        idx_vec = indices[::max(1, int(VECTOR_EVERY))]

                        if len(idx_vec) == 0:
                            continue

                        X = data_std[idx_vec, i0]
                        Y = data_std[idx_vec, i1]
                        Z = data_std[idx_vec, i2]
                        U = state['ijk_all_up'][idx_vec, 0]
                        Vv = state['ijk_all_up'][idx_vec, 1]
                        W = state['ijk_all_up'][idx_vec, 2]

                        norms = np.linalg.norm(np.column_stack([U, Vv, W]), axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1.0, norms)
                        U, Vv, W = U/norms[:,0], Vv/norms[:,0], W/norms[:,0]

                        color = lines[j].get_color()
                        if VECTOR_NORMALIZE:
                            qobj = ax.quiver(X, Y, Z, U, Vv, W, length=VECTOR_SCALE*1.5, normalize=True,
                                      arrow_length_ratio=ARROW_LENGTH_RATIO, linewidths=1.0, color=color)
                        else:
                            qobj = ax.quiver(X, Y, Z, VECTOR_SCALE*1.5*U, VECTOR_SCALE*1.5*Vv, VECTOR_SCALE*1.5*W,
                                      length=1.0, normalize=False,
                                      arrow_length_ratio=ARROW_LENGTH_RATIO, linewidths=1.0, color=color)
                        state['quiver_objs'].append(qobj)
                elif not state['show_vectors']:
                    # Alle Vektoren entfernen
                    for qobj in state['quiver_objs']:
                        qobj.remove()
                    state['quiver_objs'].clear()

                ax.set_title(f"Visualisierung Trajektorie ({idx+1}/{len(lines)})")
                fig.canvas.draw_idle()

            def toggle_vectors(event):
                state['show_vectors'] = not state['show_vectors']
                vectors_button.label.set_text('Vektoren AUS' if state['show_vectors'] else 'Vektoren AN')
                slider_update(file_slider.val)

            vectors_button.on_clicked(toggle_vectors)
            file_slider.on_changed(slider_update)

        else:
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

    ax.set_xlabel(f"{axes[0].upper()} in mm")
    ax.set_ylabel(f"{axes[1].upper()} in mm")
    ax.set_zlabel(f"{axes[2].upper()} in mm")
    ax.set_box_aspect((
        np.ptp(data_std[:, i0]) if np.ptp(data_std[:, i0]) > 0 else 1.0,
        np.ptp(data_std[:, i1]) if np.ptp(data_std[:, i1]) > 0 else 1.0,
        np.ptp(data_std[:, i2]) if np.ptp(data_std[:, i2]) > 0 else 1.0
    ))
    plt.title("Visualisierung Trajektorie")
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close('all')


# Daten laden
data_std, data, meta = load_xyzijk_path(
    INPUT_PATH_OR_DIR,
    INPUT_ORDER,
    GLOB_PATTERN,
    output_path=OUTPUT_TRAJEKTORIE_PATH,
)

# Visualisierung
visualize_trajektorie(data_std, meta)

# Trajektorie exportieren
if EXPORT_TRAJEKTORIE:
    write_trajektorie_from_xyzijk(OUTPUT_TRAJEKTORIE_PATH, data_std, meta, write_abc=WRITE_A_B_C,
                            dec_xyz=DECIMALS_XYZ, dec_abc=DECIMALS_ABC, drop_duplicates=DROP_DUPLICATES,
                            dup_tol=DUP_TOL, z_lift=Z_LIFT, long_move_mm=LONG_MOVE_MM)
    print(f"Trajektorie geschrieben nach: {OUTPUT_TRAJEKTORIE_PATH}")
