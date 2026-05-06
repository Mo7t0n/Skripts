"""
STL -> Tetraedergitter (.tet) + Auswahl-Datei (.txt)
- .tet:   "N vertices" / "E tets" + Punkte + "4 i j k l"
- .txt:   "<NodeIndex1b>:<fixed>:<handle>:"
"""

from __future__ import annotations
from pathlib import Path
import sys, numpy as np
import trimesh
import tetgen

# Ein-/Ausgabe
INPUT_PATH = r"input_stl/Kegel_v5.stl"  # Datei (STL)
OUTPUT_PATH = r"output_tet/Kegel_v5.tet"  # Ausgabedatei (.tet); wenn leer -> <input>.tet
Y_UP = True  # True: (x,y,z)->(x,z,y), Up-Achse = Y
ANGLE_DEG = 20.0  # handle=1 wenn Fläche flacher als ANGLE (zur Up-Achse)
FIX_ON_HANDLE = True  # fixed = 1, wenn handle = 1


def load_and_repair_stl(path: Path) -> "trimesh.Trimesh":
    """
    STL laden, reparieren und ggf. Löcher schließen
    :param path: Pfad zur STL-Datei
    :return: repariertes trimesh.Trimesh-Objekt
    """

    mesh = trimesh.load_mesh(path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"'{path}' ist kein gültiges Dreiecksnetz.")
    mesh.remove_unreferenced_vertices()
    unique, _ = trimesh.grouping.unique_rows(mesh.faces)
    mesh.update_faces(unique)
    mesh.update_faces(mesh.nondegenerate_faces())
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        print("Warnung: Netz ist nicht geschlossen.", file=sys.stderr)
    return mesh


def to_y_up(nodes: np.ndarray) -> np.ndarray:
    """
    Koordinaten von Z-up nach Y-up umwandeln
    :param nodes: Punktkoordinaten (N x 3)
    :return: transformierte Koordinaten (x, z, y)
    """
    nodes = np.asarray(nodes, float)
    if nodes.shape[1] < 3: raise ValueError("Erwarte 3D-Koordinaten (x,y,z).")
    return nodes[:, [0, 2, 1]]  # (x, z, y)


def tetrahedralize(verts: np.ndarray, faces: np.ndarray,
                   quality: float = 1.2, max_volume: float | None = None):
    """
    Erzeugt ein Tetraedergitter mit TetGen
    :param verts: Knotenkoordinaten (N x 3)
    :param faces: Dreiecksflächen (M x 3, Indizes auf verts)
    :param quality: Qualitätsfaktor (TetGen-Option q)
    :param max_volume: maximale Tetraeder-Volumenvorgabe (TetGen-Option a)
    :return: (nodes, tets) als NumPy-Arrays
    """
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    t = tetgen.TetGen(verts, faces)

    opts: dict = {"plc": True, "quality": True, "minratio": quality}
    if max_volume and max_volume > 0:
        opts["fixedvolume"] = True
        opts["maxvolume"] = max_volume

    t.tetrahedralize(**opts)

    nodes = np.asarray(t.node)
    tets = np.asarray(t.elem)
    return nodes, tets


# Handle/Fixed

def compute_surface_handle_flags(nodes: np.ndarray, tets: np.ndarray,
                                 angle_deg: float, up_axis: str, exclude_bottom: bool = True) -> np.ndarray:
    """
    Markiert Knoten, die zu flachen Oberflächen gehören (Handles)
    :param nodes: Punktkoordinaten (N x 3)
    :param tets: Tetraeder-Konnektivität (M x 4)
    :param angle_deg: maximaler Neigungswinkel zur Up-Achse
    :param up_axis: 'x', 'y' oder 'z' (Up-Achse)
    :param exclude_bottom: untere Fläche ausschließen
    :return: Array (N,) mit Flags 0/1 für handle
    """
    nodes = np.asarray(nodes, float)
    N = int(nodes.shape[0])

    tet = np.asarray(tets, int)
    if tet.ndim != 2 or tet.shape[1] < 4: raise ValueError("Tet-Form ungültig.")
    tet = tet[:, :4]
    mn, mx = tet.min(), tet.max()
    if mn >= 1 and mx <= N:
        tet = tet - 1
    elif not (mn >= 0 and mx < N):
        raise ValueError("Tet-Indizes außerhalb [0..N-1] / [1..N].")

    f0, f1, f2, f3 = tet[:, [1, 2, 3]], tet[:, [0, 3, 2]], tet[:, [0, 1, 3]], tet[:, [0, 2, 1]]
    faces_all = np.vstack((f0, f1, f2, f3))
    faces_sorted = np.sort(faces_all, axis=1)
    _, inv, counts = np.unique(faces_sorted, axis=0, return_inverse=True, return_counts=True)
    boundary = faces_all[counts[inv] == 1]
    if boundary.size == 0: return np.zeros(N, np.int8)

    v0, v1, v2 = nodes[boundary[:, 0]], nodes[boundary[:, 1]], nodes[boundary[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(n, axis=1)
    good = area2 > (1e-12 * max(1.0, float(area2.max())))
    if not np.any(good): return np.zeros(N, np.int8)
    boundary, n = boundary[good], n[good]

    axis = {"x": 0, "y": 1, "z": 2}[up_axis.lower()]
    n_up = np.abs(n[:, axis]) / (np.linalg.norm(n, axis=1) + 1e-15)
    slope = np.degrees(np.arccos(np.clip(n_up, -1.0, 1.0)))  # 0° flach, 90° vertikal
    flat = slope < float(angle_deg)  # "flacher als ANGLE"

    if exclude_bottom and boundary.size:
        y = nodes[:, axis]
        ymin, ymax = float(y.min()), float(y.max())
        tol = max(1e-8, (ymax - ymin) * 0.01)  # 1% Höhe
        on_bottom = (y[boundary[:, 0]] <= ymin + tol) & (y[boundary[:, 1]] <= ymin + tol) & (
                    y[boundary[:, 2]] <= ymin + tol)
        flat &= ~on_bottom

    handle = np.zeros(N, np.int8)
    if np.any(flat):
        handle[np.unique(boundary[flat].ravel())] = 1
    return handle


# Writer

def write_tet(path: Path, nodes: np.ndarray, tets: np.ndarray, decimals: int = 6):
    """
    Schreibt ein Tetraedergitter im .tet-Format
    :param path: Ausgabepfad (.tet)
    :param nodes: Punktkoordinaten
    :param tets: Tetraeder-Konnektivität
    :param decimals: Nachkommastellen für Ausgabe
    :return: Pfad zur geschriebenen Datei
    """
    path = Path(path)
    if path.suffix.lower() != ".tet": path = path.with_suffix(".tet")
    N, E = int(nodes.shape[0]), int(tets.shape[0])

    def ffix(x: float) -> str:
        s = f"{float(x):.{decimals}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    # ggf. auf 0-basig normalisieren
    tet = np.asarray(tets, int)
    mn, mx = tet.min(), tet.max()
    if mn >= 1 and mx <= N:
        tet = tet - 1
    elif not (mn >= 0 and mx < N):
        raise ValueError(f"Ungültiger Indexbereich: min={mn}, max={mx}, N={N}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{N} vertices\n{E} tets\n")
        for p in nodes: f.write(f"{ffix(p[0])} {ffix(p[1])} {ffix(p[2])}\n")
        for e in tet:   f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
    return path


def write_selection(path_txt: Path, nodes: np.ndarray, tets: np.ndarray,
                    angle_deg: float, up_axis: str, fix_on_handle: bool):
    """
    Schreibt Auswahl-Datei mit Flags für fixed und handle
    :param path_txt: Ausgabepfad (.txt)
    :param nodes: Punktkoordinaten
    :param tets: Tetraeder-Konnektivität
    :param angle_deg: Winkel zur Up-Achse für handle-Erkennung
    :param up_axis: Up-Achse ('x', 'y', 'z')
    :param fix_on_handle: fixed=1, wenn handle=1
    :return: Pfad zur geschriebenen Datei
    """
    N = int(nodes.shape[0])
    handle = compute_surface_handle_flags(nodes, tets, angle_deg=angle_deg, up_axis=up_axis, exclude_bottom=True)
    fixed = np.zeros(N, np.int8)
    if fix_on_handle: fixed = np.maximum(fixed, handle)
    with open(path_txt, "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(f"{i + 1}:{int(fixed[i])}:{int(handle[i])}:\n")
    return path_txt


# Main

def main():
    """
    Hauptablauf: STL einlesen, tetraedrisieren und Auswahl-Dateien erzeugen
    """
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        print(f"Eingabedatei nicht gefunden: {in_path}", file=sys.stderr);
        sys.exit(1)
    out_path = Path(OUTPUT_PATH) if OUTPUT_PATH else in_path.with_suffix(".tet")

    mesh = load_and_repair_stl(in_path)
    verts = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)

    nodes, tets = tetrahedralize(verts, faces, quality=1.2, max_volume=None)

    # Koordinaten ggf. in Y-up bringen
    up_axis = "y" if Y_UP else "z"
    if Y_UP: nodes = to_y_up(nodes)

    try:
        tet_path = write_tet(out_path, nodes, tets)
        print(f"OK: .tet -> {tet_path}")
    except Exception as e:
        print(f"Fehler beim Schreiben .tet: {e}", file=sys.stderr);
        sys.exit(2)

    sel_path = f"output_selectionfile/{tet_path.with_suffix('.txt').name}"
    try:
        sel_path = write_selection(sel_path, nodes, tets,
                                   angle_deg=ANGLE_DEG, up_axis=up_axis, fix_on_handle=FIX_ON_HANDLE)
        print(f"OK: Auswahl-Datei -> {sel_path}")
    except Exception as e:
        print(f"Fehler beim Schreiben .txt: {e}", file=sys.stderr);
        sys.exit(3)


if __name__ == "__main__":
    main()
