import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation
import warnings
warnings.filterwarnings('ignore')

# Ein-/Ausgabe
INPUT_GEO_FILE = "output_geo_code/Kegel_v4.geo"   # Pfad zur .geo Datei (relativ zum Skript)

# Darstellungsmodus
# 'platform'  → Plattform-Pose (Originalkoordinaten aus dem Geo-File)
# 'toolpath'  → Werkzeugbahn / Düsenposition (Umkehrtransformation)
# 'both'      → beide Modi nacheinander
VISUALIZE_MODE = 'toolpath'

# Kamera der 3D-Ansicht (Grad)
CAMERA_ELEV_DEG = 20
CAMERA_AZIM_DEG = 135

# Beschleunigung: True = schnelle Darstellung via vorab erzeugte Linien-Artists.
# Bei Problemen mit der Darstellung auf False setzen (Kompatibilitätsmodus).
ENABLE_FAST_RENDER = True

# Gewichtung der Darstellungstiefe (2D + 3D): Schwerpunkt vs. kameranächster Punkt
DEPTH_WEIGHT_CENTROID = 0.7
DEPTH_WEIGHT_NEAREST = 0.3

# Standard-Offsets aus py_geo_code_parser.py
DEFAULT_BED_OFFSET  = (-0.19494689, 0.40624396, -186.25046989)
DEFAULT_TEST_OFFSET = (0.0, 0.0, 0.0)

# Liniendicke (in mm)
LINE_WIDTH_PRINT  = 5.0   # Liniendicke beim Drucken, in mm
LINE_WIDTH_TRAVEL = 0.05  # Liniendicke bei Leerfahrten, in mm

# GIF-Geschwindigkeit (Frames pro Sekunde): höher = schneller, niedriger = langsamer
GIF_FPS = 20

# Zufallsfarben für Druckblöcke
# Ganzzahl → reproduzierbare Farben; None → bei jedem Start neue Farben
COLOR_RANDOM_SEED = 42


def mm_to_pt(mm: float) -> float:
    """Konvertiert Millimeter in Matplotlib-Linienpunkte (1 pt = 1/72 Zoll)."""
    return mm * 72.0 / 25.4


def generate_block_colors(num_print_blocks: int, seed=COLOR_RANDOM_SEED) -> list:
    """Erzeugt eine reproduzierbare Liste zufälliger RGB-Farben für Druckblöcke."""
    rng = np.random.default_rng(seed)
    colors = rng.random((max(num_print_blocks, 1), 3)) * 0.7 + 0.3
    return [tuple(c) for c in colors]


def blend_depth(centroid_value: float, nearest_value: float) -> float:
    """Mischt Schwerpunkt- und kameranächste Tiefe gemäß globaler Gewichtung."""
    w_sum = DEPTH_WEIGHT_CENTROID + DEPTH_WEIGHT_NEAREST
    if w_sum <= 1e-12:
        return nearest_value
    return (DEPTH_WEIGHT_CENTROID * centroid_value + DEPTH_WEIGHT_NEAREST * nearest_value) / w_sum


class GeoCodeVisualizer:
    """Visualisiert Geo-Code Dateien mit Druckstatus und Liniendicke."""
    
    def __init__(self, filepath: str,
                 as_toolpath: bool = False,
                 bed_offset: Tuple = DEFAULT_BED_OFFSET,
                 test_offset: Tuple = DEFAULT_TEST_OFFSET):
        """
        :param filepath: Pfad zur .geo Datei
        :param as_toolpath: True  → Koordinaten per Umkehrtransformation in
                                    Werkzeugbahn-Raum (Düsenposition) umrechnen
                            False → Plattform-Pose direkt visualisieren
        :param bed_offset:   Offset Rotationszentrum-Düse (aus geo_code_parser)
        :param test_offset:  Extruder-Offset (aus geo_code_parser)
        """
        self.filepath = Path(filepath)
        self.commands = []
        self.extruder_state = False
        self.as_toolpath = as_toolpath
        self.bed_offset = np.array(bed_offset)
        self.test_offset = np.array(test_offset)
        self._shared_colors: Optional[list] = None
        self.parse_file()

    def set_shared_colors(self, colors: list):
        """Setzt vorab berechnete Farben, die für beide Visualisierungen identisch sind."""
        self._shared_colors = colors
    
    @staticmethod
    def compute_toolpath_pose(px: float, py: float, pz: float,
                              pa: float, pb: float, pc: float,
                              bed_offset: np.ndarray,
                              test_offset: np.ndarray) -> Tuple:
        """
        Umkehrung von compute_platform_pose aus py_geo_code_parser.py.

        Im geo-File gespeicherte Plattform-Pose:
          pa = euler[2] (X-Achse), pb = euler[1] (Y), pc = euler[0] (Z)
          → Rotation.from_euler('ZYX', [pc, pb, pa])

        Vorwärts-Transformation war:
          T_nozzle  = [R_nozzle | new_nozzle_pos]
          T_platform = inv(T_nozzle)

        Rückwärts:
          T_nozzle = inv(T_platform)
          new_nozzle_pos = -R_plat^T @ [px,py,pz]
          tool_tip = new_nozzle_pos + bed_offset
                     - R_nozzle.apply(bed_offset) + test_offset
        """
        # Plattform-Rotation aus gespeicherten Euler-Winkeln (ZYX: [pc, pb, pa])
        R_plat = Rotation.from_euler('ZYX', [pc, pb, pa], degrees=True)

        # T_nozzle = inv(T_platform)  →  R_nozzle = R_plat^-1,  t_nozzle = -R_plat^T @ t_platform
        R_nozzle = R_plat.inv()
        new_nozzle_pos = -(R_plat.as_matrix().T @ np.array([px, py, pz]))

        # Werkzeugspitze zurückrechnen
        tool_tip = new_nozzle_pos + bed_offset - R_nozzle.apply(bed_offset) + test_offset

        # Winkel der Düse zurückrechnen: R = from_euler('ZYX', [c, b, a])
        euler = R_nozzle.as_euler('ZYX', degrees=True)  # [Z, Y, X]
        c_ang = euler[0]
        b_ang = euler[1]
        a_ang = euler[2]

        return tool_tip[0], tool_tip[1], tool_tip[2], a_ang, b_ang, c_ang

    def parse_file(self):
        """Parst die Geo-Code Datei und wendet ggf. Umkehrtransformation an."""
        self.commands = []
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('EXTRUDER_'):
                    # Extruder Befehl
                    self.extruder_state = 'ON' in line
                    self.commands.append({
                        'type': 'extruder',
                        'state': 'ON' if 'ON' in line else 'OFF',
                    })
                elif line.startswith('LA'):
                    # Linear Axis Befehl: LA x y z a b c
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            a = float(parts[4]) if len(parts) > 4 else 0.0
                            b = float(parts[5]) if len(parts) > 5 else 0.0
                            c = float(parts[6]) if len(parts) > 6 else 0.0

                            # Umkehrtransformation: Plattform-Pose → Düsenposition
                            if self.as_toolpath:
                                x, y, z, a, b, c = self.compute_toolpath_pose(
                                    x, y, z, a, b, c,
                                    self.bed_offset, self.test_offset
                                )

                            self.commands.append({
                                'type': 'move',
                                'x': x,
                                'y': y,
                                'z': z,
                                'a': a,
                                'b': b,
                                'c': c,
                                'extruder_on': self.extruder_state,
                            })
                        except (ValueError, IndexError):
                            pass
    
    def get_blocks(self) -> List[Dict]:
        """Gruppiert Moves in zusammenhängende Blöcke basierend auf Extruder-Status"""
        moves = [cmd for cmd in self.commands if cmd['type'] == 'move']
        
        blocks = []
        if not moves:
            return blocks
        
        # Starte ersten Block
        current_block = {
            'extruder_on': moves[0]['extruder_on'],
            'moves': [moves[0]],
        }
        
        # Gruppiere aufeinanderfolgende Moves
        for i in range(1, len(moves)):
            move = moves[i]
            
            # Wenn sich Extruder-Status ändert, neuer Block
            if move['extruder_on'] != current_block['extruder_on']:
                last_move = current_block['moves'][-1]
                blocks.append(current_block)
                # Letzten Punkt des Vorgängers als Startpunkt übernehmen,
                # damit der Druckpfad lückenlos von der Anfahrposition beginnt.
                current_block = {
                    'extruder_on': move['extruder_on'],
                    'moves': [last_move, move],
                }
            else:
                current_block['moves'].append(move)
        
        # Letzten Block hinzufügen
        blocks.append(current_block)
        
        return blocks
    
    def get_block_color(self, blocks: List[Dict], block_idx: int):
        """Gibt eine Farbe für einen Block zurück – zufällig für Druckblöcke, grau für Leerfahrten."""
        block = blocks[block_idx]

        if not block['extruder_on']:
            return 'gray'

        print_block_idx = sum(1 for b in blocks[:block_idx + 1] if b['extruder_on']) - 1

        # Geteilte Farben verwenden (falls vorab gesetzt), sonst frisch generieren
        if self._shared_colors is not None:
            colors = self._shared_colors
        else:
            total_print_blocks = sum(1 for b in blocks if b['extruder_on'])
            colors = generate_block_colors(total_print_blocks)

        return colors[print_block_idx]
    
    def create_gif_animation(self, output_file: str, fps: int = GIF_FPS, dpi: int = 80,
                               line_width_print: float = LINE_WIDTH_PRINT,
                               line_width_travel: float = LINE_WIDTH_TRAVEL):
        """Erstellt eine GIF-Animation mit jedem Block als Frame.
        
        Koordinatenraum hängt vom im Konstruktor gesetzten 'as_toolpath' Flag ab.

        :param line_width_print:  Liniendicke beim Drucken, in mm
        :param line_width_travel: Liniendicke bei Leerfahrten, in mm
        """
        coord_label = 'Werkzeugbahn (Düsenposition)' if self.as_toolpath else 'Plattform-Pose'
        
        blocks = self.get_blocks()
        
        if not blocks:
            print("Keine Blöcke gefunden!")
            return
        
        # Blockdaten einmalig vorbereiten (vermeidet wiederholte Listen-/Farb-Berechnung pro Frame)
        total_print_blocks = sum(1 for b in blocks if b['extruder_on'])
        shared_colors = self._shared_colors if self._shared_colors is not None else generate_block_colors(total_print_blocks)

        prepared_blocks = []
        print_idx = 0
        for seq_idx, block in enumerate(blocks):
            moves = block['moves']
            coords = np.array([[m['x'], m['y'], m['z']] for m in moves], dtype=float)

            if block['extruder_on']:
                color = shared_colors[print_idx]
                print_idx += 1
                linewidth = mm_to_pt(line_width_print)
            else:
                color = 'gray'
                linewidth = mm_to_pt(line_width_travel)

            mean_xyz = np.mean(coords, axis=0)
            zorder_xy_centroid = float(mean_xyz[2])
            zorder_xy_nearest = float(np.max(coords[:, 2]))
            zorder_xz_centroid = float(-mean_xyz[1])
            zorder_xz_nearest = float(-np.min(coords[:, 1]))
            zorder_yz_centroid = float(-mean_xyz[0])
            zorder_yz_nearest = float(-np.min(coords[:, 0]))
            prepared_blocks.append({
                'seq_idx': seq_idx,
                'coords': coords,
                'color': color,
                'linewidth': linewidth,
                'extruder_on': block['extruder_on'],
                # Tiefenwerte für korrekte Überlappung in den 2D-Projektionen:
                # je Ansicht Gewichtung über DEPTH_WEIGHT_CENTROID / DEPTH_WEIGHT_NEAREST.
                # XY-Ansicht (von oben): höhere Z = Vordergrund → hoher zorder
                # XZ-Ansicht (von vorne, Tiefe = Y): kleine Y = Vordergrund → hoher zorder
                # YZ-Ansicht (von der Seite, Tiefe = X): kleine X = Vordergrund → hoher zorder
                'zorder_xy': blend_depth(zorder_xy_centroid, zorder_xy_nearest),
                'zorder_xz': blend_depth(zorder_xz_centroid, zorder_xz_nearest),
                'zorder_yz': blend_depth(zorder_yz_centroid, zorder_yz_nearest),
                # Tiefe relativ zur Kamera (wird später für Painter's-Sort in 3D gesetzt)
                'mean_xyz': mean_xyz,
            })

        all_coords = np.vstack([b['coords'] for b in prepared_blocks])
        x_min, y_min, z_min = np.min(all_coords, axis=0)
        x_max, y_max, z_max = np.max(all_coords, axis=0)
        
        # Puffer
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        z_range = z_max - z_min if z_max != z_min else 1
        
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        z_min -= z_range * 0.1
        z_max += z_range * 0.1

        # Einheitliche 3D-Skalierung: alle Achsen erhalten dieselbe Spannweite.
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        z_center = 0.5 * (z_min + z_max)
        max_half_span = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        x3_min, x3_max = x_center - max_half_span, x_center + max_half_span
        y3_min, y3_max = y_center - max_half_span, y_center + max_half_span
        z3_min, z3_max = z_center - max_half_span, z_center + max_half_span
        
        frames = []
        print(f"Erstelle {len(blocks)} Frames...")

        # Tiefenvektor der Kamera: von der Szene in Richtung Kamera
        # Für die 3D-Sortierung verwenden wir die gleiche einstellbare Mischung
        # aus Schwerpunkt-Tiefe und dem blicknächsten Punkt eines Blocks.
        _elev_r = np.radians(CAMERA_ELEV_DEG)
        _azim_r = np.radians(CAMERA_AZIM_DEG)
        _view_dir = np.array([
            np.cos(_elev_r) * np.cos(_azim_r),
            np.cos(_elev_r) * np.sin(_azim_r),
            np.sin(_elev_r),
        ])
        for pb in prepared_blocks:
            centroid_depth = float(np.dot(pb['mean_xyz'], _view_dir))
            nearest_depth = float(np.max(pb['coords'] @ _view_dir))
            pb['depth_3d'] = blend_depth(centroid_depth, nearest_depth)

        # Einmalige Figure; 3D-Achse wird pro Frame neu gezeichnet (Painter's Sort),
        # 2D-Achsen bleiben inkrementell mit zorder.
        fig = plt.figure(figsize=(16, 12), dpi=dpi)

        # 3D Subplot
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')

        # 2D Subplots
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)

        ELEV, AZIM = CAMERA_ELEV_DEG, CAMERA_AZIM_DEG

        def _setup_ax3d(ax, title=''):
            """(Neu-)Einrichten der 3D-Achse nach cla()."""
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim(x3_min, x3_max)
            ax.set_ylim(y3_min, y3_max)
            ax.set_zlim(z3_min, z3_max)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=ELEV, azim=AZIM)
            if title:
                ax.set_title(title)

        # Statisches Axis-Setup
        _setup_ax3d(ax3d)

        ax_xy.set_xlabel('X (mm)')
        ax_xy.set_ylabel('Y (mm)')
        ax_xy.set_xlim(x_min, x_max)
        ax_xy.set_ylim(y_min, y_max)
        ax_xy.set_title('Draufsicht (X-Y)')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect('equal', adjustable='box')

        ax_xz.set_xlabel('X (mm)')
        ax_xz.set_ylabel('Z (mm)')
        ax_xz.set_xlim(x_min, x_max)
        ax_xz.set_ylim(z_min, z_max)
        ax_xz.set_title('Seitenansicht (X-Z)')
        ax_xz.grid(True, alpha=0.3)
        ax_xz.set_aspect('equal', adjustable='box')
        ax_xz.invert_xaxis()

        ax_yz.set_xlabel('Y (mm)')
        ax_yz.set_ylabel('Z (mm)')
        ax_yz.set_xlim(y_min, y_max)
        ax_yz.set_ylim(z_min, z_max)
        ax_yz.set_title('Seitenansicht (Y-Z)')
        ax_yz.grid(True, alpha=0.3)
        ax_yz.set_aspect('equal', adjustable='box')
        ax_yz.invert_xaxis()

        seq_to_block = {pb['seq_idx']: pb for pb in prepared_blocks}

        if ENABLE_FAST_RENDER:
            # Linien nur einmal erzeugen und anschließend frameweise einblenden.
            depth_sorted = sorted(prepared_blocks, key=lambda b: b['depth_3d'])
            for pb in depth_sorted:
                c = pb['coords']
                pb['artist_3d'] = ax3d.plot(
                    c[:, 0], c[:, 1], c[:, 2],
                    color=pb['color'], linewidth=pb['linewidth'], alpha=1.0, visible=False
                )[0]
                pb['artist_xy'] = ax_xy.plot(
                    c[:, 0], c[:, 1],
                    color=pb['color'], linewidth=pb['linewidth'], alpha=1.0,
                    zorder=pb['zorder_xy'], visible=False
                )[0]
                pb['artist_xz'] = ax_xz.plot(
                    c[:, 0], c[:, 2],
                    color=pb['color'], linewidth=pb['linewidth'], alpha=1.0,
                    zorder=pb['zorder_xz'], visible=False
                )[0]
                pb['artist_yz'] = ax_yz.plot(
                    c[:, 1], c[:, 2],
                    color=pb['color'], linewidth=pb['linewidth'], alpha=1.0,
                    zorder=pb['zorder_yz'], visible=False
                )[0]

        from PIL import Image
        last_visible_seq = -1
        for block_idx, prep in enumerate(prepared_blocks):
            # Zeige Fortschritt
            if (block_idx + 1) % max(1, len(prepared_blocks) // 10) == 0:
                print(f"  Frame {block_idx + 1}/{len(prepared_blocks)}")

            block_idx_text = f"{block_idx + 1:03d}"

            if ENABLE_FAST_RENDER:
                # Nur neu hinzugekommene Sequenzen sichtbar schalten.
                for seq_idx in range(last_visible_seq + 1, block_idx + 1):
                    pb = seq_to_block.get(seq_idx)
                    if pb is None:
                        continue
                    pb['artist_3d'].set_visible(True)
                    pb['artist_xy'].set_visible(True)
                    pb['artist_xz'].set_visible(True)
                    pb['artist_yz'].set_visible(True)
                last_visible_seq = block_idx

                ax3d.set_title("3D-Ansicht")
            else:
                coords = prep['coords']
                color = prep['color']
                linewidth = prep['linewidth']

                xs = coords[:, 0]
                ys = coords[:, 1]
                zs = coords[:, 2]

                # 3D: alle bisherigen Blöcke nach Kamera-Tiefe sortiert neu zeichnen (Painter's Algorithm)
                title_3d = "3D-Ansicht"
                ax3d.cla()
                _setup_ax3d(ax3d, title=title_3d)
                visible = prepared_blocks[:block_idx + 1]
                for pb in sorted(visible, key=lambda b: b['depth_3d']):  # fern → nah
                    c = pb['coords']
                    ax3d.plot(c[:, 0], c[:, 1], c[:, 2],
                              color=pb['color'], linewidth=pb['linewidth'], alpha=1.0)

                # 2D: inkrementell mit zorder für korrekte Überlappung
                # zorder steuert die Zeichenreihenfolge in den 2D-Ansichten:
                # Vordergrund-Linien (näher am Betrachter) erhalten einen höheren zorder-Wert.
                ax_xy.plot(xs, ys, color=color, linewidth=linewidth, alpha=1.0,
                           zorder=prep['zorder_xy'])
                ax_xz.plot(xs, zs, color=color, linewidth=linewidth, alpha=1.0,
                           zorder=prep['zorder_xz'])
                ax_yz.plot(ys, zs, color=color, linewidth=linewidth, alpha=1.0,
                           zorder=prep['zorder_yz'])
            fig.suptitle(
                f'Geo-Code Animation {coord_label} – Block {block_idx_text} von {len(prepared_blocks)}',
                fontsize=14, fontweight='bold'
            )

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            frames.append(Image.fromarray(frame))

        plt.close(fig)

        # Speichere GIF/TIFF
        print(f"\nSpeichere GIF mit {len(frames)} Frames...")
        frame_duration = int(1000 / fps)
        images = frames
        if images:
            images[0].save(
                output_file,
                save_all=True,
                append_images=images[1:],
                duration=frame_duration,
                loop=0
            )
            print(f"✓ GIF gespeichert: {output_file}")

            # Speichere als mehrseitiges TIFF (Bildreihe in einer Datei)
            tiff_file = output_file.replace('.gif', '.tiff')
            images[0].save(
                tiff_file,
                save_all=True,
                append_images=images[1:],
                compression='lzw',
            )
            print(f"✓ TIFF gespeichert: {tiff_file}")
            
            # Speichere auch das letzte Bild separat
            if images:
                last_image_file = output_file.replace('.gif', '_final.png')
                images[-1].save(last_image_file)
                print(f"✓ Finales Bild gespeichert: {last_image_file}")
    
    
    def create_statistics(self) -> Dict:
        """Erstellt Statistiken über den Druckprozess"""
        moves = [cmd for cmd in self.commands if cmd['type'] == 'move']
        if not moves:
            return {
                'Gesamt_Befehle': 0,
                'Druck_Befehle': 0,
                'Leerfahrt_Befehle': 0,
                'Gesamt_Bloecke': 0,
                'Druck_Bloecke': 0,
                'Fahr_Bloecke': 0,
                'Druck_zu_Fahrt_Verhaeltnis': '0.0%',
                'X_Bereich': '0.00 bis 0.00 (Spanne: 0.00 mm)',
                'Y_Bereich': '0.00 bis 0.00 (Spanne: 0.00 mm)',
                'Z_Bereich': '0.00 bis 0.00 (Spanne: 0.00 mm)',
            }
        
        extruder_mask = np.array([m['extruder_on'] for m in moves], dtype=bool)
        coords = np.array([[m['x'], m['y'], m['z']] for m in moves], dtype=float)

        on_count = int(np.count_nonzero(extruder_mask))
        off_count = len(moves) - on_count
        
        blocks = self.get_blocks()
        on_blocks = [b for b in blocks if b['extruder_on']]
        off_blocks = [b for b in blocks if not b['extruder_on']]

        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        spans = maxs - mins
        
        stats = {
            'Gesamt_Befehle': len(moves),
            'Druck_Befehle': on_count,
            'Leerfahrt_Befehle': off_count,
            'Gesamt_Bloecke': len(blocks),
            'Druck_Bloecke': len(on_blocks),
            'Fahr_Bloecke': len(off_blocks),
            'Druck_zu_Fahrt_Verhaeltnis': f"{on_count / len(moves) * 100:.1f}%",
            'X_Bereich': f"{mins[0]:.2f} bis {maxs[0]:.2f} (Spanne: {spans[0]:.2f} mm)",
            'Y_Bereich': f"{mins[1]:.2f} bis {maxs[1]:.2f} (Spanne: {spans[1]:.2f} mm)",
            'Z_Bereich': f"{mins[2]:.2f} bis {maxs[2]:.2f} (Spanne: {spans[2]:.2f} mm)",
        }
        
        return stats


def run_visualization(geo_file: Path, as_toolpath: bool,
                      bed_offset: Tuple = DEFAULT_BED_OFFSET,
                      test_offset: Tuple = DEFAULT_TEST_OFFSET,
                      fps: int = GIF_FPS, dpi: int = 80,
                      line_width_print: float = LINE_WIDTH_PRINT,
                      line_width_travel: float = LINE_WIDTH_TRAVEL,
                      shared_colors: Optional[list] = None):
    """Erstellt GIF + finales PNG für einen Koordinatenraum."""

    mode_label   = 'Toolpath'   if as_toolpath else 'Platform'
    mode_display = 'Werkzeugbahn (Düsenposition)' if as_toolpath else 'Plattform-Pose'
    out_dir = geo_file.parent / geo_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    output_gif   = out_dir / f'{geo_file.stem}_BlockAnimation_{mode_label}.gif'

    print(f"\n{'='*55}")
    print(f"  Modus: {mode_display}")
    print(f"{'='*55}")

    visualizer = GeoCodeVisualizer(
        str(geo_file),
        as_toolpath=as_toolpath,
        bed_offset=bed_offset,
        test_offset=test_offset,
    )
    if shared_colors is not None:
        visualizer.set_shared_colors(shared_colors)

    stats = visualizer.create_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nErstelle GIF-Animation ({mode_label})...\n")
    visualizer.create_gif_animation(str(output_gif), fps=fps, dpi=dpi,
                                    line_width_print=line_width_print,
                                    line_width_travel=line_width_travel)

    print(f"  → Ordner: {out_dir}")
    print(f"  → GIF:   {output_gif.name}")
    print(f"  → TIFF:  {output_gif.stem}.tiff")
    print(f"  → PNG:   {output_gif.stem}_final.png")


def main():
    """Hauptfunktion – erstellt Visualisierungen gemäß VISUALIZE_MODE."""

    geo_file = Path(__file__).parent / INPUT_GEO_FILE

    if not geo_file.exists():
        print(f"Datei nicht gefunden: {geo_file}")
        return

    print(f"Lade Datei: {geo_file}")

    mode = VISUALIZE_MODE.strip().lower()
    if mode not in ('platform', 'toolpath', 'both'):
        print(f"Unbekannter VISUALIZE_MODE: '{VISUALIZE_MODE}' – bitte 'platform', 'toolpath' oder 'both' verwenden.")
        return

    # Farben einmal vorab berechnen, damit beide Modi identische Farben verwenden
    _tmp = GeoCodeVisualizer(str(geo_file), as_toolpath=False)
    _blocks = _tmp.get_blocks()
    num_print_blocks = sum(1 for b in _blocks if b['extruder_on'])
    shared_colors = generate_block_colors(num_print_blocks)

    if mode in ('platform', 'both'):
        run_visualization(geo_file, as_toolpath=False, shared_colors=shared_colors)

    if mode in ('toolpath', 'both'):
        run_visualization(geo_file, as_toolpath=True, shared_colors=shared_colors)

    stem = geo_file.stem
    out_dir = geo_file.parent / stem
    print("\n" + "="*55)
    if mode == 'platform':
        print(" Plattform-Pose erfolgreich erstellt!")
        print("="*55)
        print(f"  {out_dir / (stem + '_BlockAnimation_Platform.gif')}")
    elif mode == 'toolpath':
        print(" Werkzeugbahn erfolgreich erstellt!")
        print("="*55)
        print(f"  {out_dir / (stem + '_BlockAnimation_Toolpath.gif')}")
    else:
        print(" BEIDE VISUALISIERUNGEN ERFOLGREICH ERSTELLT!")
        print("="*55)
        print(f"  {out_dir / (stem + '_BlockAnimation_Platform.gif')}  – Plattform-Pose")
        print(f"  {out_dir / (stem + '_BlockAnimation_Toolpath.gif')}  – Werkzeugbahn")
    print()


if __name__ == '__main__':
    main()
