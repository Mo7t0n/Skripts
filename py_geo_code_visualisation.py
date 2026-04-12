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

# Standard-Offsets aus py_geo_code_parser.py
DEFAULT_BED_OFFSET  = (-0.19494689, 0.40624396, -186.25046989)
DEFAULT_TEST_OFFSET = (0.0, 0.0, 0.0)

# Liniendicke (in mm)
LINE_WIDTH_PRINT  = 4.0   # Liniendicke beim Drucken (Extruder AN), in mm
LINE_WIDTH_TRAVEL = 0.1  # Liniendicke bei Leerfahrten (Extruder AUS), in mm

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


class GeoCodeVisualizer:
    """Visualisiert Geo-Code Dateien mit Extruder-Status und Liniendicke."""
    
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
    
    def create_gif_animation(self, output_file: str, fps: int = 5, dpi: int = 80,
                               line_width_print: float = LINE_WIDTH_PRINT,
                               line_width_travel: float = LINE_WIDTH_TRAVEL):
        """Erstellt eine GIF-Animation mit jedem Block als Frame.
        
        Koordinatenraum hängt vom im Konstruktor gesetzten 'as_toolpath' Flag ab.

        :param line_width_print:  Liniendicke beim Drucken (Extruder AN), in mm
        :param line_width_travel: Liniendicke bei Leerfahrten (Extruder AUS), in mm
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
        for block in blocks:
            moves = block['moves']
            coords = np.array([[m['x'], m['y'], m['z']] for m in moves], dtype=float)

            if block['extruder_on']:
                color = shared_colors[print_idx]
                print_idx += 1
                linewidth = mm_to_pt(line_width_print)
            else:
                color = 'gray'
                linewidth = mm_to_pt(line_width_travel)

            prepared_blocks.append({
                'coords': coords,
                'color': color,
                'linewidth': linewidth,
                'extruder_on': block['extruder_on'],
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
        
        frames = []
        print(f"Erstelle {len(blocks)} Frames...")
        # Einmalige Figure; pro Frame nur den nächsten Block ergänzen und Bild aus dem Canvas ziehen
        fig = plt.figure(figsize=(16, 12), dpi=dpi)

        # 3D Subplot
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')

        # 2D Subplots
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)

        # Statisches Axis-Setup
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Z (mm)')
        ax3d.set_xlim(x_min, x_max)
        ax3d.set_ylim(y_min, y_max)
        ax3d.set_zlim(z_min, z_max)
        ax3d.view_init(elev=20, azim=45)

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

        ax_yz.set_xlabel('Y (mm)')
        ax_yz.set_ylabel('Z (mm)')
        ax_yz.set_xlim(y_min, y_max)
        ax_yz.set_ylim(z_min, z_max)
        ax_yz.set_title('Seitenansicht (Y-Z)')
        ax_yz.grid(True, alpha=0.3)
        ax_yz.set_aspect('equal', adjustable='box')

        from PIL import Image
        for block_idx, prep in enumerate(prepared_blocks):
            # Zeige Fortschritt
            if (block_idx + 1) % max(1, len(prepared_blocks) // 10) == 0:
                print(f"  Frame {block_idx + 1}/{len(prepared_blocks)}")

            coords = prep['coords']
            color = prep['color']
            linewidth = prep['linewidth']

            xs = coords[:, 0]
            ys = coords[:, 1]
            zs = coords[:, 2]

            # Nur neuen Block hinzufügen (kumulative Darstellung entsteht automatisch)
            ax3d.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=1.0)
            ax_xy.plot(xs, ys, color=color, linewidth=linewidth, alpha=1.0)
            ax_xz.plot(xs, zs, color=color, linewidth=linewidth, alpha=1.0)
            ax_yz.plot(ys, zs, color=color, linewidth=linewidth, alpha=1.0)

            ax3d.set_title(
                f'3D View - Block {block_idx + 1}/{len(prepared_blocks)}\nExtruder: {"AN" if prep["extruder_on"] else "AUS"}'
            )
            fig.suptitle(
                f'Geo-Code Animation [{coord_label}] – Block {block_idx + 1} von {len(prepared_blocks)}',
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
                'Extruder_AN_Befehle': 0,
                'Extruder_AUS_Befehle': 0,
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
            'Extruder_AN_Befehle': on_count,
            'Extruder_AUS_Befehle': off_count,
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
                      fps: int = 5, dpi: int = 80,
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
