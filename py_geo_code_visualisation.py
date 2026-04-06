import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation
import warnings
warnings.filterwarnings('ignore')

# Ein-/Ausgabe
INPUT_GEO_FILE = "output_geo_code/Kegel_v3.geo"   # Pfad zur .geo Datei (relativ zum Skript)

# Darstellungsmodus
# 'platform'  → Plattform-Pose (Originalkoordinaten aus dem Geo-File)
# 'toolpath'  → Werkzeugbahn / Düsenposition (Umkehrtransformation)
# 'both'      → beide Modi nacheinander
VISUALIZE_MODE = 'both'

# Standard-Offsets aus py_geo_code_parser.py
DEFAULT_BED_OFFSET  = (24.01192936, -23.95110169, 184.52700323)
DEFAULT_TEST_OFFSET = (0.0, 0.0, 0.0)

# Liniendicke (in mm)
LINE_WIDTH_PRINT  = 3.0   # Liniendicke beim Drucken (Extruder AN), in mm
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
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        self.commands = []
        for line in lines:
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
                blocks.append(current_block)
                current_block = {
                    'extruder_on': move['extruder_on'],
                    'moves': [move],
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
        
        import tempfile
        import os
        
        blocks = self.get_blocks()
        
        if not blocks:
            print("Keine Blöcke gefunden!")
            return
        
        # Sammle alle Koordinaten für Skalierung
        all_x = []
        all_y = []
        all_z = []
        
        for block in blocks:
            for move in block['moves']:
                all_x.append(move['x'])
                all_y.append(move['y'])
                all_z.append(move['z'])
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)
        
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
        
        # Erstelle temporären Ordner für Bilder
        temp_dir = tempfile.mkdtemp()
        frames = []
        print(f"Erstelle {len(blocks)} Frames...")
        
        try:
            for block_idx, block in enumerate(blocks):
                # Zeige Fortschritt
                if (block_idx + 1) % max(1, len(blocks) // 10) == 0:
                    print(f"  Frame {block_idx + 1}/{len(blocks)}")
                
                # Erstelle Figure
                fig = plt.figure(figsize=(16, 12), dpi=dpi)
                
                # 3D Subplot
                ax3d = fig.add_subplot(2, 2, 1, projection='3d')
                
                # 2D Subplots
                ax_xy = fig.add_subplot(2, 2, 2)
                ax_xz = fig.add_subplot(2, 2, 3)
                ax_yz = fig.add_subplot(2, 2, 4)
                
                # Zeichne alle bisherigen Blöcke (kumulativ)
                for prev_idx in range(block_idx + 1):
                    prev_block = blocks[prev_idx]
                    
                    # Erhalte Farbe basierend auf Extruder-Status und Block-Index
                    color = self.get_block_color(blocks, prev_idx)
                    
                    # Bei Graustufen-Farbe für graues Fahren
                    if prev_block['extruder_on']:
                        color_3d = color
                        color_2d = color
                        linewidth = mm_to_pt(line_width_print)
                    else:
                        color_3d = 'gray'
                        color_2d = 'gray'
                        linewidth = mm_to_pt(line_width_travel)
                    
                    # Extrahiere Koordinaten
                    moves = prev_block['moves']
                    xs = [m['x'] for m in moves]
                    ys = [m['y'] for m in moves]
                    zs = [m['z'] for m in moves]
                    
                    # 3D Plot
                    ax3d.plot(xs, ys, zs, color=color_3d, linewidth=linewidth, alpha=1.0)
                    
                    # 2D XY
                    ax_xy.plot(xs, ys, color=color_2d, linewidth=linewidth, alpha=1.0)
                    
                    # 2D XZ
                    ax_xz.plot(xs, zs, color=color_2d, linewidth=linewidth, alpha=1.0)
                    
                    # 2D YZ
                    ax_yz.plot(ys, zs, color=color_2d, linewidth=linewidth, alpha=1.0)
                    

                
                # 3D Axis setup
                ax3d.set_xlabel('X (mm)')
                ax3d.set_ylabel('Y (mm)')
                ax3d.set_zlabel('Z (mm)')
                ax3d.set_xlim(x_min, x_max)
                ax3d.set_ylim(y_min, y_max)
                ax3d.set_zlim(z_min, z_max)
                ax3d.set_title(f'3D View - Block {block_idx + 1}/{len(blocks)}\nExtruder: {"AN" if block["extruder_on"] else "AUS"}')
                ax3d.view_init(elev=20, azim=45)
                
                # 2D XY
                ax_xy.set_xlabel('X (mm)')
                ax_xy.set_ylabel('Y (mm)')
                ax_xy.set_xlim(x_min, x_max)
                ax_xy.set_ylim(y_min, y_max)
                ax_xy.set_title('Draufsicht (X-Y)')
                ax_xy.grid(True, alpha=0.3)
                ax_xy.set_aspect('equal', adjustable='box')
                
                # 2D XZ
                ax_xz.set_xlabel('X (mm)')
                ax_xz.set_ylabel('Z (mm)')
                ax_xz.set_xlim(x_min, x_max)
                ax_xz.set_ylim(z_min, z_max)
                ax_xz.set_title('Seitenansicht (X-Z)')
                ax_xz.grid(True, alpha=0.3)
                ax_xz.set_aspect('equal', adjustable='box')
                
                # 2D YZ
                ax_yz.set_xlabel('Y (mm)')
                ax_yz.set_ylabel('Z (mm)')
                ax_yz.set_xlim(y_min, y_max)
                ax_yz.set_ylim(z_min, z_max)
                ax_yz.set_title('Seitenansicht (Y-Z)')
                ax_yz.grid(True, alpha=0.3)
                ax_yz.set_aspect('equal', adjustable='box')
                
                fig.suptitle(
                    f'Geo-Code Animation [{coord_label}] – Block {block_idx + 1} von {len(blocks)}',
                    fontsize=14, fontweight='bold'
                )
                
                # Speichere Frame als Datei
                frame_file = os.path.join(temp_dir, f'frame_{block_idx:04d}.png')
                fig.savefig(frame_file, dpi=dpi, bbox_inches='tight')
                frames.append(frame_file)
                
                plt.close(fig)
            
            # Lade Bilder und speichere als GIF
            print(f"\nSpeichere GIF mit {len(frames)} Frames...")
            frame_duration = int(1000 / fps)
            
            from PIL import Image
            images = [Image.open(f) for f in frames]
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
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    
    def create_statistics(self) -> Dict:
        """Erstellt Statistiken über den Druckprozess"""
        moves = [cmd for cmd in self.commands if cmd['type'] == 'move']
        
        on_moves = [m for m in moves if m['extruder_on']]
        off_moves = [m for m in moves if not m['extruder_on']]
        
        blocks = self.get_blocks()
        on_blocks = [b for b in blocks if b['extruder_on']]
        off_blocks = [b for b in blocks if not b['extruder_on']]
        
        # Berechne räumliche Statistiken
        all_x = [m['x'] for m in moves]
        all_y = [m['y'] for m in moves]
        all_z = [m['z'] for m in moves]
        
        stats = {
            'Gesamt_Befehle': len(moves),
            'Extruder_AN_Befehle': len(on_moves),
            'Extruder_AUS_Befehle': len(off_moves),
            'Gesamt_Bloecke': len(blocks),
            'Druck_Bloecke': len(on_blocks),
            'Fahr_Bloecke': len(off_blocks),
            'Druck_zu_Fahrt_Verhaeltnis': f"{len(on_moves) / len(moves) * 100:.1f}%",
            'X_Bereich': f"{min(all_x):.2f} bis {max(all_x):.2f} (Spanne: {max(all_x) - min(all_x):.2f} mm)",
            'Y_Bereich': f"{min(all_y):.2f} bis {max(all_y):.2f} (Spanne: {max(all_y) - min(all_y):.2f} mm)",
            'Z_Bereich': f"{min(all_z):.2f} bis {max(all_z):.2f} (Spanne: {max(all_z) - min(all_z):.2f} mm)",
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
    output_gif   = geo_file.parent / f'Kegel_v3_BlockAnimation_{mode_label}.gif'

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

    print(f"  → GIF:   {output_gif.name}")
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

    print("\n" + "="*55)
    if mode == 'platform':
        print(" Plattform-Pose erfolgreich erstellt!")
        print("="*55)
        print("  Kegel_v3_BlockAnimation_Platform.gif  – Plattform-Pose")
    elif mode == 'toolpath':
        print(" Werkzeugbahn erfolgreich erstellt!")
        print("="*55)
        print("  Kegel_v3_BlockAnimation_Toolpath.gif  – Werkzeugbahn")
    else:
        print(" BEIDE VISUALISIERUNGEN ERFOLGREICH ERSTELLT!")
        print("="*55)
        print("  Kegel_v3_BlockAnimation_Platform.gif  – Plattform-Pose")
        print("  Kegel_v3_BlockAnimation_Toolpath.gif  – Werkzeugbahn")
    print()


if __name__ == '__main__':
    main()
