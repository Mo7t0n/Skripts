import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class GeoCodeVisualizer:
    """Visualisiert Geo-Code Dateien mit Extruder-Status und Liniendicke"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.commands = []
        self.extruder_state = False
        self.parse_file()
    
    def parse_file(self):
        """Parst die Geo-Code Datei"""
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
    
    def get_block_color(self, blocks: List[Dict], block_idx: int) -> str:
        """Generiert eine Farbe für einen Block basierend auf Extruder-Status"""
        block = blocks[block_idx]
        
        if not block['extruder_on']:
            return 'gray'
        
        # Zähle nur die Druck-Blöcke bis zu diesem Index
        print_blocks_so_far = sum(1 for b in blocks[:block_idx + 1] if b['extruder_on'])
        total_print_blocks = sum(1 for b in blocks if b['extruder_on'])
        
        # Verwende HSV-Farbmodell für kontinuierliche Farbverteilung
        hue = (print_blocks_so_far - 1) / max(1, total_print_blocks - 1)
        
        # Konvertiere HSV zu RGB
        color = cm.hsv(hue)
        return color
    
    def create_gif_animation(self, output_file: str, fps: int = 5, dpi: int = 80):
        """Erstellt eine GIF-Animation mit jedem Block als Frame"""
        
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
                        linewidth = 3
                    else:
                        color_3d = 'gray'
                        color_2d = 'gray'
                        linewidth = 1
                    
                    # Extrahiere Koordinaten
                    moves = prev_block['moves']
                    xs = [m['x'] for m in moves]
                    ys = [m['y'] for m in moves]
                    zs = [m['z'] for m in moves]
                    
                    # 3D Plot
                    ax3d.plot(xs, ys, zs, color=color_3d, linewidth=linewidth, alpha=0.8)
                    
                    # 2D XY
                    ax_xy.plot(xs, ys, color=color_2d, linewidth=linewidth, alpha=0.8)
                    
                    # 2D XZ
                    ax_xz.plot(xs, zs, color=color_2d, linewidth=linewidth, alpha=0.8)
                    
                    # 2D YZ
                    ax_yz.plot(ys, zs, color=color_2d, linewidth=linewidth, alpha=0.8)
                    
                    # Markiere aktuellen Block
                    if prev_idx == block_idx:
                        # Start und Ende des aktuellen Blocks
                        ax3d.scatter([xs[0]], [ys[0]], [zs[0]], color='lime', s=100, marker='D', edgecolors='darkgreen', linewidths=2)
                        ax3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='cyan', s=100, marker='D', edgecolors='darkblue', linewidths=2)
                        
                        ax_xy.scatter([xs[0]], [ys[0]], color='lime', s=100, marker='D', edgecolors='darkgreen', linewidths=2)
                        ax_xy.scatter([xs[-1]], [ys[-1]], color='cyan', s=100, marker='D', edgecolors='darkblue', linewidths=2)
                        
                        ax_xz.scatter([xs[0]], [zs[0]], color='lime', s=100, marker='D', edgecolors='darkgreen', linewidths=2)
                        ax_xz.scatter([xs[-1]], [zs[-1]], color='cyan', s=100, marker='D', edgecolors='darkblue', linewidths=2)
                        
                        ax_yz.scatter([ys[0]], [zs[0]], color='lime', s=100, marker='D', edgecolors='darkgreen', linewidths=2)
                        ax_yz.scatter([ys[-1]], [zs[-1]], color='cyan', s=100, marker='D', edgecolors='darkblue', linewidths=2)
                
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
                
                fig.suptitle(f'Geo-Code Animation - Block {block_idx + 1} von {len(blocks)}', fontsize=16, fontweight='bold')
                
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


def main():
    """Hauptfunktion für die Visualisierung"""
    
    # Pfad zur Geo-Code Datei
    geo_file = Path(__file__).parent / 'output_geo_code' / 'Kegel_v3.geo'
    
    if not geo_file.exists():
        print(f"Datei nicht gefunden: {geo_file}")
        return
    
    print(f"Lade Datei: {geo_file}")
    visualizer = GeoCodeVisualizer(str(geo_file))
    
    # Zeige Statistiken
    stats = visualizer.create_statistics()
    print("\n" + "="*50)
    print("      DRUCKPROZESS STATISTIKEN - KEGEL_V3")
    print("="*50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Erstelle GIF-Animation
    print("Erstelle GIF-Animation blockweise...\n")
    output_file = Path(__file__).parent / 'Kegel_v3_BlockAnimation.gif'
    visualizer.create_gif_animation(str(output_file), fps=5, dpi=80)
    
    print("\n" + "="*50)
    print("  ✅ VISUALISIERUNG ERFOLGREICH ERSTELLT!")
    print("="*50)
    print(f"\nÖffne die GIF-Datei: {output_file.name}")
    print("  → Jeder Frame zeigt einen Druckblock")
    print("  → Rote Linien = Extruder AN (Drucken)")
    print("  → Graue Linien = Extruder AUS (Fahren)")
    print("\n")


if __name__ == '__main__':
    main()
