import numpy as np
from scipy.spatial.transform import Rotation

"""
Generieren des benutzerdefiniertes Bewegungsformat (.geo)

Konvertiert G-Code (G0/G1 + Achsen + Rotationen) in ein einfaches
Bewegungsformat. Dabei wird die Plattformpose für den Hexapod MiniHex so berechnet, dass die
Düse beim Rotieren denselben Punkt berührt.
"""

# Ein-/Ausgabe
INPUT_PATH = 'output_trajektorie/Kegel_v6_5x1.txt'
OUTPUT_PATH = 'output_geo_code/Kegel_v6_5x1.geo'

# Geschwindigkeitskonstante (mm/min)
SPEED_CONSTANT = 8.7 * 60

# Maximale erlaubte Rotationen
MAX_ROT_X = 20.0
MAX_ROT_Y = 20.0
MAX_ROT_Z = 15.0

# Offset des Rotationszentrums
BED_OFFSET_X = 0.45643577
BED_OFFSET_Y = 1.58638469 
BED_OFFSET_Z = -160.01789573 

# Offset zur Extruder-Position für Testzwecke
TEST_OFFSET_X = 0
TEST_OFFSET_Y = 0
TEST_OFFSET_Z = 0 # 20 für Testzwecke, damit die Düse nicht auf dem Bett schleift

# Extruder-Temperatur und temperaturabhängiger Z-Offset
# Offset steigt linear von 0 mm bei 0 °C an: offset_z = TEMP_OFFSET_Z_SLOPE * EXTRUDER_TEMPERATURE
EXTRUDER_TEMPERATURE = 200  # Extruder-Temperatur in °C
TEMP_OFFSET_Z_SLOPE = 0.66 / 400  # Steigung: mm pro °C (Kalibrierwert vom Benutzer eintragen)

# Deaktiviert die Neigung
DISABLE_ROTATION = False

def strip_comments(line):
    return line.split(';')[0].strip()

def calculate_temp_offset_z(temperature, slope=TEMP_OFFSET_Z_SLOPE):
    """
    Berechnet den Z-Offset basierend auf der Extruder-Temperatur.
    Lineare Kalibrierung: offset_z = slope * temperature
    :param temperature: Extruder-Temperatur (°C)
    :param slope: Steigung (mm/°C)
    :return: Z-Offset in mm
    """
    return slope * temperature


def parse_gcode_line(line):
    """
    Liest eine G-/M-Code-Zeile und extrahiert Parameter.
    Unterstützt: G/M, X/Y/Z, A/B/C, F.
    :return: dict z.B. {'command':'G1','X':1.0,'F':3000}
    """
    components = {}
    tokens = line.strip().split()
    for token in tokens:
        if token.startswith('G') or token.startswith('M'):
            components['command'] = token
        elif token.startswith('X'):
            components['X'] = float(token[1:])
        elif token.startswith('Y'):
            components['Y'] = float(token[1:])
        elif token.startswith('Z'):
            components['Z'] = float(token[1:])
        elif token.startswith('A'):
            components['A'] = float(token[1:])
        elif token.startswith('B'):
            components['B'] = float(token[1:])
        elif token.startswith('C'):
            components['C'] = float(token[1:])
        elif token.startswith('F'):
            components['F'] = float(token[1:])
    return components


def compute_platform_pose(x, y, z, a, b, c):
    """
    Berechnet Plattform-Pose (Inverse der Düsenpose).
    :param x,y,z: Position (mm)
    :param a,b,c: Rotationen (°)
    :return: (px, py, pz, pa, pb, pc) Plattformposition (mm) und -rotation (°)
    """
    bed_rot_offset = np.array([BED_OFFSET_X, BED_OFFSET_Y, BED_OFFSET_Z])
    test_offset = np.array([TEST_OFFSET_X, TEST_OFFSET_Y, TEST_OFFSET_Z])

    dx_buildspace = np.array([x, y, z]) + test_offset

    R_B = Rotation.from_euler('ZYX', [c, b, a], degrees=True)
    R_H = R_B.inv()

    rot_hexapod = R_H.as_euler('xyz', degrees=True)

    dx_hexapod = R_H.apply(bed_rot_offset) - R_H.apply(dx_buildspace) - bed_rot_offset

    temp_offset_z = calculate_temp_offset_z(EXTRUDER_TEMPERATURE)

    return dx_hexapod[0], dx_hexapod[1], dx_hexapod[2] - temp_offset_z, rot_hexapod[0], rot_hexapod[1], rot_hexapod[2]

def convert_to_custom_code(gcode_lines, max_rot_x, max_rot_y, max_rot_z):
    """
    Wandelt Eingabe-Zeilen in Bewegungsbefehle um.
    Berechnet Plattformpositionen und begrenzt Rotationen.
    """
        
    custom_code = ['EXTRUDER_OFF','LA 0.0 0.0 -300.0 0.0 0.0 0.0']  # Startpose
    custom_ende_code = ''
    last_speed = None
    speed_mode_set = False

    curr_x = 0.0
    curr_y = 0.0
    curr_z = 0.0
    curr_a = 0.0
    curr_b = 0.0
    curr_c = 0.0

    prev_x = 0.0
    prev_y = 0.0
    prev_z = 0.0
    total_distance = 0.0

    for line in gcode_lines:
        line = strip_comments(line)
        if not line:
            continue

        components = parse_gcode_line(line)

        cmd = components.get('command')
        if cmd == 'M101':
            custom_code.append('EXTRUDER_ON')
            continue
        if cmd == 'M103':
            custom_code.append('EXTRUDER_OFF')
            continue

        if 'F' in components:
            speed = components['F']
            if speed != last_speed:
                if not speed_mode_set:
                    custom_code.append(f'VF {speed} 1')
                    speed_mode_set = True
                else:
                    custom_code.append(f'VF {speed}')
                last_speed = speed

        if cmd in ['G0', 'G1', 'G00', 'G01']:
            curr_x = components.get('X', curr_x)
            curr_y = components.get('Y', curr_y)
            curr_z = components.get('Z', curr_z)

            if DISABLE_ROTATION:
                curr_a = 0.0
                curr_b = 0.0
                curr_c = 0.0
            else:
                curr_a = components.get('A', curr_a)
                curr_b = components.get('B', curr_b)
                curr_c = components.get('C', curr_c)

            rot_a = np.clip(curr_a, -max_rot_x, max_rot_x)
            rot_b = np.clip(curr_b, -max_rot_y, max_rot_y)
            rot_c = np.clip(curr_c, -max_rot_z, max_rot_z)

            px, py, pz, pa, pb, pc = compute_platform_pose(
                curr_x, curr_y, curr_z, rot_a, rot_b, rot_c)

            custom_code.append(f'LA {px:.5f} {py:.5f} {pz:.5f} {pa:.5f} {pb:.5f} {pc:.5f}')

            segment = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)
            total_distance += segment
            prev_x, prev_y, prev_z = curr_x, curr_y, curr_z

    custom_code.append(custom_ende_code)

    return custom_code, total_distance


def main(input_filename, output_filename, max_rot_x, max_rot_y, max_rot_z):
    with open(input_filename, 'r') as infile:
        gcode_lines = infile.readlines()

    custom_code, total_distance = convert_to_custom_code(gcode_lines, max_rot_x, max_rot_y, max_rot_z)

    total_time_min = total_distance / SPEED_CONSTANT
    hours = int(total_time_min // 60)
    minutes = total_time_min % 60
    print(f'Gesamtstrecke : {total_distance:.2f} mm')
    print(f'Geschwindigkeit: {SPEED_CONSTANT:.1f} mm/min')
    print(f'Gesamtzeit    : {hours} h {minutes:.1f} min')

    with open(output_filename, 'w') as outfile:
        outfile.write('\n'.join(custom_code))


if __name__ == '__main__':
    main(
        INPUT_PATH,
        OUTPUT_PATH,
        MAX_ROT_X,
        MAX_ROT_Y,
        MAX_ROT_Z
    )
