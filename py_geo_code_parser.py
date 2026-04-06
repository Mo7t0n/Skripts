import numpy as np
from scipy.spatial.transform import Rotation

"""
Generieren des benutzerdefiniertes Bewegungsformat (.geo)

Konvertiert G-Code (G0/G1 + Achsen + Rotationen) in ein einfaches
Bewegungsformat. Dabei wird die Plattformpose für den Hexapod MiniHex so berechnet, dass die
Düse beim Rotieren denselben Punkt berührt.
"""

# Ein-/Ausgabe
INPUT_PATH = 'output_trajektorie/Kegel_v3.txt'
OUTPUT_PATH = 'output_geo_code/Kegel_v3.geo'

# Maximale erlaubte Rotationen
MAX_ROT_X = 25.0
MAX_ROT_Y = 25.0
MAX_ROT_Z = 15.0

# Offset des Rotationszentrums
BED_OFFSET_X = 24.01192936
BED_OFFSET_Y = -23.95110169
BED_OFFSET_Z = 184.52700323

# Offset zur Extruder-Position für Testzwecke
TEST_OFFSET_X = 0
TEST_OFFSET_Y = 0
TEST_OFFSET_Z = 0

def strip_comments(line):
    return line.split(';')[0].strip()

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


def compute_platform_pose(x, y, z, a, b, c, bed_offset_xyz=(-3.13421237, -9.74438965, -178.13481531), test_offset_xyz=(10.0, -1.5, 55.0)):
    """
    Berechnet Plattform-Pose (Inverse der Düsenpose).
    :param x,y,z: Position (mm)
    :param a,b,c: Rotationen (°)
    :param bed_offset_xyz: Abstand Rotationszentrum–Düse (mm)
    :param test_offset_xyz: Offset für Extruder-Position (mm)
    :return: (px,py,pz,pa,pb,pc)
    """
    bed_offset = np.array(bed_offset_xyz)
    test_offset = np.array(test_offset_xyz)

    tool_tip = np.array([x, y, z])

    R = Rotation.from_euler('ZYX', [c, b, a], degrees=True)

    rot_center = tool_tip - bed_offset
    rotated_offset = R.apply(bed_offset)
    new_nozzle_pos = rot_center + rotated_offset - test_offset

    T_nozzle = np.eye(4)
    T_nozzle[:3, :3] = R.as_matrix()
    T_nozzle[:3, 3] = new_nozzle_pos

    T_plattform = np.linalg.inv(T_nozzle)

    pos = T_plattform[:3, 3]
    rot = R.from_matrix(T_plattform[:3, :3])
    euler = rot.as_euler('ZYX', degrees=True)

    return pos[0], pos[1], pos[2], euler[2], euler[1], euler[0]

def convert_to_custom_code(gcode_lines, max_rot_x, max_rot_y, max_rot_z, bed_offset=(-3.13421237, -9.74438965, -178.13481531), test_offset=(10.0, -1.5, 55.0)):
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

            rot_x = np.clip(components.get('A', 0.0), -max_rot_x, max_rot_x)
            rot_y = np.clip(components.get('B', 0.0), -max_rot_y, max_rot_y)
            rot_z = np.clip(components.get('C', 0.0), -max_rot_z, max_rot_z)

            px, py, pz, pa, pb, pc = compute_platform_pose(
                curr_x, curr_y, curr_z, rot_x, rot_y, rot_z, bed_offset, test_offset
            )

            custom_code.append(f'LA {px:.5f} {py:.5f} {pz:.5f} {pa:.5f} {pb:.5f} {pc:.5f}')

    custom_code.append(custom_ende_code)

    return custom_code


def main(input_filename, output_filename, max_rot_x, max_rot_y, max_rot_z, bed_offset, test_offset):
    with open(input_filename, 'r') as infile:
        gcode_lines = infile.readlines()

    custom_code = convert_to_custom_code(gcode_lines, max_rot_x, max_rot_y, max_rot_z, bed_offset, test_offset)

    with open(output_filename, 'w') as outfile:
        outfile.write('\n'.join(custom_code))


if __name__ == '__main__':
    BED_OFFSET = (BED_OFFSET_X, BED_OFFSET_Y, BED_OFFSET_Z)
    TEST_OFFSET = (TEST_OFFSET_X, TEST_OFFSET_Y, TEST_OFFSET_Z)

    main(
        INPUT_PATH,
        OUTPUT_PATH,
        MAX_ROT_X,
        MAX_ROT_Y,
        MAX_ROT_Z,
        BED_OFFSET,
        TEST_OFFSET
    )
