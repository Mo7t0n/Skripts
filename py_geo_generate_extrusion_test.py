"""
Generiert einen Extrusionstest: 5 parallele Linien auf dem geraden Druckbett.
Zwischen den Linien wird der Extruder abgeschaltet.

Parameter:
    LINE_HEIGHT     – Höhe der Linien über dem Druckbett in mm
    TRAVEL_HEIGHT   – Fahrhöhe zwischen den Linien in mm
    LINE_LENGTH     – Länge jeder Linie in mm
    LINE_SPACING    – Abstand zwischen den Linien in mm
    NUM_LINES       – Anzahl der parallelen Linien
    POINTS_BASE     – Anzahl Punkte auf der ersten Linie
    POINTS_STEP     – Zuwachs der Punktanzahl pro Linie
    S_X_OFFSET      – X-Versatz der S-Kurve relativ zu den Linien (mm)
    S_AMPLITUDE     – Amplitude der S-Kurve in X-Richtung (mm)
    S_SPACING_MIN   – Minimaler Punktabstand auf der S-Kurve (mm)
    S_SPACING_MAX   – Maximaler Punktabstand auf der S-Kurve (mm)
"""

OUTPUT_PATH = 'output_geo_code/Extrusionstest.geo'

from py_geo_code_parser import calculate_temp_offset_z, EXTRUDER_TEMPERATURE

# ── Parameter ────────────────────────────────────────────────────────────────
LINE_HEIGHT   = 1.0     # Höhe der Drucklinien über dem Druckbett (mm)
TRAVEL_HEIGHT = 20.0    # Fahrhöhe zwischen den Linien (mm)
LINE_LENGTH   = 200.0   # Länge jeder Linie in mm
LINE_SPACING  = 30.0    # Abstand zwischen den Linien (mm)
NUM_LINES     = 3       # Anzahl der parallelen Linien
POINTS_BASE   = 2       # Punkte auf der ersten Linie
POINTS_STEP   = 25      # Zuwachs der Punktanzahl pro folgende Linie
S_X_OFFSET    = 30.0    # Mindestabstand zwischen letzter Linie und S-Kurve (mm)
S_AMPLITUDE   = 35.0    # Amplitude der S-Kurve in X-Richtung (mm)
S_SPACING_MIN = 1.0     # Minimaler Bogenpunktabstand am Anfang der S-Kurve (mm)
S_SPACING_MAX = 5.0     # Maximaler Bogenpunktabstand am Ende der S-Kurve (mm)

# ── Parameter für Rechteck-Treppenmuster ──────────────────────────────────────
RECT_SIZE     = 40.0    # Größe des Rechtecks (±rect_size in X und Y) (mm)
LINE_WIDTH    = 5.0     # Breite der Drucklinien (mm)
RECT_NUM_STEPS = 5      # Anzahl der Treppenstufen
# ─────────────────────────────────────────────────────────────────────────────

def fmt(x, y, z, rx=0.0, ry=0.0, rz=0.0):
    return f"LA {x:.5f} {y:.5f} {z:.5f} {rx:.5f} {ry:.5f} {rz:.5f}"

def draw_filled_rectangle(lines, x_min, x_max, y_min, y_max, z, line_width):
    """Zeichnet ein gefülltes Rechteck mit Rahmen und Zickzack-Füllung.

    lines: Liste, zu der die Befehle hinzugefügt werden
    x_min, x_max, y_min, y_max: Rechteck-Grenzen
    z: Höhe zum Drucken
    line_width: Breite der Drucklinien für innere Füllung
    """
    # Fahrt zur Startposition
    lines.append(fmt(x_min, y_min, z - TRAVEL_HEIGHT))  # Fahrhöhe (20mm unter z)
    lines.append(fmt(x_min, y_min, z))
    lines.append("EXTRUDER_ON")

    # Rechteck-Rahmen
    lines.append(fmt(x_max, y_min, z))  # Unten
    lines.append(fmt(x_max, y_max, z))  # Rechts
    lines.append(fmt(x_min, y_max, z))  # Oben
    lines.append(fmt(x_min, y_min, z))  # Links

    lines.append("EXTRUDER_OFF")
    lines.append(fmt(x_min, y_min, z - TRAVEL_HEIGHT))

    # Innere Füllung
    x_inner_min = x_min + line_width
    x_inner_max = x_max - line_width
    y_inner_min = y_min + line_width
    y_inner_max = y_max - line_width

    lines.append(fmt(x_inner_min, y_inner_min, z - TRAVEL_HEIGHT))
    lines.append(fmt(x_inner_min, y_inner_min, z))
    lines.append("EXTRUDER_ON")

    # Horizontale Fülllinien mit Zickzack
    y = y_inner_min
    going_right = True

    while y <= y_inner_max:
        if going_right:
            lines.append(fmt(x_inner_max, y, z))
        else:
            lines.append(fmt(x_inner_min, y, z))

        y += line_width
        if y <= y_inner_max:
            # U-förmiger Übergang zur nächsten Linie
            lines.append(fmt(x_inner_max if going_right else x_inner_min, y, z))
            lines.append(fmt(x_inner_min if going_right else x_inner_max, y, z))
            lines.append(fmt(x_inner_min if going_right else x_inner_max, y, z))
            going_right = not going_right

    lines.append("EXTRUDER_OFF")
    lines.append(fmt(x_inner_min, y_inner_min, z - TRAVEL_HEIGHT))


def generate_rectangle_stairs(output_file, rect_size=RECT_SIZE, line_width=LINE_WIDTH,
                             step_height=LINE_HEIGHT, num_steps=RECT_NUM_STEPS):
    """Generiert ein Rechteck mit Linienfüllung und Treppenstufen in einer Ecke.
    """
    temp_offset_z = calculate_temp_offset_z(EXTRUDER_TEMPERATURE)
    z_print = - LINE_HEIGHT - temp_offset_z
    z_travel = - TRAVEL_HEIGHT - temp_offset_z

    lines = []

    # Startposition
    lines.append("EXTRUDER_OFF")
    lines.append("LA 0.0 0.0 -300.0 0.0 0.0 0.0")

    x_min, x_max = -rect_size, rect_size
    y_min, y_max = -rect_size, rect_size

    # ── Erste Schicht (volle Größe) ────────────────────────────────────────
    draw_filled_rectangle(lines, x_min, x_max, y_min, y_max, z_print, line_width)

    # ── Mehrere Schichten mit Treppenmuster in Ecke ────────────────────────────
    for layer in range(1, num_steps + 1):
        z_layer = - (layer + 1) * step_height - temp_offset_z

        # Rechteck wird in dieser Ecke kleiner (oben rechts)
        reduction = layer * (x_max - (x_min + line_width)) / (num_steps + 1)
        x_layer_max = x_max - reduction
        y_layer_max = y_max - reduction
        x_layer_min = x_min + line_width
        y_layer_min = y_min + line_width

        draw_filled_rectangle(lines, x_layer_min, x_layer_max, y_layer_min, y_layer_max, z_layer, line_width)

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def s_curve_points(y_start, y_end, amplitude, spacing_min, spacing_max):
    """Erzeugt Punkte einer S-Kurve (zwei halbe Sinusbögen) mit variablem Abstand.
    Der Punktabstand wächst linear von spacing_min (Anfang) auf spacing_max (Ende).
    Gibt Liste von (x_offset, y)-Paaren zurück, x_offset relativ zur Mittellinie."""
    import math
    total_len = y_end - y_start
    # Akkumuliere Bogenpositionen mit linear wachsendem Abstand
    positions = [0.0]
    s = 0.0
    while True:
        t = s / total_len                          # Fortschritt 0..1
        step = spacing_min + t * (spacing_max - spacing_min)
        s += step
        if s >= total_len:
            break
        positions.append(s)
    positions.append(total_len)                    # Endpunkt immer exakt

    pts = []
    for s in positions:
        t = s / total_len                          # 0..1
        y = y_start + s
        # S-Form: erste Hälfte schwingt nach außen (+), zweite nach innen (-)
        x_off = -amplitude * math.sin(2 * math.pi * t)
        pts.append((x_off, y))
    return pts


def generate():
    temp_offset_z = calculate_temp_offset_z(EXTRUDER_TEMPERATURE)
    z_print  = -LINE_HEIGHT - temp_offset_z
    z_travel = -TRAVEL_HEIGHT - temp_offset_z
    y_start  = -LINE_LENGTH / 2.0
    y_end    =  LINE_LENGTH / 2.0

    lines = []

    # Startposition (Parkposition)
    lines.append("EXTRUDER_OFF")
    lines.append("LA 0.0 0.0 -300.0 0.0 0.0 0.0")

    for i in range(NUM_LINES):
        x = (i - (NUM_LINES - 1) / 2.0) * LINE_SPACING  # zentriert um 0
        n_points = POINTS_BASE + i * POINTS_STEP          # Punkte auf dieser Linie

        # Anfahren Startpunkt auf Fahrhöhe
        lines.append(fmt(x, y_start, z_travel))

        # Absenken auf Druckhöhe
        lines.append(fmt(x, y_start, z_print))

        # Extruder einschalten und Linie mit Zwischenpunkten drucken
        lines.append("EXTRUDER_ON")
        for j in range(1, n_points + 1):
            y = y_start + j * LINE_LENGTH / n_points
            lines.append(fmt(x, y, z_print))

        # Extruder ausschalten und auf Fahrhöhe anheben
        lines.append("EXTRUDER_OFF")
        lines.append(fmt(x, y_end, z_travel))

    # ── S-Kurve ───────────────────────────────────────────────────────────────
    # x_s_center so positionieren, dass der linke Ausschlag (-amplitude) noch
    # mindestens S_X_OFFSET von der letzten parallelen Linie entfernt bleibt.
    x_last = ((NUM_LINES - 1) - (NUM_LINES - 1) / 2.0) * LINE_SPACING
    x_s_center = x_last + S_X_OFFSET + S_AMPLITUDE

    s_pts = s_curve_points(y_start, y_end, S_AMPLITUDE, S_SPACING_MIN, S_SPACING_MAX)

    # Anfahren des ersten Punkts auf Fahrhöhe
    x0, y0 = s_pts[0]
    lines.append("EXTRUDER_OFF")
    lines.append(fmt(x_s_center + x0, y0, z_travel))
    lines.append(fmt(x_s_center + x0, y0, z_print))

    # S-Kurve drucken
    lines.append("EXTRUDER_ON")
    for x_off, y in s_pts:  # alle Punkte inkl. Startpunkt im Druckblock
        lines.append(fmt(x_s_center + x_off, y, z_print))

    # Extruder aus und auf Fahrhöhe
    lines.append("EXTRUDER_OFF")
    x_last_pt, y_last_pt = s_pts[-1]
    lines.append(fmt(x_s_center + x_last_pt, y_last_pt, z_travel))

    # Zurück zur Parkposition
    lines.append("EXTRUDER_OFF")
    lines.append("LA 0.0 0.0 -300.0 0.0 0.0 0.0")

    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    generate()
    print(f'Extrusionstest generiert: {OUTPUT_PATH}')

    rect_output = 'output_geo_code/Rechteck_Treppe_test.geo'
    generate_rectangle_stairs(rect_output)
    print(f'Rechteck-Treppenmuster generiert: {rect_output}')
