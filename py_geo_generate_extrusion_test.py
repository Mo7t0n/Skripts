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
# ─────────────────────────────────────────────────────────────────────────────

def fmt(x, y, z, rx=0.0, ry=0.0, rz=0.0):
    return f"LA {x:.5f} {y:.5f} {z:.5f} {rx:.5f} {ry:.5f} {rz:.5f}"

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
    z_print  = -LINE_HEIGHT
    z_travel = -TRAVEL_HEIGHT
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
