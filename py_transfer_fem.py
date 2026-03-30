import os, re, sys, numpy as np

# Ein-/Ausgabe
INPUT_TET = r"output_tet/Kegel_v3.tet"
INPUT_STRESS_DIR = r"input_fem"
OUTPUT_TXT = r"output_fem/SR_Kegel_v3.txt"
USE_CORNER_CENTROIDS = True

Q = np.array([[1, 0, 0],
              [0, 0, 1],
              [0, -1, 0]], float)


def read_tet_ascii(p):
    import re, numpy as np
    L = [l.strip() for l in open(p, "r", errors="ignore") if l.strip() and not l.lstrip().startswith("#")]
    iv = next(i for i, s in enumerate(L) if re.search(r"\bvertices\b", s, re.I))
    it = next(i for i, s in enumerate(L) if re.search(r"\btets\b", s, re.I))
    nV = int(re.search(r"\d+", L[iv]).group());
    nT = int(re.search(r"\d+", L[it]).group());
    i = max(iv, it) + 1

    V = np.zeros((nV, 3), float)
    for k in range(nV):
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", L[i + k])
        V[k] = tuple(map(float, nums[:3]))
    i += nV

    T = np.zeros((nT, 4), int)
    for k in range(nT):
        ids = re.findall(r"-?\d+", L[i + k])
        if ids and ids[0] == '4': ids = ids[1:]  # Zeilen wie: "4 a b c d"
        T[k] = tuple(map(int, ids[:4]))
    if T.min() == 1: T -= 1
    return V, T


def _blk(pat, s):
    m = re.search(pat, s, re.S);
    if not m: raise RuntimeError("Block fehlt: " + pat)
    return m.group(1)


def parse_vtu_ascii_corners(p):
    s = open(p, "r", errors="ignore").read()
    if 'format="ascii"' not in s and "format='ascii'" not in s: raise RuntimeError(f"{p}: kein ASCII-VTU")
    pts = np.fromstring(_blk(r"<Points>.*?<DataArray[^>]*>(.*?)</DataArray>.*?</Points>", s), sep=" ",
                        dtype=float).reshape(-1, 3)
    cells = _blk(r"<Cells>(.*?)</Cells>", s)
    conn = np.fromstring(_blk(r'Name="connectivity"[^>]*>(.*?)</DataArray>', cells), sep=" ", dtype=int)
    offs = np.fromstring(_blk(r'Name="offsets"[^>]*>(.*?)</DataArray>', cells), sep=" ", dtype=int)
    types = np.fromstring(_blk(r'Name="types"[^>]*>(.*?)</DataArray>', cells), sep=" ", dtype=int)
    corners = [];
    st = 0
    for i, off in enumerate(offs):
        ids = conn[st:off];
        st = off
        if types[i] in (10, 24): corners.append(np.array(ids[:4], int))
    return pts, corners, s


def centroid_from_corners(pts, corners):
    C = np.empty((len(corners), 3), float)
    for i, ids in enumerate(corners): C[i] = pts[ids].mean(0)
    return C


def first_scalar_for_cells(vtu_text, corners):
    m = re.search(r"<CellData[^>]*>(.*?)</CellData>", vtu_text, re.S)
    if m:
        dm = re.search(r'<DataArray[^>]*>(.*?)</DataArray>', m.group(1), re.S)
        if dm:
            arr = np.fromstring(dm.group(1), sep=" ", dtype=float)
            if arr.size == len(corners): return arr
    m = re.search(r"<PointData[^>]*>(.*?)</PointData>", vtu_text, re.S)
    if not m: raise RuntimeError("keine CellData/PointData")
    dm = re.search(r'<DataArray[^>]*>(.*?)</DataArray>', m.group(1), re.S)
    if not dm: raise RuntimeError("PointData leer")
    pt = np.fromstring(dm.group(1), sep=" ", dtype=float)
    vals = np.empty(len(corners), float)
    for i, ids in enumerate(corners): vals[i] = pt[ids].mean()
    return vals


def apply_frame_Q(Q, t, X): return X @ Q.T + t


def nn_map(srcC, refC):
    idx = np.empty(len(refC), int)
    for i, y in enumerate(refC):
        d = srcC - y;
        idx[i] = int(np.argmin(np.einsum("ij,ij->i", d, d)))
    return idx


def reorder_scalar_to_reference(vtu_path, Q, t, refC):
    pts, corners, s = parse_vtu_ascii_corners(vtu_path)
    myC = centroid_from_corners(apply_frame_Q(Q, t, pts), corners)
    idx = nn_map(myC, refC)
    a = first_scalar_for_cells(s, corners)
    if a.shape[0] != myC.shape[0]: raise RuntimeError(f"{os.path.basename(vtu_path)}: Zellzahl mismatch")
    return a[idx]


def safe_load_scalar(path, Q, t, refC): return reorder_scalar_to_reference(path, Q, t, refC) if os.path.isfile(
    path) else None


def main():
    if not os.path.isfile(INPUT_TET): raise RuntimeError(f"fehlt: {INPUT_TET}")
    Vd, Td = read_tet_ascii(INPUT_TET);
    Nd = Td.shape[0];
    dstC = Vd[Td].mean(1)

    ref = os.path.join(INPUT_STRESS_DIR, "XX.vtu")
    if not os.path.isfile(ref): raise RuntimeError(f"fehlt: {ref}")
    Ps_ref, corners_ref, s_ref = parse_vtu_ascii_corners(ref)
    if not corners_ref: raise RuntimeError("keine Tetraeder")
    srcC0 = centroid_from_corners(Ps_ref, corners_ref) if USE_CORNER_CENTROIDS else Ps_ref

    def bbox_center(A):  # A: (N,3)
        return 0.5 * (A.min(0) + A.max(0))

    src_center = bbox_center(srcC0 @ Q.T)  # nach Achsentausch
    dst_center = bbox_center(dstC)  # Zielnetz
    t = dst_center - src_center

    Ps_ref_al = apply_frame_Q(Q, t, Ps_ref)
    srcC = centroid_from_corners(Ps_ref_al, corners_ref) if USE_CORNER_CENTROIDS else Ps_ref_al
    idxD2S = nn_map(srcC, dstC)

    def load(name):
        return safe_load_scalar(os.path.join(INPUT_STRESS_DIR, name), Q, t, srcC)

    s_xx, s_yy, s_zz, s_xy, s_yz, s_zx = load("XX.vtu"), load("YY.vtu"), load("ZZ.vtu"), load("XY.vtu"), load(
        "YZ.vtu"), load(
        "ZX.vtu")

    if not any(a is not None for a in (s_xx, s_yy, s_zz, s_xy, s_yz, s_zx)): raise RuntimeError("keine Komponenten")
    n = len(srcC);
    z = lambda a: a if a is not None else np.zeros(n)
    sxx, syy, szz, sxy, syz, sxz = z(s_xx), z(s_yy), z(s_zz), z(s_xy), z(s_yz), z(s_zx)

    sigma = np.stack([
        np.stack([sxx, sxy, sxz], 1),
        np.stack([sxy, syy, syz], 1),
        np.stack([sxz, syz, szz], 1),
    ], 1).astype(float)
    sigma_p = np.einsum("ij,njk,kl->nil", Q, sigma, Q.T)
    sxx_p, syy_p, szz_p = sigma_p[:, 0, 0], sigma_p[:, 1, 1], sigma_p[:, 2, 2]
    sxy_p, sxz_p, syz_p = sigma_p[:, 0, 1], sigma_p[:, 0, 2], sigma_p[:, 1, 2]

    sig_path = os.path.join(INPUT_STRESS_DIR, "sigma_max.vtu")
    if os.path.isfile(sig_path):
        smax = safe_load_scalar(sig_path, Q, t, srcC)
        if smax.shape[0] != n: raise RuntimeError("sigma_max: Zellzahl mismatch")
    else:
        smax = np.linalg.eigvalsh(sigma_p)[:, 2]

    sxx_d, syy_d, szz_d = sxx_p[idxD2S], syy_p[idxD2S], szz_p[idxD2S]
    sxy_d, sxz_d, syz_d = sxy_p[idxD2S], sxz_p[idxD2S], syz_p[idxD2S]
    smax_d = smax[idxD2S]

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, "w") as f:
        for i in range(Nd):
            f.write(
                f"{i},{smax_d[i]:.9g},{sxx_d[i]:.9g},{syy_d[i]:.9g},{szz_d[i]:.9g},{sxy_d[i]:.9g},{sxz_d[i]:.9g},{syz_d[i]:.9g}\n")

    A = np.vstack([sxx_p, syy_p, szz_p, sxy_p, sxz_p, syz_p]).T
    zeros = np.mean(np.isclose(A, 0, atol=1e-14), 1) == 1.0
    print("OK ->", OUTPUT_TXT, "| #dest:", Nd, "| #src:", n, "| det(Q):", round(float(np.linalg.det(Q)), 0), "| t", t,
          "| scalar:", ("file" if os.path.isfile(sig_path) else "eig"), "| #zero-tensors:",
          int(np.count_nonzero(zeros)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr);
        sys.exit(1)
