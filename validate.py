#!/usr/bin/env python3
"""validate.py - printability report for the STLs in a folder."""
import sys, os, glob, json
import numpy as np, trimesh
folder = sys.argv[1] if len(sys.argv) > 1 else 'stl'
rows = []
for f in sorted(glob.glob(os.path.join(folder, '*.stl'))):
    m = trimesh.load(f, process=False)
    m.merge_vertices(digits_vertex=8)      # weld exactly like a slicer does
    nz = m.face_normals[:, 2]
    zc = m.triangles_center[:, 2]
    over = (nz < -0.707) & (zc > 0.1)
    # exclude magnet pockets (inside r<... near side walls at z<4): count only above z=4.2
    over_feat = over & (zc > 4.2)
    rows.append((os.path.basename(f)[:-4], bool(m.is_watertight), len(m.split(only_watertight=False)),
                 m.bounds[1][2], m.volume / 1000.0, float(m.area_faces[over_feat].sum())))
print(f"{'tile':34s} {'wt':3s} {'bodies':6s} {'height':>7s} {'cm3':>6s} {'overhang_mm2':>12s}")
bad = 0
for n, wt, nb, h, v, o in rows:
    flag = '' if (wt and nb == 1) else '  <-- CHECK'
    bad += bool(flag)
    print(f"{n:34s} {'ok' if wt else 'NO':3s} {nb:6d} {h:7.1f} {v:6.2f} {o:12.1f}{flag}")
print(f"\n{len(rows)} tiles, {bad} flagged, max height {max(r[3] for r in rows):.1f} mm")
