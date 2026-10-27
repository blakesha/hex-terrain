#!/usr/bin/env python3
"""plate.py - arrange tiles on a hex-packed print plate.

  python3 plate.py stl/plains.stl stl/hills.stl ... --bed 256 256 --gap 3 --out plates
  python3 plate.py "stl/road_*_plains.stl" --bed 220 220          (globs are expanded)

Tiles are packed on a hex lattice (rows offset by half a tile) and split over as many
plates as needed. Writes plates/plate_01.stl, ... and plate_01.txt listing what is where.
"""
import argparse, glob, os, sys
import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--bed', type=float, nargs=2, default=[256, 256], metavar=('X', 'Y'))
    ap.add_argument('--gap', type=float, default=3.0, help='mm between tiles')
    ap.add_argument('--margin', type=float, default=5.0, help='mm kept clear at the bed edge')
    ap.add_argument('--out', default='plates')
    a = ap.parse_args()
    files = []
    for f in a.files:
        files += sorted(glob.glob(f)) or [f]
    meshes = [trimesh.load(f, process=False) for f in files]
    w = max(m.bounds[1][0] - m.bounds[0][0] for m in meshes)          # width across flats (x)
    h = max(m.bounds[1][1] - m.bounds[0][1] for m in meshes)          # point to point (y)
    px, py = w + a.gap, 0.75 * h + a.gap * 0.87                       # lattice spacing
    bx, by = a.bed[0] - 2 * a.margin, a.bed[1] - 2 * a.margin
    cells = []
    row = 0
    while h + row * py <= by + 1e-6:
        off = (px / 2) if row % 2 else 0.0
        col = 0
        while w + off + col * px <= bx + 1e-6:
            cells.append((a.margin + w / 2 + off + col * px, a.margin + h / 2 + row * py))
            col += 1
        row += 1
    if not cells:
        sys.exit('bed too small for one tile')
    os.makedirs(a.out, exist_ok=True)
    per = len(cells)
    for pi in range(0, len(meshes), per):
        batch = list(range(pi, min(pi + per, len(meshes))))
        placed, lines = [], []
        for k, i in enumerate(batch):
            m = meshes[i].copy()
            ctr = (m.bounds[0] + m.bounds[1]) / 2
            m.apply_translation([cells[k][0] - ctr[0], cells[k][1] - ctr[1], -m.bounds[0][2]])
            placed.append(m)
            lines.append(f'{os.path.splitext(os.path.basename(files[i]))[0]:36s} x={cells[k][0]:6.1f} y={cells[k][1]:6.1f}')
        name = os.path.join(a.out, f'plate_{pi // per + 1:02d}')
        trimesh.util.concatenate(placed).export(name + '.stl')
        open(name + '.txt', 'w').write('\n'.join(lines) + '\n')
        print(f'{name}.stl  {len(batch)} tiles ({per} fit on a {a.bed[0]:.0f} x {a.bed[1]:.0f} bed)')


if __name__ == '__main__':
    main()
