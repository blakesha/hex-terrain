#!/usr/bin/env python3
"""multicolour.py - split a tile into two STLs at a height, e.g. snow caps for multi-material printers.

  python3 multicolour.py stl/mountains.stl stl/mountains_b.stl --z 14 --out stl/multicolour

Writes <name>_body.stl and <name>_cap.stl. Both keep their original coordinates, so load them
together in the slicer and assign a different filament to each. The cut is a flat plane;
anything above --z becomes the cap."""
import argparse, os
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--z', type=float, default=14.0)
    ap.add_argument('--out', default='stl/multicolour')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.files:
        m = trimesh.load(f, process=False)
        tri = m.vertices.astype(np.float32)[m.faces]                    # STL stores unshared vertices: weld exactly
        u, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
        mf = Manifold(Mesh(vert_properties=u, tri_verts=inv.reshape(-1, 3).astype(np.uint32)))
        top, bot = mf.split_by_plane([0, 0, 1], a.z)
        n = os.path.splitext(os.path.basename(f))[0]
        for part, tag in ((bot, 'body'), (top, 'cap')):
            if part.is_empty():
                continue
            mm = part.to_mesh()
            trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts, process=False).export(os.path.join(a.out, f'{n}_{tag}.stl'))
        print(n, 'cap volume', round(top.volume(), 1), 'mm3')


if __name__ == '__main__':
    main()
