#!/usr/bin/env python3
"""
generate.py - build the hex tile STL library.

  python3 generate.py --set test        # a small representative set
  python3 generate.py --set all         # the full catalogue
  python3 generate.py --only river_curve_gentle_hills mountains

Options: --flat 40 --no-magnets --magnet-d 3 --magnet-t 1 --out stl
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hexlib import Params
from tiles import build_tile
import trimesh


def finalize(m):
    """Weld exact float32 coordinates (as a slicer does), then discard any
    zero-volume fin shells left by boolean operations. Returns (mesh, dropped_faces)."""
    tri = m.vertices.astype(np.float32)[m.faces]
    u, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    w = trimesh.Trimesh(u.astype(np.float64), inv.reshape(-1, 3), process=False)
    parts = w.split(only_watertight=False)
    if len(parts) == 1:
        return w, 0
    parts.sort(key=lambda p: -len(p.faces))
    dropped = sum(len(p.faces) for p in parts[1:])
    assert all(abs(p.volume) < 0.5 for p in parts[1:]), 'a real floating piece was found - not a fin'
    return parts[0], dropped


def weld_check(m):
    """weld exact float32 coordinates like a slicer and test the result"""
    tri = m.vertices.astype(np.float32)[m.faces]
    u, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    w = trimesh.Trimesh(u.astype(np.float64), inv.reshape(-1, 3), process=False)
    return bool(w.is_watertight and w.volume > 0), len(w.split(only_watertight=False))

PATTERNS = {
    'dead_end':     [0],
    'straight':     [0, 3],
    'curve_gentle': [0, 2],
    'curve_sharp':  [0, 1],
    'fork_y':       [0, 2, 4],
    'fork_t':       [0, 1, 3],      # mirror it in the slicer for the other hand
    'cross':        [0, 1, 3, 4],
}


def catalogue():
    cat = {}
    for t in ('plains', 'hills', 'forest', 'mountains', 'desert', 'swamp', 'water'):
        cat[t] = dict(terrain=t, seed=1)
    for t in ('hills', 'forest', 'mountains'):
        cat[t + '_b'] = dict(terrain=t, seed=7)
    for pat, ex in PATTERNS.items():
        for t in ('plains', 'hills', 'forest', 'desert'):
            cat[f'road_{pat}_{t}'] = dict(terrain=t, road=ex, seed=3)
        for t in ('plains', 'hills', 'forest'):
            cat[f'river_{pat}_{t}'] = dict(terrain=t, river=ex, seed=4)
    cat['bridge_plains'] = dict(terrain='plains', river=[0, 3], road=[1, 4], bridge=True, seed=4)   # same river as river_straight_plains
    cat['ford_plains'] = dict(terrain='plains', river=[0, 3], road=[1, 4], ford=True, seed=4)      # same river as river_straight_plains
    # coasts: water on the first n edges (0..n-1); land on the rest.  The same tiles are the
    # shores of lakes - surround water tiles with them.
    for n in (1, 2, 3, 4, 5):
        for t in ('plains', 'hills', 'forest'):
            cat[f'coast_{n}_{t}'] = dict(terrain=t, water=dict(kind='coast', edges=list(range(n))), seed=60 + n)
    for n, rv in ((1, 3), (2, 4), (3, 4)):                       # river mouths on the land edge facing the sea
        for t in ('plains', 'hills', 'forest'):
            cat[f'coast_{n}_river_{t}'] = dict(terrain=t, river=[rv], water=dict(kind='coast', edges=list(range(n))), seed=60 + n)
    # inland lakes: a pond in the middle of the tile, with rivers running through it
    for t in ('plains', 'forest'):
        cat[f'lake_pond_{t}'] = dict(terrain=t, water=dict(kind='pond'), seed=70)
        for pat, ex in (('1', [0]), ('straight', [0, 3]), ('curve_gentle', [0, 2]), ('curve_sharp', [0, 1]), ('fork_y', [0, 2, 4])):
            cat[f'lake_river_{pat}_{t}'] = dict(terrain=t, river=ex, water=dict(kind='pond'), seed=70)
    cat['village_0'] = dict(terrain='plains', feature='village', seed=11)
    cat['village_1'] = dict(terrain='plains', feature='village', road=[0], seed=11)
    cat['village_2'] = dict(terrain='plains', feature='village', road=[0, 3], seed=12)
    cat['city_1'] = dict(terrain='plains', feature='city', road=[0], seed=21)
    cat['city_2'] = dict(terrain='plains', feature='city', road=[0, 3], seed=22)
    cat['city_3'] = dict(terrain='plains', feature='city', road=[0, 2, 4], seed=23)
    cat['fort_1'] = dict(terrain='plains', feature='fort', road=[0], seed=31)
    cat['temple_0'] = dict(terrain='plains', feature='temple', seed=41)
    cat['temple_1'] = dict(terrain='plains', feature='temple', road=[0], seed=41)
    cat['ziggurat_0'] = dict(terrain='plains', feature='ziggurat', seed=42)
    cat['ziggurat_1'] = dict(terrain='plains', feature='ziggurat', road=[0], seed=42)
    cat['dungeon_1'] = dict(terrain='plains', feature='dungeon', road=[0], seed=51)
    return cat


TEST = ['plains', 'hills', 'mountains', 'forest', 'water', 'road_curve_gentle_forest',
        'river_curve_sharp_hills', 'river_fork_y_plains', 'bridge_plains', 'ford_plains',
        'village_2', 'city_2', 'fort_1', 'temple_1', 'ziggurat_1', 'dungeon_1',
        'coast_3_plains', 'coast_1_hills', 'coast_3_river_plains', 'lake_river_straight_plains']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='test', choices=['test', 'all'])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default='stl')
    ap.add_argument('--flat', type=float, default=40.0)
    ap.add_argument('--no-magnets', action='store_true')
    ap.add_argument('--magnets-per-edge', type=int, default=2, choices=[1, 2])
    ap.add_argument('--magnet-d', type=float, default=3.0)
    ap.add_argument('--magnet-t', type=float, default=1.0)
    ap.add_argument('--res', type=float, default=0.3)
    ap.add_argument('--organic', type=float, default=1.0, help='meander of roads/rivers, 0 = geometric')
    a = ap.parse_args()

    p = Params(flat=a.flat, magnets=not a.no_magnets, magnet_d=a.magnet_d,
               magnet_t=a.magnet_t, res=a.res, magnets_per_edge=a.magnets_per_edge, organic=a.organic)
    cat = catalogue()
    names = a.only if a.only else (TEST if a.set == 'test' else list(cat))
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for n in names:
        t0 = time.time()
        m, dropped = finalize(build_tile(p=p, **cat[n]))
        path = os.path.join(a.out, n + '.stl')
        m.export(path)
        ok, bodies = weld_check(m)
        ok = ok and bodies == 1
        report[n] = dict(watertight=ok, volume_mm3=round(float(m.volume), 1),
                         height_mm=round(float(m.bounds[1][2]), 2), tris=len(m.faces),
                         seconds=round(time.time() - t0, 1))
        print(f"{n:32s} {'OK ' if ok else 'BAD'} h={report[n]['height_mm']:5.1f}mm "
              f"tris={len(m.faces):7d} {report[n]['seconds']}s", flush=True)
    json.dump(report, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
