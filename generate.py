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
from hexlib import Params, MATERIALS
from tiles import build_tile
import trimesh


def finalize(m):
    """Weld exact float32 coordinates (as a slicer does), then discard any
    zero-volume fin shells left by boolean operations. Returns (mesh, dropped_faces).
    The per-face material tags (m.face_attributes['mat']) follow the surviving faces."""
    tri = m.vertices.astype(np.float32)[m.faces]
    u, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    w = trimesh.Trimesh(u.astype(np.float64), inv.reshape(-1, 3), process=False)
    mat = np.asarray(m.face_attributes['mat']) if 'mat' in m.face_attributes else None
    comps = trimesh.graph.connected_components(w.face_adjacency, nodes=np.arange(len(w.faces)), min_len=1)
    if len(comps) > 1:
        comps.sort(key=lambda ix: -len(ix))
        dropped = sum(len(ix) for ix in comps[1:])
        assert all(abs(w.submesh([ix], only_watertight=False, append=True).volume) < 0.5 for ix in comps[1:]), \
            'a real floating piece was found - not a fin'
        keep = np.sort(comps[0])
        w = trimesh.Trimesh(w.vertices, w.faces[keep], process=False)
        mat = mat[keep] if mat is not None else None
    else:
        dropped = 0
    if mat is not None:
        w.face_attributes['mat'] = mat
    w.metadata.update(m.metadata)
    return w, dropped


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
    cat['village_farm_0'] = dict(terrain='farmland', feature='village', seed=13)
    cat['village_farm_1'] = dict(terrain='farmland', feature='village', road=[0], seed=13)
    cat['village_farm_2'] = dict(terrain='farmland', feature='village', road=[0, 3], seed=14)
    cat['fort_1'] = dict(terrain='plains', feature='fort', road=[0], seed=31)
    cat['temple_0'] = dict(terrain='plains', feature='temple', seed=41)
    cat['temple_1'] = dict(terrain='plains', feature='temple', road=[0], seed=41)
    cat['ziggurat_0'] = dict(terrain='plains', feature='ziggurat', seed=42)
    cat['ziggurat_1'] = dict(terrain='plains', feature='ziggurat', road=[0], seed=42)
    cat['dungeon_1'] = dict(terrain='plains', feature='dungeon', road=[0], seed=51)
    # coast styles
    for n in (1, 2, 3, 4, 5):
        for t in ('plains', 'hills', 'forest'):
            cat[f'cliff_coast_{n}_{t}'] = dict(terrain=t, style='cliff', water=dict(kind='coast', edges=list(range(n))), seed=90 + n)
        cat[f'sand_coast_{n}'] = dict(terrain='desert', style='sand', water=dict(kind='coast', edges=list(range(n))), seed=100 + n)
        cat[f'marsh_coast_{n}'] = dict(terrain='swamp', water=dict(kind='coast', edges=list(range(n))), seed=110 + n)
    for n, rv in ((1, 3), (2, 4), (3, 4)):
        cat[f'cliff_coast_{n}_river_hills'] = dict(terrain='hills', style='cliff', river=[rv], water=dict(kind='coast', edges=list(range(n))), seed=90 + n)
    # islands
    for t in ('plains', 'hills', 'forest'):
        cat[f'island_{t}'] = dict(terrain=t, water=dict(kind='island'), seed=120)
    cat['island_lighthouse'] = dict(terrain='hills', feature='lighthouse', water=dict(kind='island'), seed=121)
    # waterfall and mountain passes / gorges
    cat['waterfall_mountains'] = dict(terrain='mountains', river=[0, 3], cascade=True, seed=4)
    for pat, ex in PATTERNS.items():
        cat[f'pass_{pat}_mountains'] = dict(terrain='mountains', road=ex, seed=3)
        cat[f'gorge_{pat}_mountains'] = dict(terrain='mountains', river=ex, seed=4)
    # mirror images of the one-handed forks
    for t in ('plains', 'hills', 'forest', 'desert'):
        cat[f'road_fork_t_{t}_mirror'] = dict(terrain=t, road=[0, 5, 3], seed=3)
    for t in ('plains', 'hills', 'forest'):
        cat[f'river_fork_t_{t}_mirror'] = dict(terrain=t, river=[0, 5, 3], seed=4)
    # --- terrains added to the original seven ---------------------------------------------------
    NEW = ('tundra', 'glacier', 'jungle', 'farmland', 'forest_hills', 'forest_mountains')
    for i, t in enumerate(NEW):
        cat[t] = dict(terrain=t, seed=1)
        cat[t + '_b'] = dict(terrain=t, seed=7)
    for t in ('swamp', 'tundra', 'jungle', 'farmland', 'forest_hills'):               # roads over the rest
        for pat, ex in PATTERNS.items():
            cat[f'road_{pat}_{t}'] = dict(terrain=t, road=ex, seed=3)
        cat[f'road_fork_t_{t}_mirror'] = dict(terrain=t, road=[0, 5, 3], seed=3)
    for t in ('desert', 'swamp', 'tundra', 'jungle', 'farmland', 'forest_hills'):     # rivers over the rest
        for pat, ex in PATTERNS.items():
            cat[f'river_{pat}_{t}'] = dict(terrain=t, river=ex, seed=4)
        cat[f'river_fork_t_{t}_mirror'] = dict(terrain=t, river=[0, 5, 3], seed=4)
    for pat in ('straight', 'curve_gentle', 'fork_y'):
        cat[f'road_{pat}_forest_mountains'] = dict(terrain='forest_mountains', road=PATTERNS[pat], seed=3)
        cat[f'river_{pat}_forest_mountains'] = dict(terrain='forest_mountains', river=PATTERNS[pat], seed=4)
    for n in (1, 2, 3, 4, 5):
        cat[f'coast_{n}_mountains'] = dict(terrain='mountains', water=dict(kind='coast', edges=list(range(n))), seed=60 + n)
    # mirrored coasts (the shore reflected about its axis) and river mouths on the other side
    for n in (2, 3, 4):
        for t in ('plains', 'hills', 'forest', 'mountains'):
            cat[f'coast_{n}_{t}_mirror'] = dict(terrain=t, water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=60 + n)
        for t in ('plains', 'hills', 'forest'):
            cat[f'cliff_coast_{n}_{t}_mirror'] = dict(terrain=t, style='cliff', water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=90 + n)
        cat[f'sand_coast_{n}_mirror'] = dict(terrain='desert', style='sand', water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=100 + n)
        cat[f'marsh_coast_{n}_mirror'] = dict(terrain='swamp', water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=110 + n)
    for n, rv in ((2, 3), (3, 4)):
        for t in ('plains', 'hills', 'forest'):
            cat[f'coast_{n}_river_{t}_mirror'] = dict(terrain=t, river=[rv], water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=60 + n)
        cat[f'cliff_coast_{n}_river_hills_mirror'] = dict(terrain='hills', style='cliff', river=[rv], water=dict(kind='coast', edges=list(range(n)), mirror=True), seed=90 + n)
    cat['ruins_plains'] = dict(terrain='plains', feature='ruins', seed=81)
    cat['ruins_forest'] = dict(terrain='forest', feature='ruins', seed=82)
    cat['ruins_road_plains'] = dict(terrain='plains', feature='ruins', road=[0, 3], seed=83)
    cat['mine_1'] = dict(terrain='plains', feature='mine', road=[0], seed=84)
    cat['stones_0'] = dict(terrain='plains', feature='stones', seed=85)
    cat['stones_2'] = dict(terrain='plains', feature='stones', road=[0, 3], seed=85)
    cat['watchtower_0'] = dict(terrain='plains', feature='watchtower', seed=86)
    cat['watchtower_1'] = dict(terrain='plains', feature='watchtower', road=[0], seed=86)
    cat['harbour_0'] = dict(terrain='plains', feature='harbour', water=dict(kind='coast', edges=[0, 1, 2]), seed=63)
    cat['harbour_1'] = dict(terrain='plains', feature='harbour', road=[4], water=dict(kind='coast', edges=[0, 1, 2]), seed=63)
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
        if 'mat' in m.face_attributes:         # materials + terrain grids for preview.py (the STL is unchanged)
            np.savez_compressed(os.path.join(a.out, n + '.mat.npz'), mat=m.face_attributes['mat'],
                                names=np.array(MATERIALS), **m.metadata.get('grid', {}))
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
