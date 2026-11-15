#!/usr/bin/env python3
"""preview.py - software-rendered, material-shaded previews of STL tiles (no GPU needed).

   python3 preview.py out.png a.stl b.stl ...   [--cols 4] [--size 420] [--ssaa 2]
   python3 preview.py hero.png stl/mountains.stl --size 1200        # one big picture

If generate.py wrote a <name>.mat.npz next to the STL (it does), the renderer knows which part of the
tile is terrain, base, tree, roof, wall ... and shades each with a procedural material: grass, rock,
snow, sand, dirt road, water with reflections, stone courses, roof tiles, wood. Lighting is a soft-shadowed
sun plus sky light with ambient occlusion, tone mapped. Without the sidecar every face is shaded
as generic terrain, which still works for any STL."""
import os, argparse
import numpy as np
import trimesh
from PIL import Image, ImageDraw

MATERIALS = ['stone', 'terrain', 'base', 'conifer', 'broadleaf', 'roof', 'plaster', 'wood', 'rock', 'dark', 'thatch']
M = {n: i for i, n in enumerate(MATERIALS)}
SMOOTH = {M['terrain']: 1, M['conifer']: 2, M['broadleaf']: 2, M['rock']: 3}     # material -> smoothing group
SUN = np.array([-0.45, -0.60, 0.66]); SUN /= np.linalg.norm(SUN)
SUN_COL = np.array([1.0, 0.90, 0.74]) * 2.4
SKY_COL = np.array([0.50, 0.62, 0.85]) * 0.80
BOUNCE_COL = np.array([0.45, 0.38, 0.30]) * 0.42
WATER_LEVEL = 4.55                    # tiles sit on a 5 mm base; river beds and water tiles are at 4 mm


# ---------------------------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------------------------
def _hash(ix, iy, s):
    n = np.sin(ix * 127.1 + iy * 311.7 + s * 74.7) * 43758.5453
    return n - np.floor(n)


def vnoise(x, y, s=0):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy, s), _hash(ix + 1, iy, s)
    c, d = _hash(ix, iy + 1, s), _hash(ix + 1, iy + 1, s)
    return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v


def fbm(x, y, s=0, octaves=3):
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise(x, y, s + 13 * o)
        norm += amp
        x, y, amp = x * 2.03 + 17.1, y * 2.03 + 5.3, amp * 0.5
    return tot / norm


def vnoise3(x, y, z, s=0):
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    u, v, w = [t * t * (3 - 2 * t) for t in (fx, fy, fz)]
    def h(a, b, c):
        n = np.sin(a * 127.1 + b * 311.7 + c * 191.3 + s * 74.7) * 43758.5453
        return n - np.floor(n)
    def lerp(a, b, t):
        return a + (b - a) * t
    x00 = lerp(h(ix, iy, iz), h(ix + 1, iy, iz), u)
    x10 = lerp(h(ix, iy + 1, iz), h(ix + 1, iy + 1, iz), u)
    x01 = lerp(h(ix, iy, iz + 1), h(ix + 1, iy, iz + 1), u)
    x11 = lerp(h(ix, iy + 1, iz + 1), h(ix + 1, iy + 1, iz + 1), u)
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)


def fbm3(x, y, z, s=0, octaves=3):
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise3(x, y, z, s + 13 * o)
        norm += amp
        x, y, z, amp = x * 2.03 + 17.1, y * 2.03 + 5.3, z * 2.03 + 9.7, amp * 0.5
    return tot / norm


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t[:, None]


def col(*c):
    return np.array(c, np.float64)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
class Tile:
    def __init__(self, path, offset=(0.0, 0.0)):
        m = trimesh.load(path, process=False)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.V = np.asarray(m.vertices, np.float64) + np.array([offset[0], offset[1], 0.0])
        self.F = np.asarray(m.faces, np.int64)
        side = os.path.splitext(path)[0] + '.mat.npz'
        self.mat = None
        self.grid = None
        self.ground = 'grass'
        self.paved = False
        if os.path.exists(side):
            d = np.load(side)
            if len(d['mat']) == len(self.F):
                self.mat = d['mat'].astype(np.int64)
                if 'Z' in d.files:
                    from scipy.ndimage import distance_transform_edt
                    g = {k: d[k] for k in d.files if k not in ('mat', 'names', 'ground', 'water', 'paved')}
                    g['xs'] = g['xs'] + offset[0]
                    g['ys'] = g['ys'] + offset[1]
                    wet = g['Z'] < WATER_LEVEL
                    g['shore'] = (distance_transform_edt(wet) * float(g['xs'][1] - g['xs'][0])
                                  if wet.any() and not wet.all() else np.full(wet.shape, 9.0))
                    self.grid = g
                    self.ground = str(d['ground'])
                    self.paved = bool('paved' in d.files and d['paved'])
        if self.mat is None:                      # no sidecar: flat things at the bottom are base, the rest terrain
            c = self.V[self.F].mean(1)
            fn = np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
            fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
            self.mat = np.where((np.abs(fn[:, 2]) < 0.5) & (c[:, 2] < 4.9) | (c[:, 2] < 0.05), M['base'], M['terrain'])


def smooth_normals(V, F, mat):
    """per-face-vertex normals: smooth inside terrain / foliage / boulders (with a crease angle), flat elsewhere"""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)                              # area weighted
    fnu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    VN = np.repeat(fnu[:, None, :], 3, axis=1)
    grp = np.array([SMOOTH.get(int(k), 0) for k in range(len(MATERIALS))])[mat]
    for g in set(SMOOTH.values()):
        fi = np.where(grp == g)[0]
        if not len(fi):
            continue
        key = np.round(V[F[fi]].reshape(-1, 3) * 2000).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        acc = np.zeros((inv.max() + 1, 3))
        np.add.at(acc, inv, np.repeat(fn[fi], 3, axis=0))
        avg = acc[inv].reshape(-1, 3, 3)
        avg /= np.linalg.norm(avg, axis=2, keepdims=True) + 1e-12
        ok = (avg * fnu[fi][:, None, :]).sum(2) > 0.7             # crease angle ~45 deg
        VN[fi] = np.where(ok[..., None], avg, VN[fi])
    return VN, fnu


# ---------------------------------------------------------------------------------------------
# rasteriser (z-buffer with barycentrics)
# ---------------------------------------------------------------------------------------------
def raster(px, py, dp, F, W, H):
    zb = np.full((H, W), -1e30, np.float32)
    fid = np.full((H, W), -1, np.int32)
    B0 = np.zeros((H, W), np.float32)
    B1 = np.zeros((H, W), np.float32)
    X, Y, D = px[F], py[F], dp[F]
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    x0m = np.clip(np.floor(X.min(1)), 0, W - 1).astype(int).tolist()
    x1m = np.clip(np.ceil(X.max(1)), 0, W - 1).astype(int).tolist()
    y0m = np.clip(np.floor(Y.min(1)), 0, H - 1).astype(int).tolist()
    y1m = np.clip(np.ceil(Y.max(1)), 0, H - 1).astype(int).tolist()
    inside = ((X.max(1) >= 0) & (X.min(1) <= W) & (Y.max(1) >= 0) & (Y.min(1) <= H) & (np.abs(den) > 1e-9)).tolist()
    Xl, Yl, Dl, denl = X.tolist(), Y.tolist(), D.tolist(), den.tolist()
    for f in range(len(F)):
        if not inside[f]:
            continue
        xa, xb, ya, yb = x0m[f], x1m[f], y0m[f], y1m[f]
        (x0, x1, x2), (y0, y1, y2), (d0, d1, d2), dn = Xl[f], Yl[f], Dl[f], denl[f]
        gx = np.arange(xa, xb + 1)[None, :] + 0.5
        gy = np.arange(ya, yb + 1)[:, None] + 0.5
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / dn
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / dn
        l2 = 1 - l0 - l1
        m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not m.any():
            continue
        d = l0 * d0 + l1 * d1 + l2 * d2
        sub = zb[ya:yb + 1, xa:xb + 1]
        upd = m & (d > sub)
        if not upd.any():
            continue
        sub[upd] = d[upd]
        fid[ya:yb + 1, xa:xb + 1][upd] = f
        B0[ya:yb + 1, xa:xb + 1][upd] = np.broadcast_to(l0, upd.shape)[upd]
        B1[ya:yb + 1, xa:xb + 1][upd] = np.broadcast_to(l1, upd.shape)[upd]
    return zb, fid, B0, B1


class Occupancy:
    """Voxel grid of the solid (built by scan-converting the mesh column by column). Used for sun shadows
    and ambient occlusion, so overhangs (eaves, tree canopies, tower galleries) shade what is under them."""
    def __init__(self, V, F, res=0.15, pad=14.0):
        self.res = res
        self.o = np.array([V[:, 0].min() - pad, V[:, 1].min() - pad, -0.3])
        W = int((V[:, 0].max() + pad - self.o[0]) / res) + 1
        H = int((V[:, 1].max() + pad - self.o[1]) / res) + 1
        D = int((V[:, 2].max() + 1.0 - self.o[2]) / res) + 1
        self.shape = (D, H, W)
        X = (V[F, 0] - self.o[0]) / res
        Y = (V[F, 1] - self.o[1]) / res
        Z = V[F, 2]
        den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
        e1 = V[F[:, 1]] - V[F[:, 0]]
        e2 = V[F[:, 2]] - V[F[:, 0]]
        nzs = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]                 # sign: up-facing (+) / down-facing (-)
        x0 = np.floor(X.min(1)).astype(int).tolist(); x1 = np.ceil(X.max(1)).astype(int).tolist()
        y0 = np.floor(Y.min(1)).astype(int).tolist(); y1 = np.ceil(Y.max(1)).astype(int).tolist()
        Xl, Yl, Zl, dl, sg = X.tolist(), Y.tolist(), Z.tolist(), den.tolist(), np.sign(nzs).tolist()
        ii, jj, kk, ss = [], [], [], []
        for f in range(len(F)):
            if abs(dl[f]) < 1e-9 or sg[f] == 0:
                continue
            (xa, xb, xc), (ya, yb, yc), (za, zb_, zc), dn = Xl[f], Yl[f], Zl[f], dl[f]
            gx = np.arange(x0[f], x1[f] + 1)[None, :] + 0.5
            gy = np.arange(y0[f], y1[f] + 1)[:, None] + 0.5
            l0 = ((yb - yc) * (gx - xc) + (xc - xb) * (gy - yc)) / dn
            l1 = ((yc - ya) * (gx - xc) + (xa - xc) * (gy - yc)) / dn
            l2 = 1 - l0 - l1
            m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not m.any():
                continue
            z = l0 * za + l1 * zb_ + l2 * zc
            jj_, ii_ = np.nonzero(m)
            ii.append(ii_ + x0[f]); jj.append(jj_ + y0[f]); kk.append(np.rint((z[m] - self.o[2]) / res).astype(int))
            ss.append(np.full(len(ii_), sg[f], np.int8))
        ii, jj, kk, ss = map(np.concatenate, (ii, jj, kk, ss))
        ok = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < H) & (kk >= 0) & (kk < D)
        diff = np.zeros(self.shape, np.int8)
        np.add.at(diff, (kk[ok], jj[ok], ii[ok]), -ss[ok])             # up-facing surface ends the solid, down-facing starts it
        solid = np.cumsum(diff, axis=0, dtype=np.int16) > 0
        self.sharp = solid.ravel()
        from scipy.ndimage import gaussian_filter
        self.soft = gaussian_filter(solid.astype(np.float32), 3.0).astype(np.float16).ravel()

    def idx(self, Q):
        D, H, W = self.shape
        i = np.clip(((Q[:, 0] - self.o[0]) / self.res).astype(np.int64), 0, W - 1)
        j = np.clip(((Q[:, 1] - self.o[1]) / self.res).astype(np.int64), 0, H - 1)
        k = np.clip(((Q[:, 2] - self.o[2]) / self.res).astype(np.int64), 0, D - 1)
        return (k * H + j) * W + i

    def inside(self, Q):
        return self.sharp[self.idx(Q)]


def sun_visibility(P, occ):
    """soft sun shadow: sharp near the surface, the penumbra widens with distance (blurred voxels further out)"""
    vis = np.ones(len(P))
    t = 0.3
    while t < 45.0:
        i = occ.idx(P + SUN * t)
        o = occ.sharp[i].astype(np.float32) if t < 2.0 else occ.soft[i].astype(np.float32)
        vis = np.minimum(vis, 1.0 - sstep(0.15, 0.75, o))
        t *= 1.2
    return vis


def ambient_occlusion(P, N, occ, strength=1.0):
    """cosine-weighted hemisphere probes (random per-pixel twist, averaged away by supersampling)"""
    n = len(P)
    ref = np.where((np.abs(N[:, 2]) < 0.9)[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    tg = np.cross(N, ref)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    bt = np.cross(N, tg)
    twist = (_hash(np.arange(n) % 9973, np.arange(n) // 9973, 5.0)) * 2 * np.pi
    hit = np.zeros(n)
    wsum = 0.0
    for th, nphi in ((0.45, 3), (0.95, 5)):
        for a in range(nphi):
            phi = twist + 2 * np.pi * a / nphi
            d = (np.sin(th) * np.cos(phi))[:, None] * tg + (np.sin(th) * np.sin(phi))[:, None] * bt + np.cos(th) * N
            for s, w in ((0.35, 1.0), (0.8, 0.9), (1.5, 0.75), (2.7, 0.55), (4.5, 0.35)):
                hit += w * occ.inside(P + d * s)
                wsum += w
    return np.clip(1 - strength * 1.6 * hit / wsum, 0.04, 1)


# ---------------------------------------------------------------------------------------------
# materials: albedo (linear) and, for water, a reflectivity flag
# ---------------------------------------------------------------------------------------------
def tile_lookup(tiles, tid, P, which, nearest=False):
    """terrain height / road weight under points P, from the sidecar grids (NaN if unknown)"""
    out = np.full(len(P), np.nan)
    for k, t in enumerate(tiles):
        if t.grid is None:
            continue
        sel = tid == k
        if not sel.any():
            continue
        if which not in t.grid:
            continue
        xs, ys, g = t.grid['xs'], t.grid['ys'], t.grid[which]
        fx = np.clip((P[sel, 0] - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.001)
        fy = np.clip((P[sel, 1] - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        if nearest:
            out[sel] = g[np.rint(fy).astype(int).clip(0, len(ys) - 1), np.rint(fx).astype(int).clip(0, len(xs) - 1)]
            continue
        u, v = fx - i, fy - j
        out[sel] = (g[j, i] * (1 - u) + g[j, i + 1] * u) * (1 - v) + (g[j + 1, i] * (1 - u) + g[j + 1, i + 1] * u) * v
    return out


def shade_materials(P, N, mat, tid, tiles):
    """returns albedo (n,3), water mask, ripple-perturbed normals, shore distance"""
    n = len(P)
    alb = np.tile(col(0.5, 0.5, 0.5), (n, 1))
    water = np.zeros(n, bool)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    nz = N[:, 2]
    gnd = np.array([t.ground for t in tiles])[tid]
    gz = tile_lookup(tiles, tid, P, 'Z')
    road = tile_lookup(tiles, tid, P, 'road')
    shore = tile_lookup(tiles, tid, P, 'shore')
    urban = tile_lookup(tiles, tid, P, 'urban')
    fieldv = tile_lookup(tiles, tid, P, 'field', nearest=True)
    hedge = tile_lookup(tiles, tid, P, 'hedge')
    crev = tile_lookup(tiles, tid, P, 'crev')
    plaza = tile_lookup(tiles, tid, P, 'plaza')
    paved = np.array([t.paved for t in tiles])[tid]
    N = N.copy()

    # ---- terrain ---------------------------------------------------------------------------
    s = mat == M['terrain']
    if s.any():
        xs, ys, zs, nzs, g = x[s], y[s], z[s], nz[s], gnd[s]
        big = fbm(xs * 0.22, ys * 0.22, 3)                      # patches
        mid = fbm(xs * 0.9, ys * 0.9, 7)
        fine = fbm3(xs * 4.5, ys * 4.5, zs * 4.5, 11, 2)
        a = np.zeros((s.sum(), 3))
        grass = mix(col(0.09, 0.18, 0.05), col(0.19, 0.27, 0.075), sstep(0.3, 0.7, big))
        grass = mix(grass, col(0.33, 0.30, 0.11), sstep(0.62, 0.8, mid) * 0.5)   # dry patches
        sand = mix(col(0.40, 0.29, 0.16), col(0.53, 0.40, 0.23), sstep(0.3, 0.7, big))
        swamp = mix(col(0.06, 0.08, 0.035), col(0.14, 0.15, 0.06), sstep(0.3, 0.75, big))
        swamp = mix(swamp, col(0.12, 0.085, 0.05), sstep(0.55, 0.75, mid) * 0.8)       # mud
        forest = mix(col(0.07, 0.13, 0.04), col(0.15, 0.20, 0.07), sstep(0.3, 0.7, big))
        hills = mix(col(0.14, 0.23, 0.06), col(0.28, 0.32, 0.10), sstep(0.35, 0.65, big))
        tundra = mix(col(0.15, 0.145, 0.08), col(0.27, 0.25, 0.15), sstep(0.3, 0.7, big))
        tundra = mix(tundra, col(0.42, 0.40, 0.33), sstep(0.6, 0.8, mid) * 0.5)                   # lichen
        jungle = mix(col(0.035, 0.11, 0.03), col(0.09, 0.20, 0.05), sstep(0.3, 0.7, big))
        ice = mix(col(0.25, 0.40, 0.53), col(0.44, 0.57, 0.67), sstep(0.3, 0.75, big))
        ice = mix(ice, col(0.05, 0.17, 0.32), sstep(0.15, 0.7, np.nan_to_num(crev[s], nan=0.0)) * 0.9)     # crevasse blue
        fid = np.floor(np.nan_to_num(fieldv[s], nan=0.0))
        kd = np.rint((np.nan_to_num(fieldv[s], nan=0.0) - fid) * 10).astype(int) % 4
        crop = np.array([[0.20, 0.13, 0.075], [0.10, 0.22, 0.05], [0.13, 0.23, 0.065], [0.40, 0.31, 0.10]])[kd]
        crop = crop * (0.82 + 0.34 * _hash(fid, 0 * fid, 17.0))[:, None] * (0.85 + 0.3 * fine)[:, None]
        farm = np.where(np.isnan(fieldv[s])[:, None], grass, crop)
        hg = np.nan_to_num(hedge[s], nan=0.0)
        farm = mix(farm, mix(col(0.035, 0.09, 0.03), col(0.08, 0.17, 0.045), fine), sstep(0.35, 0.7, hg))
        ground = grass
        for nm, cc in (('sand', sand), ('swamp', swamp), ('forest', forest), ('hills', hills), ('tundra', tundra),
                       ('jungle', jungle), ('ice', ice), ('farmland', farm)):
            ground = np.where((g == nm)[:, None], cc, ground)
        ground = ground * (0.82 + 0.36 * fine)[:, None]
        # rock: steep faces, and altitude on mountains
        rel = zs - 5.0
        slope = 1 - nzs
        rock_amt = sstep(0.16, 0.34, slope) * np.where((g == 'ice') | (g == 'farmland'), 0.0, 1.0)
        rock_amt = np.maximum(rock_amt, np.where(g == 'mountain', sstep(2.5, 7.0, rel + 2.2 * (mid - 0.5)), 0.0))
        strata = fbm3(xs * 0.8, ys * 0.8, zs * 0.8, 21, 2)
        band = 0.5 + 0.5 * np.sin(zs * 5.0 + 3.0 * strata)
        rock = mix(col(0.17, 0.15, 0.13), col(0.34, 0.30, 0.26), 0.35 * band + 0.65 * fine)
        redrock = mix(col(0.46, 0.27, 0.18), col(0.62, 0.40, 0.26), 0.4 * band + 0.6 * fine)
        rock = np.where((g == 'sand')[:, None], redrock, rock)
        a = mix(ground, rock, rock_amt)
        # tundra snow patches
        tsn = sstep(0.60, 0.72, big + 0.25 * (mid - 0.5)) * sstep(0.35, 0.12, slope) * (g == 'tundra')
        a = mix(a, col(0.80, 0.83, 0.87) * (0.9 + 0.1 * fine)[:, None], np.clip(tsn, 0, 1))
        # snow
        snow_line = 15.6 + 2.5 * (mid - 0.5)
        snow = sstep(snow_line - 0.5, snow_line + 0.9, zs) * sstep(0.28, 0.12, slope) * (g == 'mountain')
        snow += sstep(snow_line + 2.0, snow_line + 3.2, zs) * (g == 'mountain') * 0.8
        a = mix(a, col(0.86, 0.88, 0.92), np.clip(snow, 0, 1))
        # dirt road
        rd = road[s]
        rd = np.where(np.isnan(rd), 0.0, rd)
        dirt = mix(col(0.40, 0.31, 0.20), col(0.52, 0.43, 0.30), fine)
        a = mix(a, dirt, sstep(0.35, 0.75, rd) * (1 - sstep(0.12, 0.3, slope)))
        # trodden earth in towns and yards, cobbles on paved streets
        ub = np.nan_to_num(urban[s], nan=0.0)
        earth = mix(col(0.20, 0.155, 0.10), col(0.30, 0.25, 0.17), fine)
        a = mix(a, earth, sstep(0.25, 0.8, ub) * (1 - sstep(0.2, 0.4, slope)) * 0.9)
        plz = np.nan_to_num(plaza[s], nan=0.0)
        pv = paved[s] & ((rd > 0.3) | (plz > 0.3))
        rd = np.maximum(rd, plz)
        if pv.any():
            cx, cy = xs[pv] * 2.6, ys[pv] * 2.6 + 0.5 * np.floor(xs[pv] * 2.6)
            fx_, fy_ = cx - np.floor(cx), cy - np.floor(cy)
            gap = np.maximum(sstep(0.12, 0.02, np.minimum(fx_, 1 - fx_)), sstep(0.12, 0.02, np.minimum(fy_, 1 - fy_)))
            cob = col(0.30, 0.285, 0.26) * (0.7 + 0.6 * _hash(np.floor(cx), np.floor(cy), 29.0))[:, None]
            cob = mix(cob, cob * 0.35, gap)
            a[pv] = mix(a[pv], cob, sstep(0.3, 0.6, rd[pv]))
        # wet banks: just above the water line
        lvl = zs
        wet = sstep(WATER_LEVEL + 0.55, WATER_LEVEL, lvl) * sstep(5.5, 5.0, zs)
        a = mix(a, a * 0.45 * np.array([1.0, 0.95, 0.9]), np.clip(wet, 0, 1) * 0.8)
        a = np.where((g == 'sand')[:, None] & (lvl < 5.12)[:, None], mix(a, col(0.55, 0.45, 0.30), np.clip(wet, 0, 1)), a)
        alb[s] = a
        # water: anything lying at (or below) the water level
        w = (zs < WATER_LEVEL)
        ws = np.where(s)[0][w]
        water[ws] = True
        # tiny surface bump on terrain
        e = 0.06
        zz = zs * 6.0
        b0 = fbm3(xs * 6.0, ys * 6.0, zz, 31, 2)
        Ns = N[s]
        Ns[:, 0] -= (fbm3((xs + e) * 6.0, ys * 6.0, zz, 31, 2) - b0) / e * 0.07
        Ns[:, 1] -= (fbm3(xs * 6.0, (ys + e) * 6.0, zz, 31, 2) - b0) / e * 0.07
        Ns[:, 2] -= (fbm3(xs * 6.0, ys * 6.0, (zs + e) * 6.0, 31, 2) - b0) / e * 0.07
        N[s] = Ns / np.linalg.norm(Ns, axis=1, keepdims=True)

    # ---- base slab -------------------------------------------------------------------------
    s = mat == M['base']
    if s.any():
        grain = 0.5 + 0.5 * np.sin((y[s] * 5.0 + z[s] * 0.7) + 4 * fbm(x[s] * 0.5, z[s] * 3.0, 41))
        alb[s] = mix(col(0.15, 0.085, 0.05), col(0.26, 0.15, 0.085), grain * 0.6 + 0.4 * fbm(x[s] * 2, y[s] * 2, 43))

    # ---- foliage ---------------------------------------------------------------------------
    for key, lo, hi in (('conifer', col(0.045, 0.13, 0.06), col(0.10, 0.25, 0.10)),
                        ('broadleaf', col(0.13, 0.26, 0.06), col(0.30, 0.42, 0.10))):
        s = mat == M[key]
        if not s.any():
            continue
        h = np.where(np.isnan(gz[s]), 5.0, gz[s])
        above = z[s] - h
        clump = fbm3(x[s] * 2.2, y[s] * 2.2, z[s] * 2.2, 51, 3)
        tint = fbm(x[s] * 0.35, y[s] * 0.35, 53)
        a = mix(lo, hi, np.clip(0.25 + 0.75 * clump, 0, 1))
        a = a * (0.85 + 0.3 * tint)[:, None]
        a = np.where((gnd[s] == 'jungle')[:, None], a * np.array([0.55, 0.78, 0.6]), a)
        trunk = sstep(1.35, 0.95, above)
        a = mix(a, mix(col(0.18, 0.11, 0.06), col(0.26, 0.17, 0.09), clump), trunk)
        alb[s] = a
        e = 0.05
        k = 7.0 if key == 'broadleaf' else 9.0
        b0 = fbm3(x[s] * k, y[s] * k, z[s] * k, 161, 2)
        Ns = N[s]
        for ax in range(3):
            d = [0.0, 0.0, 0.0]; d[ax] = e
            Ns[:, ax] -= (fbm3((x[s] + d[0]) * k, (y[s] + d[1]) * k, (z[s] + d[2]) * k, 161, 2) - b0) / e * 0.09
        N[s] = Ns / np.linalg.norm(Ns, axis=1, keepdims=True)

    s = mat == M['thatch']
    if s.any():
        st = fbm3(x[s] * 1.5, y[s] * 1.5, z[s] * 6.0, 171, 3)
        alb[s] = mix(col(0.30, 0.21, 0.07), col(0.58, 0.45, 0.17), st)
    s = mat == M['dark']
    if s.any():
        alb[s] = col(0.025, 0.02, 0.018) * (0.7 + 0.6 * fbm3(x[s] * 3, y[s] * 3, z[s] * 3, 151))[:, None]

    # ---- roofs, plaster, wood --------------------------------------------------------------
    s = mat == M['roof']
    if s.any():
        cell = _hash(np.floor(x[s] / 3.2), np.floor(y[s] / 3.2), 61)
        pal = np.array([[0.34, 0.14, 0.09], [0.19, 0.21, 0.25], [0.40, 0.31, 0.16], [0.28, 0.15, 0.10]])
        base = pal[(cell * 4).astype(int) % 4]
        rows = 0.80 + 0.20 * np.sin(z[s] * 17.0)
        shingle = 0.88 + 0.24 * vnoise(x[s] * 5 + z[s] * 3, y[s] * 5, 63)
        alb[s] = base * (rows * shingle)[:, None]
    s = mat == M['plaster']
    if s.any():
        cell = _hash(np.floor(x[s] / 3.2), np.floor(y[s] / 3.2), 71)
        pal = np.array([[0.72, 0.66, 0.52], [0.80, 0.76, 0.66], [0.68, 0.54, 0.34], [0.76, 0.70, 0.58]])
        base = pal[(cell * 4).astype(int) % 4]
        grime = 0.8 + 0.4 * fbm3(x[s] * 2.5, y[s] * 2.5, z[s] * 2.5, 73)
        near_g = np.where(np.isnan(gz[s]), 0.0, sstep(0.9, 0.0, z[s] - gz[s]))
        alb[s] = base * (grime * (1 - 0.35 * near_g))[:, None]
    s = mat == M['wood']
    if s.any():
        u = x[s] * 0.8 + y[s] * 1.3
        grain = 0.5 + 0.5 * np.sin(u * 22.0 + 5.0 * fbm3(x[s] * 3, y[s] * 3, z[s] * 3, 81))
        alb[s] = mix(col(0.20, 0.12, 0.06), col(0.38, 0.25, 0.13), grain)

    # ---- stone: coursed masonry on vertical faces, weathered grey elsewhere -------------------
    for key, lo, hi, blocks in (('stone', col(0.28, 0.265, 0.24), col(0.42, 0.39, 0.34), True),
                                ('rock', col(0.27, 0.25, 0.23), col(0.46, 0.42, 0.37), False)):
        s = mat == M[key]
        if not s.any():
            continue
        nn = N[s]
        hz = np.hypot(nn[:, 0], nn[:, 1]) + 1e-9
        u = (x[s] * -nn[:, 1] + y[s] * nn[:, 0]) / hz
        v = z[s]
        n1 = fbm3(x[s] * 1.7, y[s] * 1.7, z[s] * 1.7, 91, 3)
        a = mix(lo, hi, np.clip(n1 * 1.2, 0, 1))
        if blocks:
            course = v / 0.62
            row = np.floor(course)
            bu = u / 1.15 + 0.5 * (row % 2) + 0.31 * _hash(row, 0 * row, 93)
            fc, fb = course - row, bu - np.floor(bu)
            mortar = np.maximum(sstep(0.10, 0.02, fc), sstep(0.07, 0.01, fb)) * sstep(0.7, 0.45, np.abs(nn[:, 2]))
            blk = 0.82 + 0.3 * _hash(np.floor(bu), row, 95)
            a = a * blk[:, None]
            a = mix(a, a * 0.45, mortar)
        alb[s] = a

    return alb, water, N, shore


# ---------------------------------------------------------------------------------------------
# camera + shading
# ---------------------------------------------------------------------------------------------
def render_scene(tiles, size=420, az=-35, el=42, ssaa=2, span=25.0, exposure=0.85, shadows=True):
    S = size * ssaa
    V = np.vstack([t.V for t in tiles])
    off = np.cumsum([0] + [len(t.V) for t in tiles])
    F = np.vstack([t.F + o for t, o in zip(tiles, off[:-1])])
    mat = np.concatenate([t.mat for t in tiles])
    tfid = np.concatenate([np.full(len(t.F), k) for k, t in enumerate(tiles)])
    VN, fnu = smooth_normals(V, F, mat)

    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rm = Rx @ Rz
    w = V @ Rm.T
    sc = S * 0.46 / span
    px, py, dp = S / 2 + w[:, 0] * sc, S * 0.60 - w[:, 2] * sc, -w[:, 1]
    to_cam = Rm.T @ np.array([0.0, -1.0, 0.0])                    # world direction towards the camera
    front = fnu @ to_cam > -1e-6
    fidx = np.where(front)[0]
    zb, fid_l, B0, B1 = raster(px, py, dp, F[fidx], S, S)
    fid = np.where(fid_l >= 0, fidx[np.clip(fid_l, 0, None)], -1)
    cov = fid >= 0

    occ = Occupancy(V, F)
    ys_, xs_ = np.mgrid[0:S, 0:S]
    sx, sy = (xs_ + 0.5 - S / 2) / sc, (S * 0.60 - (ys_ + 0.5)) / sc
    img = np.zeros((S, S, 3))

    # ---- table surface under everything ----------------------------------------------------
    Mi = Rm.T
    t = -(Mi[2, 0] * sx + Mi[2, 2] * sy) / Mi[2, 1]
    fl = np.stack([Mi[0, 0] * sx + Mi[0, 1] * t + Mi[0, 2] * sy, Mi[1, 0] * sx + Mi[1, 1] * t + Mi[1, 2] * sy,
                   np.zeros_like(sx)], -1)
    fm = ~cov
    Pf = fl[fm]
    felt = 0.40 + 0.06 * fbm(Pf[:, 0] * 3.0, Pf[:, 1] * 3.0, 101, 3)
    alb_f = felt[:, None] * np.array([1.0, 0.97, 0.92])
    Nf = np.tile([0.0, 0.0, 1.0], (len(Pf), 1))
    Pf0 = Pf + np.array([0, 0, 0.3])
    vis_f = sun_visibility(Pf0, occ) if shadows else np.ones(len(Pf))
    ao_f = ambient_occlusion(Pf0, Nf, occ, 1.1) if shadows else np.ones(len(Pf))
    lit = alb_f * (SUN_COL * (SUN[2] * vis_f)[:, None] + (SKY_COL * 0.9 * ao_f[:, None]))
    r = np.hypot(sx[fm], sy[fm]) / span
    lit *= (1.0 - 0.42 * np.clip(r - 0.35, 0, 1) ** 1.3 * 1.4)[:, None]
    img[fm] = lit

    # ---- the tiles -------------------------------------------------------------------------
    f = fid[cov]
    b0, b1 = B0[cov].astype(np.float64), B1[cov].astype(np.float64)
    b2 = 1 - b0 - b1
    tri = F[f]
    P = V[tri[:, 0]] * b0[:, None] + V[tri[:, 1]] * b1[:, None] + V[tri[:, 2]] * b2[:, None]
    N = VN[f, 0] * b0[:, None] + VN[f, 1] * b1[:, None] + VN[f, 2] * b2[:, None]
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    mt, tid = mat[f], tfid[f]
    alb, water, N, shore = shade_materials(P, N, mt, tid, tiles)

    Pb = P + N * 0.30
    vis = sun_visibility(Pb, occ) if shadows else np.ones(len(P))
    ao = ambient_occlusion(Pb, N, occ) if shadows else np.ones(len(P))
    V_ = to_cam[None, :]
    # water: flatten the (carved) bank normals and add ripples
    if water.any():
        pw = P[water]
        rip = np.stack([fbm(pw[:, 0] * 1.6 + 0.4 * pw[:, 1], pw[:, 1] * 3.2, 121, 3) - 0.5,
                        fbm(pw[:, 0] * 3.2, pw[:, 1] * 1.6 - 0.4 * pw[:, 0], 123, 3) - 0.5], 1)
        Nw = N[water] * np.array([0.55, 0.55, 1.0])
        Nw[:, 0] += rip[:, 0] * 0.22
        Nw[:, 1] += rip[:, 1] * 0.22
        N[water] = Nw / np.linalg.norm(Nw, axis=1, keepdims=True)
    ndl = np.clip(N @ SUN, 0, 1)
    up = 0.5 + 0.5 * N[:, 2]
    amb = SKY_COL[None, :] * up[:, None] + BOUNCE_COL[None, :] * (1 - up)[:, None]
    ao_d = 0.35 + 0.65 * ao
    color = alb * (SUN_COL[None, :] * (ndl * vis * ao_d)[:, None] + amb * ao[:, None] * 1.0)
    # specular + reflections on water, a little sheen on roofs
    hv = SUN + to_cam
    hv /= np.linalg.norm(hv)
    ndh = np.clip(N @ hv, 0, 1)
    if water.any():
        vw = np.abs(N[water] @ to_cam)
        fres = 0.03 + 0.97 * (1 - vw) ** 5
        sd = np.where(np.isnan(shore[water]), 9.0, shore[water])
        shallow = sstep(3.2, 0.0, sd)                                    # lighter, greener towards the shore
        deep = mix(col(0.018, 0.075, 0.095), col(0.040, 0.16, 0.18), np.clip(fbm(P[water, 0] * 0.4, P[water, 1] * 0.4, 131), 0, 1))
        deep = mix(deep, col(0.10, 0.30, 0.27), shallow * 0.75)
        refl = SKY_COL * 1.35 * (0.6 + 0.4 * N[water][:, 2:3] ** 2)
        # shallow water lets the (light) bed show through; the bank AO darkens it a little
        bed = alb[water] * 0.5
        wc = deep * (SUN_COL * 0.18 * (ndl[water] * vis[water])[:, None] + 0.75 * amb[water]) + bed * 0.20 * amb[water]
        wc = wc * (1 - fres[:, None]) + refl * fres[:, None] * (0.45 + 0.55 * ao[water])[:, None]
        wc += (SUN_COL * (ndh[water] ** 220 * vis[water] * 3.0)[:, None]) * 0.9
        foam = sstep(0.55, 0.05, sd) * (0.55 + 0.45 * vnoise(P[water, 0] * 5.0, P[water, 1] * 5.0, 141))
        wc = mix(wc, col(0.55, 0.60, 0.58) * (0.5 + 0.5 * vis[water][:, None]), foam * 0.6)
        color[water] = wc
    sheen = (mt == M['roof']) | (mt == M['wood']) | ((mt == M['terrain']) & np.array([t.ground == 'ice' for t in tiles])[tid])
    color[sheen] += (SUN_COL[None, :] * (ndh[sheen] ** 30 * vis[sheen] * 0.07)[:, None])
    img[cov] = color

    # ---- tone map, gamma, resolve ----------------------------------------------------------
    x = img * exposure
    x = x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
    x = np.clip(x, 0, 1) ** (1 / 2.2)
    out = Image.fromarray((x * 255 + 0.5).astype(np.uint8))
    return out.resize((size, size), Image.LANCZOS) if ssaa > 1 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--size', type=int, default=420)
    ap.add_argument('--ssaa', type=int, default=2)
    ap.add_argument('--az', type=float, default=-35)
    ap.add_argument('--el', type=float, default=42)
    ap.add_argument('--span', type=float, default=25.0, help='half-width of the view in mm')
    ap.add_argument('--exposure', type=float, default=0.85)
    ap.add_argument('--no-shadows', action='store_true', help='skip shadows and ambient occlusion (faster)')
    ap.add_argument('--group', action='store_true', help='render the first four STLs side by side on the hex lattice, as one scene')
    a = ap.parse_args()
    kw = dict(size=a.size, az=a.az, el=a.el, ssaa=a.ssaa, span=a.span, exposure=a.exposure, shadows=not a.no_shadows)
    if a.group:
        cells = [(0, 0), (1, 0), (0, 1), (-1, 1)]
        tiles = [Tile(f, (40.0 * i + 20.0 * j, 34.641 * j)) for f, (i, j) in zip(a.files, cells)]
        cx = np.mean([t.V[:, 0].mean() for t in tiles]); cy = np.mean([t.V[:, 1].mean() for t in tiles])
        for t in tiles:
            t.V[:, 0] -= cx; t.V[:, 1] -= cy
            if t.grid is not None:
                t.grid['xs'] = t.grid['xs'] - cx; t.grid['ys'] = t.grid['ys'] - cy
        render_scene(tiles, **kw).save(a.out)
        print('wrote', a.out)
        return
    ims = []
    for f in a.files:
        im = render_scene([Tile(f)], **kw)
        ImageDraw.Draw(im).text((8, 6), os.path.splitext(os.path.basename(f))[0], fill=(235, 235, 235))
        ims.append(im)
    rows = (len(ims) + a.cols - 1) // a.cols
    sheet = Image.new('RGB', (min(a.cols, len(ims)) * a.size, rows * a.size))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % a.cols) * a.size, (i // a.cols) * a.size))
    sheet.save(a.out)
    print('wrote', a.out, sheet.size)


if __name__ == '__main__':
    main()
