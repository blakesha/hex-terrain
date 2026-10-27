"""
tiles.py - terrain, roads/rivers, feature structures and the tile builder.
"""
import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from manifold3d import Manifold, CrossSection

from hexlib import *


# ==========================================================================
# build context
# ==========================================================================
class Ctx:
    def __init__(self, p, seed):
        self.p, self.seed = p, seed
        self.rng = np.random.RandomState(seed)
        self.a = p.a
        R = self.a / np.cos(30 * DEG) + 1.0
        n = int(np.ceil(2 * R / p.res)) + 1
        self.xs = np.linspace(-R, R, n)
        self.ys = np.linspace(-R, R, n)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.hd = hexdist(self.X, self.Y)
        self.mask = smoothstep((self.a - p.rim - self.hd) / p.fade)
        self.road = None
        self.river = None
        self.items = []            # occupied circles (x, y, r)
        self.Z = None
        self._ground = None
        self.water_s = None        # signed distance to a shoreline (negative = open water)
        self._ws = None

    def s_at(self, x, y):
        if self._ws is None:
            self._ws = RegularGridInterpolator((self.ys, self.xs), self.water_s, bounds_error=False, fill_value=1e3)
        return float(self._ws([[y, x]])[0])

    def ground(self, x, y):
        if self._ground is None:
            self._ground = RegularGridInterpolator((self.ys, self.xs), self.Z,
                                                   bounds_error=False, fill_value=None)
        return float(self._ground([[y, x]])[0])

    def free(self, x, y, r, margin=1.6, gap=0.3):
        p = self.p
        if hexdist(np.array(x), np.array(y)) + r > self.a - margin:
            return False
        if self.road is not None:
            d, _ = self.road.query(np.array([x]), np.array([y]))
            if d[0] < p.road_w / 2 + r + 0.3:
                return False
        if self.river is not None:
            d, _ = self.river.query(np.array([x]), np.array([y]))
            if d[0] < p.river_w / 2 + 1.2 + r:
                return False
        if self.water_s is not None and self.s_at(x, y) < r + 1.2:
            return False
        for (ox, oy, orad) in self.items:
            if np.hypot(x - ox, y - oy) < r + orad + gap:
                return False
        return True

    def claim(self, x, y, r):
        self.items.append((x, y, r))


# ==========================================================================
# terrain (returns height above base level, and a level offset that is NOT
# faded at the rim - used only by water)
# ==========================================================================
def t_plains(c):
    n = 0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 1, 3, 0.07)
    return 0.7 * n, 0.0


def t_hills(c):
    n = 0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 2, 3, 0.04, gain=0.4)
    n2 = 0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 3, 2, 0.12, gain=0.4)
    h = 7.5 * smoothstep((n - 0.25) / 0.6) + 0.25 * n2
    return h, 0.0


def t_mountains(c):
    rng = c.rng
    r = ridged(c.X, c.Y, c.seed + 4, 4, 0.06)
    n = 0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 5, 3, 0.1)
    env = np.zeros_like(c.X)
    for k in range(3):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 8)
        cx, cy = rad * np.cos(ang), rad * np.sin(ang)
        s = rng.uniform(6.5, 9.5)
        amp = rng.uniform(0.65, 1.0) if k else 1.0
        env = np.maximum(env, amp * np.exp(-((c.X - cx) ** 2 + (c.Y - cy) ** 2) / (2 * s * s)))
    h = 19.0 * env * (0.30 + 0.70 * r) + 2.5 * env * n
    return h, 0.0


def t_desert(c):
    rng = c.rng
    th = rng.uniform(0, np.pi)
    warp = 2.5 * fbm(c.X, c.Y, c.seed + 6, 2, 0.05)
    u = (c.X * np.cos(th) + c.Y * np.sin(th)) * 0.45 + warp
    saw = (u / (2 * np.pi)) % 1.0
    dune = np.where(saw < 0.7, saw / 0.7, (1 - saw) / 0.3)      # gentle windward, steep lee
    amp = 0.4 + 1.8 * (0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 7, 2, 0.05))
    return amp * smoothstep(dune), 0.0


def t_swamp(c):
    n = 0.5 + 0.5 * fbm(c.X, c.Y, c.seed + 8, 3, 0.12)
    pools = fbm(c.X, c.Y, c.seed + 9, 2, 0.08)
    h = 1.6 * n ** 2 - 1.0 * smoothstep((-0.05 - pools) / 0.25)
    return h, 0.0


def t_water(c):
    """Wavy crest lines from lattice-periodic waves (wave vectors from the hex reciprocal lattice).
    The phases are fixed, not per tile, so every water tile carries the same pattern and it
    runs unbroken across the shared edge of neighbouring tiles. Troughs sit exactly at the
    river-floor level, so rivers and lakes meet cleanly."""
    F, s3 = c.p.flat, np.sqrt(3.0)
    b1 = np.array([1.0, -1.0 / s3]) / F
    b2 = np.array([0.0, 2.0 / s3]) / F
    rng = np.random.RandomState(20261027)
    s = np.zeros_like(c.X)
    for (m, n), w in (((2, 1), 1.0), ((3, 0), 0.9), ((3, 1), 0.8), ((4, 0), 0.6), ((3, 2), 0.5)):
        k = 2 * np.pi * (m * b1 + n * b2)
        for r in range(3):                       # k and -k are the same wave
            th = r * 60 * DEG
            kx = k[0] * np.cos(th) - k[1] * np.sin(th)
            ky = k[0] * np.sin(th) + k[1] * np.cos(th)
            s += w * np.cos(kx * c.X + ky * c.Y + rng.uniform(0, 2 * np.pi))
    s = s / 1.5                                   # ~unit spread (fixed constant: identical on every tile)
    return 0.35 * np.exp(-(s / 0.5) ** 2), -c.p.river_depth


TERRAINS = dict(plains=t_plains, hills=t_hills, forest=lambda c: tuple([0.45 * t_hills(c)[0] + 0.3, 0.0]),
                mountains=t_mountains, desert=t_desert, swamp=t_swamp, water=t_water)


# ==========================================================================
# trees
# ==========================================================================
def scatter_trees(c, n, p_conifer=0.7, rmin=1.5, rmax=2.9, area=17.5, tries=400):
    rng, parts = c.rng, []
    placed = 0
    for _ in range(tries):
        if placed >= n:
            break
        x, y = rng.uniform(-area, area, 2)
        R = rng.uniform(rmin, rmax)
        if not c.free(x, y, R * 0.8, gap=-0.5):
            continue
        z = c.ground(x, y) - 0.5
        H = rng.uniform(6.0, 10.5) * (0.75 + 0.25 * R / rmax)
        if rng.rand() < p_conifer:
            t = conifer(rng, H, R, tiers=int(rng.choice([3, 4])))
        else:
            t = broadleaf(rng, H * 0.85, R * 0.95)
        parts.append(t.rotate([0, 0, rng.uniform(0, 360)]).translate([x, y, z]))
        c.claim(x, y, R * 0.8)
        placed += 1
    return parts


def f_forest(c):
    return scatter_trees(c, 26)


# ==========================================================================
# houses and villages
# ==========================================================================
def house(x, y, z, w, d, wh, rh, ang, pyramid=False):
    walls = box(0, 0, -1.0, w, d, wh + 1.0)
    door = box(0, -d / 2 + 0.1, 0.1, 0.9, 0.6, 1.4)
    walls = walls - door
    eave = 0.35
    if pyramid:
        roof = pyramid_roof(w + 2 * eave, d + 2 * eave, rh)
    elif w >= d:
        roof = gable_roof(d, w + 2 * eave, rh, eave).rotate([0, 0, 90])
    else:
        roof = gable_roof(w, d + 2 * eave, rh, eave)
    m = walls + roof.translate([0, 0, wh - 0.05])
    return m.rotate([0, 0, ang]).translate([x, y, z])


def pack_houses(c, n, size=(3.0, 4.6, 2.4, 3.6), wh=(2.0, 2.8), rh=(1.4, 2.2),
                radius=14.0, align_road=True, tries=700, extra_free=lambda x, y: True):
    rng, parts = c.rng, []
    for _ in range(tries):
        if len(parts) >= n:
            break
        x, y = rng.uniform(-radius, radius, 2)
        w, d = rng.uniform(size[0], size[1]), rng.uniform(size[2], size[3])
        r = 0.5 * np.hypot(w, d) + 0.1
        if not extra_free(x, y) or not c.free(x, y, r):
            continue
        if align_road and c.road is not None:
            _, ta = c.road.query(np.array([x]), np.array([y]))
            ang = np.degrees(ta[0]) + rng.uniform(-8, 8)
            # door faces the road: keep door (-y local) on the road side
            dx, dy = x, y
            rx, ry = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            side = dx * (-ry) + dy * rx
            if side > 0:
                ang += 180
        else:
            ang = np.degrees(np.arctan2(y, x)) + 90 + rng.uniform(-10, 10)
        z = c.ground(x, y)
        parts.append(house(x, y, z, w, d, rng.uniform(*wh), rng.uniform(*rh), ang,
                           pyramid=(abs(w - d) < 0.5 and rng.rand() < 0.5)))
        c.claim(x, y, r)
    return parts


def f_village(c):
    parts = pack_houses(c, 8)
    parts += scatter_trees(c, 8, p_conifer=0.3, rmin=1.4, rmax=2.2)
    return parts


# ==========================================================================
# city
# ==========================================================================
def tower(x, y, z, r, h, roof_h, zlow=None):
    zb = z - 0.5 if zlow is None else zlow
    return cyl(x, y, zb, r, z + h - zb, seg=14) + cyl(x, y, z + h - 0.05, r + 0.3, roof_h + 0.05, r2=0.05, seg=14)


def wall_ring(inner_a, thick, height, gates, gate_w=5.0, z0=0.0):
    ring = hex_prism(inner_a + thick, height + 1, z0 - 1.0) - hex_prism(inner_a, height + 3, z0 - 2.0)
    for e in gates:
        g = box(inner_a + thick / 2, 0, z0 - 2.0, thick + 2.0, gate_w, height + 6).rotate([0, 0, 60 * e])
        ring = ring - g
    return ring


def f_city(c):
    wall_a, th, wh = 15.0, 1.1, 3.6
    gates = c.exits
    parts = [wall_ring(wall_a, th, wh, gates, z0=c.ground(0, 0))]
    zc = c.ground(0, 0)
    vr = (wall_a + th / 2) / np.cos(30 * DEG)
    for k in range(6):
        ang = (30 + 60 * k) * DEG
        parts.append(tower(vr * np.cos(ang), vr * np.sin(ang), zc, 1.5, 7.0, 2.8))
    for e in gates:                      # gate-house towers flank each gate
        for s in (-1, 1):
            ang = 60 * e * DEG
            gx = (wall_a + th / 2) * np.cos(ang) - s * 3.0 * np.sin(ang)
            gy = (wall_a + th / 2) * np.sin(ang) + s * 3.0 * np.cos(ang)
            parts.append(tower(gx, gy, zc, 1.1, 5.6, 2.0))
    inner = lambda x, y: hexdist(np.array(x), np.array(y)) < wall_a - 1.4
    # a minster / keep somewhere near the middle
    rng = c.rng
    for _ in range(80):
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(4.5, 8)
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        if c.free(x, y, 3.6):
            a0 = np.degrees(ang) + 90
            nave = box(0, 0, -1.0, 7.0, 3.4, 5.0) + gable_roof(3.4, 7.8, 2.0, 0.4).rotate([0, 0, 90]).translate([0, 0, 4.0 - 0.05])
            spire = box(-3.0, 0, -1.0, 2.2, 2.2, 9.0) + pyramid_roof(3.0, 3.0, 3.6).translate([-3.0, 0, 8.0 - 0.05])
            parts.append((nave + spire).rotate([0, 0, a0]).translate([x, y, c.ground(x, y)]))
            c.claim(x, y, 4.2)
            break
    parts += pack_houses(c, 24, size=(2.3, 3.4, 2.1, 3.0), wh=(1.8, 3.0), rh=(1.2, 1.8),
                         radius=wall_a, tries=1500, extra_free=inner)
    return parts


# ==========================================================================
# fort
# ==========================================================================
def merlons(x0, y0, x1, y1, z, thick, pitch=1.8, mw=0.9, mh=0.7):
    L = np.hypot(x1 - x0, y1 - y0)
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    out = []
    for s in np.arange(mw, L - mw / 2, pitch):
        t = s / L
        out.append(box(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z, mw, thick, mh, ang))
    return out


def h_fort(c):
    d = np.hypot(c.X, c.Y)
    return 3.0 * smoothstep((14.0 - d) / 7.0)


def f_fort(c):
    z = c.ground(0, 0)
    zl = c.p.base - 0.5
    half, th, wh = 7.4, 1.2, 4.0
    parts = []
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    gate_w = 4.4
    for i in range(4):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        L = np.hypot(x1 - x0, y1 - y0)
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if 'gate' in c.fort_flags and x0 == half and x1 == half:      # +x face gets the gate
            seg = (L - gate_w) / 2
            for sgn in (-1, 1):
                off = sgn * (gate_w / 2 + seg / 2)
                parts.append(box(mx, my + off, zl, th, seg, z + wh - zl))
                parts += merlons(half, my + off - seg / 2, half, my + off + seg / 2, z + wh, th)
        else:
            parts.append(box(mx, my, zl, L, th, z + wh - zl, ang))
            parts += merlons(x0, y0, x1, y1, z + wh, th)
    for (x, y) in corners:
        parts.append(tower(x, y, z, 2.0, 7.5, 3.0, zlow=zl))
    keep = box(-2.0, 0, z - 1, 6.0, 6.0, 9.0) + box(-2.0, 0, z + 8.0, 6.6, 6.6, 0.0 + 0.5) \
        + pyramid_roof(6.0, 6.0, 3.2).translate([-2.0, 0, z + 8.5 - 0.05])
    parts.append(keep)
    parts += [tower(-4.4, 2.6, z, 1.0, 11.2, 1.8)]
    parts += scatter_trees(c, 5, p_conifer=0.4, rmin=1.4, rmax=2.2, area=17.5)
    return parts


# ==========================================================================
# temples
# ==========================================================================
def f_temple(c):
    z = c.ground(0, 0)
    parts = []
    steps = [(17.0, 12.0), (15.2, 10.2), (13.4, 8.4)]
    for i, (sx, sy) in enumerate(steps):
        parts.append(box(0, 0, z - 1, sx, sy, 1.0 + 0.8 * (i + 1)))
    top = z + 2.4
    colH = 5.0
    xs = np.linspace(-5.8, 5.8, 6)
    ys = np.linspace(-3.4, 3.4, 4)
    pos = [(x, y) for x in xs for y in (-3.4, 3.4)] + [(x, y) for x in (-5.8, 5.8) for y in ys[1:-1]]
    for (x, y) in pos:
        parts.append(cyl(x, y, top - 0.1, 0.55, colH + 0.2, seg=12))
    parts.append(box(0, 0, top, 9.2, 4.0, colH + 0.1))                     # cella
    parts.append(box(0, 0, top + colH, 12.8, 8.0, 0.7))                    # entablature
    ped = gable_roof(7.6, 12.4, 2.2, 0.0).rotate([0, 0, 90]).translate([0, 0, top + colH + 0.7 - 0.05])
    parts.append(ped)
    parts.append(box(4.65, 0, top, 0.5, 1.6, 2.6))                         # door block (relief)
    c.claim(0, 0, 10.2)
    parts += scatter_trees(c, 9, p_conifer=0.35, rmin=1.4, rmax=2.3)
    return parts


def f_ziggurat(c):
    z = c.ground(0, 0)
    parts = []
    sizes, th = [16.0, 12.8, 9.6, 6.4], 2.2
    for i, s in enumerate(sizes):
        parts.append(box(0, 0, z - 1, s, s, (i + 1) * th))
    top = z - 1 + 4 * th
    parts.append(box(0, 0, top, 3.0, 3.0, 1.8) + gable_roof(3.0, 3.6, 1.2, 0.3).translate([0, 0, top + 1.8 - 0.05]))
    n, st = 16, 0.55
    pts = [(12.0, 0.0)]
    for k in range(n):
        pts += [(12.0 - k * st, (k + 1) * st), (12.0 - (k + 1) * st, (k + 1) * st)]
    pts += [(12.0 - n * st, 0.0)]
    stair = Manifold.extrude(CrossSection([np.array(pts)]), 2.4).rotate([90, 0, 0]).translate([0, 1.2, z - 1])
    parts.append(stair)
    c.claim(0, 0, 10.5)
    parts += scatter_trees(c, 6, p_conifer=0.2, rmin=1.4, rmax=2.0)
    return parts


# ==========================================================================
# dungeon entrance (faces edge 0)
# ==========================================================================
XF = 4.0          # x of the facade


def h_dungeon(c):
    d = np.hypot(c.X, c.Y)
    plat = 7.8 * smoothstep((12.5 - d) / 6.5)
    r = ridged(c.X, c.Y, c.seed + 20, 4, 0.12)
    rough = 4.0 * smoothstep((13.0 - d) / 8.0) * r
    r2 = ridged(c.X, c.Y, c.seed + 21, 3, 0.09)
    return plat * (0.78 + 0.42 * r2) + 0.8 * rough


def f_dungeon(c):
    p = c.p
    zl = p.base - 0.5
    floor = p.base + p.road_h
    parts = []
    facade = box(XF + 0.5, 0, zl, 3.2, 8.0, floor + 7.2 - zl) + box(XF + 0.5, 0, floor + 7.2 - 0.05, 3.5, 8.4, 0.8 + 0.05)
    w, hsq = 3.6, 2.6
    tun = box(XF - 1.7, 0, floor, 9.0, w, hsq)
    top = cyl(0, 0, -4.5, w / 2, 9.0, seg=24).rotate([0, 90, 0]).translate([XF - 1.7, 0, floor + hsq])
    cut = tun + top
    c.carve = cut
    parts.append(facade - cut)
    c.claim(0.0, 0.0, 11.0)
    rng = c.rng
    for _ in range(9):
        x, y = rng.uniform(XF + 3.0, XF + 9.0), rng.uniform(-6, 6)
        if c.free(x, y, 0.8, gap=-0.3):
            parts.append(rock(rng, x, y, c.ground(x, y) - 0.1, rng.uniform(0.6, 1.1)))
            c.claim(x, y, 0.8)
    parts += scatter_trees(c, 4, p_conifer=0.6, rmin=1.3, rmax=2.0, area=17)
    return parts



# ==========================================================================
# harbour (coast tile, water on edges 0-2, quay and pier along the 60-degree axis)
# ==========================================================================
def harbour_axis(c):
    """unit vector toward the sea and the distance from the centre at which that axis meets the shore"""
    u = np.array([np.cos(60 * DEG), np.sin(60 * DEG)])
    rs = np.linspace(-14, 14, 561)
    ss = np.array([c.s_at(*(u * r)) for r in rs])
    idx = np.where(ss[:-1] * ss[1:] <= 0)[0]
    return u, (rs[idx[0]] if len(idx) else 0.0)


def boat(rng, x, y, z, ang):
    hull = np.array([[-2.0, -0.7], [1.0, -0.85], [2.4, 0.0], [1.0, 0.85], [-2.0, 0.7]])
    m = Manifold.extrude(CrossSection([hull]), 1.3).translate([0, 0, z])
    m = m + box(-0.7, 0, z + 1.25, 1.5, 1.0, 0.75)
    return m.rotate([0, 0, ang]).translate([x, y, 0])


def f_harbour(c):
    p, rng = c.p, c.rng
    u, r0 = harbour_axis(c)
    tv = np.array([-u[1], u[0]])
    ang = 60.0
    top, zlow = p.base + 0.9, p.base - 1.6
    parts = []
    q = u * (r0 - 0.3)
    parts.append(box(q[0], q[1], zlow, 3.8, 11.0, top - zlow, ang))                  # quay
    jet = u * (r0 + 5.5)
    parts.append(box(jet[0], jet[1], zlow, 9.0, 2.2, top - zlow, ang))               # stone jetty
    for off in (-4.0, 0.0, 4.0):                                                       # bollards
        b = q - u * 1.2 + tv * off
        parts.append(cyl(b[0], b[1], top - 0.1, 0.45, 0.9, seg=10))
    for side in (-1, 1):                                                               # boats moored in the water
        pos = u * (r0 + 5.0) + tv * side * rng.uniform(4.0, 5.0)
        if c.s_at(*pos) < -2.5:
            parts.append(boat(rng, pos[0], pos[1], p.base - 1.1, ang + rng.uniform(-12, 12) + (0 if side > 0 else 0)))
    c.claim(q[0], q[1], 6.8)
    parts += pack_houses(c, 6, radius=15.0, extra_free=lambda x, y: c.s_at(x, y) > 4.0)
    parts += scatter_trees(c, 5, p_conifer=0.3, rmin=1.4, rmax=2.2)
    return parts


# ==========================================================================
# island lighthouse (water tile with an island in the middle)
# ==========================================================================
def f_lighthouse(c):
    z = c.ground(0, 0)
    zl = c.p.base - 0.5
    parts = [Manifold.cylinder(z + 10.2 - zl, 2.3, 1.5, 24).translate([0, 0, zl])]            # tapered tower
    parts.append(Manifold.extrude(CrossSection([np.array([[np.cos(a) * 1.7, np.sin(a) * 1.7] for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)])]),
                                  0.9, 0, 0.0, (1.35, 1.35)).translate([0, 0, z + 9.9]))     # chamfered gallery
    parts.append(cyl(0, 0, z + 10.7, 1.0, 1.7, seg=16))                                      # lamp room
    parts.append(cyl(0, 0, z + 12.35, 1.35, 1.5, r2=0.05, seg=16))                           # cap
    parts.append(house(3.8, 2.4, c.ground(3.8, 2.4), 3.0, 2.4, 1.9, 1.4, 25.0))              # keeper's cottage
    c.claim(0, 0, 3.0); c.claim(3.8, 2.4, 2.6)
    return parts


# ==========================================================================
# ruins
# ==========================================================================
def _broken(rng, m, x, y, z_top, length, thick, ang):
    """take a few bites out of the top of a wall piece so it looks collapsed"""
    for _ in range(rng.randint(1, 3)):
        s = rng.uniform(-length / 2 + 0.4, length / 2 - 0.4)
        bite = box(s, 0, z_top - rng.uniform(0.5, 1.2), rng.uniform(0.8, 1.6), thick + 0.4, 3.0)
        m = m - bite
    return m


def f_ruins(c):
    rng, p = c.rng, c.p
    z0 = c.ground(0, 0)
    zl = p.base - 0.5
    parts = []
    W, D = 11.0, 8.0
    sides = [((-W / 2, -D / 2), (W / 2, -D / 2)), ((W / 2, -D / 2), (W / 2, D / 2)),
             ((W / 2, D / 2), (-W / 2, D / 2)), ((-W / 2, D / 2), (-W / 2, -D / 2))]
    for (x0, y0), (x1, y1) in sides:
        L = np.hypot(x1 - x0, y1 - y0)
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        cuts = np.sort(rng.uniform(0.2, 0.8, rng.randint(1, 3)))
        edges = np.concatenate([[0.0], cuts, [1.0]])
        for a, b in zip(edges[:-1], edges[1:]):
            if rng.rand() < 0.25:
                continue                                   # a missing stretch of wall
            seg = (b - a) * L - 0.9
            if seg < 1.0:
                continue
            tm = (a + b) / 2
            cx, cy = x0 + (x1 - x0) * tm, y0 + (y1 - y0) * tm
            top = z0 + rng.uniform(1.4, 4.6)
            w = box(0, 0, zl, seg, 1.0, top - zl)
            w = _broken(rng, w, 0, 0, top - zl + zl, seg, 1.0, 0) if False else w
            for _ in range(rng.randint(1, 3)):             # collapsed crown
                s = rng.uniform(-seg / 2 + 0.4, seg / 2 - 0.4)
                w = w - box(s, 0, top - rng.uniform(0.5, 1.1), rng.uniform(0.8, 1.6), 1.4, 3.0)
            parts.append(w.rotate([0, 0, ang]).translate([cx, cy, 0]))
    for (x, y) in ((-3.4, -1.0), (2.6, 1.6), (4.2, -2.4)):                           # column stubs
        h = rng.uniform(2.0, 4.8)
        col = cyl(0, 0, zl, 0.62, z0 + h - zl, seg=12)
        col = col - box(0.3, 0, z0 + h - 0.4, 2.4, 2.4, 1.0).rotate([0, rng.uniform(15, 35), 0])
        parts.append(col.translate([x, y, 0]))
    ax, ay = 0.0, D / 2 + 3.0                                                            # a free-standing arch
    parts.append(box(ax - 2.0, ay, zl, 1.1, 1.1, z0 + 5.2 - zl))
    parts.append(box(ax + 2.0, ay, zl, 1.1, 1.1, z0 + 5.2 - zl))
    parts.append(box(ax, ay, z0 + 4.7, 5.4, 1.1, 1.0))
    group = union(parts).rotate([0, 0, rng.uniform(-25, 25)])
    out = [group]
    c.claim(0, 0, 8.6); c.claim(0, D / 2 + 3.0, 3.6)
    for _ in range(9):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(1.0, 8.0)
        x, y = r * np.cos(a), r * np.sin(a)
        out.append(rock(rng, x, y, c.ground(x, y) - 0.1, rng.uniform(0.5, 1.0)))
    return out


# ==========================================================================
# mine entrance (re-uses the dungeon hill, faces edge 0)
# ==========================================================================
def f_mine(c):
    p, rng = c.p, c.rng
    zl = p.base - 0.5
    floor = p.base + p.road_h
    parts = []
    xf = XF + 1.7                                                  # the cut face made by the approach clearing
    w, hsq = 3.4, 3.0
    c.carve = box(XF - 2.5, 0, floor, 9.5, w, hsq)                  # square adit into the hill
    for s in (-1, 1):                                              # timber frame
        parts.append(box(xf, s * (w / 2 + 0.25), zl, 0.9, 0.9, floor + hsq + 0.3 - zl))
    parts.append(box(xf, 0, floor + hsq, 0.9, w + 1.4, 0.8))
    for k in range(3):                                             # braces along the tunnel roof
        parts.append(box(xf - 1.6 - 1.7 * k, 0, floor + hsq - 0.5, 0.6, w + 0.2, 0.5))
    parts.append(box(xf + 2.2, 0.15, floor - 0.1, 1.4, 0.95, 0.65))                       # ore cart
    parts.append(cyl(xf + 2.2 - 0.45, 0.15 - 0.55, floor - 0.1, 0.32, 0.12, seg=10))
    parts.append(cyl(xf + 2.2 + 0.45, 0.15 + 0.55, floor - 0.1, 0.32, 0.12, seg=10))
    for _ in range(8):                                             # spoil heap
        x, y = rng.uniform(XF + 3.0, XF + 7.0), rng.uniform(4.2, 8.0)
        if c.free(x, y, 0.9, gap=-0.5):
            parts.append(rock(rng, x, y, c.ground(x, y) - 0.2, rng.uniform(0.7, 1.4)))
            c.claim(x, y, 0.9)
    hx, hy = XF + 4.5, -7.2
    if c.free(hx, hy, 2.6):
        parts.append(house(hx, hy, c.ground(hx, hy), 3.6, 2.6, 2.1, 1.5, 12.0))
        c.claim(hx, hy, 2.6)
    parts += scatter_trees(c, 5, p_conifer=0.7, rmin=1.3, rmax=2.0, area=17)
    return parts


# ==========================================================================
# standing stones
# ==========================================================================
def h_stones(c):
    d = np.hypot(c.X, c.Y)
    return 1.0 * smoothstep((10.5 - d) / 4.0)


def _stone(rng, x, y, z, w, d, h, ang, taper=0.62):
    poly = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    m = Manifold.extrude(CrossSection([poly]), h, 0, 0.0, (taper, taper + 0.08))
    return m.rotate([0, 0, ang]).translate([x, y, z])


def f_stones(c):
    rng = c.rng
    parts, R = [], 7.2
    n = 9
    for k in range(n):
        a = 2 * np.pi * k / n + rng.uniform(-0.08, 0.08)
        x, y = R * np.cos(a), R * np.sin(a)
        if c.road is not None:
            d, _ = c.road.query(np.array([x]), np.array([y]))
            if d[0] < 3.2:
                continue                                     # leave the road a gap through the ring
        parts.append(_stone(rng, x, y, c.ground(x, y) - 0.6, rng.uniform(1.5, 2.0), rng.uniform(0.9, 1.2),
                            rng.uniform(4.2, 6.2), np.degrees(a) + 90 + rng.uniform(-6, 6)))
    zc = c.ground(0, 0) - 0.6                                  # central trilithon and altar
    parts.append(_stone(rng, -1.7, 0, zc, 1.5, 1.1, 6.4, 90))
    parts.append(_stone(rng, 1.7, 0, zc, 1.5, 1.1, 6.4, 90))
    parts.append(box(0, 0, zc + 6.0, 4.8, 1.3, 1.0))
    parts.append(box(0, -3.0, c.ground(0, -3.0) - 0.6, 2.4, 1.3, 1.5))
    c.claim(0, 0, 9.6)
    parts += scatter_trees(c, 7, p_conifer=0.4, rmin=1.4, rmax=2.3)
    return parts


# ==========================================================================
# watchtower on a mound inside a low wall (gate faces edge 0)
# ==========================================================================
def h_watch(c):
    d = np.hypot(c.X, c.Y)
    return 3.0 * smoothstep((12.5 - d) / 6.0)


def f_watchtower(c):
    p = c.p
    z = c.ground(0, 0)
    zl = p.base - 0.5
    sq = np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]])
    tw = Manifold.extrude(CrossSection([sq]), z + 9.5 - zl, 0, 0.0, (0.8, 0.8)).translate([0, 0, zl])
    tw = tw - box(2.5, 0, z, 1.0, 1.3, 2.0)                                       # door
    for (sx, sy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):                           # arrow slits
        tw = tw - box(2.3 * sx, 2.3 * sy, z + 5.2, 0.7 if sx else 0.4, 0.4 if sx else 0.7, 1.4)
    g_sq = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    gal = Manifold.extrude(CrossSection([g_sq]), 1.3, 0, 0.0, (1.65, 1.65)).translate([0, 0, z + 9.0])
    parts = [tw, gal]
    for sx in np.linspace(-2.9, 2.9, 5):                                           # merlons
        for sy in (-3.0, 3.0):
            parts.append(box(sx, sy, z + 10.25, 0.9, 0.7, 0.7))
            parts.append(box(sy, sx, z + 10.25, 0.7, 0.9, 0.7))
    for sx in (-2.3, 2.3):
        for sy in (-2.3, 2.3):
            parts.append(box(sx, sy, z + 10.2, 0.7, 0.7, 2.3))                     # roof posts
    parts.append(pyramid_roof(6.2, 6.2, 1.9).translate([0, 0, z + 12.45]))
    ring = cyl(0, 0, zl, 9.0, z + 1.6 - zl, seg=48) - cyl(0, 0, zl - 1, 8.1, z + 6, seg=48)
    ring = ring - box(8.55, 0, zl - 1, 2.6, 3.2, 20)                               # gate
    parts.append(ring)
    c.claim(0, 0, 9.6)
    parts += scatter_trees(c, 5, p_conifer=0.5, rmin=1.4, rmax=2.2)
    return parts


FEATURES = {
    'forest':   dict(parts=f_forest),
    'village':  dict(parts=f_village),
    'city':     dict(parts=f_city),
    'fort':     dict(parts=f_fort, height=h_fort),
    'temple':   dict(parts=f_temple),
    'ziggurat': dict(parts=f_ziggurat),
    'dungeon':  dict(parts=f_dungeon, height=h_dungeon),
    'harbour':  dict(parts=f_harbour),
    'lighthouse': dict(parts=f_lighthouse),
    'ruins':    dict(parts=f_ruins),
    'mine':     dict(parts=f_mine, height=h_dungeon),
    'stones':   dict(parts=f_stones, height=h_stones),
    'watchtower': dict(parts=f_watchtower, height=h_watch),
}


# ==========================================================================
# bridge deck
# ==========================================================================
def bridge_parts(c, road_ang_deg):
    """Deck and parapets where the (straight) road crosses the (meandering) river:
    centred on the point where the river centreline meets the road line."""
    p = c.p
    th = np.radians(road_ang_deg)
    u = np.array([np.cos(th), np.sin(th)])                 # along the road
    n = np.array([-np.sin(th), np.cos(th)])                # across it
    q = np.zeros(2)
    best = 1e9
    for poly in c.river.polys:
        s = poly @ n
        for i in np.where(s[:-1] * s[1:] <= 0)[0]:         # river centreline crosses the road line
            f = s[i] / (s[i] - s[i + 1] + 1e-12)
            pt = poly[i] + f * (poly[i + 1] - poly[i])
            if np.hypot(*pt) < best:
                best, q = np.hypot(*pt), pt
    top = p.base + p.road_h
    L = 14.0
    deck = box(0, 0, top - 0.8, L, p.road_w, 0.8)
    rail = box(0, p.road_w / 2 - 0.3, top, L, 0.6, 0.7) + box(0, -p.road_w / 2 + 0.3, top, L, 0.6, 0.7)
    return [(deck + rail).rotate([0, 0, road_ang_deg]).translate([q[0], q[1], 0])]


# ==========================================================================
# water bodies: coasts (water reaching whole edges) and inland ponds
# ==========================================================================
def _vertex(c, k):
    R = c.a / np.cos(30 * DEG)
    return R * np.array([np.cos((30 + 60 * k) * DEG), np.sin((30 + 60 * k) * DEG)])


def coast_signed(c, edges, rng, organic):
    """Signed distance to a shoreline that separates the water edges `edges` (consecutive,
    anticlockwise) from the land. The shoreline runs between the two tile corners where the
    water edges end, bulging by an amount that depends on how many edges are water."""
    from matplotlib.path import Path
    a, n, i0 = c.a, len(edges), edges[0]
    phi = np.radians(60 * (i0 + (n - 1) / 2.0))
    u = np.array([np.cos(phi), np.sin(phi)])
    Vs, Ve = _vertex(c, i0 - 1), _vertex(c, i0 + n - 1)
    apex_r = {1: a - 9.5, 2: 6.0, 3: 0.0, 4: -6.0, 5: -14.0}[n]
    apex = u * (apex_r + rng.uniform(-1, 1) * 1.2 * organic)
    ctrl = 2 * apex - 0.5 * (Vs + Ve)
    t = np.linspace(0, 1, 160)[:, None]
    curve = (1 - t) ** 2 * Vs + 2 * (1 - t) * t * ctrl + t ** 2 * Ve
    dense, sarc = PathNet._resample(curve)
    tang = np.gradient(dense, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    ph = rng.uniform(0, 2 * np.pi, 2)
    wig = 1.6 * np.sin(2 * np.pi * sarc / rng.uniform(14, 20) + ph[0]) + 0.7 * np.sin(2 * np.pi * sarc / rng.uniform(7, 10) + ph[1])
    taper = smoothstep(np.minimum(sarc, sarc[-1] - sarc) / 6.0)          # shoreline meets the corners cleanly
    dense = dense + (organic * wig * taper)[:, None] * nrm
    dense = gaussian_filter1d(dense, sigma=2.2 / 0.4, axis=0, mode='nearest')
    k_out = (a + 3.0) / a                     # push the water polygon past the tile edge so the clip never
    chain = [_vertex(c, i0 - 1)] + [k_out * _vertex(c, i0 - 1 + j) for j in range(n + 1)] + [_vertex(c, i0 + n - 1)]
    poly = np.array(chain + list(dense[::-1][1:-1]))   # sees a half-land, half-water boundary
    inside = Path(poly).contains_points(np.c_[c.X.ravel(), c.Y.ravel()]).reshape(c.X.shape)
    d, _ = PathNet([dense]).query(c.X, c.Y)
    s = np.where(inside, -d, d)
    return gaussian_filter(s, 0.8 / c.p.res, mode='nearest')


def pond_signed(c, rng, organic, r0=8.5):
    c0 = rng.uniform(-1.5, 1.5, 2) * organic
    th = np.arctan2(c.Y - c0[1], c.X - c0[0])
    r = np.hypot(c.X - c0[0], c.Y - c0[1])
    ph = rng.uniform(0, 2 * np.pi, 3)
    Rth = r0 * (1 + organic * (0.16 * np.sin(2 * th + ph[0]) + 0.10 * np.sin(3 * th + ph[1]) + 0.06 * np.sin(5 * th + ph[2])))
    return gaussian_filter(r - Rth, 0.8 / c.p.res, mode='nearest')


def apply_water(c, Z, level_free_edges, bw=2.6):
    """Blend the open-water surface (river-floor level, rippled, ripples fading to nothing at the
    shore) into the land surface Z with a gentle beach. Land edges keep the flat tile rim."""
    p, s = c.p, c.water_s
    ripple, lvl = t_water(c)
    fade = smoothstep(-s / 3.0)
    Zw = p.base + lvl + ripple * fade
    t = smoothstep(s / bw)                                     # beach: water level -> land level
    for i in level_free_edges:                                 # land edges: hold the flat rim
        d_i = c.a - (c.X * np.cos(60 * i * DEG) + c.Y * np.sin(60 * i * DEG))
        t = np.maximum(t, smoothstep((1.2 - d_i) / 0.8) * smoothstep(s + 1.0))
    return Zw * (1 - t) + Z * t



def add_cascade(c, Z):
    """A stepped stream that tumbles down the mountain flank into the river."""
    p = c.p
    dr, _ = c.river.query(c.X, c.Y)
    cand = (dr > 8.0) & (dr < 13.0) & (hexdist(c.X, c.Y) < c.a - 7.0)
    idx = np.argmax(np.where(cand, Z, -1e9))
    sy, sx = np.unravel_index(idx, Z.shape)
    S = np.array([c.xs[sx], c.ys[sy]])
    # end: the river-bank point on the line towards the river
    gy, gx = np.gradient(dr, c.ys, c.xs)
    P = [S.copy()]
    q = S.copy()
    for _ in range(400):
        ix = np.clip(np.searchsorted(c.xs, q[0]), 0, len(c.xs) - 1)
        iy = np.clip(np.searchsorted(c.ys, q[1]), 0, len(c.ys) - 1)
        g = np.array([gx[iy, ix], gy[iy, ix]])
        if dr[iy, ix] < p.river_w / 2 + 0.6:
            break
        q = q - 0.5 * g / (np.linalg.norm(g) + 1e-9)
        P.append(q.copy())
    P = np.array(P)
    ds = np.hypot(*np.diff(P, axis=0).T).cumsum()
    ds = np.r_[0, ds]
    wig = 1.4 * np.sin(2 * np.pi * ds / 11.0) * smoothstep(ds / 4.0) * smoothstep((ds[-1] - ds) / 4.0)
    nv = np.gradient(P, axis=0); nv = np.stack([-nv[:, 1], nv[:, 0]], 1)
    nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9
    P = P + wig[:, None] * nv
    g = np.array([float(RegularGridInterpolator((c.ys, c.xs), Z)([[y, x]])[0]) for x, y in P])
    g = np.minimum.accumulate(g)
    step = np.maximum(np.floor((g - p.base) / 1.4) * 1.4 + p.base, p.base)    # staircase of little falls
    step = np.minimum.accumulate(step)
    Zn = Z.copy()
    R = 1.5
    for (x, y), fl in zip(P, step):
        ix0, ix1 = np.searchsorted(c.xs, [x - R - 0.5, x + R + 0.5])
        iy0, iy1 = np.searchsorted(c.ys, [y - R - 0.5, y + R + 0.5])
        sub = (slice(iy0, iy1), slice(ix0, ix1))
        d = np.hypot(c.X[sub] - x, c.Y[sub] - y)
        floor = fl + 0.9 * smoothstep((d - R * 0.55) / (R * 0.9))
        Zn[sub] = np.minimum(Zn[sub], np.maximum(floor, p.base))
    c.cascade_pts = P
    return Zn

# ==========================================================================
# the builder
# ==========================================================================
def build_tile(terrain='plains', road=None, river=None, feature=None, seed=0,
               p=None, bridge=False, ford=False, water=None, cascade=False, style=None):
    p = p or Params()
    c = Ctx(p, seed)
    c.exits = list(road or [])
    c.fort_flags = ('gate',) if feature == 'fort' else ()

    # rigid geometry where walls / gates have to line up with the path; the road on a bridge or
    # ford tile is straight, but its river is the same meandering river as the plain river tile
    rigid = feature in ('city', 'fort', 'temple', 'ziggurat', 'dungeon', 'mine', 'watchtower')
    org = 0.0 if rigid else p.organic * (0.5 if feature == 'village' else 1.0)
    org_road = 0.0 if (bridge or ford) else org
    land_edges = list(range(6))
    if water:                                  # {'kind': 'coast', 'edges': [0, 1, 2]} or {'kind': 'pond'}
        wr = np.random.RandomState(seed + 303)
        if water['kind'] == 'coast':
            c.water_s = coast_signed(c, water['edges'], wr, p.organic)
            land_edges = [i for i in range(6) if i not in water['edges']]
        elif water['kind'] == 'island':        # land in the middle, open water to every edge
            c.water_s = -pond_signed(c, wr, p.organic, r0=9.5)
            land_edges = []
        else:
            c.water_s = pond_signed(c, wr, p.organic)
    level = 0.0
    h, level = TERRAINS[terrain](c)
    h = np.asarray(h, float)
    if c.water_s is not None:
        h = h * smoothstep(c.water_s / 4.0)    # no relief in the water; land rises away from the shore
    if feature and 'height' in FEATURES[feature]:
        h = h + FEATURES[feature]['height'](c)
    if style == 'cliff' and c.water_s is not None:   # plateau that drops steeply into the sea
        rr = 0.75 + 0.5 * ridged(c.X, c.Y, seed + 9, 3, 0.09)
        h = h + 3.8 * rr * smoothstep((c.water_s - 0.3) / 1.2)
    if terrain != 'water':
        h = h * c.mask                 # flat level rim (water keeps its ripples to the edge)

    # --- road & river networks --------------------------------------------
    if road:
        if feature in ('dungeon', 'mine'):
            e = (p.a + 3.0)
            c.road = PathNet([[np.array([e, 0.0]), np.array([XF + 1.5, 0.0])]])
        else:
            c.road = PathNet.from_exits(road, p.a, rng=np.random.RandomState(seed + 101),
                                        organic=0.6 * org_road, wamp=0.10)
    if river:
        end = None
        if water and water['kind'] == 'coast' and len(river) == 1:       # river mouth: run on into the sea
            dv = np.array([np.cos(60 * river[0] * DEG), np.sin(60 * river[0] * DEG)])
            for lam in np.linspace(p.a, -p.a, 400):
                if c.s_at(*(dv * lam)) < 0:
                    end = dv * (lam - 1.5)
                    break
        c.river = PathNet.from_exits(river, p.a, rng=np.random.RandomState(seed + 202),
                                     organic=org, wamp=0.22, end=end)

    Z = p.base + level + h
    if c.road is not None:
        d, _ = c.road.query(c.X, c.Y)
        d = gaussian_filter(d, 0.9 / p.res, mode='nearest')      # no creases on inner bends
        if feature in ('dungeon', 'mine'):      # approach cut only in front of the facade
            flat = 1 - smoothstep((c.X - (XF + 1.0)) / 1.0) * (1 - smoothstep((np.abs(c.Y) - 4.2) / 2.0))
        else:
            flat = smoothstep((d - p.road_w / 2 - 0.5) / 5.0)
        Z = p.base + level + h * flat
        rm = smoothstep((p.road_w / 2 + 0.5 - d) / 1.0)
    else:
        rm = 0.0
    if c.river is not None:
        d, _ = c.river.query(c.X, c.Y)
        d = gaussian_filter(d, 0.9 / p.res, mode='nearest')      # no creases on inner bends
        hw = p.river_w / 2
        valley = smoothstep((d - hw) / 7.0)
        Z = p.base + level + (Z - p.base - level) * valley
        carve = smoothstep((hw + 1.0 - d) / 1.8)
        if ford and c.road is not None:
            carve = carve * (1 - 0.75 * rm)
        Z = Z - p.river_depth * carve
        if bridge:
            rm = rm * (1 - carve)
    else:
        carve = 0.0
    Z = Z + p.road_h * rm
    if cascade and c.river is not None:
        Z = add_cascade(c, Z)
    if c.water_s is not None:
        bw = {'cliff': 0.9, 'sand': 3.6}.get(style, 2.6)
        Z = apply_water(c, Z, land_edges, bw)
    c.Z = Z

    # --- solid ------------------------------------------------------------
    solid = heightfield_solid(c.xs, c.ys, Z)
    ch = p.chamfer
    def ring(a, z):
        R = a / np.cos(30 * DEG)
        return [[R * np.cos((30 + 60 * k) * DEG), R * np.sin((30 + 60 * k) * DEG), z] for k in range(6)]
    clip = Manifold.hull_points(np.array(ring(p.a - ch, 0.0) + ring(p.a, ch) + ring(p.a, 60.0)))   # chamfered hex prism
    tile = solid ^ clip

    # --- magnet pockets ---------------------------------------------------
    if p.magnets:
        depth = p.magnet_t + p.magnet_fit
        r = p.magnet_d / 2 + p.magnet_fit / 2
        offs = [0.0] if p.magnets_per_edge == 1 else [-p.magnet_pitch / 2, p.magnet_pitch / 2]
        for i in range(6):
            for off in offs:
                pk = Manifold.cylinder(depth + 1.0, r, r, 32).rotate([0, 90, 0]) \
                    .translate([p.a - depth, off, 1.9]).rotate([0, 0, 60 * i])
                tile = tile - pk

    # --- features ---------------------------------------------------------
    parts = []
    if feature:
        parts += FEATURES[feature]['parts'](c)
        if feature in ('dungeon', 'mine') and getattr(c, 'carve', None) is not None:
            tile = tile - c.carve
    if terrain == 'forest':
        parts += f_forest(c)
    if bridge:
        parts += bridge_parts(c, 60.0 * river_cross_edge(road))
    if parts:
        feat = union(parts)
        feat = feat ^ hex_prism(p.a - 0.05, 80.0, 0.5)       # nothing may poke past the tile edge
        if feature in ('dungeon', 'mine') and getattr(c, 'carve', None) is not None:
            feat = feat - c.carve
        tile = tile + feat
    tile = tile.simplify(0.001)
    comps = tile.decompose()
    if len(comps) > 1:                 # drop dust / zero-volume slivers left by boolean ops
        keep = [m for m in comps if m.volume() > 1.0]
        tile = union(keep)
    return to_trimesh(tile)


def river_cross_edge(road):
    return road[0]
