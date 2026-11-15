"""
buildings.py - miniature buildings and the street layouts of villages and cities.

Buildings are built in a local frame: footprint centred on the origin, z = 0 at ground level (walls
sink 1.2 mm below it so they sit on sloping ground), the door on the -y side.  Each part is tagged with
its material *before* openings are cut (the cutters carry the 'dark' tag), see tag() in hexlib.

Printability: window recesses are 0.45 mm wide and about 0.3 deep, timber framing stands 0.07 proud,
the jetty overhangs 0.28 mm, no roof slope is shallower than about 40 degrees from vertical.
"""
import numpy as np
from manifold3d import Manifold, CrossSection
from hexlib import *


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def solid(cx, cy, z0, w, d, h, mat, ang=0.0):
    return tag(box(cx, cy, z0, w, d, h, ang), mat)


def dark(cx, cy, z0, w, d, h, ang=0.0):
    return tag(box(cx, cy, z0, w, d, h, ang), 'dark')


def _tri(span, rh, length, y0):
    """triangular prism along +y from y0 to y0+length, base at z=0, centred on x"""
    poly = np.array([[-span / 2, 0], [span / 2, 0], [0, rh]])
    return Manifold.extrude(CrossSection([poly]), length).rotate([90, 0, 0]).translate([0, y0 + length, 0])


def gabled(span, length, rh, over=0.28, mat='roof', infill='plaster'):
    """Gable roof, ridge along y, base z=0, centred.  The two end faces are flush with y=+-length/2
    and filled with `infill` triangles inside a thin barge-board of roof material."""
    roof = tag(gable_roof(span, length, rh, over), mat)
    sp, k, ln = span + 2 * over, 0.80, 0.3
    cut = _tri(sp * k, rh * k, ln + 0.02, -length / 2 - 0.01) + _tri(sp * k, rh * k, ln + 0.02, length / 2 - ln - 0.01)
    fill = tag(_tri(sp * k, rh * k, ln, -length / 2) + _tri(sp * k, rh * k, ln, length / 2 - ln), infill)
    return (roof - cut) + fill


def lean_to(length, width, rise, mat='roof'):
    """single-pitch roof along x; the high side is at -y (against a taller wall), low side at +y"""
    poly = np.array([[-width / 2, 0], [width / 2, 0], [-width / 2, rise]])
    p = Manifold.extrude(CrossSection([poly]), length).rotate([90, 0, 0]).translate([0, length / 2, 0])
    return tag(p.rotate([0, 0, 90]), mat)           # ridge along x; (the section's x becomes world y)


# --------------------------------------------------------------------------
# houses
# --------------------------------------------------------------------------
def townhouse(rng, w, d, floors=2, jetty=True, ridge='y', timber=False, chimney=True, rh=None,
              thatch=False, wall='plaster', door_w=0.72, door_h=1.1, door_x=None, over=0.28, plinth=True):
    """A plastered (optionally half-timbered, jettied) house: windows, door and step, plinth, chimney."""
    fh = rng.uniform(1.15, 1.32)
    z_low = fh + (0.3 if floors == 1 else 0.0)
    jet = 0.28 if (jetty and floors > 1) else 0.0
    walls = solid(0, 0, -1.2, w, d, 1.2 + z_low, wall)
    top, uh = z_low, 0.0
    if floors > 1:
        uh = (floors - 1) * fh
        walls = walls + solid(0, -jet / 2, z_low - 0.01, w, d + jet, uh + 0.01, wall)
        top = z_low + uh
    dtop = d + jet
    yc = -jet / 2 if floors > 1 else 0.0
    if door_x is None:
        door_x = 0.0 if w < 2.6 else rng.choice([-1, 0, 1]) * w * 0.22
    frame = []
    if timber and floors > 1:
        yf = -(d / 2 + jet)
        n = max(2, int(round(w / 0.95)))
        px = np.linspace(-w / 2 + 0.12, w / 2 - 0.12, n + 1)
        for x in px:
            frame.append(solid(x, yf, z_low, 0.17, 0.14, uh, 'wood'))
        frame.append(solid(0, yf, z_low - 0.06, w + 0.02, 0.14, 0.2, 'wood'))
        frame.append(solid(0, yf, top - 0.2, w + 0.02, 0.14, 0.2, 'wood'))
        for i in range(n):
            bw = px[i + 1] - px[i]
            ln = float(np.hypot(bw, uh)) * 1.25
            ang = np.degrees(np.arctan2(uh, bw)) * (1 if (i + int(w * 7)) % 2 else -1)
            br = Manifold.cube([ln, 0.14, 0.14]).translate([-ln / 2, -0.07, -0.07]).rotate([0, -ang, 0])
            br = br.translate([(px[i] + px[i + 1]) / 2, yf, z_low + uh / 2]) ^ box((px[i] + px[i + 1]) / 2, yf, z_low, bw, 0.3, uh)
            frame.append(tag(br, 'wood'))
    body = union([walls] + frame)
    # openings
    ww, wht = 0.46, 0.56
    cuts = [dark(door_x, -d / 2 + 0.1, 0.0, door_w, 0.5, door_h)]
    nwin = max(1, int(round(w / 1.25)))
    for f in range(floors):
        zc = f * fh + fh / 2 - wht / 2 + 0.1
        yface = -d / 2 if f == 0 else -(d / 2 + jet)
        for i in range(nwin):
            x = (i + 0.5) / nwin * w - w / 2
            if not (f == 0 and abs(x - door_x) < (door_w + ww) / 2 + 0.12):
                cuts.append(dark(x, yface + 0.1, zc, ww, 0.46, wht))
            if rng.rand() < 0.75:
                cuts.append(dark(x, d / 2 - 0.1, zc, ww, 0.46, wht))
        for sx in (-1, 1):
            cuts.append(dark(sx * (w / 2 - 0.1), yc * (f > 0), zc, 0.46, ww, wht))
    body = body - union(cuts)
    parts = [body]
    if plinth:
        parts.append(solid(0, 0, -1.2, w + 0.14, d + 0.14, 1.5, 'stone') - union(cuts[:1]))
    parts.append(solid(door_x, -d / 2 - 0.2, -0.1, door_w + 0.3, 0.4, 0.22, 'stone'))      # sunk into the plinth: overlap, not tangent
    span, length = (w, dtop) if ridge == 'y' else (dtop, w)
    if rh is None:
        rh = float(np.clip(span * 0.56, 1.15, 2.3))
    roof = gabled(span, length, rh, over, 'thatch' if thatch else 'roof', wall)
    if ridge == 'x':
        roof = roof.rotate([0, 0, 90])
    parts.append(roof.translate([0, yc, top - 0.04]))
    if chimney:
        if ridge == 'x':
            cx, cy = rng.choice([-1, 1]) * (w / 2 - 0.35), yc
        else:
            cx, cy = 0.0, yc + rng.choice([-1, 1]) * (dtop / 2 - 0.5)
        parts.append(box(cx, cy, top - 0.5, 0.46, 0.46, rh + 0.5 + 0.35))
        parts.append(box(cx, cy, top + rh + 0.3, 0.6, 0.6, 0.12))
    return union(parts)


def l_house(rng, w, d, **kw):
    """main range plus a cross-wing at the back"""
    main = townhouse(rng, w, d, ridge='x', **kw)
    w2, d2 = rng.uniform(1.8, 2.3), rng.uniform(2.0, 2.8)
    wing = townhouse(rng, w2, d2, floors=1, jetty=False, ridge='y', chimney=False, plinth=False, door_x=0.0)
    sx = rng.choice([-1, 1]) * (w / 2 - w2 / 2 - 0.2)
    return union([main, wing.rotate([0, 0, 180]).translate([sx, d / 2 + d2 / 2 - 0.3, 0])]), d + d2 - 0.3


def barn(rng, w=5.0, d=3.0):
    return townhouse(rng, w, d, floors=1, jetty=False, ridge='x', chimney=False, thatch=rng.rand() < 0.6,
                     wall='wood', door_w=1.5, door_h=1.5, door_x=0.0, rh=1.7, plinth=True)


# --------------------------------------------------------------------------
# churches
# --------------------------------------------------------------------------
def church(rng):
    """village church: nave, west tower with spire, chancel, porch, buttresses. Nave runs along x, porch on -y."""
    L, W, wh = 4.4, 2.3, 2.3
    parts = [solid(0, 0, -1.2, L, W, 1.2 + wh, 'stone')]
    parts.append(gabled(W, L, 1.45, 0.22, 'roof', 'stone').rotate([0, 0, 90]).translate([0, 0, wh - 0.04]))
    parts.append(solid(L / 2 + 0.85, 0, -1.2, 1.8, 1.7, 1.2 + wh - 0.45, 'stone'))                          # chancel
    parts.append(gabled(1.7, 1.8, 1.0, 0.15, 'roof', 'stone').rotate([0, 0, 90]).translate([L / 2 + 0.85, 0, wh - 0.45 - 0.04]))
    tx = -L / 2 - 0.55
    parts.append(solid(tx, 0, -1.2, 1.8, 1.8, 1.2 + 4.6, 'stone'))                                          # tower
    parts.append(solid(tx, 0, 3.3, 2.0, 2.0, 0.3, 'stone'))                                                  # belfry string course
    parts.append(tag(pyramid_roof(2.2, 2.2, 3.0), 'roof').translate([tx, 0, 4.6 - 0.05]))                   # spire
    for x in (-1.5, -0.5, 0.5, 1.5):
        for s in (-1, 1):
            parts.append(solid(x, s * (W / 2 + 0.1), -1.2, 0.36, 0.36, 1.2 + 1.5, 'stone'))                 # buttresses
    px = 0.6
    parts.append(solid(px, -W / 2 - 0.55, -1.2, 1.3, 1.1, 1.2 + 1.5, 'stone'))                              # porch
    parts.append(gabled(1.3, 1.1, 0.65, 0.12, 'roof', 'stone').translate([px, -W / 2 - 0.55, 1.5 - 0.04]))
    cuts = [dark(px, -W / 2 - 1.05, 0.0, 0.62, 0.5, 1.0)]
    for x in (-1.0, 1.7):
        for s in (-1, 1):
            if not (s < 0 and abs(x - px) < 0.9):
                cuts.append(dark(x, s * (W / 2 - 0.1), 0.65, 0.4, 0.46, 1.15))
    cuts.append(dark(L / 2 + 1.7, 0, 0.55, 0.4, 0.5, 0.95))
    for s in (-1, 1):                                                                                          # belfry openings
        cuts.append(dark(tx, s * 0.82, 3.7, 0.42, 0.45, 0.7))
        cuts.append(dark(tx + s * 0.82, 0, 3.7, 0.45, 0.42, 0.7))
    return union(parts) - union(cuts)


def cathedral(rng, transept=True):
    """cruciform minster: aisled nave, transepts, choir with apse, crossing tower and spire, twin west towers"""
    wh, rh = 3.7, 1.9
    parts = [solid(-0.4, 0, -1.2, 7.2, 2.4, 1.2 + wh, 'stone')]
    parts.append(gabled(2.4, 7.2, rh, 0.22, 'roof', 'stone').rotate([0, 0, 90]).translate([-0.4, 0, wh - 0.04]))
    for s in (-1, 1):                                                                                          # aisles
        parts.append(solid(-0.4, s * 1.75, -1.2, 7.2, 1.15, 1.2 + 1.9, 'stone'))
        ro = lean_to(7.2, 1.35, 0.7)
        parts.append((ro if s < 0 else ro.rotate([0, 0, 180])).translate([-0.4, s * 1.75, 1.9 - 0.03]))
        for x in np.linspace(-3.2, 2.4, 6):                                                                    # buttresses
            parts.append(solid(x, s * 2.45, -1.2, 0.4, 0.45, 1.2 + 2.3, 'stone'))
    if transept:
        parts.append(solid(0.7, 0, -1.2, 2.4, 6.8, 1.2 + wh, 'stone'))
        parts.append(gabled(2.4, 6.8, rh, 0.22, 'roof', 'stone').translate([0.7, 0, wh - 0.04]))
    parts.append(solid(4.5, 0, -1.2, 2.6, 2.3, 1.2 + wh - 0.4, 'stone'))                                      # choir
    parts.append(gabled(2.3, 2.6, 1.7, 0.2, 'roof', 'stone').rotate([0, 0, 90]).translate([4.5, 0, wh - 0.4 - 0.04]))
    parts.append(Manifold.cylinder(1.2 + wh - 0.9, 1.15, 1.1, 8).translate([5.8, 0, -1.2]))                    # apse
    parts.append(tag(Manifold.cylinder(1.4, 1.35, 0.1, 8), 'roof').translate([5.8, 0, wh - 0.9 - 0.04]))
    parts.append(solid(0.7, 0, -1.2, 2.5, 2.5, 1.2 + wh + 2.4, 'stone'))                                       # crossing tower
    parts.append(solid(0.7, 0, wh + 1.0, 2.7, 2.7, 0.3, 'stone'))
    parts.append(tag(pyramid_roof(2.9, 2.9, 4.4), 'roof').translate([0.7, 0, wh + 2.4 - 0.05]))
    for s in (-1, 1):                                                                                          # west towers
        parts.append(solid(-3.9, s * 1.55, -1.2, 1.9, 1.9, 1.2 + 6.0, 'stone'))
        parts.append(tag(pyramid_roof(2.2, 2.2, 2.2), 'roof').translate([-3.9, s * 1.55, 6.0 - 0.05]))
    cuts = [dark(-4.05, 0, 0.0, 0.9, 0.9, 1.4)]
    cuts.append(tag(Manifold.cylinder(0.5, 0.55, 0.55, 14), 'dark').rotate([0, 90, 0]).translate([-4.0, 0, 3.7]))   # rose window
    for s in (-1, 1):
        for x in (-2.6, -1.2, 0.2 if not transept else None, 2.2, 3.4):
            if x is None or (transept and -0.5 < x < 1.9):
                continue
            cuts.append(dark(x, s * (1.2 - 0.1), 2.75, 0.4, 0.46, 0.8))                                          # clerestory
            if x < 3.0:
                cuts.append(dark(x, s * (2.3 - 0.1), 0.55, 0.38, 0.46, 1.0))                                     # aisle
        cuts.append(dark(0.7, s * 3.3, 1.0, 0.7, 0.5, 1.8))                                                      # transept ends
        cuts.append(dark(-3.9, s * 2.45, 1.2, 0.4, 0.5, 0.9))
        cuts.append(dark(-3.9 - 0.9, s * 1.55, 4.4, 0.5, 0.42, 0.9))
        cuts.append(dark(-3.9, s * 2.45, 4.4, 0.42, 0.5, 0.9))
        cuts.append(dark(0.7 + s * 1.2, 0, 4.9, 0.5, 0.42, 0.9))
        cuts.append(dark(0.7, s * 1.2, 4.9, 0.42, 0.5, 0.9))
    cuts.append(dark(6.7, 0, 0.55, 0.4, 0.5, 1.0))
    return union(parts) - union(cuts)


# --------------------------------------------------------------------------
# small things
# --------------------------------------------------------------------------
def haystack(x, y, z, r=0.8):
    return tag(cyl(x, y, z - 0.2, r, 0.7, r2=r * 0.95, seg=12), 'thatch') + tag(cyl(x, y, z + 0.5, r * 0.95, 0.75, r2=0.08, seg=12), 'thatch')


def low_wall(x0, y0, x1, y1, z, thick=0.4, h=0.5, mat='stone'):
    L = float(np.hypot(x1 - x0, y1 - y0))
    ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    return tag(box((x0 + x1) / 2, (y0 + y1) / 2, z - 0.3, L + thick, thick, h + 0.3, ang), mat)


def well(x, y, z):
    ring = cyl(x, y, z - 0.3, 0.52, 0.85, seg=14) - cyl(x, y, z + 0.4, 0.34, 0.6, seg=14)
    posts = [solid(x + s * 0.42, y, z + 0.4, 0.12, 0.12, 1.1, 'wood') for s in (-1, 1)]
    roof = gabled(0.9, 0.6, 0.35, 0.05, 'roof', 'wood').translate([x, y, z + 1.45])
    return union([ring] + posts + [roof])


def fountain(x, y, z):
    basin = cyl(x, y, z - 0.3, 1.25, 0.95, seg=8) - cyl(x, y, z + 0.35, 0.95, 0.7, seg=8)
    pillar = cyl(x, y, z, 0.28, 1.5, r2=0.2, seg=10) + cyl(x, y, z + 1.4, 0.5, 0.18, r2=0.3, seg=10)
    return basin + pillar


def stall(x, y, z, ang, rng):
    m = solid(0, 0, 0, 1.1, 0.7, 0.5, 'wood')
    m = m + tag(gabled(0.95, 1.2, 0.35, 0.12, 'roof', 'roof').rotate([0, 0, 90]), 'roof').translate([0, 0, 0.95])
    for sx in (-0.5, 0.5):
        for sy in (-0.3, 0.3):
            m = m + solid(sx, sy, 0.5, 0.1, 0.1, 0.5, 'wood')
    return m.rotate([0, 0, ang]).translate([x, y, z])


def garden(rng, w, d, kind):
    """garden at the back of a plot, local frame like a building but centred on the garden: w x d"""
    parts = []
    t = 0.38
    mat = 'broadleaf' if rng.rand() < 0.5 else 'stone'
    h = 0.62 if mat == 'broadleaf' else 0.5
    parts += [solid(0, d / 2, -0.3, w, t, h + 0.3, mat), solid(-w / 2, 0, -0.3, t, d, h + 0.3, mat), solid(w / 2, 0, -0.3, t, d, h + 0.3, mat)]
    if kind == 'beds':
        for k in range(3):
            parts.append(solid(0, -d / 2 + 0.45 + k * 0.5, -0.1, w - 1.0, 0.26, 0.3, 'broadleaf'))
    elif kind == 'hay':
        parts.append(haystack(0, 0, 0.0, min(0.85, w / 2 - 0.4)))
    elif kind == 'orchard':
        for sx in (-1, 1):
            parts.append(broadleaf(rng, 3.0, 0.9).translate([sx * w * 0.22, 0, -0.5]))
    return union(parts)


def tower(x, y, z, r, h, roof_h, zlow=None, rng=None, slits=2):
    """round tower: plinth, arrow slits, corbelled roof of slate"""
    zb = z - 0.5 if zlow is None else zlow
    body = tag(cyl(x, y, zb, r, z + h - zb, seg=16), 'stone') + tag(cyl(x, y, zb, r + 0.22, z + 0.7 - zb, r2=r + 0.05, seg=16), 'stone')
    body = body + tag(cyl(x, y, z + h - 0.35, r + 0.3, 0.35, r2=r + 0.16, seg=16), 'stone')
    cuts = []
    phi0 = np.radians(rng.uniform(0, 360)) if rng is not None else 0.0
    for k in range(slits):
        phi = phi0 + np.pi * k
        cuts.append(dark(x + (r - 0.1) * np.cos(phi), y + (r - 0.1) * np.sin(phi), z + h * 0.45, 0.45, 0.3, 0.9, np.degrees(phi)))
    roof = tag(cyl(x, y, z + h - 0.05, r + 0.3, roof_h + 0.05, r2=0.05, seg=16), 'roof')
    return (body - union(cuts)) + roof


# --------------------------------------------------------------------------
# placement: occupancy of rectangular plots
# --------------------------------------------------------------------------
class Plots:
    def __init__(self, c, limit=None, road_hw=None):
        self.c = c
        self.occ = np.zeros(c.X.shape, bool)
        self.limit = (c.a - 1.6) if limit is None else limit
        self.road_hw = c.p.road_w / 2 + 0.25 if road_hw is None else road_hw

    def mask(self, x, y, w, d, ang, pad=0.0):
        c = self.c
        r = 0.5 * np.hypot(w, d) + pad + 0.5
        i0, i1 = np.searchsorted(c.xs, [x - r, x + r])
        j0, j1 = np.searchsorted(c.ys, [y - r, y + r])
        sl = (slice(max(j0, 0), j1), slice(max(i0, 0), i1))
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        dx, dy = c.X[sl] - x, c.Y[sl] - y
        u, v = dx * ca + dy * sa, -dx * sa + dy * ca
        return sl, (np.abs(u) <= w / 2 + pad) & (np.abs(v) <= d / 2 + pad)

    def ok(self, x, y, w, d, ang, pad=0.1):
        c = self.c
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        loc = np.array([[sx * w / 2, sy * d / 2] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
        pts = np.stack([x + loc[:, 0] * ca - loc[:, 1] * sa, y + loc[:, 0] * sa + loc[:, 1] * ca], 1)
        if (hexdist(pts[:, 0], pts[:, 1]) > self.limit).any():
            return False
        if c.road is not None:
            dd, _ = c.road.query(pts[:, 0], pts[:, 1])
            if (dd < self.road_hw).any():
                return False
        if c.river is not None:
            dd, _ = c.river.query(pts[:, 0], pts[:, 1])
            if (dd < c.p.river_w / 2 + 0.9).any():
                return False
        if c.water_s is not None and any(c.s_at(px, py) < 1.4 for px, py in pts):
            return False
        if c.Z is not None:
            g = [c.ground(px, py) for px, py in pts]
            if max(g) - min(g) > 1.5:
                return False
        sl, m = self.mask(x, y, w, d, ang, pad)
        return not (self.occ[sl] & m).any()

    def claim(self, x, y, w, d, ang, pad=0.1):
        sl, m = self.mask(x, y, w, d, ang, pad)
        self.occ[sl] |= m
        self.c.claim(x, y, 0.5 * np.hypot(w, d) * 0.85)

    def ground_z(self, x, y, w, d, ang):
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        zs = [self.c.ground(x + lx * ca - ly * sa, y + lx * sa + ly * ca)
              for lx in (-w / 2, 0, w / 2) for ly in (-d / 2, 0, d / 2)]
        return min(zs)


def _along(poly, s_target, s_cum):
    p = np.array([np.interp(s_target, s_cum, poly[:, 0]), np.interp(s_target, s_cum, poly[:, 1])])
    q = np.array([np.interp(s_target + 0.3, s_cum, poly[:, 0]), np.interp(s_target + 0.3, s_cum, poly[:, 1])])
    q0 = np.array([np.interp(s_target - 0.3, s_cum, poly[:, 0]), np.interp(s_target - 0.3, s_cum, poly[:, 1])])
    t = q - q0
    return p, t / (np.linalg.norm(t) + 1e-12)


def frontage(c, plots, poly, rng, make, sides=(1, -1), gap=(0.0, 0.3), setback=(0.8, 1.4), wr=(2.0, 3.0),
             dr=(2.6, 3.4), p=1.0, s0=3.0, s1=None, hw=None, on_place=None):
    """Walk along a street and put buildings facing it on the given sides.  make(rng, w, d) -> (manifold, info)."""
    poly = np.asarray(poly)
    s_cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    hw = c.p.road_w / 2 if hw is None else hw
    out = []
    for side in sides:
        pos = s0
        end = s_cum[-1] if s1 is None else s1
        while pos < end - 1.0:
            w, d = rng.uniform(*wr), rng.uniform(*dr)
            if rng.rand() > p:
                pos += w * 0.7 + rng.uniform(*gap)
                continue
            pt, t = _along(poly, pos + w / 2, s_cum)
            n = np.array([-t[1], t[0]])
            off = hw + rng.uniform(*setback) + d / 2
            cx, cy = pt + side * n * off
            ang = np.degrees(np.arctan2(t[1], t[0])) + (0.0 if side > 0 else 180.0)
            if plots.ok(cx, cy, w, d, ang):
                z = plots.ground_z(cx, cy, w, d, ang)
                m, dd = make(rng, w, d)
                out.append(m.rotate([0, 0, ang + rng.uniform(-0.35, 0.35)]).translate([cx, cy, z + rng.uniform(-0.03, 0.03)]))   # no exactly shared edges
                plots.claim(cx, cy, w, d, ang)
                if on_place:
                    out += on_place(rng, cx, cy, z, w, dd if dd else d, ang)
                pos += w + rng.uniform(*gap)
            else:
                pos += 0.6
    return out


def back_garden(c, plots):
    def on_place(rng, cx, cy, z, w, d, ang):
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        gd = rng.uniform(1.8, 2.6)
        gw = w + rng.uniform(0.2, 1.0)
        # local +y of the building is its back: (-sin, cos)
        gx, gy = cx + (d / 2 + gd / 2 + 0.05) * (-sa), cy + (d / 2 + gd / 2 + 0.05) * ca
        if not plots.ok(gx, gy, gw, gd, ang, pad=0.05):
            return []
        plots.claim(gx, gy, gw, gd, ang, pad=0.05)
        g = garden(rng, gw, gd, rng.choice(['beds', 'beds', 'hay', 'orchard', 'plain']))
        return [g.rotate([0, 0, ang]).translate([gx, gy, c.ground(gx, gy)])]
    return on_place


# --------------------------------------------------------------------------
# village
# --------------------------------------------------------------------------
def _village_house(rng, w, d):
    r = rng.rand()
    if r < 0.10 and w > 2.6:
        m, dd = l_house(rng, w, d, floors=1, jetty=False, thatch=rng.rand() < 0.5)
        return m, dd
    if r < 0.20:
        return barn(rng, rng.uniform(3.6, 4.6), min(d, 3.0)), None
    two = rng.rand() < 0.45
    return townhouse(rng, w, d, floors=2 if two else 1, jetty=two and rng.rand() < 0.6, ridge='x' if rng.rand() < 0.65 else 'y',
                     timber=two and rng.rand() < 0.6, thatch=rng.rand() < 0.45), None


def village_parts(c):
    rng = c.rng
    plots = Plots(c)
    parts = []
    polys = list(c.road.polys) if c.road is not None else []
    # a ring lane stands in for the street of a village without a road: houses gather round a green
    if not polys:
        R = 8.2
        th = np.linspace(0, 2 * np.pi, 160)
        polys = [np.stack([R * np.cos(th), R * np.sin(th)], 1)]
        plots.road_hw = 0.0
        hw, sides = 0.0, (-1,)
        green = True
    else:
        hw, sides, green = c.p.road_w / 2, (1, -1), False
    # the church first, on the longest street, so it keeps a good plot
    main = max(polys, key=lambda q: len(q))
    s_cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(main, axis=0).T))]
    placed_church = False
    for frac in rng.permutation([0.34, 0.5, 0.62, 0.42, 0.7]):
        for side in rng.permutation(sides):
            pt, t = _along(main, s_cum[-1] * frac, s_cum)
            n = np.array([-t[1], t[0]])
            fw, fd = 7.0, 3.6
            off = hw + 1.7 + fd / 2
            cx, cy = pt + side * n * off
            ang = np.degrees(np.arctan2(t[1], t[0])) + (0.0 if side > 0 else 180.0)
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            cx, cy = cx + 0.3 * ca, cy + 0.3 * sa
            if plots.ok(cx, cy, fw, fd, ang):
                z = plots.ground_z(cx, cy, fw, fd, ang)
                parts.append(church(rng).rotate([0, 0, ang]).translate([cx, cy, z]))
                plots.claim(cx, cy, fw, fd, ang)
                # churchyard wall at the front is left open for the lane
                placed_church = True
                break
        if placed_church:
            break
    if green:
        parts.append(well(0.0, 0.0, c.ground(0, 0)))
    for poly in polys:
        parts += frontage(c, plots, poly, rng, _village_house, sides=sides, gap=(0.7, 2.4), setback=(0.8, 1.8),
                          wr=(2.2, 3.3), dr=(2.6, 3.5), p=0.8, s0=3.2, hw=hw, on_place=back_garden(c, plots))
    # a second, back row of small cottages and outbuildings where there is room
    for _ in range(60):
        if len(parts) > 16:
            break
        x, y = rng.uniform(-15, 15, 2)
        w, d = rng.uniform(1.9, 2.6), rng.uniform(2.0, 2.8)
        ang = rng.choice([0, 30, 60, 90, 120, 150]) + rng.uniform(-6, 6)
        if plots.ok(x, y, w, d, ang, pad=0.5):
            m = townhouse(rng, w, d, floors=1, jetty=False, ridge='x', thatch=rng.rand() < 0.6)
            parts.append(m.rotate([0, 0, ang]).translate([x, y, plots.ground_z(x, y, w, d, ang)]))
            plots.claim(x, y, w, d, ang, pad=0.5)
    for _ in range(40):                                                    # haystacks and a well by the lane
        x, y = rng.uniform(-14, 14, 2)
        if plots.ok(x, y, 1.8, 1.8, 0.0, pad=0.2):
            parts.append(haystack(x, y, c.ground(x, y), rng.uniform(0.6, 0.85)))
            plots.claim(x, y, 1.8, 1.8, 0.0, pad=0.2)
            if len(c.items) > 40:
                break
    from tiles import scatter_trees
    parts += scatter_trees(c, 9, p_conifer=0.25, rmin=1.2, rmax=2.0, hmin=4.5, hmax=7.5)
    # trodden earth around buildings (colour only)
    from scipy.ndimage import gaussian_filter
    c.extra['urban'] = np.clip(gaussian_filter(plots.occ.astype(np.float32), 1.6 / c.p.res) * 2.2, 0, 1).astype(np.float32)
    return parts


# --------------------------------------------------------------------------
# city
# --------------------------------------------------------------------------
CATH_W, CATH_D, CATH_OX = 12.0, 7.0, 0.9          # cathedral footprint (with transepts) and its offset from the origin


def city_streets(c, net, wall_a=15.0):
    """Main streets plus perpendicular side lanes.  The cathedral site is chosen first, beside the main street,
    and the lanes keep clear of it.  Stores c.cath = (x, y, angle, transept)."""
    rng = np.random.RandomState(c.seed + 77)
    c.road = net
    plots = Plots(c, limit=wall_a - 1.5, road_hw=c.p.road_w / 2 + 0.15)
    main = net.polys[0]
    s_cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(main, axis=0).T))]
    centre_s = float(s_cum[np.argmin(np.hypot(main[:, 0], main[:, 1]))])
    pc, t = _along(main, centre_s, s_cum)
    n = np.array([-t[1], t[0]])
    ang = float(np.degrees(np.arctan2(t[1], t[0])))
    c.cath = None
    for tr in (True, False):
        for side in rng.permutation([1, -1]):
            for u0 in (1.0, 0.0, 2.0, -1.0):
                for v0 in (6.6, 6.0, 7.2):
                    org = pc + t * u0 + n * side * v0
                    ctr = org + t * CATH_OX
                    w, d = CATH_W, (CATH_D if tr else 5.0)
                    if plots.ok(ctr[0], ctr[1], w, d, ang, pad=0.2):
                        c.cath = (float(org[0]), float(org[1]), ang, tr, float(ctr[0]), float(ctr[1]), w, d)
                        plots.claim(ctr[0], ctr[1], w, d, ang, pad=0.2)
                        break
                if c.cath:
                    break
            if c.cath:
                break
        if c.cath:
            break
    polys = [np.asarray(q) for q in net.polys]
    widths = [np.ones(len(q)) for q in polys]
    # side lanes
    for q in list(net.polys):
        sc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        c0 = float(sc[np.argmin(np.hypot(q[:, 0], q[:, 1]))])
        pos = c0 + rng.choice([-1, 1]) * 6.5
        side = int(rng.choice([-1, 1]))
        for k in range(8):
            pt, tt = _along(q, pos, sc)
            if hexdist(np.array([pt[0]]), np.array([pt[1]]))[0] < wall_a - 4.0 and abs(pos - c0) > 5.0:
                nn = np.array([-tt[1], tt[0]]) * side
                pts = [pt]
                ph = rng.uniform(0, 6.28)
                for i in range(1, 80):
                    s = 0.4 * i
                    p_ = pt + nn * s + tt * (0.7 * np.sin(s / 3.0 + ph) * min(1.0, s / 4.0))
                    if hexdist(np.array([p_[0]]), np.array([p_[1]]))[0] > wall_a - 1.9:
                        break
                    if c.cath is not None:
                        ca, sa = np.cos(np.radians(c.cath[2])), np.sin(np.radians(c.cath[2]))
                        dx, dy = p_[0] - c.cath[4], p_[1] - c.cath[5]
                        if abs(dx * ca + dy * sa) < c.cath[6] / 2 + 1.0 and abs(-dx * sa + dy * ca) < c.cath[7] / 2 + 1.0:
                            break
                    pts.append(p_)
                if len(pts) > 12:
                    polys.append(np.array(pts))
                    widths.append(np.full(len(pts), 0.6))
            pos += rng.uniform(6.5, 8.0) * rng.choice([-1, 1] if k == 0 else [1])
            side = -side if rng.rand() < 0.6 else side
            if pos > sc[-1] - 3:
                break
    return PathNet(polys, widths)


def _merlons(x0, y0, x1, y1, z, gate_mid=None, gate_half=3.2, thick=0.55):
    L = float(np.hypot(x1 - x0, y1 - y0))
    ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    out = []
    for s in np.arange(0.9, L - 0.6, 1.5):
        t = s / L
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if gate_mid is not None and np.hypot(x - gate_mid[0], y - gate_mid[1]) < gate_half:
            continue
        out.append(box(x, y, z, 0.8, thick, 0.62, ang))
    return out


def city_parts(c):
    from tiles import wall_ring, scatter_trees
    rng = c.rng
    wall_a, th, wh = 15.0, 1.1, 3.6
    gates = c.exits
    z0 = c.ground(0, 0)
    parts = [tag(wall_ring(wall_a, th, wh, gates, z0=z0), 'stone')]
    vr = (wall_a + th / 2) / np.cos(30 * DEG)
    verts = [np.array([vr * np.cos((30 + 60 * k) * DEG), vr * np.sin((30 + 60 * k) * DEG)]) for k in range(6)]
    for k in range(6):                                                    # crenellations round the wall walk
        v0, v1 = verts[k], verts[(k + 1) % 6]
        mid = (v0 + v1) / 2
        gm = mid if (k in gates) else None
        parts += _merlons(v0[0], v0[1], v1[0], v1[1], z0 + wh, gm)
    for k in range(6):
        parts.append(tower(verts[k][0], verts[k][1], z0, 1.5, 7.0, 2.8, rng=rng))
    for e in gates:                                                       # gatehouses: flanking towers and a lintel
        ang = 60 * e * DEG
        for s in (-1, 1):
            gx = (wall_a + th / 2) * np.cos(ang) - s * 3.0 * np.sin(ang)
            gy = (wall_a + th / 2) * np.sin(ang) + s * 3.0 * np.cos(ang)
            parts.append(tower(gx, gy, z0, 1.1, 5.6, 2.0, rng=rng, slits=1))
        mx, my = (wall_a + th / 2) * np.cos(ang), (wall_a + th / 2) * np.sin(ang)
        parts.append(solid(0, 0, z0 + 2.6, th + 0.5, 5.8, wh - 2.6 + 0.35, 'stone').rotate([0, 0, 60 * e]).translate([mx, my, 0]))
    plots = Plots(c, limit=wall_a - 1.5, road_hw=c.p.road_w / 2 * 0.62 + 0.1)
    # market square and its furniture
    sl, m = plots.mask(0, 0, 7.0, 7.0, 0.0)
    rr = np.hypot(c.X[sl], c.Y[sl]) < 3.5
    plots.occ[sl] |= rr
    parts.append(fountain(0.0, 0.0, z0 + 0.0))
    for k in range(4):
        a = np.radians(45 + 90 * k + rng.uniform(-8, 8))
        sx, sy = 2.55 * np.cos(a), 2.55 * np.sin(a)
        if c.road is not None:
            dd, _ = c.road.query(np.array([sx]), np.array([sy]))
            if dd[0] < c.p.road_w / 2 + 0.6:
                continue
        parts.append(stall(sx, sy, z0, np.degrees(a) + 90, rng))
    # the cathedral planned by city_streets, beside the square
    if c.cath is not None:
        ox, oy, cang, tr, cx_, cy_, cw, cd = c.cath
        parts.append(cathedral(rng, tr).rotate([0, 0, cang]).translate([ox, oy, plots.ground_z(cx_, cy_, cw, cd, cang)]))
        plots.claim(cx_, cy_, cw, cd, cang, pad=0.2)

    def make(rng, w, d):
        floors = int(rng.choice([2, 2, 3]))
        return townhouse(rng, w, d, floors=floors, jetty=True, ridge='y' if rng.rand() < 0.65 else 'x',
                         timber=rng.rand() < 0.55, chimney=rng.rand() < 0.8, rh=rng.uniform(1.3, 2.0)), None
    for k, poly in enumerate(c.road.polys):                               # terraces along every street and lane
        lane = k >= len(gates)
        parts += frontage(c, plots, poly, rng, make, sides=(1, -1), gap=(0.0, 0.12), setback=(0.25, 0.5),
                          wr=(1.9, 2.7), dr=(2.7, 3.4), p=0.95, s0=0.5 if lane else 2.0,
                          hw=c.p.road_w / 2 * (0.6 if lane else 1.0))
    for _ in range(900):                                                  # fill the blocks behind with smaller houses
        if len(parts) > 150:
            break
        x, y = rng.uniform(-14, 14, 2)
        w, d = rng.uniform(1.8, 2.5), rng.uniform(2.0, 2.8)
        ang = rng.choice([0, 30, 60, 90, 120, 150]) + rng.uniform(-5, 5)
        if plots.ok(x, y, w, d, ang, pad=0.12):
            m = townhouse(rng, w, d, floors=int(rng.choice([1, 2])), jetty=rng.rand() < 0.5, ridge='x' if rng.rand() < 0.5 else 'y',
                          timber=rng.rand() < 0.4, chimney=rng.rand() < 0.7)
            parts.append(m.rotate([0, 0, ang]).translate([x, y, plots.ground_z(x, y, w, d, ang)]))
            plots.claim(x, y, w, d, ang, pad=0.12)
    for _ in range(30):                                                   # a few trees in the yards
        x, y = rng.uniform(-13, 13, 2)
        if plots.ok(x, y, 1.4, 1.4, 0.0, pad=0.1):
            parts.append(broadleaf(rng, rng.uniform(3.2, 4.6), rng.uniform(0.9, 1.3)).translate([x, y, c.ground(x, y) - 0.5]))
            plots.claim(x, y, 1.4, 1.4, 0.0, pad=0.1)
    from scipy.ndimage import gaussian_filter
    inside = smoothstep((wall_a - 0.2 - hexdist(c.X, c.Y)) / 1.5)
    c.extra['urban'] = np.clip(inside * 0.85, 0, 1).astype(np.float32)
    c.extra['plaza'] = smoothstep((3.7 - np.hypot(c.X, c.Y)) / 0.5).astype(np.float32)
    return parts
