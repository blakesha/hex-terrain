"""
hexlib.py - core of the hex tile generator.

Coordinate system (mm): tile centre at (0,0); Z=0 is the print bed.
Pointy-top hexagon: the six edge midpoints are at angles 0,60,...,300 deg
(edge index i -> angle 60*i).  Every road/river crosses an edge at its
midpoint, perpendicular to the edge, with a fixed cross-section, so any
tile joins any neighbour.
"""
from dataclasses import dataclass
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection, Mesh, OpType
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, gaussian_filter

DEG = np.pi / 180.0


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    flat: float = 40.0          # flat-to-flat width of the nominal hex cell
    clearance: float = 0.1      # trimmed off every side so tiles fit after printing
    base: float = 5.0           # base slab thickness
    magnets: bool = True        # side pockets for disc magnets
    magnets_per_edge: int = 2   # 2 = an N/S pair per edge, so ANY edge mates with ANY edge in ANY rotation
    magnet_pitch: float = 8.0   # spacing between the pair
    magnet_d: float = 3.0
    magnet_t: float = 1.0
    magnet_fit: float = 0.15    # extra clearance for glue / printing tolerance
    res: float = 0.3            # heightmap grid resolution
    rim: float = 2.0            # perfectly flat rim width
    fade: float = 10.0          # distance over which terrain rises from the rim
    road_w: float = 4.0
    road_h: float = 0.5
    river_w: float = 6.0
    river_depth: float = 1.0
    chamfer: float = 0.4        # bottom-edge chamfer (reduces elephant's foot)
    organic: float = 1.0        # 0 = geometric roads/rivers, 1 = meandering

    @property
    def a(self):                # actual apothem of the printed tile
        return self.flat / 2 - self.clearance


# --------------------------------------------------------------------------
# noise (vectorised gradient noise)
# --------------------------------------------------------------------------
def _perlin(x, y, seed):
    rng = np.random.RandomState(seed)
    p = rng.permutation(256)
    p = np.concatenate([p, p])
    xi = np.floor(x).astype(int)
    yi = np.floor(y).astype(int)
    xf, yf = x - xi, y - yi
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    u, v = fade(xf), fade(yf)
    xi &= 255
    yi &= 255
    aa, ab = p[p[xi] + yi], p[p[xi] + yi + 1]
    ba, bb = p[p[xi + 1] + yi], p[p[xi + 1] + yi + 1]

    def g(h, dx, dy):
        th = h * (2 * np.pi / 256.0)
        return np.cos(th) * dx + np.sin(th) * dy

    x1 = (1 - u) * g(aa, xf, yf) + u * g(ba, xf - 1, yf)
    x2 = (1 - u) * g(ab, xf, yf - 1) + u * g(bb, xf - 1, yf - 1)
    return (1 - v) * x1 + v * x2


def fbm(X, Y, seed, octaves=4, freq=0.1, lac=2.0, gain=0.5):
    total, amp, norm, f = 0.0, 1.0, 0.0, freq
    for o in range(octaves):
        total = total + amp * _perlin(X * f + 11.3 * o, Y * f + 7.1 * o, seed + 31 * o)
        norm += amp
        amp *= gain
        f *= lac
    return np.clip(total / norm * 1.6, -1, 1)      # ~[-1,1]


def ridged(X, Y, seed, octaves=5, freq=0.06, lac=2.1, gain=0.5):
    total, amp, norm, f = 0.0, 1.0, 0.0, freq
    w = 1.0
    for o in range(octaves):
        n = 1.0 - np.abs(_perlin(X * f + 5.7 * o, Y * f + 3.3 * o, seed + 17 * o) * 1.6)
        n = np.clip(n, 0, 1) ** 2
        n = n * w
        w = np.clip(n * 1.6, 0, 1)
        total = total + amp * n
        norm += amp
        amp *= gain
        f *= lac
    return np.clip(total / norm, 0, 1)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# hex helpers
# --------------------------------------------------------------------------
_N = np.array([[np.cos(60 * i * DEG), np.sin(60 * i * DEG)] for i in range(6)])


def hexdist(X, Y):
    """Apothem-style hex 'radius' of a point (== a on the tile boundary)."""
    return np.max(np.stack([X * _N[i, 0] + Y * _N[i, 1] for i in range(6)]), axis=0)


def hex_prism(a, h, z0=0.0, scale=1.0):
    R = a / np.cos(30 * DEG)
    ang = (30 + 60 * np.arange(6)) * DEG
    poly = np.stack([R * np.cos(ang), R * np.sin(ang)], 1)
    return Manifold.extrude(CrossSection([poly]), h, 0, 0.0, (scale, scale)).translate([0, 0, z0])


# --------------------------------------------------------------------------
# road / river paths
# --------------------------------------------------------------------------
class PathNet:
    """A network of polylines with a per-vertex width scale.
    query() returns the distance to the network *divided by the local width scale*,
    so callers keep using fixed width constants while the path swells and narrows."""

    def __init__(self, polys, widths=None):
        self.polys = [np.asarray(p, float) for p in polys]
        A, B, W = [], [], []
        for i, p in enumerate(self.polys):
            w = np.ones(len(p)) if widths is None else np.asarray(widths[i], float)
            for k in range(len(p) - 1):
                A.append(p[k]); B.append(p[k + 1]); W.append(0.5 * (w[k] + w[k + 1]))
        self.A, self.B, self.W = np.array(A), np.array(B), np.array(W)
        self.D = self.B - self.A
        self.L2 = (self.D ** 2).sum(1) + 1e-12
        self.ang = np.arctan2(self.D[:, 1], self.D[:, 0])

    @staticmethod
    def _resample(poly, step=0.4):
        poly = np.asarray(poly, float)
        seg = np.hypot(*np.diff(poly, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1] / step) + 1)
        t = np.linspace(0, s[-1], n)
        return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1), t

    @staticmethod
    def from_exits(exits, a, ext=3.0, rng=None, organic=0.0, wamp=0.0):
        """Paths from edge midpoints (always leaving the edge perpendicular, at the
        same place and width) into the tile.  `organic` scales the meander, which
        fades to nothing near every edge so neighbouring tiles still line up."""
        rng = rng if rng is not None else np.random.RandomState(0)
        dirs = [np.array([np.cos(60 * e * DEG), np.sin(60 * e * DEG)]) for e in exits]
        pts = [(a + ext) * d for d in dirs]
        base, center_fixed = [], False
        if len(exits) == 1:
            end = dirs[0] * (rng.uniform(0.0, 4.0) * organic)
            base.append(np.array([pts[0], end]))
        elif len(exits) == 2:
            P0, P3 = pts
            span = np.linalg.norm(P0 - P3)
            k0, k1 = [span * (0.30 + 0.30 * (1 - organic) + 0.18 * organic * rng.rand()) for _ in range(2)]
            if np.allclose(P0, -P3):                 # straight through: keep the control points on the line
                k0 = k1 = span / 3
            P1, P2 = P0 - dirs[0] * k0, P3 - dirs[1] * k1
            t = np.linspace(0, 1, 80)[:, None]
            base.append((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)
        else:
            c = rng.uniform(-1, 1, 2) * 1.2 * organic
            center_fixed = True
            for d, pt in zip(dirs, pts):
                base.append(np.array([pt, d * 9.0, c]))
        out, wid = [], []
        for poly in base:
            dense, s = PathNet._resample(poly)
            if organic > 0:
                tang = np.gradient(dense, axis=0)
                tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
                nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
                ph = rng.uniform(0, 2 * np.pi, 4)
                wig = (2.3 * np.sin(2 * np.pi * s / rng.uniform(15, 22) + ph[0])
                       + 0.4 * np.sin(2 * np.pi * s / rng.uniform(8.0, 11.0) + ph[1]))
                hd = hexdist(dense[:, 0], dense[:, 1])
                taper = smoothstep((a - hd - 1.0) / 8.0)
                if center_fixed or len(exits) == 1:        # junctions and dead ends stay put
                    taper = taper * smoothstep(np.hypot(*(dense - poly[-1]).T) / 6.0)
                dense = dense + (organic * wig * taper)[:, None] * nrm
                # round off any tight kinks (keeps the radial run-in at the edge: it is straight anyway)
                dense = gaussian_filter1d(dense, sigma=2.2 / 0.4, axis=0, mode='nearest')
                wsc = 1.0 + wamp * (0.65 * np.sin(2 * np.pi * s / rng.uniform(8, 13) + ph[2])
                                    + 0.35 * np.sin(2 * np.pi * s / rng.uniform(4, 6) + ph[3])) * taper
            else:
                wsc = np.ones(len(dense))
            out.append(dense); wid.append(wsc)
        return PathNet(out, wid)

    def query(self, X, Y):
        """width-normalised distance to the network and the local tangent angle"""
        P = np.stack([np.ravel(X), np.ravel(Y)], 1)
        best = np.full(len(P), 1e9)
        bi = np.zeros(len(P), int)
        for k in range(len(self.A)):
            t = np.clip(((P - self.A[k]) @ self.D[k]) / self.L2[k], 0, 1)
            q = self.A[k] + t[:, None] * self.D[k]
            d = np.hypot(P[:, 0] - q[:, 0], P[:, 1] - q[:, 1]) / self.W[k]
            m = d < best
            best[m] = d[m]
            bi[m] = k
        return best.reshape(np.shape(X)), self.ang[bi].reshape(np.shape(X))


# --------------------------------------------------------------------------
# mesh <-> manifold
# --------------------------------------------------------------------------
def to_manifold(mesh):
    return Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                         tri_verts=np.asarray(mesh.faces, np.uint32)))


def to_trimesh(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.array(mm.vert_properties)[:, :3], np.array(mm.tri_verts), process=False)


def clean_mesh(t):
    """merge duplicate vertices and drop zero-area sliver triangles (left by booleans)"""
    t = trimesh.Trimesh(t.vertices.astype(np.float32).astype(np.float64), t.faces, process=False)
    t.merge_vertices(digits_vertex=5)
    t.update_faces(t.area_faces > 1e-9)
    t.remove_unreferenced_vertices()
    return t


def union(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return Manifold.batch_boolean(parts, OpType.Add)


def heightfield_solid(xs, ys, Z):
    """Closed solid under heightmap Z (ny,nx) down to z=0."""
    ny, nx = Z.shape
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    bot = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    verts = np.vstack([top, bot])
    off = nx * ny
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1),
         np.stack([a, c, b], 1) + off, np.stack([a, d, c], 1) + off]

    def strip(t0, t1, flip):
        b0, b1 = t0 + off, t1 + off
        if not flip:
            return [np.stack([b0, b1, t1], 1), np.stack([b0, t1, t0], 1)]
        return [np.stack([b1, b0, t0], 1), np.stack([b1, t0, t1], 1)]

    F += strip(idx[0, :-1], idx[0, 1:], False)        # y min
    F += strip(idx[-1, :-1], idx[-1, 1:], True)       # y max
    F += strip(idx[:-1, 0], idx[1:, 0], True)         # x min
    F += strip(idx[:-1, -1], idx[1:, -1], False)      # x max
    mesh = trimesh.Trimesh(verts, np.vstack(F), process=False)
    return to_manifold(mesh)


# --------------------------------------------------------------------------
# primitives used by the structures / trees
# --------------------------------------------------------------------------
def box(cx, cy, z0, w, d, h, ang=0.0):
    m = Manifold.cube([w, d, h]).translate([-w / 2, -d / 2, 0])
    if ang:
        m = m.rotate([0, 0, ang])
    return m.translate([cx, cy, z0])


def cyl(cx, cy, z0, r, h, r2=None, seg=16):
    r2 = r if r2 is None else r2
    return Manifold.cylinder(h, r, max(r2, 0.02), seg).translate([cx, cy, z0])


def gable_roof(span, length, rh, overhang=0.0):
    """Gable roof spanning `span` in x, ridge running along y, base at z=0."""
    poly = np.array([[-span / 2 - overhang, 0], [span / 2 + overhang, 0], [0, rh]])
    return Manifold.extrude(CrossSection([poly]), length).rotate([90, 0, 0]).translate([0, length / 2, 0])


def pyramid_roof(w, d, rh):
    poly = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    return Manifold.extrude(CrossSection([poly]), rh, 0, 0.0, (0.03, 0.03))


def lathe(profile, n=10, jitter=0.0, rng=None):
    """Revolve (r,z) profile (bottom->top, first/last r==0 are poles)."""
    prof = np.asarray(profile, float)
    th = np.arange(n) * 2 * np.pi / n
    verts, rings = [], []
    for k, (r, z) in enumerate(prof):
        if r == 0:
            verts.append([0, 0, z]); rings.append(('pole', len(verts) - 1))
        else:
            start = len(verts)
            j = 1.0 + (jitter * (rng.rand(n) * 2 - 1) if jitter else 0)
            rr = r * j
            for i in range(n):
                verts.append([rr[i] * np.cos(th[i]), rr[i] * np.sin(th[i]), z])
            rings.append(('ring', start))
    faces = []
    for k in range(len(prof) - 1):
        (t0, s0), (t1, s1) = rings[k], rings[k + 1]
        for j in range(n):
            j2 = (j + 1) % n
            if t0 == 'pole':
                faces.append([s0, s1 + j2, s1 + j])
            elif t1 == 'pole':
                faces.append([s0 + j, s0 + j2, s1])
            else:
                faces.append([s0 + j, s0 + j2, s1 + j2])
                faces.append([s0 + j, s1 + j2, s1 + j])
    return to_manifold(trimesh.Trimesh(np.array(verts), np.array(faces), process=False))


def conifer(rng, H, R, tiers=3):
    rt = 0.55
    zt = max(0.9, 0.16 * H)
    pts = [(0, 0), (rt, 0), (rt, zt)]
    z, r = zt, rt
    radii = [R * (1 - 0.24 * k) for k in range(tiers)]
    out, rprev = 0.0, rt
    for rk in radii:
        out += 1.25 * max(0.0, rk - rprev)
        rprev = 0.3 * rk
    th = max(1.1, (H - zt - out) / tiers)
    for k, rk in enumerate(radii):
        z = z + 1.25 * max(0.0, rk - r)   # ~39 deg underside (printable without supports)
        pts.append((rk, z))
        last = (k == tiers - 1)
        r = 0.0 if last else 0.3 * rk
        z = z + th
        pts.append((r, z))
    return lathe(pts, n=11, jitter=0.10, rng=rng)


def broadleaf(rng, H, R):
    rt = 0.55
    zt = max(1.0, 0.3 * H)
    zc = zt + 1.25 * (R - rt)
    pts = [(0, 0), (rt, 0), (rt, zt), (R, zc)]
    hh = max(1.5, H - zc)
    for ph in (25, 50, 70, 85):
        pts.append((R * np.cos(ph * DEG), zc + hh * np.sin(ph * DEG)))
    pts.append((0, zc + hh))
    return lathe(pts, n=12, jitter=0.14, rng=rng)


def rock(rng, x, y, z, r):
    s = Manifold.sphere(r, 8).scale([1.0, rng.uniform(0.7, 1.1), rng.uniform(0.5, 0.8)])
    return s.rotate([0, 0, rng.uniform(0, 360)]).translate([x, y, z])
