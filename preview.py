#!/usr/bin/env python3
"""preview.py - software-rendered contact sheet of STL tiles (no GPU needed).
   python3 preview.py out.png a.stl b.stl ...   [--cols 4] [--size 420]"""
import sys, os, argparse
import numpy as np
import trimesh
from PIL import Image, ImageDraw


def render(mesh, size=420, az=-35, el=42, ssaa=2):
    S = size * ssaa
    v = mesh.vertices.copy()
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    w = (Rx @ Rz @ v.T).T                       # x right, y = depth (away), z up on screen
    # after tilt: screen x = w[:,0], screen y = w[:,2], depth = -w[:,1]
    sx, sy, dp = w[:, 0], w[:, 2], -w[:, 1]
    sc = S * 0.46 / 25.0
    px = S / 2 + sx * sc
    py = S * 0.60 - sy * sc
    F = mesh.faces
    n = mesh.face_normals
    nw = (Rx @ Rz @ n.T).T
    light = np.array([-0.5, -0.55, 0.65]); light /= np.linalg.norm(light)
    lam = np.clip((Rx @ Rz @ n.T).T @ (Rx @ Rz @ light), 0, 1)
    lam = 0.28 + 0.72 * lam
    zb = np.full((S, S), -1e9)
    img = np.zeros((S, S, 3), np.float32)
    bg = np.array([236, 233, 226], np.float32)
    img[:] = bg
    base = np.array([196, 168, 132], np.float32)
    for f in range(len(F)):
        i0, i1, i2 = F[f]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        xmin, xmax = int(max(0, np.floor(min(x0, x1, x2)))), int(min(S - 1, np.ceil(max(x0, x1, x2))))
        ymin, ymax = int(max(0, np.floor(min(y0, y1, y2)))), int(min(S - 1, np.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        l2 = 1 - l0 - l1
        m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not m.any():
            continue
        d = l0 * dp[i0] + l1 * dp[i1] + l2 * dp[i2]
        sub = zb[ymin:ymax + 1, xmin:xmax + 1]
        upd = m & (d > sub)
        if not upd.any():
            continue
        sub[upd] = d[upd]
        zz = (mesh.vertices[i0][2] + mesh.vertices[i1][2] + mesh.vertices[i2][2]) / 3
        tint = np.array([1.0, 1.0, 1.0]) * (0.92 + 0.012 * min(zz, 12))
        img[ymin:ymax + 1, xmin:xmax + 1][upd] = np.clip(base * lam[f] * tint, 0, 255)
    out = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--size', type=int, default=420)
    ap.add_argument('--az', type=float, default=-35)
    ap.add_argument('--el', type=float, default=42)
    a = ap.parse_args()
    tiles = []
    for f in a.files:
        im = render(trimesh.load(f), a.size, a.az, a.el)
        d = ImageDraw.Draw(im)
        d.text((8, 6), os.path.splitext(os.path.basename(f))[0], fill=(40, 40, 40))
        tiles.append(im)
    rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = Image.new('RGB', (a.cols * a.size, rows * a.size), (236, 233, 226))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % a.cols) * a.size, (i // a.cols) * a.size))
    sheet.save(a.out)
    print('wrote', a.out, sheet.size)


if __name__ == '__main__':
    main()
