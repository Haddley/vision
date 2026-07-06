#!/usr/bin/env python3
"""Render perspective previews from inside the generated cubemap.

Implements the OpenGL cubemap face-selection math (GL spec table 8.19),
so what you see here is exactly what samplerCube will show in the app.

Usage: python3 tools/preview.py [outdir] [skybox|skybox_dim]
"""

import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
SKY = os.path.join(HERE, "..", "assets",
                   sys.argv[2] if len(sys.argv) > 2 else "skybox")

faces = {}
for name in ["px", "nx", "py", "ny", "pz", "nz"]:
    faces[name] = np.asarray(Image.open(os.path.join(SKY, name + ".png"))
                             .convert("RGB"))
FS = faces["px"].shape[0]


def sample(dirs):
    """dirs: (...,3) array of ray directions -> (...,3) uint8 colors."""
    rx, ry, rz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    ax, ay, az = np.abs(rx), np.abs(ry), np.abs(rz)
    out = np.zeros(dirs.shape[:-1] + (3,), dtype=np.uint8)

    sel_x = (ax >= ay) & (ax >= az)
    sel_y = (ay > ax) & (ay >= az)
    sel_z = ~(sel_x | sel_y)

    def put(mask, name, sc, tc, ma):
        if not mask.any():
            return
        s = (sc[mask] / ma[mask] + 1) / 2
        t = (tc[mask] / ma[mask] + 1) / 2
        col = np.clip((s * (FS - 1)).astype(int), 0, FS - 1)
        row = np.clip((t * (FS - 1)).astype(int), 0, FS - 1)
        out[mask] = faces[name][row, col]

    put(sel_x & (rx > 0), "px", -rz, -ry, ax)
    put(sel_x & (rx <= 0), "nx", rz, -ry, ax)
    put(sel_y & (ry > 0), "py", rx, rz, ay)
    put(sel_y & (ry <= 0), "ny", rx, -rz, ay)
    put(sel_z & (rz > 0), "pz", rx, -ry, az)
    put(sel_z & (rz <= 0), "nz", -rx, -ry, az)
    return out


def render(yaw_deg, pitch_deg, w=960, h=720, fov_deg=95):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    f = 1 / np.tan(np.radians(fov_deg) / 2)
    xs = np.linspace(-1, 1, w) * (w / h)
    ys = np.linspace(1, -1, h)
    gx, gy = np.meshgrid(xs, ys)
    d = np.stack([gx, gy, -f * np.ones_like(gx)], axis=-1)
    # pitch about X, then yaw about Y
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    y, z = cp * y - sp * z, sp * y + cp * z
    x, z = cy_ * x + sy_ * z, -sy_ * x + cy_ * z
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return Image.fromarray(sample(dirs))


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else HERE
    views = {
        "view_front": (0, 0),      # -Z: Snellen chart
        "view_back": (180, 0),     # +Z: reception
        "view_left": (90, 0),      # -X: frames display
        "view_right": (-90, 0),    # +X: exam area
        "view_up": (0, 60),
        "view_down": (0, -60),
    }
    for name, (yaw, pitch) in views.items():
        render(yaw, pitch).save(os.path.join(outdir, name + ".png"))
        print("wrote", name + ".png")
