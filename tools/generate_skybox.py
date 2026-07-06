#!/usr/bin/env python3
"""Generate an optometrist-office cubemap skybox (6 PNG faces).

The cube IS the room: each face is a wall, the ceiling, or the floor.
Faces are saved pre-oriented for OpenGL cubemap targets:

    px.png -> GL_TEXTURE_CUBE_MAP_POSITIVE_X  (right wall,  +X)
    nx.png -> ..NEGATIVE_X                    (left wall,   -X)
    py.png -> ..POSITIVE_Y                    (ceiling,     +Y)
    ny.png -> ..NEGATIVE_Y                    (floor,       -Y)
    pz.png -> ..POSITIVE_Z                    (back wall,   +Z)
    nz.png -> ..NEGATIVE_Z                    (front wall,  -Z)

Walls are authored "as seen from the room center" and then mirrored
horizontally, because GL cubemap faces are defined as viewed from
outside the cube.
"""

import os
from PIL import (Image, ImageChops, ImageDraw, ImageEnhance, ImageFont,
                 ImageOps)

S = 2048  # face size in pixels
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
OUT_BRIGHT = os.path.join(ASSETS, "skybox")
OUT_DIM = os.path.join(ASSETS, "skybox_dim")  # lights down for the eye test

# ---------------------------------------------------------------- palette
WALL_TOP = (247, 244, 238)
WALL_BOT = (225, 220, 208)
TEAL = (46, 139, 139)
TEAL_DARK = (33, 107, 107)
TEAL_DEEP = (24, 82, 82)
WOOD = (185, 155, 114)
WOOD_DARK = (150, 120, 84)
BASE = (138, 131, 117)     # baseboard
TRIM = (252, 250, 246)     # crown trim
GRAY = (90, 92, 96)
GRAY_LIGHT = (160, 163, 168)
INK = (40, 42, 46)

FONT_CANDIDATES_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

_font_cache = {}

def font(size, bold=True):
    key = (size, bold)
    if key not in _font_cache:
        for path in (FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES):
            if os.path.exists(path):
                _font_cache[key] = ImageFont.truetype(path, size)
                break
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------- shared
BASEBOARD_H = 64
CROWN_H = 46

def new_wall(accent=False):
    """Blank wall with vertical gradient, crown trim and baseboard."""
    img = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(img)
    top = TEAL if accent else WALL_TOP
    bot = TEAL_DARK if accent else WALL_BOT
    for y in range(S):
        t = y / S
        d.line([(0, y), (S, y)], fill=lerp(top, bot, t * 0.9))
    # crown trim
    d.rectangle([0, 0, S, CROWN_H], fill=TRIM)
    d.line([(0, CROWN_H), (S, CROWN_H)], fill=(200, 196, 186), width=4)
    # baseboard
    d.rectangle([0, S - BASEBOARD_H, S, S], fill=BASE)
    d.line([(0, S - BASEBOARD_H), (S, S - BASEBOARD_H)],
           fill=(110, 104, 92), width=5)
    return img, d


def edge_ao(img, side=0.22, top=0.12, bottom=0.30, border=230):
    """Darken near the four borders so cube corners shade consistently.

    Faces meeting at an edge must use the same strength on that edge:
    wall sides <-> wall sides, wall top <-> ceiling, wall bottom <-> floor.
    """
    mask = Image.new("L", (S, S), 255)
    md = ImageDraw.Draw(mask)
    for i in range(border):
        f = (1 - i / border) ** 2
        md.line([(i, 0), (i, S)], fill=int(255 * (1 - side * f)))
    for i in range(border):
        f = (1 - i / border) ** 2
        md.line([(S - 1 - i, 0), (S - 1 - i, S)], fill=int(255 * (1 - side * f)))
    cols = mask.copy()
    mask = Image.new("L", (S, S), 255)
    md = ImageDraw.Draw(mask)
    for i in range(border):
        f = (1 - i / border) ** 2
        md.line([(0, i), (S, i)], fill=int(255 * (1 - top * f)))
    for i in range(border):
        f = (1 - i / border) ** 2
        md.line([(0, S - 1 - i), (S, S - 1 - i)], fill=int(255 * (1 - bottom * f)))
    mask = ImageChops.darker(cols, mask)
    black = Image.new("RGB", (S, S), (0, 0, 0))
    return Image.composite(img, black, mask)


def dim_face(img):
    """Lights-down version of a face: dark, slightly cool, desaturated."""
    img = ImageEnhance.Brightness(img).enhance(0.30)
    img = ImageEnhance.Color(img).enhance(0.70)
    r, g, b = img.split()
    r = r.point(lambda v: int(v * 0.88))
    g = g.point(lambda v: int(v * 0.94))
    b = b.point(lambda v: min(255, int(v * 1.12)))
    return Image.merge("RGB", (r, g, b))


def glow(img, box, color=(255, 236, 190), reach=140, strength=70, radius=40):
    """Soft light spill around a rect (for things still lit in the dark)."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    for g in range(reach, 0, -4):
        a = int(strength * (1 - g / reach) ** 1.6)
        od.rounded_rectangle([x0 - g, y0 - g, x1 + g, y1 + g],
                             radius=radius + g, fill=color + (a,))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def save_wall(img, name, out):
    """Author view -> stored cubemap face (mirror horizontally)."""
    img = edge_ao(img)
    ImageOps.mirror(img).save(os.path.join(out, name), optimize=True)
    print("wrote", os.path.basename(out) + "/" + name)


def save_flat(img, name, strength, out):
    img = edge_ao(img, side=strength, top=strength, bottom=strength)
    img.save(os.path.join(out, name), optimize=True)
    print("wrote", os.path.basename(out) + "/" + name)


# ---------------------------------------------------------------- widgets
def draw_glasses(d, cx, cy, w, color, style="round"):
    """A pair of eyeglasses, front view, centered at (cx, cy)."""
    lw = max(6, w // 16)
    lens_w = int(w * 0.40)
    lens_h = int(lens_w * 0.62)
    gap = int(w * 0.08)
    lx0 = cx - gap // 2 - lens_w
    rx0 = cx + gap // 2
    ytop = cy - lens_h // 2
    r = lens_h // 2 if style == "round" else lens_h // 5
    for x0 in (lx0, rx0):
        d.rounded_rectangle([x0, ytop, x0 + lens_w, ytop + lens_h],
                            radius=r, outline=color, width=lw,
                            fill=(235, 242, 245))
    # bridge
    d.arc([cx - gap, ytop - gap, cx + gap, ytop + gap],
          start=200, end=340, fill=color, width=lw)
    # temple stubs
    d.line([(lx0 - int(w * 0.06), ytop + lens_h // 4), (lx0, ytop + lens_h // 5)],
           fill=color, width=lw)
    d.line([(rx0 + lens_w, ytop + lens_h // 5),
            (rx0 + lens_w + int(w * 0.06), ytop + lens_h // 4)],
           fill=color, width=lw)


def draw_plant(d, cx, base_y, h):
    pot_w = int(h * 0.36)
    pot_h = int(h * 0.30)
    d.polygon([(cx - pot_w // 2, base_y - pot_h), (cx + pot_w // 2, base_y - pot_h),
               (cx + int(pot_w * 0.38), base_y), (cx - int(pot_w * 0.38), base_y)],
              fill=(176, 108, 74))
    d.rectangle([cx - pot_w // 2, base_y - pot_h, cx + pot_w // 2,
                 base_y - pot_h + int(pot_h * 0.18)], fill=(196, 128, 92))
    top = base_y - pot_h
    green = (74, 128, 82)
    green2 = (94, 152, 100)
    for dx, tilt, gh, col in [(-40, -30, 0.62, green), (40, 30, 0.62, green),
                              (-15, -12, 0.72, green2), (15, 12, 0.72, green2),
                              (0, 0, 0.80, green)]:
        tip_x = cx + dx + tilt
        tip_y = top - int(h * gh)
        d.polygon([(cx + dx - 22, top), (cx + dx + 22, top), (tip_x, tip_y)],
                  fill=col)


def draw_chair_front(d, cx, floor_y, w):
    """Simple waiting-room chair, front view."""
    h_back = int(w * 1.05)
    h_seat = int(w * 0.42)
    seat_th = int(w * 0.14)
    leg_w = int(w * 0.07)
    d.rounded_rectangle([cx - w // 2, floor_y - h_back, cx + w // 2,
                         floor_y - h_seat + seat_th // 2],
                        radius=w // 8, fill=TEAL_DARK)
    d.rounded_rectangle([cx - w // 2 - 8, floor_y - h_seat,
                         cx + w // 2 + 8, floor_y - h_seat + seat_th],
                        radius=seat_th // 2, fill=TEAL)
    for lx in (cx - w // 2 + leg_w, cx + w // 2 - 2 * leg_w):
        d.rectangle([lx, floor_y - h_seat + seat_th, lx + leg_w, floor_y],
                    fill=(70, 70, 74))


def draw_framed(d, x0, y0, x1, y1, frame=(120, 96, 70), mat=(250, 249, 245)):
    d.rectangle([x0, y0, x1, y1], fill=frame)
    b = max(8, (x1 - x0) // 28)
    d.rectangle([x0 + b, y0 + b, x1 - b, y1 - b], fill=mat)
    return (x0 + b, y0 + b, x1 - b, y1 - b)


def draw_eye_logo(d, cx, cy, w):
    h = int(w * 0.52)
    d.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
              fill=(252, 252, 250), outline=TEAL_DEEP, width=max(6, w // 22))
    r = int(h * 0.42)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TEAL)
    r2 = int(r * 0.45)
    d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=INK)
    rg = max(4, r // 5)
    d.ellipse([cx + r2 // 3, cy - r2 + rg, cx + r2 // 3 + 2 * rg,
               cy - r2 + 3 * rg], fill=(255, 255, 255))


# ---------------------------------------------------------------- front (-Z)
CHART_BOX = (S // 2 - 470, 540, S // 2 + 470, 1200)


def draw_chart(d):
    """Digital acuity display, set up for a double-vision test: the
    Worth 4-Dot pattern (red top, green sides, white bottom) on a black
    screen. Emissive, so it is drawn after dimming and stays lit."""
    x0, y0, x1, y1 = CHART_BOX
    d.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=(24, 25, 28))
    sx0, sy0 = x0 + 34, y0 + 34
    sx1, sy1 = x1 - 34, y1 - 34
    d.rectangle([sx0, sy0, sx1, sy1], fill=(6, 7, 9))
    d.ellipse([x1 - 74, y1 - 26, x1 - 58, y1 - 10], fill=(70, 200, 90))

    # dots sit above the bottom band (chart px 1076..1156), which the app
    # overdraws at runtime with the live prism readout (see PRISM_BAND)
    mx, my = (sx0 + sx1) // 2, (sy0 + sy1) // 2 - 60
    r = 52
    dots = [(mx, my - 160, (225, 55, 45)),        # red, top
            (mx - 260, my, (60, 200, 90)),        # green, left
            (mx + 260, my, (60, 200, 90)),        # green, right
            (mx, my + 160, (240, 240, 238))]      # white, bottom
    for dx, dy, col in dots:
        for g in range(20, 0, -2):                # soft emissive halo
            halo = lerp((6, 7, 9), col, (1 - g / 20) * 0.35)
            d.ellipse([dx - r - g, dy - r - g, dx + r + g, dy + r + g],
                      fill=halo)
        d.ellipse([dx - r, dy - r, dx + r, dy + r], fill=col)


# Runtime prism readout: the app overlays one row of this atlas onto the
# chart's bottom band (authored face pixels x 724..1324, y 1076..1156 on
# the -Z wall). Row i = prism strength at 5*i percent, 0..200.
PRISM_LABEL_ROWS = 41
PRISM_LABEL_W, PRISM_LABEL_ROW_H = 600, 80


def build_prism_labels():
    img = Image.new("RGB", (PRISM_LABEL_W,
                            PRISM_LABEL_ROWS * PRISM_LABEL_ROW_H), (6, 7, 9))
    d = ImageDraw.Draw(img)
    f = font(30)
    for i in range(PRISM_LABEL_ROWS):
        pct = 5 * i
        if pct == 0:
            label = "PRISM OFF"
        else:
            label = "PRISM %d%%  ·  %.2fPD V  ·  %.2fPD H" % (
                pct, 5.5 * pct / 100.0, 1.0 * pct / 100.0)
        d.text((PRISM_LABEL_W // 2, i * PRISM_LABEL_ROW_H +
                PRISM_LABEL_ROW_H // 2), label, font=f,
               fill=(150, 158, 168), anchor="mm")
    img.save(os.path.join(ASSETS, "prism_labels.png"), optimize=True)
    print("wrote prism_labels.png")


def build_front(dim, out):
    img, d = new_wall()
    cx = S // 2

    # --- flanking posters
    ix0, iy0, ix1, iy1 = draw_framed(d, 170, 560, 550, 1060)
    # eye anatomy poster
    ecx, ecy = (ix0 + ix1) // 2, (iy0 + iy1) // 2 - 60
    d.ellipse([ecx - 140, ecy - 140, ecx + 140, ecy + 140],
              fill=(240, 244, 248), outline=INK, width=6)
    d.pieslice([ecx + 55, ecy - 88, ecx + 230, ecy + 88], 130, 230,
               fill=(214, 226, 238), outline=INK, width=5)
    d.ellipse([ecx - 132, ecy - 52, ecx - 28, ecy + 52], fill=TEAL, outline=INK, width=4)
    d.ellipse([ecx - 106, ecy - 26, ecx - 54, ecy + 26], fill=INK)
    for ly in (iy1 - 160, iy1 - 116, iy1 - 72):
        d.line([(ix0 + 40, ly), (ix1 - 40, ly)], fill=(170, 174, 180), width=8)
    d.text(((ix0 + ix1) // 2, iy0 + 40), "THE HUMAN EYE", font=font(30),
           fill=INK, anchor="mm")

    ix0, iy0, ix1, iy1 = draw_framed(d, S - 550, 560, S - 170, 1060)
    draw_glasses(d, (ix0 + ix1) // 2, (iy0 + iy1) // 2 - 40, 260, INK, style="rect")
    d.text(((ix0 + ix1) // 2, iy1 - 110), "SEE THE", font=font(44), fill=TEAL_DEEP, anchor="mm")
    d.text(((ix0 + ix1) // 2, iy1 - 56), "DIFFERENCE", font=font(44), fill=TEAL_DEEP, anchor="mm")

    # --- credenza below the chart
    top = 1660
    d.rectangle([cx - 430, top, cx + 430, S - BASEBOARD_H], fill=WOOD)
    d.rectangle([cx - 450, top - 30, cx + 450, top], fill=WOOD_DARK)
    for i in range(3):
        dx0 = cx - 410 + i * 280
        d.rectangle([dx0, top + 40, dx0 + 260, S - BASEBOARD_H - 40],
                    outline=WOOD_DARK, width=8)
        d.rectangle([dx0 + 90, top + 130, dx0 + 170, top + 150], fill=WOOD_DARK)
    # tissue box + lens spray on top
    d.rectangle([cx - 300, top - 110, cx - 160, top - 30], fill=(230, 235, 240))
    d.rectangle([cx - 260, top - 96, cx - 200, top - 74], fill=(160, 200, 210))
    d.rectangle([cx + 200, top - 140, cx + 240, top - 30], fill=TEAL)
    d.rectangle([cx + 208, top - 170, cx + 232, top - 140], fill=(200, 202, 206))

    if dim:
        img = dim_face(img)
        # light spill from the digital chart's screen
        img = glow(img, CHART_BOX, color=(185, 205, 230), reach=120,
                   strength=60)
        d = ImageDraw.Draw(img)
    draw_chart(d)

    save_wall(img, "nz.png", out)


# ---------------------------------------------------------------- back (+Z)
def build_back(dim, out):
    img, d = new_wall()
    cx = S // 2

    # sign with eye logo
    draw_eye_logo(d, cx, 300, 300)
    d.text((cx, 520), "HADDLEY  OPTOMETRY", font=font(96), fill=TEAL_DEEP, anchor="mm")
    d.line([(cx - 560, 600), (cx + 560, 600)], fill=TEAL, width=8)

    # door on the left
    dx0, dy0, dx1 = 220, 730, 660
    d.rectangle([dx0 - 26, dy0 - 26, dx1 + 26, S - BASEBOARD_H], fill=(205, 200, 190))
    d.rectangle([dx0, dy0, dx1, S - BASEBOARD_H], fill=(122, 106, 85))
    for py0, py1 in [(dy0 + 60, dy0 + 420), (dy0 + 480, dy0 + 840)]:
        d.rectangle([dx0 + 50, py0, dx1 - 50, py1], outline=(95, 82, 66), width=10)
    d.ellipse([dx1 - 90, 1330, dx1 - 46, 1374], fill=(200, 202, 206))
    d.text(((dx0 + dx1) // 2, dy0 - 70), "EXAM 2", font=font(40), fill=GRAY, anchor="mm")

    # diplomas above the desk
    for fx in (900, 1230):
        ix0, iy0, ix1, iy1 = draw_framed(d, fx, 860, fx + 260, 1180,
                                         frame=(60, 52, 44))
        d.text(((ix0 + ix1) // 2, iy0 + 60), "DOCTOR OF", font=font(24, bold=False),
               fill=GRAY, anchor="mm")
        d.text(((ix0 + ix1) // 2, iy0 + 100), "OPTOMETRY", font=font(26),
               fill=INK, anchor="mm")
        for ly in (iy0 + 150, iy0 + 185, iy0 + 220):
            d.line([(ix0 + 36, ly), (ix1 - 36, ly)], fill=(180, 178, 172), width=5)
        d.ellipse([(ix0 + ix1) // 2 - 26, iy1 - 90, (ix0 + ix1) // 2 + 26, iy1 - 38],
                  fill=(212, 175, 55))

    # clock
    d.ellipse([1560, 880, 1760, 1080], fill=(252, 252, 250), outline=INK, width=10)
    d.line([(1660, 980), (1660, 910)], fill=INK, width=10)
    d.line([(1660, 980), (1712, 1000)], fill=INK, width=8)

    # reception desk
    tx0, tx1, ttop = 820, 1560, 1430
    d.rectangle([tx0, ttop, tx1, S - BASEBOARD_H], fill=TEAL_DARK)
    d.rectangle([tx0, ttop, tx1, ttop + 200], fill=TEAL)
    d.rectangle([tx0 - 36, ttop - 44, tx1 + 36, ttop], fill=(238, 234, 226))
    # small monitor on the counter
    d.rectangle([tx0 + 120, ttop - 260, tx0 + 400, ttop - 70], fill=(50, 52, 56))
    d.rectangle([tx0 + 140, ttop - 240, tx0 + 380, ttop - 90], fill=(120, 180, 190))
    d.rectangle([tx0 + 240, ttop - 70, tx0 + 280, ttop - 44], fill=(70, 72, 76))
    # bell + card holder
    d.pieslice([tx1 - 300, ttop - 130, tx1 - 200, ttop - 30], 180, 360, fill=(212, 175, 55))
    d.rectangle([tx1 - 160, ttop - 100, tx1 - 90, ttop - 44], fill=(230, 232, 235))

    # waiting chairs + plant
    draw_chair_front(d, 1700, S - BASEBOARD_H, 230)
    draw_plant(d, 1950 - 40, S - BASEBOARD_H, 520)

    if dim:
        img = dim_face(img)
    save_wall(img, "pz.png", out)


# ---------------------------------------------------------------- left (-X)
def build_left(dim, out):
    img, d = new_wall()
    cx = S // 2

    d.text((cx, 260), "FRAMES", font=font(120), fill=TEAL_DEEP, anchor="mm")
    d.line([(cx - 340, 350), (cx + 340, 350)], fill=TEAL, width=8)

    # display panel
    px0, px1 = 220, S - 420
    d.rectangle([px0, 430, px1, 1620], fill=(238, 234, 226),
                outline=(212, 207, 196), width=6)

    colors = [INK, (139, 90, 43), TEAL_DARK, (150, 40, 46), (36, 54, 92),
              (139, 90, 43), INK, TEAL_DARK]
    styles = ["round", "rect"]
    for row in range(3):
        shelf_y = 780 + row * 380
        d.rectangle([px0 + 30, shelf_y, px1 - 30, shelf_y + 26], fill=WOOD)
        d.rectangle([px0 + 30, shelf_y + 26, px1 - 30, shelf_y + 40],
                    fill=(200, 194, 182))
        for i in range(5):
            gx = px0 + 170 + i * ((px1 - px0 - 340) // 4)
            c = colors[(row * 3 + i) % len(colors)]
            draw_glasses(d, gx, shelf_y - 32, 200, c, styles[(row + i) % 2])

    # price tags
    for i, tag in enumerate(["$129", "$189", "$249"]):
        ty = 806 + i * 380
        d.rectangle([px1 - 160, ty - 210, px1 - 60, ty - 150], fill=(252, 252, 250),
                    outline=(190, 186, 178), width=4)
        d.text((px1 - 110, ty - 180), tag, font=font(30), fill=GRAY, anchor="mm")

    # tall mirror on the right
    mx0, mx1 = S - 330, S - 130
    d.rounded_rectangle([mx0 - 16, 520, mx1 + 16, 1740], radius=40, fill=(198, 192, 180))
    d.rounded_rectangle([mx0, 536, mx1, 1724], radius=32, fill=(205, 222, 228))
    d.line([(mx0 + 30, 700), (mx0 + 110, 590)], fill=(235, 244, 246), width=18)
    d.line([(mx0 + 40, 800), (mx0 + 150, 650)], fill=(235, 244, 246), width=10)

    # bench below display
    bx0, bx1 = 700, 1500
    d.rectangle([bx0, 1760, bx1, 1820], fill=WOOD_DARK)
    d.rectangle([bx0 + 40, 1820, bx0 + 90, S - BASEBOARD_H], fill=(70, 70, 74))
    d.rectangle([bx1 - 90, 1820, bx1 - 40, S - BASEBOARD_H], fill=(70, 70, 74))
    d.rectangle([bx0 + 60, 1700, bx0 + 340, 1760], fill=TEAL)      # cushion
    d.rectangle([bx1 - 340, 1700, bx1 - 60, 1760], fill=TEAL)

    if dim:
        img = dim_face(img)
    save_wall(img, "nx.png", out)


# ---------------------------------------------------------------- right (+X)
def build_right(dim, out):
    img, d = new_wall()

    # room plaque
    d.rounded_rectangle([700, 220, 1150, 340], radius=24, fill=TEAL_DEEP)
    d.text((925, 280), "EXAM  1", font=font(64), fill=(252, 252, 250), anchor="mm")

    floor_y = S - BASEBOARD_H

    # ---- instrument stand + phoropter
    pole_x = 560
    d.rectangle([pole_x - 20, 470, pole_x + 20, floor_y], fill=(72, 74, 78))
    d.polygon([(pole_x - 150, floor_y), (pole_x + 150, floor_y),
               (pole_x + 90, floor_y - 46), (pole_x - 90, floor_y - 46)],
              fill=(58, 60, 64))
    d.rectangle([pole_x - 20, 470, pole_x + 480, 510], fill=(72, 74, 78))  # arm
    d.rectangle([pole_x + 420, 510, pole_x + 460, 610], fill=(72, 74, 78))  # drop

    # phoropter head (front view)
    pcx, pcy = pole_x + 440, 800
    d.rounded_rectangle([pcx - 240, pcy - 190, pcx + 240, pcy + 130],
                        radius=60, fill=(96, 99, 105))
    for side in (-1, 1):
        ccx = pcx + side * 118
        d.ellipse([ccx - 108, pcy - 108, ccx + 108, pcy + 108],
                  fill=(120, 123, 130), outline=(60, 62, 66), width=8)
        d.ellipse([ccx - 62, pcy - 62, ccx + 62, pcy + 62],
                  fill=(38, 40, 46), outline=(140, 144, 152), width=6)
        d.ellipse([ccx - 30, pcy - 44, ccx - 6, pcy - 20], fill=(90, 120, 140))
        # dials
        d.ellipse([ccx - 132, pcy - 26, ccx - 84, pcy + 22], fill=(210, 175, 60))
        d.ellipse([ccx + 84, pcy - 26, ccx + 132, pcy + 22], fill=(210, 175, 60))
    d.rectangle([pcx - 30, pcy - 250, pcx + 30, pcy - 190], fill=(72, 74, 78))
    d.ellipse([pcx - 26, pcy + 96, pcx + 26, pcy + 148], fill=(60, 62, 66))

    # ---- exam chair (profile, facing the phoropter)
    ccx = 1420
    d.polygon([(ccx - 130, floor_y), (ccx + 130, floor_y),
               (ccx + 80, floor_y - 60), (ccx - 80, floor_y - 60)],
              fill=(58, 60, 64))
    d.rectangle([ccx - 40, floor_y - 260, ccx + 40, floor_y - 60], fill=(80, 82, 88))
    d.rounded_rectangle([ccx - 250, floor_y - 420, ccx + 210, floor_y - 260],
                        radius=50, fill=(52, 96, 96))          # seat
    d.rounded_rectangle([ccx + 60, floor_y - 980, ccx + 250, floor_y - 300],
                        radius=60, fill=(52, 96, 96))          # backrest
    d.rounded_rectangle([ccx + 80, floor_y - 1120, ccx + 240, floor_y - 960],
                        radius=40, fill=(40, 78, 78))          # headrest
    d.rounded_rectangle([ccx - 240, floor_y - 500, ccx + 40, floor_y - 440],
                        radius=24, fill=(40, 78, 78))          # armrest
    d.polygon([(ccx - 250, floor_y - 400), (ccx - 480, floor_y - 340),
               (ccx - 480, floor_y - 280), (ccx - 250, floor_y - 300)],
              fill=(52, 96, 96))                               # leg rest

    # ---- desk with monitor
    tx0, tx1, ttop = 1660, S - 120, floor_y - 500
    d.rectangle([tx0, ttop, tx1, ttop + 36], fill=WOOD_DARK)
    d.rectangle([tx0 + 20, ttop + 36, tx0 + 70, floor_y], fill=(70, 70, 74))
    d.rectangle([tx1 - 70, ttop + 36, tx1 - 20, floor_y], fill=(70, 70, 74))
    mcx = (tx0 + tx1) // 2
    d.rectangle([mcx - 20, ttop - 110, mcx + 20, ttop - 40], fill=(70, 72, 76))
    d.rectangle([mcx - 90, ttop - 40, mcx + 90, ttop - 20], fill=(70, 72, 76))

    # computer monitor: screen off, nothing to display
    d.rectangle([mcx - 150, ttop - 330, mcx + 150, ttop - 110], fill=(50, 52, 56))
    d.rectangle([mcx - 132, ttop - 312, mcx + 132, ttop - 128], fill=(15, 16, 18))

    if dim:
        img = dim_face(img)
    save_wall(img, "px.png", out)


# ---------------------------------------------------------------- ceiling (+Y)
def build_ceiling(dim, out):
    # stored orientation: top of image = -Z (front), left = -X
    img = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(img)
    for y in range(S):
        t = abs(y - S / 2) / (S / 2)
        d.line([(0, y), (S, y)], fill=lerp((251, 250, 246), (238, 236, 229), t * 0.8))

    # ceiling grid
    for i in range(1, 4):
        p = i * S // 4
        d.line([(p, 0), (p, S)], fill=(226, 223, 215), width=6)
        d.line([(0, p), (S, p)], fill=(226, 223, 215), width=6)

    # four recessed light panels (soft glow when on, dark glass when off)
    for qx in (S // 4, 3 * S // 4):
        for qy in (S // 4, 3 * S // 4):
            if dim:
                d.rounded_rectangle([qx - 330, qy - 210, qx + 330, qy + 210],
                                    radius=36, fill=(52, 54, 58),
                                    outline=(44, 46, 50), width=8)
                continue
            for g in range(60, 0, -6):
                a = 1 - g / 60
                col = lerp((250, 249, 244), (255, 253, 238), a)
                d.rounded_rectangle([qx - 330 - g, qy - 210 - g,
                                     qx + 330 + g, qy + 210 + g],
                                    radius=40 + g, fill=col)
            d.rounded_rectangle([qx - 330, qy - 210, qx + 330, qy + 210],
                                radius=36, fill=(255, 253, 240),
                                outline=(228, 224, 212), width=8)
            d.line([(qx - 330, qy), (qx + 330, qy)], fill=(246, 242, 226), width=4)

    # sprinklers + smoke detector
    for sx, sy in [(S // 2, S // 4), (S // 2, 3 * S // 4)]:
        d.ellipse([sx - 22, sy - 22, sx + 22, sy + 22], fill=(200, 60, 50))
    d.ellipse([S // 2 - 60, S // 2 - 60, S // 2 + 60, S // 2 + 60],
              fill=(240, 238, 232), outline=(210, 206, 196), width=6)

    if dim:
        img = dim_face(img)
    save_flat(img, "py.png", 0.12, out)


# ---------------------------------------------------------------- floor (-Y)
def build_floor(dim, out):
    # stored orientation: top of image = +Z (back), left = -X
    img = Image.new("RGB", (S, S), (231, 225, 211))
    d = ImageDraw.Draw(img)
    tile = S // 8
    for i in range(8):
        for j in range(8):
            shade = (i + j) % 2
            col = (233, 227, 214) if shade else (224, 218, 203)
            d.rectangle([i * tile, j * tile, (i + 1) * tile, (j + 1) * tile], fill=col)
    for i in range(9):
        p = min(i * tile, S - 1)
        d.line([(p, 0), (p, S)], fill=(206, 199, 184), width=5)
        d.line([(0, p), (S, p)], fill=(206, 199, 184), width=5)

    # round rug at room center
    cx = cy = S // 2
    r = 560
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TEAL_DEEP)
    d.ellipse([cx - r + 44, cy - r + 44, cx + r - 44, cy + r - 44],
              outline=(216, 210, 196), width=10)
    d.ellipse([cx - r + 120, cy - r + 120, cx + r - 120, cy + r - 120], fill=TEAL_DARK)
    draw_eye_logo(d, cx, cy, 420)

    if dim:
        img = dim_face(img)
    save_flat(img, "ny.png", 0.30, out)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    for dim, out in ((False, OUT_BRIGHT), (True, OUT_DIM)):
        os.makedirs(out, exist_ok=True)
        build_front(dim, out)
        build_back(dim, out)
        build_left(dim, out)
        build_right(dim, out)
        build_ceiling(dim, out)
        build_floor(dim, out)
    build_prism_labels()
    print("done ->", os.path.abspath(ASSETS))
