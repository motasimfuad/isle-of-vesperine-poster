"""Travel poster for Vesperine, an island that does not exist.

Pure standard library: a small anti-aliased polygon rasteriser, a hand-built
geometric display typeface, and a PNG encoder. Deterministic output.
"""
import math
import os
import struct
import zlib
from array import array

W, H = 768, 1024
SS = 4  # vertical sub-samples per pixel row


def hexc(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def ramp(stops, t):
    """stops: list of (pos, colour) sorted by pos."""
    if t <= stops[0][0]:
        return stops[0][1]
    for i in range(1, len(stops)):
        p1, c1 = stops[i]
        if t <= p1:
            p0, c0 = stops[i - 1]
            return mix(c0, c1, (t - p0) / (p1 - p0))
    return stops[-1][1]


class Canvas:
    def __init__(self):
        n = W * H
        self.r = array('d', [0.0]) * n
        self.g = array('d', [0.0]) * n
        self.b = array('d', [0.0]) * n
        self.acc = [0.0] * (W + 2)
        self.dif = [0.0] * (W + 2)

    def row_fill(self, y, col):
        o = y * W
        self.r[o:o + W] = array('d', [col[0]]) * W
        self.g[o:o + W] = array('d', [col[1]]) * W
        self.b[o:o + W] = array('d', [col[2]]) * W

    def fill(self, contours, col, alpha=1.0, shade=None, add=False):
        """Nonzero-winding fill of one or more closed contours with AA."""
        edges = []
        ymin, ymax = 1e9, -1e9
        for pts in contours:
            n = len(pts)
            for i in range(n):
                x0, ya = pts[i]
                x1, yb = pts[(i + 1) % n]
                if ya == yb:
                    continue
                d = 1
                if ya > yb:
                    x0, ya, x1, yb = x1, yb, x0, ya
                    d = -1
                edges.append((ya, yb, x0, (x1 - x0) / (yb - ya), d))
                ymin = min(ymin, ya)
                ymax = max(ymax, yb)
        if not edges:
            return
        y0 = max(0, int(ymin))
        y1 = min(H - 1, int(ymax))
        acc, dif = self.acc, self.dif
        w = 1.0 / SS
        R, G, B = self.r, self.g, self.b
        for y in range(y0, y1 + 1):
            lo, hi = W + 1, -1
            for k in range(SS):
                sy = y + (k + 0.5) / SS
                xs = []
                for (ya, yb, x0, dx, d) in edges:
                    if ya <= sy < yb:
                        xs.append((x0 + (sy - ya) * dx, d))
                if not xs:
                    continue
                xs.sort()
                wind = 0
                start = 0.0
                for (x, d) in xs:
                    pw = wind
                    wind += d
                    if pw == 0 and wind != 0:
                        start = x
                    elif pw != 0 and wind == 0:
                        xa = max(0.0, start)
                        xb = min(float(W), x)
                        if xb <= xa:
                            continue
                        ia = int(xa)
                        ib = int(xb)
                        if ia < lo:
                            lo = ia
                        if ib > hi:
                            hi = ib
                        if ia == ib:
                            acc[ia] += (xb - xa) * w
                        else:
                            acc[ia] += (ia + 1 - xa) * w
                            dif[ia + 1] += w
                            dif[ib] -= w
                            acc[ib] += (xb - ib) * w
            if hi < 0:
                continue
            hi = min(hi, W - 1)
            run = 0.0
            o = y * W
            for i in range(lo, hi + 1):
                run += dif[i]
                cv = acc[i] + run
                acc[i] = 0.0
                dif[i] = 0.0
                if cv > 0.002:
                    a = (cv if cv < 1.0 else 1.0) * alpha
                    c = shade(i + 0.5, y + 0.5) if shade else col
                    j = o + i
                    if add:
                        R[j] += c[0] * a
                        G[j] += c[1] * a
                        B[j] += c[2] * a
                    else:
                        R[j] += (c[0] - R[j]) * a
                        G[j] += (c[1] - G[j]) * a
                        B[j] += (c[2] - B[j]) * a
            # clear the tail sentinel slots
            acc[hi + 1] = 0.0
            dif[hi + 1] = 0.0
            if hi + 2 <= W + 1:
                dif[hi + 2] = 0.0

    def glow(self, cx, cy, radius, col, strength, power=2.0, sy=1.0):
        """Additive radial glow, falloff (1 - d/r)^power."""
        x0 = max(0, int(cx - radius))
        x1 = min(W - 1, int(cx + radius))
        y0 = max(0, int(cy - radius * sy))
        y1 = min(H - 1, int(cy + radius * sy))
        R, G, B = self.r, self.g, self.b
        inv = 1.0 / radius
        for y in range(y0, y1 + 1):
            dy = (y + 0.5 - cy) / sy
            dy2 = dy * dy
            o = y * W
            for x in range(x0, x1 + 1):
                dx = x + 0.5 - cx
                d = math.sqrt(dx * dx + dy2) * inv
                if d < 1.0:
                    f = (1.0 - d) ** power * strength
                    j = o + x
                    R[j] += col[0] * f
                    G[j] += col[1] * f
                    B[j] += col[2] * f


def ellipse(cx, cy, rx, ry, n=96, a0=0.0, a1=2 * math.pi):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n), cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n)]


def arc(cx, cy, rx, ry, d0, d1, n=18):
    out = []
    for i in range(n + 1):
        a = math.radians(d0 + (d1 - d0) * i / n)
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def stroke(poly, t, miter_limit=1.9):
    """Outline an open polyline of width t as one closed contour (miter joins)."""
    h = t / 2.0
    n = len(poly)
    norms = []
    for i in range(n - 1):
        dx = poly[i + 1][0] - poly[i][0]
        dy = poly[i + 1][1] - poly[i][1]
        L = math.hypot(dx, dy) or 1.0
        norms.append((-dy / L, dx / L))
    left, right = [], []
    for i in range(n):
        if i == 0:
            nx, ny = norms[0]
            m = 1.0
        elif i == n - 1:
            nx, ny = norms[-1]
            m = 1.0
        else:
            ax, ay = norms[i - 1]
            bx, by = norms[i]
            nx, ny = ax + bx, ay + by
            L = math.hypot(nx, ny) or 1.0
            nx, ny = nx / L, ny / L
            cosh = nx * ax + ny * ay
            m = min(miter_limit, 1.0 / max(cosh, 1e-3))
        px, py = poly[i]
        left.append((px + nx * h * m, py + ny * h * m))
        right.append((px - nx * h * m, py - ny * h * m))
    return left + right[::-1]


# ---------------------------------------------------------------- typeface
# Glyphs on a unit cap height (y down). Each entry: (advance width, [polylines], [ring ellipses])
def _glyphs():
    g = {}
    g['V'] = (0.78, [[(0, 0), (0.39, 1), (0.78, 0)]], [])
    g['E'] = (0.56, [[(0.56, 0), (0, 0), (0, 1), (0.56, 1)], [(0, 0.5), (0.46, 0.5)]], [])
    g['F'] = (0.54, [[(0.54, 0), (0, 0), (0, 1)], [(0, 0.5), (0.44, 0.5)]], [])
    g['L'] = (0.52, [[(0, 0), (0, 1), (0.52, 1)]], [])
    g['I'] = (0.0, [[(0, 0), (0, 1)]], [])
    g['N'] = (0.70, [[(0, 1), (0, 0), (0.70, 1), (0.70, 0)]], [])
    g['S'] = (0.58, [arc(0.29, 0.25, 0.29, 0.25, -15, -270, 22) + arc(0.29, 0.75, 0.29, 0.25, -90, 165, 22)[1:]], [])
    bowl = [(0, 1), (0, 0), (0.30, 0)] + arc(0.30, 0.27, 0.27, 0.27, -90, 90, 16)[1:] + [(0, 0.54)]
    g['P'] = (0.57, [bowl], [])
    g['R'] = (0.60, [bowl, [(0.26, 0.54), (0.60, 1)]], [])
    g['O'] = (0.92, [], [(0.46, 0.5, 0.46, 0.5)])
    g[' '] = (0.35, [], [])
    return g


GLYPHS = _glyphs()


def text_width(s, h, t, track):
    w = 0.0
    for ch in s:
        adv = GLYPHS[ch][0]
        w += adv * h + (t if ch != ' ' else 0) + track
    return w - track


def draw_text(cv, s, cx, top, h, t, track, col, shade=None):
    x = cx - text_width(s, h, t, track) / 2.0 + t / 2.0
    for ch in s:
        adv, lines, rings = GLYPHS[ch]
        contours = []
        for pl in lines:
            pts = [(x + px * h, top + py * (h - t) + t / 2.0) for (px, py) in pl]
            contours.append(stroke(pts, t))
        for (ex, ey, rx, ry) in rings:
            ccx, ccy = x + ex * h - t / 2.0 + t / 2.0, top + h / 2.0
            orx, ory = rx * h + t / 2.0 - t / 2.0, h / 2.0
            contours.append(ellipse(ccx - t / 2.0 + t / 2.0, ccy, orx, ory, 72))
            contours.append(ellipse(ccx, ccy, orx - t, ory - t, 72)[::-1])
        ring_contours = contours[len(lines):]
        for c in contours[:len(lines)]:
            cv.fill([c], col, shade=shade)
        if ring_contours:
            cv.fill(ring_contours, col, shade=shade)
        x += adv * h + (t if ch != ' ' else 0) + track


# ---------------------------------------------------------------- scene
def render():
    cv = Canvas()
    HORIZON = 600

    sky = [(0.0, hexc('#1d1640')), (0.25, hexc('#3b2363')), (0.55, hexc('#a8457a')),
           (0.78, hexc('#ec7a5e')), (1.0, hexc('#ffc27a'))]
    sea = [(0.0, hexc('#f0a07a')), (0.08, hexc('#b35d78')), (0.35, hexc('#3f3a74')), (1.0, hexc('#121a3c'))]
    for y in range(H):
        if y < HORIZON:
            cv.row_fill(y, ramp(sky, y / HORIZON))
        else:
            cv.row_fill(y, ramp(sea, (y - HORIZON) / (H - HORIZON)))

    # sun and its glow
    sx, sy = 486, 572
    cv.glow(sx, sy, 420, hexc('#ff9a5c'), 0.35, 2.2)
    cv.glow(sx, sy, 150, hexc('#ffd9a0'), 0.45, 1.6)
    cv.fill([ellipse(sx, sy, 62, 62, 120)], hexc('#fff0c8'))

    # stars in the high violet
    seed = 7
    for i in range(70):
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        x = 30 + seed % (W - 60)
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        y = 30 + seed % 260
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        r = 0.6 + (seed % 100) / 100.0 * 0.9
        cv.fill([ellipse(x, y, r, r, 10)], hexc('#fff4e0'), alpha=0.35 + 0.5 * (1 - y / 300.0))

    # long lit cloud streaks
    clouds = [(120, 410, 300, 9, 0.55), (520, 380, 260, 7, 0.5), (650, 455, 220, 6, 0.6),
              (300, 470, 340, 8, 0.45), (40, 505, 230, 5, 0.5), (560, 515, 200, 4, 0.55)]
    for (cx, cy, half, th, a) in clouds:
        pts = []
        for k in range(41):
            u = k / 40.0
            pts.append((cx - half + 2 * half * u, cy - th * math.sin(math.pi * u) ** 0.7))
        for k in range(40, -1, -1):
            u = k / 40.0
            pts.append((cx - half + 2 * half * u, cy + th * 0.5 * math.sin(math.pi * u)))
        cv.fill([pts], hexc('#ffb38a'), alpha=a)
        cv.fill([[(p[0], p[1] + th * 0.55) for p in pts]], hexc('#6c2f6a'), alpha=a * 0.6)

    # distant islands on the horizon
    cv.fill([[(560, HORIZON), (600, HORIZON - 10), (640, HORIZON - 16), (690, HORIZON - 7), (730, HORIZON)]], hexc('#8a4a78'))
    cv.fill([[(380, HORIZON), (410, HORIZON - 6), (450, HORIZON - 4), (470, HORIZON)]], hexc('#a05a7c'))

    # sea covers the bottom of the sun
    cv.fill([[(0, HORIZON), (W, HORIZON), (W, H), (0, H)]], None,
            shade=lambda x, y: ramp(sea, (y - HORIZON) / (H - HORIZON)))

    # fine swell lines across the open sea
    seed = 31
    for i in range(140):
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        t = (seed % 1000) / 1000.0
        yy = HORIZON + 8 + t * t * (H - HORIZON - 40)
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        xx = 30 + seed % (W - 60)
        ln = 8 + 40 * t
        cv.fill([[(xx - ln, yy), (xx + ln, yy), (xx + ln - 2, yy + 0.9 + t), (xx - ln + 2, yy + 0.9 + t)]],
                hexc('#c98aa6'), alpha=0.18 + 0.12 * (1 - t))

    # sun path on the water
    for i in range(46):
        yy = HORIZON + 4 + i * i * 0.19 + i * 2.2
        if yy > H - 40:
            break
        half = 50 - i * 0.6 + (i * 37 % 11)
        off = ((i * 53) % 17) - 8
        th = 1.2 + i * 0.05
        a = max(0.0, 0.85 - i * 0.017)
        cv.fill([[(sx + off - half, yy), (sx + off + half, yy), (sx + off + half - 3, yy + th), (sx + off - half + 3, yy + th)]],
                hexc('#ffd6a0'), alpha=a)

    # ---- the headland with its sea arch (left), lit from the sun on the right
    rock_top = hexc('#c2607a')
    rock_low = hexc('#3a1f45')

    def rock(x, y):
        t = min(1.0, max(0.0, (y - 300) / 420.0))
        rim = max(0.0, 1.0 - (x - 0) / 420.0) * 0.0
        return mix(rock_top, rock_low, t ** 0.7)

    head = [(0, 360), (40, 352), (90, 340), (150, 336), (205, 344), (250, 352), (300, 372), (340, 400),
            (372, 440), (392, 490), (404, 560), (410, 640), (414, 690), (392, 700), (300, 704), (180, 712), (60, 708), (0, 716)]
    cv.fill([head], None, shade=rock)
    # arch opening: shows sky and sea through the rock
    arch = arc(318, HORIZON + 30, 52, 120, 180, 360, 30) + [(370, 701), (266, 703)]
    cv.fill([arch], None, shade=lambda x, y: ramp(sky, y / HORIZON) if y < HORIZON else ramp(sea, (y - HORIZON) / (H - HORIZON)))
    # the headland's reflection, broken by ripples
    for k in range(26):
        yy = 706 + k * 6.5
        a = 0.5 * (1 - k / 26.0)
        sh = 6 * math.sin(k * 1.7)
        cv.fill([[(0, yy), (408 - k * 9 + sh, yy), (404 - k * 9 + sh, yy + 3.2), (0, yy + 3.2)]], hexc('#1f1638'), alpha=a)
    # sunlit rim on the seaward face
    rim = [(300, 372), (340, 400), (372, 440), (392, 490), (404, 560), (410, HORIZON + 30),
           (414, 690), (406, 690), (402, HORIZON + 30), (395, 560), (382, 492), (362, 446), (332, 408), (296, 380)]
    cv.fill([rim], hexc('#ffb27a'), alpha=0.8)

    # terraced village climbing the headland
    seed = 99
    houses = []
    for row in range(5):
        y = 360 + row * 34
        x = 20 + row * 9
        while x < 280 - row * 6:
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            w = 18 + seed % 16
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            h = 16 + seed % 14
            houses.append((x, y - h + (seed % 7), w, h, seed))
            x += w + 3 + seed % 6
    for (x, y, w, h, s) in houses:
        cv.fill([[(x, y), (x + w, y), (x + w, y + h + 20), (x, y + h + 20)]], hexc('#f3d6c6'))
        cv.fill([[(x + w * 0.62, y), (x + w, y), (x + w, y + h + 20), (x + w * 0.62, y + h + 20)]], hexc('#ffc59a'), alpha=0.9)
        cv.fill([[(x, y), (x + w * 0.3, y), (x + w * 0.3, y + h + 20), (x, y + h + 20)]], hexc('#9d6e9a'), alpha=0.55)
        if s % 3 == 0:  # domed roof
            cv.fill([arc(x + w / 2, y, w / 2, w / 2.4, 180, 360, 14)], hexc('#3f7fa0'))
        else:  # terracotta flat-hip roof
            cv.fill([[(x - 2, y), (x + w + 2, y), (x + w - 3, y - 5), (x + 3, y - 5)]], hexc('#c4563f'))
        if s % 2 == 0:
            cv.fill([[(x + w * 0.4, y + h * 0.4), (x + w * 0.55, y + h * 0.4), (x + w * 0.55, y + h * 0.75), (x + w * 0.4, y + h * 0.75)]],
                    hexc('#ffe08a'))
            cv.glow(x + w * 0.47, y + h * 0.57, 12, hexc('#ffb050'), 0.35)

    # ---- the landmark: a slender lantern spire on the arch's crown
    tx = 318
    base_y = 382
    TOPY = 262
    spire = [(tx - 16, base_y), (tx + 16, base_y), (tx + 9, TOPY), (tx - 9, TOPY)]
    cv.fill([spire], hexc('#f6e6d6'))
    cv.fill([[(tx + 1, base_y), (tx + 16, base_y), (tx + 9, TOPY), (tx + 1, TOPY)]], hexc('#ffbf8f'), alpha=0.85)
    for k in range(3):  # banding
        yb = base_y - 26 - k * 36
        cv.fill([[(tx - 15 + k * 1.6, yb), (tx + 15 - k * 1.6, yb), (tx + 14 - k * 1.6, yb - 9), (tx - 14 + k * 1.6, yb - 9)]],
                hexc('#7a3550'), alpha=0.85)
    # gallery and lantern
    L = TOPY  # gallery level
    cv.fill([[(tx - 16, L), (tx + 16, L), (tx + 16, L - 6), (tx - 16, L - 6)]], hexc('#3a1f45'))
    cv.glow(tx, L - 20, 150, hexc('#ffcf7a'), 0.30, 2.4)
    cv.glow(tx, L - 20, 44, hexc('#fff0b0'), 0.7, 1.4)
    cv.fill([[(tx - 8, L - 6), (tx + 8, L - 6), (tx + 8, L - 32), (tx - 8, L - 32)]], hexc('#fff6d0'))
    cv.fill([[(tx - 12, L - 32), (tx + 12, L - 32), (tx, L - 52)]], hexc('#3a1f45'))
    cv.fill([stroke([(tx, L - 52), (tx, L - 62)], 1.6)], hexc('#3a1f45'))
    # beams sweeping out to sea
    cv.fill([[(tx + 6, L - 20), (W, L - 70), (W, L + 40)]], hexc('#ffd890'), alpha=0.10, add=True)
    cv.fill([[(tx - 6, L - 20), (0, L + 10), (0, L + 80)]], hexc('#ffd890'), alpha=0.06, add=True)

    # ---- a sailing boat heading for the island
    bx, by = 600, 742
    cv.fill([[(bx - 34, by), (bx + 36, by), (bx + 26, by + 10), (bx - 26, by + 10)]], hexc('#22183a'))
    cv.fill([[(bx - 2, by - 2), (bx - 2, by - 78), (bx - 40, by - 4)]], hexc('#ffe3c0'))
    cv.fill([[(bx + 2, by - 2), (bx + 2, by - 64), (bx + 30, by - 4)]], hexc('#ffb48a'))
    cv.fill([[(bx - 30, by + 12), (bx + 30, by + 12), (bx + 22, by + 15), (bx - 22, by + 15)]], hexc('#ffd6a0'), alpha=0.4)

    # gulls
    for (gx, gy, s) in [(430, 300, 9), (455, 318, 7), (520, 290, 6), (402, 330, 5)]:
        cv.fill([stroke([(gx - s, gy - s * 0.3), (gx - s * 0.4, gy - s * 0.5), (gx, gy), (gx + s * 0.4, gy - s * 0.5), (gx + s, gy - s * 0.3)], 1.6)],
                hexc('#2a1838'))

    # ---- foreground shore rocks for depth
    fg = [(0, 880), (60, 862), (130, 870), (190, 900), (250, 940), (290, H), (0, H)]
    cv.fill([fg], hexc('#160f26'))
    fg2 = [(W, 930), (700, 925), (640, 950), (600, H), (W, H)]
    cv.fill([fg2], hexc('#160f26'))

    # ---- type
    cream = hexc('#fff0d4')
    gold = hexc('#ffc77a')
    draw_text(cv, 'ISLE OF', W / 2, 52, 20, 3.2, 14, hexc('#ffd9b0'))
    draw_text(cv, 'VESPERINE', W / 2 + 3, 88, 88, 12.5, 9, hexc('#120a24'))
    draw_text(cv, 'VESPERINE', W / 2, 85, 88, 12.5, 9, cream,
              shade=lambda x, y: mix(cream, gold, min(1.0, max(0.0, (y - 85) / 88.0))))
    # a thin rule under the name ties it to the horizon below
    cv.fill([[(W / 2 - 120, 194), (W / 2 + 120, 194), (W / 2 + 120, 195.6), (W / 2 - 120, 195.6)]], gold, alpha=0.75)
    cv.fill([ellipse(W / 2, 194.8, 3.2, 3.2, 16)], gold)

    # ---- frame
    m = 18
    cv.fill([[(0, 0), (W, 0), (W, H), (0, H)], [(m, m), (m, H - m), (W - m, H - m), (W - m, m)]], hexc('#f4e4cc'))
    m2 = 26
    cv.fill([[(m2, m2), (W - m2, m2), (W - m2, H - m2), (m2, H - m2)],
             [(m2 + 1.5, m2 + 1.5), (m2 + 1.5, H - m2 - 1.5), (W - m2 - 1.5, H - m2 - 1.5), (W - m2 - 1.5, m2 + 1.5)]],
            hexc('#f4e4cc'), alpha=0.8)

    return cv


def encode_png(cv, path):
    R, G, B = cv.r, cv.g, cv.b
    raw = bytearray()
    prev = bytes(W * 3)
    for y in range(H):
        o = y * W
        row = bytearray(W * 3)
        for x in range(W):
            j = o + x
            r = int(R[j] * 255.0 + 0.5)
            g = int(G[j] * 255.0 + 0.5)
            b = int(B[j] * 255.0 + 0.5)
            row[3 * x] = 0 if r < 0 else (255 if r > 255 else r)
            row[3 * x + 1] = 0 if g < 0 else (255 if g > 255 else g)
            row[3 * x + 2] = 0 if b < 0 else (255 if b > 255 else b)
        up = bytes((row[i] - prev[i]) & 255 for i in range(len(row)))
        raw.append(2)
        raw += up
        prev = bytes(row)
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(bytes(raw), 9)) + chunk(b'IEND', b'')
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(png)
    return len(png)


if __name__ == '__main__':
    out = os.environ.get('POSTER_OUT', '/output/entry.png')
    canvas = render()
    encode_png(canvas, out)
