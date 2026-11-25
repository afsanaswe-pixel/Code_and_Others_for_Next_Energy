# =====================================================================
# Graphical abstract - Next Energy submission
# Paste this WHOLE file into ONE Google Colab code cell and run it.
# Drawn with matplotlib (vector shapes and text); no image-generation model.
# Output: Graphical_abstract.tif (3137 x 1254 px, 600 dpi, RGB; upload this),
#         Graphical_abstract.pdf / .eps (vector), Graphical_abstract.png (preview)
# =====================================================================

# ---- 1. Arial-compatible font (Liberation Sans) --------------------------
import glob, subprocess
subprocess.run("apt-get -qq install -y fonts-liberation", shell=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import matplotlib.font_manager as fm
_fonts = glob.glob("/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf")
for _f in _fonts:
    fm.fontManager.addfont(_f)
if not _fonts:
    raise RuntimeError("Liberation Sans not found; the text will not fit the panels.")
print("Font ready:", len(_fonts), "Liberation Sans files")

# ---- 2. Draw ---------------------------------------------------------------
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
from sklearn.isotonic import IsotonicRegression

OUT = "Graphical_abstract"   # output file prefix

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})

# ------------------------------------------------------------ palette
INK, MUTED, RULE = "#1f2328", "#5b6470", "#c9ced6"
ORANGE, BLUE = "#D9541E", "#2462C4"     # prediction p / projection q
GREEN, RED = "#1B8A5A", "#B42318"       # status: holds / fails (with glyph)
BG_SOFT = "#f3f4f6"                     # context panels
BG_BLUE = "#edf3fc"                     # approach (centre of gravity)
BG_FIND = "#ffffff"
BG_GREEN = "#eaf5ef"

W, H = 13.28, 5.31
fig = plt.figure(figsize=(W / 2.54, H / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

TEXTS = []  # (artist, panel_x0, panel_x1) for the overflow check


def T(x, y, s, size=7, panel=None, weight="normal", color=INK, ha="center",
      va="center", style="normal", **kw):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                ha=ha, va=va, fontstyle=style, zorder=8, **kw)
    if panel is not None:
        TEXTS.append((t, panel))
    return t


def panel(x0, x1, label, bg, edge=None, lw=0.0):
    box = FancyBboxPatch((x0, 0.12), x1 - x0, H - 0.24,
                         boxstyle="round,pad=0,rounding_size=0.12",
                         fc=bg, ec=edge if edge else bg, lw=lw, zorder=0)
    ax.add_patch(box)
    T(x0 + 0.14, H - 0.32, label, size=7.5, weight="bold",
      color=edge if edge else MUTED, ha="left", panel=(x0, x1))
    ax.plot([x0 + 0.14, x1 - 0.14], [H - 0.52, H - 0.52], color=RULE,
            lw=0.6, zorder=1)
    return (x0, x1)


def chevron(xc, yc=2.62):
    ax.add_patch(Polygon([[xc - 0.06, yc + 0.17], [xc + 0.08, yc],
                          [xc - 0.06, yc - 0.17]], closed=True, fc=MUTED,
                         ec="none", zorder=3))


def battery(xc, yc, w, h, fill, edge=INK, level=0.0, lw=0.8):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="white",
                 ec=edge, lw=lw, zorder=4))
    ax.add_patch(Rectangle((xc - w * 0.22, yc + h / 2), w * 0.44, h * 0.09,
                 fc=edge, ec="none", zorder=4))
    if level > 0:
        pad = 0.045
        ax.add_patch(Rectangle((xc - w / 2 + pad, yc - h / 2 + pad),
                     w - 2 * pad, (h - 2 * pad) * level, fc=fill,
                     ec="none", zorder=5))


def tick_glyph(xc, yc, r=0.11, color=GREEN):
    ax.add_patch(Circle((xc, yc), r, fc=color, ec="none", zorder=6))
    ax.plot([xc - 0.5 * r, xc - 0.12 * r, xc + 0.55 * r],
            [yc + 0.02 * r, yc - 0.4 * r, yc + 0.45 * r], color="white",
            lw=1.0, solid_capstyle="round", solid_joinstyle="round", zorder=7)


def cross_glyph(xc, yc, r=0.11, color=RED):
    ax.add_patch(Circle((xc, yc), r, fc=color, ec="none", zorder=6))
    d = 0.42 * r
    ax.plot([xc - d, xc + d], [yc - d, yc + d], color="white", lw=1.0,
            solid_capstyle="round", zorder=7)
    ax.plot([xc - d, xc + d], [yc + d, yc - d], color="white", lw=1.0,
            solid_capstyle="round", zorder=7)


# ------------------------------------------------------------ layout (cm)
GAP, MARG = 0.16, 0.06
widths = [1.80, 2.20, 2.75, 3.75, 2.02]
x = MARG
bounds = []
for w_ in widths:
    bounds.append((x, x + w_)); x += w_ + GAP
P = panel(*bounds[0], "Problem", "white")
I = panel(*bounds[1], "Input", "white")
M = panel(*bounds[2], "Approach", BG_BLUE, edge=BLUE, lw=0.9)
F = panel(*bounds[3], "Key findings", "#f5f6f8", edge="#3b4350", lw=0.8)
C = panel(*bounds[4], "Contribution", "white", edge=None)
C_hdr = TEXTS[-1][0]; C_hdr.set_color(GREEN)
for a, b in zip(bounds[:-1], bounds[1:]):
    chevron((a[1] + b[0]) / 2)

# ------------------------------------------------------------ 1 Problem
pc = (P[0] + P[1]) / 2
battery(pc - 0.36, 4.12, 0.38, 0.66, fill=RULE, edge=INK, level=0.0)
T(pc - 0.36, 4.09, "?", size=11, weight="bold", color=ORANGE, panel=P)
# capacity meter, struck through = capacity not measured
gx, gy, gr = pc + 0.36, 3.98, 0.27
th = np.linspace(np.pi, 0, 50)
ax.plot(gx + gr * np.cos(th), gy + gr * np.sin(th), color=INK, lw=0.8,
        zorder=5)
ax.plot([gx - gr, gx + gr], [gy, gy], color=INK, lw=0.8, zorder=5)
for a_ in np.linspace(np.pi * 0.85, np.pi * 0.15, 5):
    ax.plot([gx + 0.78 * gr * np.cos(a_), gx + gr * np.cos(a_)],
            [gy + 0.78 * gr * np.sin(a_), gy + gr * np.sin(a_)], color=INK,
            lw=0.5, zorder=5)
ax.plot([gx, gx + 0.13], [gy, gy + 0.16], color=INK, lw=0.9, zorder=5)
ax.plot([gx - 0.3, gx + 0.3], [gy - 0.12, gy + 0.36], color=RED, lw=1.3,
        solid_capstyle="round", zorder=6)

T(pc, 3.22, "SOH estimator", size=7, panel=P)
T(pc, 2.96, "on new cells", size=7, panel=P)
T(pc, 2.36, "capacity not", size=7.5, weight="bold", panel=P)
T(pc, 2.08, "measured", size=7.5, weight="bold", panel=P)
T(pc, 1.42, "What can be", size=7.5, weight="bold", color=ORANGE, panel=P)
T(pc, 1.15, "known about", size=7.5, weight="bold", color=ORANGE, panel=P)
T(pc, 0.88, "the error?", size=7.5, weight="bold", color=ORANGE, panel=P)

# ------------------------------------------------------------ 2 Input
ic = (I[0] + I[1]) / 2
for k, (lab, lev) in enumerate([("NCM", 0.78), ("NCA", 0.6), ("LFP", 0.9)]):
    xc = ic + (k - 1) * 0.6
    battery(xc, 4.18, 0.3, 0.56, fill="#9aa3ae", edge=INK, level=lev)
    T(xc, 3.69, lab, size=7, weight="bold", panel=I)

T(ic, 3.22, "387 cells", size=11, weight="bold", panel=I)
T(ic, 2.84, "13 domains", size=7.5, panel=I)
T(ic, 2.57, "310,554 cycles", size=7.5, panel=I)

sq, gp = 0.112, 0.033
x_start = ic - (13 * sq + 12 * gp) / 2
held = 9
for k in range(13):
    ax.add_patch(Rectangle((x_start + k * (sq + gp), 1.93), sq, sq,
                 fc=ORANGE if k == held else "#aab2bc", ec="none", zorder=4))
T(x_start + held * (sq + gp) + sq / 2, 2.19, "test", size=7, weight="bold",
  color=ORANGE, panel=I)
T(x_start, 2.19, "train", size=7, color=MUTED, ha="left", panel=I)
T(ic, 1.7, "leave-one-out", size=7, panel=I)
T(ic, 1.45, "by domain", size=7, panel=I)

T(ic, 0.96, "11 estimators", size=7.5, weight="bold", panel=I)
T(ic, 0.67, "ridge regression", size=7, color=MUTED, panel=I)
T(ic, 0.42, "→ neural networks", size=7, color=MUTED, panel=I)

# ------------------------------------------------------------ 3 Approach
mc = (M[0] + M[1]) / 2
T(mc, 4.56, "Monotone projection", size=7.5, weight="bold", color=BLUE,
  panel=M)

px0, px1, py0, py1 = M[0] + 0.36, M[1] - 0.14, 2.26, 4.3
ax_in = fig.add_axes([px0 / W, py0 / H, (px1 - px0) / W, (py1 - py0) / H])
n = 160
t = np.linspace(0, 1, n)
y_true = 0.52 - 0.22 * t - 0.16 * t ** 2
rng = np.random.default_rng(3)
bias = 0.07 + 0.015 * t
wig = 0.008 * rng.standard_normal(n)
bumps = np.zeros(n)
for c0, amp in [(0.18, 0.085), (0.45, 0.075), (0.7, 0.09)]:
    m = (t >= c0) & (t < c0 + 0.12)
    bumps[m] += amp * np.exp(-(t[m] - c0) / 0.045)
p = y_true + bias + wig + bumps
q = IsotonicRegression(increasing=False).fit_transform(np.arange(n), p)
ax_in.fill_between(t, p, q, where=p > q, color="#f3bfa6", lw=0,
                   interpolate=True)
ax_in.plot(t, y_true, color=MUTED, lw=1.0, ls=(0, (3, 2)))
ax_in.plot(t, p, color=ORANGE, lw=0.9)
ax_in.plot(t, q, color=BLUE, lw=1.3)
ax_in.set_xlim(0, 1); ax_in.set_ylim(0.08, 1.0)
ax_in.set_xticks([]); ax_in.set_yticks([])
for s_ in ("top", "right"):
    ax_in.spines[s_].set_visible(False)
ax_in.spines["left"].set_color(MUTED); ax_in.spines["bottom"].set_color(MUTED)
ax_in.set_facecolor(BG_BLUE)
kx = 0.07
for j, (sym, lab, col, ls) in enumerate([
        ("p", "prediction", ORANGE, "-"),
        ("q", "projection", BLUE, "-"),
        ("y", "true, unknown", MUTED, (0, (3, 2)))]):
    yy = 0.93 - j * 0.115
    ax_in.plot([kx, kx + 0.09], [yy, yy], color=col, lw=1.2, ls=ls)
    t1 = ax_in.text(kx + 0.12, yy, sym, fontsize=7, fontweight="bold",
                    color=col if col != MUTED else INK, va="center",
                    ha="left", zorder=8)
    t2 = ax_in.text(kx + 0.2, yy, lab, fontsize=7, color=INK, va="center",
                    ha="left", zorder=8)
    TEXTS.extend([(t1, M), (t2, M)])
T(M[0] + 0.24, 3.0, "SOH", size=7, color=MUTED, rotation=90, panel=M)
T((px0 + px1) / 2, 2.12, "cycles", size=7, color=MUTED, panel=M)

T(mc, 1.84, "capacity does not", size=7, color=MUTED, style="italic",
  panel=M)
T(mc, 1.60, "increase over life", size=7, color=MUTED, style="italic",
  panel=M)
T(mc, 1.19, "distance p–q", size=9, weight="bold", panel=M)
T(mc, 0.88, "≤ true error", size=9, weight="bold", panel=M)
T(mc, 0.58, "provable lower bound,", size=7, panel=M)
T(mc, 0.34, "no capacity labels", size=7, panel=M)

# ------------------------------------------------------------ 4 Key findings
fx = F[0] + 0.15
rows = [4.72, 3.58, 2.44, 1.3]
for r_ in rows[1:]:
    ax.plot([F[0] + 0.12, F[1] - 0.12], [r_ + 0.05, r_ + 0.05], color=RULE,
            lw=0.6, zorder=1)

y0 = rows[0]
tick_glyph(fx + 0.11, y0 - 0.3)
T(fx + 0.31, y0 - 0.3, "4,257 / 4,257", size=10, weight="bold", ha="left",
  panel=F)
T(fx, y0 - 0.64, "cell–estimator pairs: bound held", size=7, ha="left",
  panel=F)
T(fx, y0 - 0.89, "error ↓ in 97–100% of cells", size=7, ha="left",
  color=MUTED, panel=F)

y0 = rows[1]
T(fx, y0 - 0.3, "0.4–8.9%", size=10, weight="bold", color=BLUE, ha="left",
  panel=F)
bx0, bw, bh = fx + 1.75, 1.6, 0.2
ax.add_patch(Rectangle((bx0, y0 - 0.4), bw, bh, fc="#d0d5dc", ec="none",
                       zorder=4))
ax.add_patch(Rectangle((bx0, y0 - 0.4), bw * 0.089, bh, fc=BLUE, ec="none",
                       zorder=5))
T(fx, y0 - 0.64, "of squared error removed", size=7, ha="left", panel=F)
T(fx, y0 - 0.89, "most error is hidden bias", size=7, weight="bold",
  ha="left", panel=F)

y0 = rows[2]
T(fx, y0 - 0.24, "Label-free warning signals", size=7, ha="left", panel=F)
tick_glyph(fx + 0.11, y0 - 0.55)
T(fx + 0.3, y0 - 0.55, "domains", size=7.5, weight="bold", ha="left",
  panel=F)
T(F[1] - 0.15, y0 - 0.55, "AUC up to 0.86", size=7.5, weight="bold",
  ha="right", panel=F)
cross_glyph(fx + 0.11, y0 - 0.86)
T(fx + 0.3, y0 - 0.86, "cells", size=7.5, weight="bold", ha="left", panel=F)
T(F[1] - 0.15, y0 - 0.86, "Spearman ρ ≤ 0.33", size=7.5, weight="bold",
  ha="right", panel=F)

y0 = rows[3]
T(fx, y0 - 0.3, "2.41% → 1.86% SOH", size=10, weight="bold", color=GREEN,
  ha="left", panel=F)
T(fx, y0 - 0.64, "online error with 3 scheduled", size=7, ha="left",
  panel=F)
T(fx, y0 - 0.89, "reference capacity checks", size=7, ha="left", panel=F)

# ------------------------------------------------------------ 5 Contribution
cc = (C[0] + C[1]) / 2
sx, sy = cc, 4.12
ax.add_patch(Polygon([[sx - 0.27, sy + 0.32], [sx + 0.27, sy + 0.32],
                      [sx + 0.27, sy - 0.02], [sx, sy - 0.36],
                      [sx - 0.27, sy - 0.02]], closed=True, fc=GREEN,
                     ec="none", zorder=4))
ax.plot([sx - 0.12, sx - 0.02, sx + 0.14], [sy + 0.02, sy - 0.1, sy + 0.14],
        color="white", lw=1.3, solid_capstyle="round",
        solid_joinstyle="round", zorder=5)
T(cc, 3.42, "Safe add-on", size=8, weight="bold", color=GREEN, panel=C)
T(cc, 3.12, "for any SOH", size=7, panel=C)
T(cc, 2.88, "estimator, plus", size=7, panel=C)
T(cc, 2.64, "an error alarm", size=7, panel=C)

ax.plot([C[0] + 0.3, C[1] - 0.3], [2.28, 2.28], color=RULE, lw=0.6)

battery(cc - 0.22, 1.66, 0.26, 0.46, fill=GREEN, edge=INK, level=0.6)
gx, gy, gr = cc + 0.22, 1.56, 0.21
th = np.linspace(np.pi, 0, 40)
ax.plot(gx + gr * np.cos(th), gy + gr * np.sin(th), color=INK, lw=0.8,
        zorder=5)
ax.plot([gx - gr, gx + gr], [gy, gy], color=INK, lw=0.8, zorder=5)
ax.plot([gx, gx + 0.1], [gy, gy + 0.14], color=INK, lw=0.9, zorder=6)
T(cc, 1.0, "A few capacity", size=7.5, weight="bold", panel=C)
T(cc, 0.73, "checks remain", size=7.5, weight="bold", panel=C)
T(cc, 0.46, "necessary", size=7.5, weight="bold", panel=C)

# ------------------------------------------------------------ outer border
BORDER_PT = 0.8                       # line width in points
d = BORDER_PT / 72 * 2.54 / 2         # half the line width, in cm
ax.add_patch(Rectangle((d, d), W - 2 * d, H - 2 * d, fill=False,
                       ec="#3b4350", lw=BORDER_PT, zorder=10, clip_on=False))

# ------------------------------------------------------------ checks
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = 0
boxes = []
for t_, (x0, x1) in TEXTS:
    bb = t_.get_window_extent(r)
    (a0, b0), (a1, b1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    boxes.append((a0, b0, a1, b1, t_.get_text()))
    if a0 < x0 + 0.06 or a1 > x1 - 0.06:
        bad += 1
        print(f"OVERFLOW {t_.get_text()!r}: {a0:.2f}-{a1:.2f} in {x0}-{x1}")
    if b0 < 0.12 or b1 > H - 0.12:
        bad += 1
        print(f"OVERFLOW (vertical) {t_.get_text()!r}: {b0:.2f}-{b1:.2f}")
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        A, B = boxes[i], boxes[j]
        if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
            print("OVERLAP", A[4], "|", B[4])
            bad += 1
sizes = sorted({t_.get_fontsize() for t_, _ in TEXTS})
print("font sizes used:", sizes, "| problems:", bad)

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".eps")
# raster copies: flatten onto white and save as RGB (Elsevier: RGB/CMYK)
import io
from PIL import Image
for ext, dpi in ((".png", 300), (".tif", 600)):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    im = Image.open(buf)
    rgb = Image.new("RGB", im.size, "white")
    rgb.paste(im, mask=im.split()[3])
    kw = {"compression": "tiff_lzw"} if ext == ".tif" else {}
    rgb.save(OUT + ext, dpi=(dpi, dpi), **kw)
    print("saved", OUT + ext, rgb.size, rgb.mode, dpi, "dpi")

# ---- 3. Preview and download (Colab) ---------------------------------------
plt.close(fig)
try:
    from IPython.display import Image as _Img, display
    display(_Img(OUT + ".png", width=1100))
    from google.colab import files
    for _ext in (".tif", ".pdf", ".eps", ".png"):
        files.download(OUT + _ext)
except ImportError:
    pass  # not running in Colab: files are saved in the working folder
