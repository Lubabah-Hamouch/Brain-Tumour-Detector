"""Draw the web client's icons: favicon.ico (16, 32, 48 px), icon.svg and apple-icon.png (180 px).

The mark is three nested squares in the finding colours, the same nesting as the BraTS regions
(whole tumour contains tumour core contains enhancing tumour), on the near-black page colour,
with thin corner brackets like a viewer's crop marks. The 16 px version is drawn pixel by pixel
without the brackets so it stays sharp in a browser tab; larger sizes are supersampled.

    python -m src.visualization.render_icons

Writes to frontend/src/app/, where Next.js picks the files up by name.
"""
from pathlib import Path

from PIL import Image, ImageDraw

from src.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "frontend" / "src" / "app"

# Colours from frontend/src/design/tokens.ts.
BACKGROUND = "#131416"
BRACKET = "#5E6268"
EDEMA = "#7A111B"
NECROTIC = "#D7263D"
ENHANCING = "#FFC2CA"

# Geometry on a 32-unit grid: (x0, y0, x1, y1, colour), drawn in order.
SQUARES = ((8, 8, 24, 24, EDEMA), (11, 11, 21, 21, NECROTIC), (14, 14, 18, 18, ENHANCING))
BRACKET_INSET, BRACKET_LENGTH, BRACKET_WIDTH = 3, 6, 1

# The 16 px icon, on its own pixel grid: the squares only.
SQUARES_16 = ((4, 4, 12, 12, EDEMA), (6, 6, 10, 10, NECROTIC), (7, 7, 9, 9, ENHANCING))


def brackets():
    """Corner brackets as rectangles on the 32-unit grid."""
    i, n, w, far = BRACKET_INSET, BRACKET_LENGTH, BRACKET_WIDTH, 32 - BRACKET_INSET
    rects = []
    for x, dx in ((i, 1), (far, -1)):
        for y, dy in ((i, 1), (far, -1)):
            rects.append((min(x, x + dx * n), min(y, y + dy * w), max(x, x + dx * n), max(y, y + dy * w)))
            rects.append((min(x, x + dx * w), min(y, y + dy * n), max(x, x + dx * w), max(y, y + dy * n)))
    return rects


def draw(size):
    """The icon at `size` px. 16 and 32 px are pixel-exact; other sizes are drawn at 8x and reduced."""
    if size == 16:
        img = Image.new("RGB", (16, 16), BACKGROUND)
        d = ImageDraw.Draw(img)
        for x0, y0, x1, y1, colour in SQUARES_16:
            d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=colour)
        return img
    big = size if size == 32 else size * 8
    k = big / 32
    img = Image.new("RGB", (big, big), BACKGROUND)
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in brackets():
        d.rectangle((round(x0 * k), round(y0 * k), round(x1 * k) - 1, round(y1 * k) - 1), fill=BRACKET)
    for x0, y0, x1, y1, colour in SQUARES:
        d.rectangle((round(x0 * k), round(y0 * k), round(x1 * k) - 1, round(y1 * k) - 1), fill=colour)
    return img if big == size else img.resize((size, size), Image.LANCZOS)


def svg():
    rect = '<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'
    parts = [rect.format(0, 0, 32, 32, BACKGROUND)]
    parts += [rect.format(x0, y0, x1 - x0, y1 - y0, BRACKET) for x0, y0, x1, y1 in brackets()]
    parts += [rect.format(x0, y0, x1 - x0, y1 - y0, c) for x0, y0, x1, y1, c in SQUARES]
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" shape-rendering="crispEdges">'
            + "".join(parts) + "</svg>\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # RGBA: some ICO decoders (including the Next.js build) reject RGB frames.
    icons = [draw(s).convert("RGBA") for s in (16, 32, 48)]
    icons[-1].save(OUT / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48)], append_images=icons[:-1])
    draw(180).convert("RGBA").save(OUT / "apple-icon.png", optimize=True)
    (OUT / "icon.svg").write_text(svg(), encoding="utf-8")
    for name in ("favicon.ico", "apple-icon.png", "icon.svg"):
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
