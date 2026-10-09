"""Generates app/static/bg/chest.svg, the stylised radiograph behind the UI.

Drawn procedurally rather than taken from the dataset: a real patient image
used as page decoration would be both a licensing question and an odd thing
to put behind the image the user is actually reading. Re-run after editing
the geometry here; the output is committed.

    python scripts/make_chest_svg.py
"""
import math
import os

W, H, CX = 800, 900, 400
OUT = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'bg', 'chest.svg')


def rib(i: int) -> str:
    """One left-side rib (viewer's left); mirrored for the right."""
    y = 200 + i * 47
    # widest through ribs 5-7, narrowing to the shoulders and the floating ribs
    w = 120 + 150 * math.sin(min(1.0, (i + 1.6) / 7.2) * math.pi / 2) - max(0, i - 8) * 26
    x0 = CX - 26
    # posterior arc rises slightly out of the spine, rolls over the lateral
    # margin, then the anterior segment runs down and in -- no closed loop
    return (f'M{x0},{y} C{CX - w * .5:.1f},{y - 28} {CX - w * .93:.1f},{y - 8} {CX - w:.1f},{y + 46} '
            f'C{CX - w * 1.02:.1f},{y + 86} {CX - w * .9:.1f},{y + 112} {CX - w * .74:.1f},{y + 128}')


def build() -> str:
    ribs = [rib(i) for i in range(11)]
    rib_paths = ''.join(
        f'<path d="{d}" stroke-width="{10 - i * .4:.1f}" opacity="{.5 - i * .025:.3f}"/>'
        for i, d in enumerate(ribs))
    rib_cores = ''.join(f'<path d="{d}"/>' for d in ribs)

    vertebrae = ''.join(
        f'<rect x="{CX - 21 + i * .25:.1f}" y="{60 + i * 37}" width="{42 - i * .5:.1f}" height="29" rx="9" '
        f'opacity="{.22 + .16 * math.sin(i / 21 * math.pi):.3f}"/>'
        for i in range(21))

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none">
<defs>
  <radialGradient id="lung" cx="50%" cy="45%" r="60%">
    <stop offset="0" stop-color="#7fc4ff" stop-opacity=".30"/>
    <stop offset=".6" stop-color="#3b82f6" stop-opacity=".12"/>
    <stop offset="1" stop-color="#1e3a8a" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="torso" cx="50%" cy="42%" r="58%">
    <stop offset="0" stop-color="#9fd2ff" stop-opacity=".16"/>
    <stop offset=".75" stop-color="#2563eb" stop-opacity=".05"/>
    <stop offset="1" stop-color="#0b1220" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="heart" cx="45%" cy="45%" r="55%">
    <stop offset="0" stop-color="#e8f4ff" stop-opacity=".34"/>
    <stop offset="1" stop-color="#93c5fd" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="bone" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#eaf5ff"/>
    <stop offset="1" stop-color="#8cc2ff"/>
  </linearGradient>
  <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="14"/></filter>
  <filter id="glow" x="-10%" y="-10%" width="120%" height="120%">
    <feGaussianBlur stdDeviation="3.2" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>

<!-- soft tissue -->
<path d="M{CX - 60},20 C{CX - 70},90 {CX - 150},120 {CX - 300},150 C{CX - 380},170 {CX - 395},260 {CX - 385},380
         L{CX - 360},880 L{CX + 360},880 L{CX + 385},380 C{CX + 395},260 {CX + 380},170 {CX + 300},150
         C{CX + 150},120 {CX + 70},90 {CX + 60},20 Z" fill="url(#torso)"/>

<!-- lung fields -->
<g filter="url(#soft)">
  <path d="M{CX - 40},170 C{CX - 150},150 {CX - 255},230 {CX - 280},420 C{CX - 295},560 {CX - 290},700 {CX - 250},770
           C{CX - 180},740 {CX - 90},735 {CX - 45},760 C{CX - 40},560 {CX - 30},320 {CX - 40},170 Z" fill="url(#lung)"/>
  <path d="M{CX + 40},170 C{CX + 150},150 {CX + 255},230 {CX + 280},420 C{CX + 295},560 {CX + 290},700 {CX + 250},770
           C{CX + 180},740 {CX + 90},735 {CX + 45},760 C{CX + 40},560 {CX + 30},320 {CX + 40},170 Z" fill="url(#lung)"/>
  <ellipse cx="{CX + 55}" cy="590" rx="125" ry="140" fill="url(#heart)"/>
</g>

<!-- shoulders -->
<g fill="url(#bone)" opacity=".16" filter="url(#soft)">
  <ellipse cx="{CX - 330}" cy="215" rx="46" ry="56"/><ellipse cx="{CX + 330}" cy="215" rx="46" ry="56"/>
</g>
<g stroke="url(#bone)" stroke-linecap="round" opacity=".28" stroke-width="9">
  <path d="M{CX - 300},235 C{CX - 270},300 {CX - 250},380 {CX - 240},450"/>
  <path d="M{CX + 300},235 C{CX + 270},300 {CX + 250},380 {CX + 240},450"/>
</g>

<!-- ribs: a wide soft stroke under a thin bright core reads as cortical bone -->
<g stroke="url(#bone)" stroke-linecap="round" filter="url(#glow)">
  <g>{rib_paths}</g>
  <g transform="translate({W},0) scale(-1,1)">{rib_paths}</g>
</g>
<g stroke="#f4faff" stroke-linecap="round" stroke-width="1.3" opacity=".38">
  <g>{rib_cores}</g>
  <g transform="translate({W},0) scale(-1,1)">{rib_cores}</g>
</g>

<!-- clavicles -->
<g stroke="url(#bone)" stroke-linecap="round" stroke-width="10" opacity=".55" filter="url(#glow)">
  <path d="M{CX - 34},176 C{CX - 90},186 {CX - 150},150 {CX - 210},152 S{CX - 285},170 {CX - 305},160"/>
  <path d="M{CX + 34},176 C{CX + 90},186 {CX + 150},150 {CX + 210},152 S{CX + 285},170 {CX + 305},160"/>
</g>

<!-- spine -->
<g fill="url(#bone)" filter="url(#glow)">{vertebrae}</g>

<!-- diaphragm -->
<g stroke="#bfe0ff" stroke-linecap="round" stroke-width="3" opacity=".42" filter="url(#glow)">
  <path d="M{CX - 290},740 C{CX - 230},650 {CX - 110},655 {CX - 40},735"/>
  <path d="M{CX + 290},755 C{CX + 230},670 {CX + 110},675 {CX + 40},750"/>
</g>
</svg>
'''


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(build())
    print('wrote', os.path.normpath(OUT))
