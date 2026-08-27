"""Mechanical overlap audit: every rendered text bbox against every other text bbox
and against every drawn artist. Reports intersections in points."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, sys, importlib
sys.path.insert(0, ".")
import fig_m_evaluation_series as M

def bb(a, r):
    try:
        e = a.get_window_extent(renderer=r)
        return None if (e.width <= 0 or e.height <= 0) else e
    except Exception:
        return None

def inter(a, b):
    x = min(a.x1, b.x1) - max(a.x0, b.x0)
    y = min(a.y1, b.y1) - max(a.y0, b.y0)
    return (x, y) if (x > 0 and y > 0) else None

_orig = M.S.save
report = {}
def _spy(fig, path):
    name = str(path).rsplit("/", 1)[-1]
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts, others = [], []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                e = bb(t, r)
                if e: texts.append((t.get_text().replace("\n", " / ")[:38], e, ax))
        for coll in list(ax.patches) + list(ax.lines):
            e = bb(coll, r)
            if e: others.append((type(coll).__name__, e, ax, coll))
    hits = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][2] is not texts[j][2]: continue
            v = inter(texts[i][1], texts[j][1])
            if v: hits.append(("TEXT/TEXT", texts[i][0], texts[j][0], v))
    # A Line2D's window_extent is the bbox of the WHOLE polyline, not the stroke,
    # so testing against it reports a panel-spanning curve as overlapping every
    # label in the panel. Test the text box against the actual segments instead.
    def seg_hits_box(ln, e):
        d = ln.get_data()
        pts = ln.get_transform().transform(list(zip(*d)))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            for t in [i / 24 for i in range(25)]:      # sample along the segment
                x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                if e.x0 <= x <= e.x1 and e.y0 <= y <= e.y1:
                    return True
        return False

    for tname, te, tax in texts:
        for oname, oe, oax, art in others:
            if tax is not oax: continue
            if oname == "Line2D":
                if seg_hits_box(art, te):
                    hits.append(("TEXT/LINE", tname, oname, (0, 0)))
            else:
                # An UNFILLED patch occupies only its outline. Testing its area
                # reports every label inside a container box as an overlap, which
                # is the common-region device, not a defect. Test edges only.
                fc = art.get_facecolor()
                filled = not (isinstance(fc, tuple) and len(fc) == 4 and fc[3] == 0)
                if filled:
                    v = inter(te, oe)
                    if v and v[0] > 1.5 and v[1] > 1.5:
                        hits.append(("TEXT/FILL", tname, oname, v))
                else:
                    edges = [((oe.x0, oe.y0), (oe.x1, oe.y0)), ((oe.x1, oe.y0), (oe.x1, oe.y1)),
                             ((oe.x1, oe.y1), (oe.x0, oe.y1)), ((oe.x0, oe.y1), (oe.x0, oe.y0))]
                    for (x1, y1), (x2, y2) in edges:
                        if any(te.x0 <= x1 + (x2 - x1) * t <= te.x1 and
                               te.y0 <= y1 + (y2 - y1) * t <= te.y1
                               for t in [i / 24 for i in range(25)]):
                            hits.append(("TEXT/EDGE", tname, oname, (0, 0)))
                            break
    report[name] = hits
    return _orig(fig, path)

M.S.save = _spy
M.main()
print()
for k in sorted(report):
    h = report[k]
    print(f"  {k}: {len(h)} overlap(s)")
    for kind, a, b, v in h:
        px = f"   overlap {v[0]:.1f} x {v[1]:.1f} px" if v[0] else "   stroke crosses the text box"
        print(f"     {kind}  '{a}'  x  '{b}'{px}")
