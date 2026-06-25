from shapely.geometry import Point
from shapely.strtree import STRtree

# Simulate your pattern: tree built on buffered geoms
base = [Point(0,0), Point(1,1)]
bufd = [g.buffer(0.1) for g in base]
tree = STRtree(bufd)

raw = tree.query(Point(0,0).buffer(0.05))  # ndarray[int] on your build
print("raw[0] type:", type(raw[0]).__name__)

# Minimal copy of the mapping logic
def as_list(x):
    import numpy as np
    return x.tolist() if hasattr(x, "dtype") and isinstance(x, np.ndarray) else list(x)

hits = as_list(raw)
if not (hasattr(hits[0], "geom_type")):
    mapped = [bufd[int(i)] for i in hits]
    print("mapped[0] is geom:", hasattr(mapped[0], "geom_type"))
else:
    print("already geometry")
