import os

# The glyph search multiplies small matrices (a few hundred bitmaps at a time). A multi-threaded
# BLAS spends more time waking its threads than computing: on a 12-thread machine one episode
# burned 3800 s of CPU for 370 s of wall time, single-threaded it took 230 s of both. The setting
# must precede the first numpy import; an explicit value in the environment wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
