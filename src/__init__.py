import os

# Small statsmodels fits run ~5x slower when numpy's BLAS spreads them over many threads.
# Must be set before numpy is imported, so it lives in the package __init__.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
