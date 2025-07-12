import importlib
import sys

import pytest

# Skip if required packages are missing
pytest.importorskip('numpy')
pytest.importorskip('scipy')

# Helper to load phi_lattice without the initial Jupyter magic
module_code = open('phi_lattice.py').read().splitlines()
if module_code and module_code[0].startswith('%%'):
    module_code = '\n'.join(module_code[1:])
else:
    module_code = '\n'.join(module_code)

spec = importlib.util.spec_from_loader('phi_lattice_clean', loader=None)
phi_lattice = importlib.util.module_from_spec(spec)
exec(module_code, phi_lattice.__dict__)
sys.modules['phi_lattice_clean'] = phi_lattice

from phi_lattice_clean import build_laplacian, Lattice, CFG

class DummyKMeans:
    def __init__(self, n_clusters, random_state=None, n_init='auto'):
        self.n_clusters = n_clusters
    def fit(self, X):
        # simple deterministic labels by ordering
        n = len(X)
        indices = np.arange(n)
        splits = np.array_split(indices, self.n_clusters)
        self.labels_ = np.empty(n, dtype=int)
        for i, sl in enumerate(splits):
            self.labels_[sl] = i
        return self

# Monkeypatch sklearn.cluster.KMeans if sklearn is unavailable
try:
    import sklearn.cluster  # type: ignore
except ModuleNotFoundError:
    import types
    sklearn = types.ModuleType('sklearn')
    cluster = types.ModuleType('cluster')
    cluster.KMeans = DummyKMeans
    sklearn.cluster = cluster
    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.cluster'] = cluster

def test_locate_returns_valid_indices():
    import numpy as np

    # Reduce minimum points per shell for small synthetic dataset
    CFG['min_pts'] = 1
    CFG['n_shells'] = 3

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    _, U = build_laplacian(X)
    lat = Lattice(X, U)

    for s_idx, clouds in enumerate(lat.clouds):
        for c_idx, indices in enumerate(clouds):
            for i in indices:
                s, c = lat.locate(X[i])
                assert s == s_idx
                assert c == c_idx
