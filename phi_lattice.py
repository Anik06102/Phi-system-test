%%writefile phi_lattice_core.py
# ===============================================================
#  φ-Lattice Cognitive Engine  —  Colab-friendly reference impl.
# ===============================================================
#  • Pure NumPy / SciPy / TensorFlow (CPU-only OK)
#  • Supervised MNIST demo  →  nested Laplacian-shell encoder
#  • Tiny CartPole demo     →  centroid-aware Q-table
#  • Argparse patched to ignore IPython’s extra -f flags
# ===============================================================

from __future__ import annotations
import argparse, pathlib, os, math, itertools, json, time
import numpy as np, scipy.linalg as la, scipy.spatial.distance as ssd
import tensorflow as tf
import gudhi as gd

# ────────────── GLOBAL CONFIG ──────────────
SEED = 42
rng  = np.random.default_rng(SEED)

CFG = dict(
    batch_size     = 512,
    n_shells       = 5,
    beta           = 0.65,
    min_pts        = 20,
    soft_clouds    = False,
    epochs         = 5,          # keep small for Colab demo
    lr             = 1e-3,
    λ_equiv        = 0.1,
    flip_prob      = 0.5,
    γ              = 0.95,
    α              = 0.1,
    min_epochs_between_rebuild = 2,
)

# ────────────── UTILITIES ──────────────
def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def make_batches(X, y, bs):
    idx = np.arange(len(X)); rng.shuffle(idx)
    for i in range(0, len(X), bs):
        sl = idx[i:i+bs]
        yield X[sl], y[sl]

# ────────────── SPECTRAL OPS ──────────────
def build_laplacian(X):
    Σ = np.cov(X, rowvar=False)        # feature-feature cov
    A = np.clip(Σ, 0, None)            # keep non-neg weights
    L = np.diag(A.sum(1)) - A
    λ, U = la.eigh(L)
    return λ, U                         # U: (F, F)

def bucket(pc2, n_bins=None):
    n_bins = n_bins or CFG['n_shells']
    qs = np.quantile(pc2, np.linspace(0, 1, n_bins+1))
    return np.digitize(pc2, qs[1:-1])   # 0 … n_bins-1

# ────────────── LATTICE ──────────────
class Lattice:
    """Partitions samples into radial shells & angular clouds."""
    def __init__(self, X, U):
        self.X, self.U = X, U
        self.shells:list[np.ndarray] = []
        self.clouds:list[list[np.ndarray]] = []
        self.centroids:list[list[np.ndarray]] = []
        self._build()

    def _build(self):
        Z   = self.X @ self.U[:, :3]       # project to first 3 comps
        pc2 = Z[:, 2]
        sidx = bucket(pc2)
        k = 0
        while True:
            pts = np.where(sidx == k)[0]
            if len(pts) < CFG['min_pts']: break
            self.shells.append(pts)

            # split by PC-1 angle into ≤4 sub-clouds
            pc1 = Z[pts, 1]
            n_cl = max(1, min(4, len(pts)//5))
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_cl, random_state=SEED, n_init='auto')
            km.fit(pc1.reshape(-1,1))

            clouds_k, cents_k = [], []
            for c in range(km.n_clusters):
                sub = pts[km.labels_ == c]
                clouds_k.append(sub)
                cents_k.append(self.X[sub].mean(0))
            self.clouds.append(clouds_k)
            self.centroids.append(cents_k)
            k += 1
            if k >= CFG['n_shells']: break

    def locate(self, vec):
        # nearest centroid (L2 in feature space)
        best = (1e9, -1, -1)
        for s, layer in enumerate(self.centroids):
            d = ssd.cdist([vec], layer).flatten()
            c = d.argmin()
            if d[c] < best[0]:
                best = (d[c], s, c)
        return best[1], best[2]

# ────────────── MODELS ──────────────
class Encoder(tf.keras.Model):
    def __init__(self, U, latent=64):
        super().__init__()
        self.U = tf.constant(U[:, :32], dtype=tf.float32, trainable=False)
        self.d1 = tf.keras.layers.Dense(128, activation='gelu')
        self.d2 = tf.keras.layers.Dense(latent)
    def call(self, x):
        spec = tf.linalg.matmul(x, self.U)
        return self.d2(self.d1(spec))

class ClassifierHead(tf.keras.Model):
    def __init__(self, n_cls): super().__init__(); self.lin = tf.keras.layers.Dense(n_cls)
    def call(self, z): return self.lin(z)

class QHead:
    def __init__(self, S, C, A):
        self.Q = np.zeros((S, C, A))
    def act(self, s, c, ε=0.1):
        return rng.integers(self.Q.shape[-1]) if rng.random() < ε else self.Q[s,c].argmax()
    def update(self, s, c, a, r, ns, nc, α=CFG['α'], γ=CFG['γ']):
        td = r + γ*self.Q[ns,nc].max() - self.Q[s,c,a]
        self.Q[s,c,a] += α*td

# ────────────── DATA ──────────────
def load_mnist(flatten=True):
    (xtr,ytr),(xte,yte) = tf.keras.datasets.mnist.load_data()
    if flatten:
        xtr, xte = xtr.reshape(-1,784), xte.reshape(-1,784)
    return (xtr/255.).astype('float32'), ytr, (xte/255.).astype('float32'), yte

# ────────────── TRAINING LOOPS ──────────────
def parity_flip(x):
    P = np.eye(x.shape[-1]); P[0,0] = -1
    return x @ P

def run_supervised():
    Xtr, ytr, Xte, yte = load_mnist()
    _, U = build_laplacian(Xtr)
    lat  = Lattice(Xtr, U)

    enc = Encoder(U); head = ClassifierHead(10)
    opt = tf.keras.optimizers.Adam(CFG['lr'])

    for ep in range(CFG['epochs']):
        for xb, yb in make_batches(Xtr, ytr, CFG['batch_size']):
            with tf.GradientTape() as tape:
                logits = head(enc(xb))
                loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(yb, logits))
                # equivariance loss
                loss += CFG['λ_equiv'] * tf.reduce_mean(
                        tf.square(logits - head(enc(parity_flip(xb)))))
            grads = tape.gradient(loss, enc.trainable_variables + head.trainable_variables)
            opt.apply_gradients(zip(grads, enc.trainable_variables + head.trainable_variables))

        acc = np.mean(np.argmax(head(enc(Xte)),1) == yte)
        print(f'Epoch {ep:02d} | Loss {loss.numpy():.4f} | Test acc {acc:.3f}')

    # return trained objects for viz
    return dict(lattice=lat, encoder=enc, U=U, X=Xtr)

def run_rl():
    import gymnasium as gym
    env = gym.make('CartPole-v1'); obs,_ = env.reset(seed=SEED)
    X0   = np.array([obs])
    _, U = build_laplacian(X0)
    lat  = Lattice(X0, U)
    q    = QHead(len(lat.centroids), len(lat.centroids[0]), env.action_space.n)

    for ep in range(5):
        obs,_ = env.reset(seed=SEED+ep); done=False; total=0
        while not done:
            s,c = lat.locate(obs)
            a   = q.act(s,c)
            obs2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ns,nc = lat.locate(obs2)
            q.update(s,c,a,r,ns,nc)
            obs   = obs2; total += r
        print(f'Episode {ep}: reward {total}')

# ────────────── MAIN (argparse patched) ──────────────
if __name__ == '__main__':
    set_seed()
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['sl','rl'], default='sl')
    args, _ = p.parse_known_args()   # swallow Colab’s “-f …kernel.json”
    (run_supervised if args.mode=='sl' else run_rl)()
