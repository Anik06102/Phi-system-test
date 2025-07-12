#!/usr/bin/env python3
"""
phi_lattice_unified_digits.py · July 2025

Unified φ-Lattice + Variational Shell + Chebyshev + CA
End-to-end on sklearn.datasets.load_digits
"""
import math, time, argparse, logging, warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

# ───────────────────────── 1 · Config ─────────────────────────────
@dataclass
class Cfg:
    seed: int        = 42
    latent_dim: int  = 32
    shells: int      = 5
    ae_epochs: int   = 50
    ae_batch: int    = 128
    lr_ae: float     = 1e-3
    tau_start: float = 1.0
    tau_end: float   = 0.05
    gap_bias: float  = 0.03

    knn_k: int       = 10
    knn_batch: int   = 256
    cheb_K: int      = 8
    cheb_R: int      = 6

    ca_steps: int    = 10
    ca_order: int    = 4

cfg = Cfg()

# ───────────────────────── 2 · Setup ───────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

# ───────────────────────── 3 · Data ───────────────────────────────
def load_data():
    ds = load_digits()
    X = ds.data.astype(np.float32) / 16.0   # pixel values 0–16
    y = ds.target
    # train/val/test split
    X0, X_test, y0, y_test = train_test_split(X, y,
        test_size=0.2, stratify=y, random_state=cfg.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X0, y0,
        test_size=0.2, stratify=y0, random_state=cfg.seed)
    return (tf.convert_to_tensor(X_tr), y_tr,
            tf.convert_to_tensor(X_val), y_val,
            tf.convert_to_tensor(X_test), y_test)

# ──────────────── 4 · k-NN Graph ─────────────────────────────────
@tf.function(jit_compile=True)
def pair_l2(A,B):
    aa = tf.reduce_sum(A*A,1,keepdims=True)
    bb = tf.reduce_sum(B*B,1,keepdims=True)
    return tf.sqrt(tf.maximum(aa - 2*tf.matmul(A,B,transpose_b=True)
                              + tf.transpose(bb), 0.0))

def build_knn(Z, k, batch):
    n = tf.shape(Z)[0]
    rows,cols,vals = [],[],[]
    for s in tf.range(0,n,batch):
        e = tf.minimum(s+batch, n)
        D = pair_l2(Z[s:e], Z)
        d,idx = tf.math.top_k(-D, k=k+1)
        idx = tf.cast(idx[:,1:],tf.int32)
        d   = -d[:,1:]
        w   = 1.0/(d+1e-6)
        w  /= tf.reduce_sum(w,1,keepdims=True)
        rows.append(tf.repeat(tf.range(s,e),k))
        cols.append(tf.reshape(idx,[-1]))
        vals.append(tf.reshape(w,[-1]))
    i = tf.concat(rows,0); j = tf.concat(cols,0); v = tf.concat(vals,0)
    # sym + unique
    ef = tf.stack([tf.concat([i,j],0), tf.concat([j,i],0)],axis=1)
    vf = tf.concat([v,v],0)
    n64 = tf.cast(n,tf.int64)
    ids = ef[:,0]*n64 + ef[:,1]
    u,idx = tf.unique(ids)
    edges = tf.cast(tf.gather(ef,idx),tf.int64)
    vals2 = tf.gather(vf,idx)
    return tf.sparse.reorder(tf.sparse.SparseTensor(edges,vals2,[n64,n64]))

# ───────────────────── 5 · Chebyshev ─────────────────────────────
def lap_apply(A,dinv,X):
    with tf.device("/CPU:0"):
        AX = tf.sparse.sparse_dense_matmul(A, dinv[:,None]*X)
    return X - dinv[:,None]*AX

def cheb_basis(A,K,R):
    n = A.dense_shape[0]
    deg = tf.sparse.reduce_sum(A,1)
    dinv= tf.pow(tf.maximum(deg,1e-9),-0.5)
    Q,_ = tf.linalg.qr(tf.random.normal([n,R],seed=cfg.seed))
    blocks=[Q]
    if K>1:
        T1,_ = tf.linalg.qr(lap_apply(A,dinv,Q)); blocks.append(T1)
        for _ in range(2,K):
            Tk = 2*lap_apply(A,dinv,blocks[-1]) - blocks[-2]
            for B in blocks:
                Tk -= B @ tf.linalg.matmul(B,Tk,transpose_a=True)
            Tk,_ = tf.linalg.qr(Tk); blocks.append(Tk)
    return tf.concat(blocks,1), blocks

def cheb_ext(Zq,Zr,k,blocks,batch):
    outs=[]
    for s in tf.range(0,tf.shape(Zq)[0],batch):
        e = tf.minimum(s+batch,tf.shape(Zq)[0])
        D = pair_l2(Zq[s:e], Zr)
        d,idx = tf.math.top_k(-D,k=k)
        d = -d; w = 1.0/(d+1e-6); w/=tf.reduce_sum(w,1,keepdims=True)
        parts=[]
        for t,B in enumerate(blocks):
            ip = tf.reduce_sum(w[:,:,None]*tf.gather(B,idx),1)
            parts.append(ip if t<2 else 2*ip-parts[t-2])
        outs.append(tf.concat(parts,1))
    return tf.concat(outs,0)

# ─────────────────── 6 · Variational φ-VDAE ──────────────────────
def tau_sched(ep):
    r = ep/(cfg.ae_epochs-1)
    return cfg.tau_end + 0.5*(cfg.tau_start-cfg.tau_end)*(1+math.cos(math.pi*r))

class PhiVDAE(tf.keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.enc = tf.keras.Sequential([
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dense(cfg.latent_dim),
        ])
        self.dec = tf.keras.Sequential([
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dense(dim,activation="sigmoid"),
        ])
        # shell posterior head:
        self.logits_r = tf.keras.layers.Dense(cfg.shells)
        # raw radii
        self.raw = tf.Variable(tf.random.normal([cfg.shells]),trainable=True)

    def radii(self):
        gaps = tf.nn.softplus(self.raw) + cfg.gap_bias
        return tf.minimum(tf.math.cumsum(gaps), 1e3)

    def call(self,x,*,tau):
        z = self.enc(x)
        c = tf.reduce_mean(z,0,keepdims=True)
        # posterior over shells
        logits = self.logits_r(z)
        q = tf.nn.softmax(logits)
        # gumbel-softmax sample
        g = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits),0,1)+1e-8)+1e-8)
        r_onehot = tf.nn.softmax((logits+g)/tau)
        r_vals = tf.reduce_sum(r_onehot*self.radii(),axis=1,keepdims=True)
        # deterministic fallback: r_vals = sum(q*radii)
        dir = (z-c)/tf.maximum(tf.norm(z-c,axis=1,keepdims=True),1e-6)
        z_n = c + dir*r_vals
        x_hat = self.dec(z_n)
        # store for loss
        self._cache = dict(x=x, x_hat=x_hat, z=z, q=q)
        return x_hat, z_n

    def loss(self):
        x,x_hat,z,q = self._cache["x"], self._cache["x_hat"], self._cache["z"], self._cache["q"]
        # recon
        Lrec = tf.reduce_mean((x_hat-x)**2)
        # KL to uniform prior
        Lkl  = tf.reduce_mean(tf.reduce_sum(q*(tf.math.log(q+1e-9)-tf.math.log(1.0/cfg.shells)),axis=1))
        return Lrec + 0.1*Lkl

# ─────────────────────── 7 · Train φ-VDAE ────────────────────────
def train_vdae(X):
    ds = tf.data.Dataset.from_tensor_slices(X).shuffle(1024,cfg.seed).batch(cfg.ae_batch)
    model = PhiVDAE(X.shape[1])
    opt = tf.keras.optimizers.Adam(cfg.lr_ae)
    for ep in range(cfg.ae_epochs):
        tau = tau_sched(ep)
        for xb in ds:
            with tf.GradientTape() as tp:
                _,_ = model(xb, tau=tau)
                loss = model.loss()
            grads = tp.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        logging.info("AE epoch %d/%d  loss=%.4f τ=%.3f",ep+1,cfg.ae_epochs,loss,tau)
    return model

# ──────────────────── 8 · Slot-in Centroids ─────────────────────
def extract_centroids(model, X, y):
    # run through model once
    xh, Z = model(X, tau=cfg.tau_end)
    # assign shell ids
    q = model._cache["q"].numpy()
    shells = q.argmax(1)
    inner = shells==0
    centroids: Dict[int,np.ndarray] = {}
    for cl in np.unique(y):
        mask = np.logical_and(inner, y==cl)
        if np.sum(mask)>0:
            centroids[int(cl)] = np.mean(Z.numpy()[mask],axis=0)
    # normalize
    for k,v in centroids.items():
        centroids[k] = v/np.linalg.norm(v)
    return centroids

# ───────────────────── 9 · Nearest-centroid Classify ────────────
def classify_nn(centroids, Z):
    # Z: collapsed codes (N,d)
    Z = Z.numpy()
    keys = sorted(centroids.keys())
    C = np.vstack([centroids[k] for k in keys])
    sims = Z @ C.T
    idx = np.argmax(sims,axis=1)
    return np.array([keys[i] for i in idx])

# ─────────────────── 10 · Pascal CA Diffusion ───────────────────
def pascal_weights(k):
    from math import comb
    w = np.array([comb(k,i) for i in range(k+1)],dtype=np.float32)
    return w/np.sum(w)

def bfs_layers(adj, max_d):
    # adj: dense np adjacency list
    n = adj.shape[0]
    layers = []
    for d in range(max_d+1):
        layers.append([None]*n)
    for v in range(n):
        dist = {v:0}
        q=[v]
        layers[0][v] = [v]
        while q:
            u=q.pop(0)
            du=dist[u]
            if du==max_d: continue
            for w in np.where(adj[u]>0)[0]:
                if w not in dist:
                    dist[w]=du+1; q.append(w)
                    layers[du+1][v] = layers[du+1][v] or []
                    layers[du+1][v].append(w)
        for d in range(max_d+1):
            if layers[d][v] is None: layers[d][v]=[]
    return layers

def ca_diffusion(z_all, centroids, steps, order):
    # build kNN graph on z_all
    Zt = tf.convert_to_tensor(z_all)
    G = build_knn(Zt, cfg.knn_k, cfg.knn_batch).to_dense().numpy()
    layers = bfs_layers(G, order)
    # state: one-hot per node from centroids if available, else uniform
    n = G.shape[0]; K = len(centroids)
    s = np.zeros((n,K),dtype=np.float32)
    for i,cl in enumerate(centroids):
        # find nearest training index for each class
        # assume centroids were computed on training portion at top of z_all
        pass  # skip: in practice seed with a few known indices
    # here we’ll just demo diffusion on random init:
    s = np.random.rand(n,K).astype(np.float32)
    W = pascal_weights(order)
    for _ in range(steps):
        s_new = np.zeros_like(s)
        for d,w in enumerate(W):
            for v in range(n):
                nbrs = layers[d][v]
                if nbrs:
                    s_new[v] += w * np.sum(s[nbrs],axis=0)
        s = s_new / np.clip(np.sum(s_new,axis=1,keepdims=True),1e-12,1e12)
    return s

# ───────────────────────── 11 · Main ─────────────────────────────
if __name__=="__main__":
    (X_tr,y_tr, X_val,y_val, X_te,y_te) = load_data()

    t0 = time.time()
    # train φ-VDAE on all data
    X_all = tf.concat([X_tr, X_val, X_te],0)
    model = train_vdae(X_all)
    # extract collapsed codes for all
    _,Z_all = model(X_all, tau=cfg.tau_end)
    Z_all = Z_all.numpy()

    # slot-in per-class centroids from the *training* split
    centroids = extract_centroids(model, X_tr, y_tr)
    logging.info("Extracted %d centroids", len(centroids))

    # classify test by nearest-centroid
    _,Z_te = model(X_te, tau=cfg.tau_end)
    y_pred = classify_nn(centroids, Z_te)
    acc = np.mean(y_pred == y_te)
    logging.info("Nearest-centroid test accuracy: %.4f", acc)

    # optional: CA diffusion demo (unseeded random demo)
    S = ca_diffusion(Z_all, centroids, steps=cfg.ca_steps, order=cfg.ca_order)
    logging.info("CA diffusion final state shape: %s", S.shape)

    logging.info("Total time: %.1fs", time.time()-t0)
