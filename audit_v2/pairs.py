"""Candidate near-duplicate pairs with all three perceptual distances.

Distances per pair (i, j):
  ph  - pHash Hamming, flip-invariant: min(H(ph_i, ph_j), H(ph_mirror_i, ph_j)), symmetrised
  dh  - dHash Hamming, same construction
  cos - cosine of L2-normalised DINOv2 embeddings
"""
import faiss
import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
PH, PHM, DH, DHM = 0, 1, 2, 3


def popcount64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP[x.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.int16)


def ham_flip(ha, hb, i, j, col, colm):
    """Flip-invariant Hamming between rows i of ha and j of hb (symmetrised)."""
    d1 = popcount64(ha[i, col] ^ hb[j, col])
    d2 = popcount64(ha[i, colm] ^ hb[j, col])
    d3 = popcount64(ha[i, col] ^ hb[j, colm])
    return np.minimum(d1, np.minimum(d2, d3))


def distances(fa, fb, i, j):
    """All three distances for index arrays i (into fa) and j (into fb)."""
    ha, hb = fa["hashes"], fb["hashes"]
    ph = ham_flip(ha, hb, i, j, PH, PHM)
    dh = ham_flip(ha, hb, i, j, DH, DHM)
    cos = np.einsum("ij,ij->i", fa["emb"][i], fb["emb"][j]).astype(np.float32)
    return ph, dh, cos


def _bin(h):
    return np.ascontiguousarray(h.astype(np.uint64)).view(np.uint8).reshape(-1, 8)


def _range_bin(index_codes, query_codes, radius):
    idx = faiss.IndexBinaryFlat(64)
    idx.add(index_codes)
    lims, _, labels = idx.range_search(query_codes, radius + 1)  # faiss keeps d < radius+1
    q = np.repeat(np.arange(len(query_codes)), np.diff(lims).astype(np.int64))
    return q, labels.astype(np.int64)


def _range_cos(emb_index, emb_query, thr, chunk=4096):
    idx = faiss.IndexFlatIP(emb_index.shape[1])
    idx.add(np.ascontiguousarray(emb_index, dtype=np.float32))
    qs, ls = [], []
    for c0 in range(0, len(emb_query), chunk):
        lims, _, labels = idx.range_search(np.ascontiguousarray(emb_query[c0:c0 + chunk], dtype=np.float32), thr)
        qs.append(np.repeat(np.arange(c0, min(c0 + chunk, len(emb_query))), np.diff(lims).astype(np.int64)))
        ls.append(labels.astype(np.int64))
    return np.concatenate(qs), np.concatenate(ls)


def candidates(fa, fb=None, r_ph=16, r_dh=16, r_cos=0.80):
    """All pairs with ph <= r_ph OR dh <= r_dh OR cos >= r_cos.

    fb=None -> pairs inside fa with i < j; otherwise pairs (i in fa, j in fb).
    Returns dict(i, j, ph, dh, cos).
    """
    same = fb is None
    fb = fa if same else fb
    I, J = [], []
    for col, colm, r in ((PH, PHM, r_ph), (DH, DHM, r_dh)):
        codes_b = _bin(fb["hashes"][:, col])
        for qcol in (col, colm):
            q, l = _range_bin(codes_b, _bin(fa["hashes"][:, qcol]), r)
            I.append(q); J.append(l)
    q, l = _range_cos(fb["emb"], fa["emb"], r_cos)
    I.append(q); J.append(l)
    i = np.concatenate(I); j = np.concatenate(J)
    if same:
        keep = i != j
        i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    key = np.unique(i * np.int64(len(fb["emb"])) + j)
    del I, J
    out = {k: [] for k in ("i", "j", "ph", "dh", "cos")}
    for c0 in range(0, len(key), 1_000_000):  # bounded RAM: emb gather is 1.5 GB per 1M pairs
        k = key[c0:c0 + 1_000_000]
        i, j = k // len(fb["emb"]), k % len(fb["emb"])
        ph, dh, cos = distances(fa, fb, i, j)
        keep = (ph <= r_ph) | (dh <= r_dh) | (cos >= r_cos)
        for n, v in zip(out, (i, j, ph, dh, cos)):
            out[n].append(v[keep])
    return {n: np.concatenate(v) for n, v in out.items()}


def is_dup(p, th):
    """Calibrated union rule; th = dict(ph=, dh=, cos=) - a missing key disables that test."""
    m = np.zeros(len(p["ph"]), dtype=bool)
    if th.get("ph") is not None:
        m |= p["ph"] <= th["ph"]
    if th.get("dh") is not None:
        m |= p["dh"] <= th["dh"]
    if th.get("cos") is not None:
        m |= p["cos"] >= th["cos"]
    return m
