"""Skip-gram with negative sampling, in NumPy. The gradients below are derived by hand
(see gradcheck.py for the finite-difference check).

For one (centre c, context o) pair with k negative words n_1..n_k:
    v = W_in[c],  u_o = W_out[o],  u_j = W_out[n_j]
    L = -log s(u_o . v) - sum_j log s(-u_j . v),      s(x) = 1 / (1 + e^-x)
Using d/dx[-log s(x)] = s(x) - 1 and d/dx[-log s(-x)] = s(x):
    dL/dv   = (s(u_o . v) - 1) u_o + sum_j s(u_j . v) u_j
    dL/du_o = (s(u_o . v) - 1) v
    dL/du_j = s(u_j . v) v
"""
import re
import collections
import numpy as np

def is_speaker(lines, i):
    """tiny-shakespeare marks a speaker label as the first line of a block: it follows a blank
    line (or starts the file) and ends with ':' ("ROMEO:", "First Citizen:", "Senators, &C:").
    Dialogue lines that happen to end with ':' ("March on, my fellows:") never follow a blank
    line, so they are kept."""
    return (i == 0 or not lines[i - 1].strip()) and lines[i].strip().endswith(':')


HEADER = re.compile(r"[A-Z0-9 ]+")


def is_header(line):
    """A page header such as "3 KING HENRY VI": capitals, digits and spaces only, with at least
    two letters. In tiny-shakespeare this matches exactly 27 lines, all "3 KING HENRY VI"; no
    speech line is written in capitals, and there are no act or scene markers or stage directions."""
    s = line.strip()
    return bool(HEADER.fullmatch(s)) and sum(ch.isalpha() for ch in s) >= 2


def tokenize(text):
    """Lower-case words. Speaker labels ("ROMEO:") are dropped, so a name only counts
    when a character actually says it; page headers ("3 KING HENRY VI") are dropped too."""
    raw = text.split('\n')
    lines = [l for i, l in enumerate(raw) if not is_speaker(raw, i) and not is_header(l)]
    toks = [w.strip("'") for w in re.findall(r"[a-z']+", '\n'.join(lines).lower())]
    return [w for w in toks if w]


def build_vocab(toks, size):
    c = collections.Counter(toks)
    words = sorted(c, key=lambda w: (-c[w], w))[:size]   # ties broken alphabetically: deterministic
    return words, np.array([c[w] for w in words], dtype=np.int64)


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # = 1/(1+e^-x), without overflow


def loss_grads(Win, Wout, c, o, neg):
    """c, o: (B,) int; neg: (B, k) int. Returns the summed loss and per-row gradients
    gv (B, D) for W_in[c], go (B, D) for W_out[o], gn (B, k, D) for W_out[neg]."""
    v = Win[c]                       # (B, D)
    uo = Wout[o]                     # (B, D)
    un = Wout[neg]                   # (B, k, D)
    so = sig(np.einsum('bd,bd->b', uo, v))           # (B,)
    sn = sig(np.einsum('bkd,bd->bk', un, v))         # (B, k)
    loss = -np.log(so + 1e-12).sum() - np.log(1.0 - sn + 1e-12).sum()
    eo = so - 1.0
    gv = eo[:, None] * uo + np.einsum('bk,bkd->bd', sn, un)
    go = eo[:, None] * v
    gn = sn[:, :, None] * v[:, None, :]
    return loss, gv, go, gn


def scatter(Win, Wout, c, o, neg, gv, go, gn):
    """Dense gradients (same shape as the matrices), adding up repeated rows."""
    dWin = np.zeros_like(Win)
    dWout = np.zeros_like(Wout)
    np.add.at(dWin, c, gv)
    np.add.at(dWout, o, go)
    np.add.at(dWout, neg.ravel(), gn.reshape(-1, Win.shape[1]))
    return dWin, dWout


def unit(W):
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def neighbours(U, i, n=10):
    """Top-n by cosine in the full space. U: unit rows (float64). Excludes i itself.
    Ties (never seen in practice) break by lower index, as in the page."""
    s = U @ U[i]
    s[i] = -np.inf
    idx = np.lexsort((np.arange(len(s)), -s))[:n]
    return [(int(j), float(s[j])) for j in idx]
