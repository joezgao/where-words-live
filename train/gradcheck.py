"""Hand-derived gradients vs central finite differences on a tiny case, float64.
Repeated indices are on purpose (word 2 is a centre twice and also a negative), so the
add-up of repeated rows is checked too."""
import json, os
import numpy as np
from w2v import loss_grads, scatter

rng = np.random.default_rng(0)
V, D = 7, 4
Win = rng.normal(0, 0.5, (V, D))
Wout = rng.normal(0, 0.5, (V, D))
c = np.array([2, 5, 2])
o = np.array([3, 1, 6])
neg = np.array([[0, 4], [2, 6], [3, 3]])

loss, gv, go, gn = loss_grads(Win, Wout, c, o, neg)
dWin, dWout = scatter(Win, Wout, c, o, neg, gv, go, gn)

def num(W, eps=1e-6):
    g = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        old = W[idx]
        W[idx] = old + eps; lp = loss_grads(Win, Wout, c, o, neg)[0]
        W[idx] = old - eps; lm = loss_grads(Win, Wout, c, o, neg)[0]
        W[idx] = old
        g[idx] = (lp - lm) / (2 * eps)
    return g

nWin, nWout = num(Win), num(Wout)
rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12))
res = {"V": V, "D": D, "batch": len(c), "k": neg.shape[1], "loss": float(loss),
       "max_rel_err_Win": rel(dWin, nWin), "max_rel_err_Wout": rel(dWout, nWout)}
res["pass"] = bool(res["max_rel_err_Win"] < 1e-6 and res["max_rel_err_Wout"] < 1e-6)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(__file__), 'gradcheck.json'), 'w'), indent=1)
assert res["pass"]
