"""Evaluate the trained vectors and export the page data.
- unit-normalise W_in, store as float16 (the page's vectors)
- neighbours (cosine, full 48-d space) for the pre-registered probes, from the float16 vectors
  exactly as the page decodes them; also how often float16 changes a top-10 list vs float64
- analogies (3CosAdd) from probes.json
- PCA to 3D, with the share of variance the three axes keep
- for the 8 words at each end of each PCA axis, how tied they are to one stretch of the text,
  as a group (one shared window; this is what the page reports) and word by word
  (localisation: the share of a word's occurrences inside its densest window of 10% of the token
  stream; the plays sit one after another in the file), against random groups of 8 vocab words
Writes data.json (for build.py), summary.json (all reported numbers) and ../test/expected.json."""
import base64, json, os
import numpy as np
from w2v import unit, neighbours, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ck = np.load(os.path.join(HERE, 'ckpt.npz'))
log = json.load(open(os.path.join(HERE, 'log.json')))
probes = json.load(open(os.path.join(HERE, 'probes.json')))
grad = json.load(open(os.path.join(HERE, 'gradcheck.json')))
words = [str(w) for w in ck['words']]
counts = ck['counts']
index = {w: i for i, w in enumerate(words)}
V, D = ck['Win'].shape

U64 = unit(ck['Win'].astype(np.float64))
h16 = U64.astype(np.float16)
U16 = h16.astype(np.float64)              # what the page sees (it decodes halves to doubles)
U = unit(U16)                             # the page re-normalises after decoding, so do we

# float16 vs float64: how many of the 10 neighbours change, over the whole vocabulary
changed = []
for i in range(V):
    a = {j for j, _ in neighbours(U64, i)}
    b = {j for j, _ in neighbours(U, i)}
    changed.append(10 - len(a & b))
changed = np.array(changed)

nn = {}
missing = []
for w in probes['probes']:
    if w not in index:
        missing.append(w); continue
    nn[w] = [(words[j], round(s, 4)) for j, s in neighbours(U, index[w])]

def analogy(a, b, c, d):
    if any(x not in index for x in (a, b, c, d)):
        return dict(q=[a, b, c, d], missing=[x for x in (a, b, c, d) if x not in index])
    q = U[index[b]] - U[index[a]] + U[index[c]]
    q /= np.linalg.norm(q)
    s = U @ q
    for x in (a, b, c):
        s[index[x]] = -np.inf
    order = np.argsort(-s, kind='stable')
    rank = int(np.where(order == index[d])[0][0]) + 1
    return dict(q=[a, b, c, d], rank=rank, top5=[words[j] for j in order[:5]], hit=rank == 1)

an = [analogy(*q) for q in probes['analogies']]

# PCA on the unit vectors (centred)
X = U - U.mean(0)
_, S, Vt = np.linalg.svd(X, full_matrices=False)
var = S ** 2 / (S ** 2).sum()
P = X @ Vt[:3].T
# PCA signs are arbitrary. Fix them by a rule, not by eye: each axis's long tail (positive skew)
# points the + way. Display only; it changes no distance, neighbour or number on the page.
skew = ((P - P.mean(0)) ** 3).mean(0)
P *= np.where(skew < 0, -1.0, 1.0)
P /= np.abs(P).max()                      # fit in [-1, 1]
pca3 = float(var[:3].sum())

# localisation of the axis ends vs random words
toks = tokenize(open(os.path.join(HERE, '../input.txt')).read())
occ = {}
for k, w in enumerate(toks):
    if w in index:
        occ.setdefault(w, []).append(k)
win = int(len(toks) * 0.1)
def loc1(w):
    p = np.array(occ[w]); j = np.searchsorted(p, p + win, side='left')
    return float((j - np.arange(len(p))).max() / len(p))
LOC = np.array([loc1(w) for w in words])
lr = np.random.default_rng(0)
rand300 = float(LOC[lr.choice(V, 300, replace=False)].mean())
groups = np.array([LOC[lr.choice(V, 8, replace=False)].mean() for _ in range(10000)])
ends = []
for k in range(3):
    o = np.argsort(P[:, k])
    e = {}
    for name, ids in (('low', o[:8]), ('high', o[::-1][:8])):
        e[name] = [words[j] for j in ids]
        e[name + '_loc'] = round(float(LOC[ids].mean()), 4)
    ends.append(e)
loc_summary = dict(window=0.1, random300=round(rand300, 4), vocab_mean=round(float(LOC.mean()), 4),
                   group8_p95=round(float(np.quantile(groups, .95)), 4),
                   group8_p99=round(float(np.quantile(groups, .99)), 4))

# Group measure (what the page reports). The per-word measure above lets each word pick its own
# busiest stretch, so it cannot see a group whose words share one play but also turn up elsewhere.
# Here the 8 words of an end must share ONE window: score = max over window starts t of
# mean_w (uses of w in [t, t+L)) / (all uses of w). Exact at token level: an optimal window can be
# slid right until its left edge sits on a use (or the end of the text), so only those starts are tried.
# The same statistic for 10,000 random 8-word groups (seed 0) gives the baseline; 5%/10%/15% windows.
raw_lines = open(os.path.join(HERE, '../input.txt')).read().split('\n')
# where each play starts: the line of its first speaker label, read off the label sequence
PLAYS = [(0, 'Coriolanus'), (5953, 'Richard III'), (11698, 'Richard II'), (15600, 'Romeo and Juliet'),
         (20333, '3 Henry VI'), (24848, "The Winter's Tale"), (29644, 'Measure for Measure'),
         (34193, 'The Taming of the Shrew'), (38565, 'The Tempest')]
play_tok = np.array([len(tokenize('\n'.join(raw_lines[:ln]))) if ln else 0 for ln, _ in PLAYS])
assert play_tok[-1] < len(toks) and np.all(np.diff(play_tok) > 0)
NT = len(toks)
POSN = {w: np.array(occ[w]) for w in words}
def group_best(ws, L):
    ps = [POSN[w] for w in ws]
    T = np.unique(np.minimum(np.concatenate(ps + [np.array([NT - L])]), NT - L))
    sc = np.zeros(len(T))
    for p in ps:
        sc += (np.searchsorted(p, T + L) - np.searchsorted(p, T)) / len(p)
    k = int(sc.argmax())
    return float(sc[k] / len(ws)), int(T[k])
def play_of(ws, t, L):
    """The play that holds most of the group's in-window share."""
    acc = np.zeros(len(PLAYS))
    for w in ws:
        p = POSN[w]; q = p[(p >= t) & (p < t + L)]
        np.add.at(acc, np.searchsorted(play_tok, q, side='right') - 1, 1.0 / len(p))
    return PLAYS[int(acc.argmax())][1], round(float(acc.max() / acc.sum()), 3)
gr = np.random.default_rng(0)
rand_groups = [[words[j] for j in gr.choice(V, 8, replace=False)] for _ in range(10000)]
gstats = {}
for frac in (0.05, 0.10, 0.15):
    L = int(NT * frac)
    R = np.array([group_best(g, L)[0] for g in rand_groups])
    gstats[frac] = (L, R)
for e in ends:
    for name in ('low', 'high'):
        g = {}
        for frac, (L, R) in gstats.items():
            share, t = group_best(e[name], L)
            pct = round(float(100 * (R < share).mean()), 1)
            if frac == 0.10:
                play, purity = play_of(e[name], t, L)
                g.update(share=round(share, 4), start=round(t / NT, 3), end=round((t + L) / NT, 3),
                         play=play, play_part=purity, pct=pct)
            else:
                g['pct_w%02d' % round(frac * 100)] = pct
        e[name + '_group'] = g
L10, R10 = gstats[0.10]
group_summary = dict(window=0.1, groups=len(rand_groups), seed=0,
                     random_mean=round(float(R10.mean()), 4), random_p95=round(float(np.quantile(R10, .95)), 4),
                     random_p99=round(float(np.quantile(R10, .99)), 4),
                     by_window={'%.2f' % f: dict(p95=round(float(np.quantile(R, .95)), 4), p99=round(float(np.quantile(R, .99)), 4))
                                for f, (L, R) in gstats.items()},
                     plays=[dict(play=n, start=round(int(t) / NT, 4)) for (ln, n), t in zip(PLAYS, play_tok)])

# a quick reference point: cosine of random pairs vs. a word's 10th neighbour
rr = np.random.default_rng(0)
ri, rj = rr.integers(0, V, 20000), rr.integers(0, V, 20000)
m = ri != rj
rand_cos = (U[ri[m]] * U[rj[m]]).sum(1)

summary = dict(
    corpus=dict(tokens=log['tokens_total'], types=log['types_total'], vocab=V, dim=D,
                coverage=round(log['coverage'], 4), min_count=log['min_count_in_vocab'],
                train_tokens=log['train_tokens'], holdout_tokens=log['holdout_tokens']),
    train=dict(epochs=log['cfg']['epochs'], pairs=log['pairs_total'], seconds=log['train_seconds'],
               window=log['cfg']['window'], neg=log['cfg']['neg'], sample=log['cfg']['sample'],
               lr=log['cfg']['lr'], batch=log['cfg']['batch'], seed=log['cfg']['seed'],
               kept_per_epoch=int(np.mean([e['kept_tokens'] for e in log['epochs']])),
               train_loss_first=round(log['epochs'][0]['train_loss'], 3),
               train_loss_final=round(log['epochs'][-1]['train_loss'], 3),
               held_loss_init=round(log['held_loss_init'], 3),
               held_loss_ep1=round(log['epochs'][0]['held_loss'], 3),
               held_loss_best=round(min(e['held_loss'] for e in log['epochs']), 3),
               held_loss_final=round(log['held_loss_final'], 3)),
    gradcheck=dict(rel_err_in=grad['max_rel_err_Win'], rel_err_out=grad['max_rel_err_Wout']),
    pca=dict(var3=round(pca3, 4), var_each=[round(float(v), 4) for v in var[:3]],
             ends=ends, loc=loc_summary, group=group_summary),
    fp16=dict(lists_unchanged=int((changed == 0).sum()), lists_one_swap=int((changed == 1).sum()),
              max_changed=int(changed.max()), mean_changed=round(float(changed.mean()), 3)),
    cos=dict(random_mean=round(float(rand_cos.mean()), 3), random_p99=round(float(np.quantile(rand_cos, .99)), 3),
             tenth_nn_median=round(float(np.median([neighbours(U, i)[-1][1] for i in range(V)])), 3)),
    analogies=an, analogy_hits=sum(1 for a in an if a.get('hit')), analogy_total=len(an),
    probes_missing=missing, neighbours=nn)
json.dump(summary, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)

data = dict(words=words, counts=[int(c) for c in counts], dim=D,
            vec=base64.b64encode(h16.tobytes()).decode(),
            pos=base64.b64encode(P.astype(np.float16).tobytes()).decode(),
            stats=summary)
data['stats'] = {k: v for k, v in summary.items() if k != 'neighbours'}
json.dump(data, open(os.path.join(HERE, 'data.json'), 'w'), separators=(',', ':'))
os.makedirs(os.path.join(HERE, '../test'), exist_ok=True)
json.dump(dict(probes=probes['probes'], neighbours=nn), open(os.path.join(HERE, '../test/expected.json'), 'w'), indent=1)

print(json.dumps({k: v for k, v in summary.items() if k not in ('neighbours', 'analogies')}, indent=1))
for a in an: print(a)
for w, l in nn.items(): print('%-8s' % w, ', '.join('%s %.2f' % x for x in l))
print('data.json bytes', os.path.getsize(os.path.join(HERE, 'data.json')))
