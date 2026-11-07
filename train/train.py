"""Train skip-gram with negative sampling on tiny-shakespeare. Fixed seed, the
standard word2vec defaults (window 5, 5 negatives, unigram^0.75, sub-sampling t=1e-3,
lr 0.025 decaying linearly to 0). Writes ckpt.npz and log.json.
Run three times with these same settings: the second run fixed the speaker-label filter in
w2v.tokenize (found by an audit), the third also removed a repeated page header. probes.json
was never changed. The shipped ckpt.npz and log.json are from the third run."""
import json, os, time
import numpy as np
from w2v import tokenize, build_vocab, loss_grads, sig

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(seed=20261106, vocab=3000, dim=48, window=5, neg=5, sample=1e-3,
           lr=0.025, epochs=15, batch=256, holdout=0.05)


def pairs_for(ids, rng, window):
    """Dynamic window as in word2vec: each centre draws b in 1..window, uses offsets 1..b."""
    b = rng.integers(1, window + 1, size=len(ids))
    C, O = [], []
    for d in range(1, window + 1):
        m = b[:-d] >= d          # centre i, context i+d
        C.append(ids[:-d][m]); O.append(ids[d:][m])
        m = b[d:] >= d           # centre i, context i-d
        C.append(ids[d:][m]); O.append(ids[:-d][m])
    return np.concatenate(C), np.concatenate(O)


def main():
    rng = np.random.default_rng(CFG['seed'])
    text = open(os.path.join(HERE, '../input.txt')).read()
    toks = tokenize(text)
    words, counts = build_vocab(toks, CFG['vocab'])
    index = {w: i for i, w in enumerate(words)}
    ids_all = np.array([index[w] for w in toks if w in index], dtype=np.int64)
    n_hold = int(len(ids_all) * CFG['holdout'])
    train_ids, hold_ids = ids_all[:-n_hold], ids_all[-n_hold:]

    # counts from the training part only, for sub-sampling and the negative table
    tc = np.bincount(train_ids, minlength=len(words)).astype(np.float64)
    freq = tc / tc.sum()
    keep_p = np.minimum(1.0, (np.sqrt(freq / CFG['sample']) + 1) * CFG['sample'] / np.maximum(freq, 1e-12))
    neg_p = tc ** 0.75; neg_p /= neg_p.sum()
    neg_cdf = np.cumsum(neg_p)
    draw_neg = lambda shape: np.minimum(np.searchsorted(neg_cdf, rng.random(shape)), len(words) - 1)

    V, D, K = len(words), CFG['dim'], CFG['neg']
    Win = (rng.random((V, D)) - 0.5) / D        # word2vec's init
    Wout = np.zeros((V, D))

    # held-out pairs: fixed window 5 (no sub-sampling), fixed negatives
    hrng = np.random.default_rng(1)
    hc, ho = pairs_for(hold_ids, np.random.default_rng(2), CFG['window'])
    hn = np.minimum(np.searchsorted(neg_cdf, hrng.random((len(hc), K))), V - 1)
    def held_loss():
        tot = 0.0
        for s in range(0, len(hc), 8192):
            tot += loss_grads(Win, Wout, hc[s:s+8192], ho[s:s+8192], hn[s:s+8192])[0]
        return tot / len(hc)

    # rough total for the linear lr decay
    est_pairs = int(keep_p[train_ids].sum() * (CFG['window'] + 1)) * CFG['epochs']
    log = dict(cfg=CFG, tokens_total=len(toks), types_total=len(set(toks)), vocab=V,
               tokens_in_vocab=int(len(ids_all)), coverage=len(ids_all) / len(toks),
               train_tokens=int(len(train_ids)), holdout_tokens=int(n_hold), holdout_pairs=int(len(hc)),
               min_count_in_vocab=int(counts[-1]), epochs=[])
    log['held_loss_init'] = held_loss()
    print('held-out loss at init %.4f (k+1)ln2 = %.4f' % (log['held_loss_init'], (K + 1) * np.log(2)))
    seen = 0; t0 = time.time()
    for ep in range(CFG['epochs']):
        ids = train_ids[rng.random(len(train_ids)) < keep_p[train_ids]]
        C, O = pairs_for(ids, rng, CFG['window'])
        perm = rng.permutation(len(C)); C, O = C[perm], O[perm]
        tot = 0.0
        for s in range(0, len(C), CFG['batch']):
            c, o = C[s:s+CFG['batch']], O[s:s+CFG['batch']]
            neg = draw_neg((len(c), K))
            lr = CFG['lr'] * max(1e-4, 1 - seen / est_pairs)
            loss, gv, go, gn = loss_grads(Win, Wout, c, o, neg)
            np.add.at(Win, c, -lr * gv)
            np.add.at(Wout, o, -lr * go)
            np.add.at(Wout, neg.ravel(), -lr * gn.reshape(-1, D))
            tot += loss; seen += len(c)
        e = dict(epoch=ep + 1, kept_tokens=int(len(ids)), pairs=int(len(C)), train_loss=tot / len(C),
                 held_loss=held_loss(), lr_end=lr, seconds=round(time.time() - t0, 1))
        log['epochs'].append(e)
        print(json.dumps(e))
    log['pairs_total'] = seen
    log['train_seconds'] = round(time.time() - t0, 1)
    log['held_loss_final'] = log['epochs'][-1]['held_loss']
    np.savez(os.path.join(HERE, 'ckpt.npz'), Win=Win, Wout=Wout, counts=counts, words=np.array(words))
    json.dump(log, open(os.path.join(HERE, 'log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
