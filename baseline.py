"""KuaiRand-Pure baselines。
  --model pop   : item popularity (official baseline, statistical only, no training)
  --model fm    : Factorization Machine (starting model; students should improve from here)
  --model random: random scores (lower bound, for checking that the evaluation code works)
Depends only on numpy. See README.md for usage.
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_bpr(self, X_pos, X_neg):
        B = len(X_pos)

        # Scores for positive and negative examples
        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)

        # BPR: we want positive score > negative score
        diff = z_pos - z_neg

        # Gradient of -log(sigmoid(pos - neg))
        g = ((sigmoid(diff) - 1.0) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        # Positive-example gradients
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(
            gV,
            X_pos,
            g[:, None, None] * (S_pos[:, None, :] - E_pos)
        )

        # Negative-example gradients have the opposite sign
        np.add.at(gW, X_neg, -g[:, None])
        np.add.at(
            gV,
            X_neg,
            -g[:, None, None] * (S_neg[:, None, :] - E_neg)
        )

        # Regularization
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW)
        ):
            M *= b1
            M += (1 - b1) * G

            Vv *= b2
            Vv += (1 - b2) * (G * G)

            P -= self.lr * (
                M / (1 - b1 ** self.t)
            ) / (
                np.sqrt(Vv / (1 - b2 ** self.t)) + eps
            )

        # b cancels in pos_score - neg_score,
        # so there is no useful bias update.

        loss = -np.mean(np.log(sigmoid(diff) + 1e-9))

        return float(loss)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def make_bpr_pairs(X, y, users, seed=0):
    rng = np.random.default_rng(seed)

    user_rows = collections.defaultdict(lambda: [[], []])

    for i, (user, label) in enumerate(zip(users, y)):
        if label == 1:
            user_rows[user][1].append(i)
        else:
            user_rows[user][0].append(i)

    pos_idx = []
    neg_idx = []

    for user, (negatives, positives) in user_rows.items():
        if not positives or not negatives:
            continue

        for p in positives:
            n = rng.choice(negatives)

            pos_idx.append(p)
            neg_idx.append(n)

    pos_idx = np.asarray(pos_idx, dtype=np.int64)
    neg_idx = np.asarray(neg_idx, dtype=np.int64)

    return X[pos_idx], X[neg_idx]

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

def run_bpr(
    splits,
    k=16,
    lr=0.001,
    epochs=40,
    bs=8192,
    patience=4,
    seed=0,
    verbose=True
):
    enc, dim = encode(splits)

    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)

    best = -1
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):

        # NEW: generate fresh negative pairs every epoch
        X_pos, X_neg = make_bpr_pairs(
            Xtr,
            ytr,
            utr,
            seed=seed + ep
        )

        if verbose and ep == 1:
            print(f"BPR training pairs: {len(X_pos):,}")

        # Shuffle the newly generated pairs
        idx = rng.permutation(len(X_pos))

        t0 = time.time()
        losses = []

        for i in range(0, len(idx), bs):

            batch = idx[i:i + bs]

            loss = m.step_bpr(
                X_pos[batch],
                X_neg[batch]
            )

            losses.append(loss)

        # Evaluate on VALIDATION data
        va = evaluate(
            uva,
            yva,
            m.predict(Xva)
        )

        if verbose:
            print(
                f"  epoch {ep:2d} | "
                f"BPR loss {np.mean(losses):.4f} | "
                f"valid GAUC {va['GAUC']:.4f} "
                f"nDCG@5 {va['nDCG@5']:.4f} "
                f"primary {va['primary']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

        # Keep the best validation model
        if va['primary'] > best + 1e-5:

            best = va['primary']
            bad = 0

            best_state = (
                m.V.copy(),
                m.W.copy(),
                np.float32(m.b)
            )

        else:
            bad += 1

            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    # Restore best model
    m.V, m.W, m.b = best_state

    return {
        'valid': evaluate(
            uva,
            yva,
            m.predict(Xva)
        ),

        'test': evaluate(
            ute,
            yte,
            m.predict(Xte)
        )
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='Data directory after extracting KuaiRand-Pure')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'bpr','random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'bpr': lambda s: run_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
