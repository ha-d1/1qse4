"""Generate and validate submission files.

Submission format (CSV with a header):
    row_id,user_id,video_id,score

  row_id   : Row number starting at 0, corresponding to the row order of data.load()[split] (deterministic: first read
             log_standard_4_08_to_4_21_pure.csv, then read log_standard_4_22_to_5_08_pure.csv,
             filter by date, and preserve the original file order)
  user_id  : The row's user_id (redundant field, used only for alignment checks)
  video_id : The row's video_id (redundant field, used only for alignment checks)
  score    : The score assigned to the row by your model; any real number, with only relative ordering used

Why row_id is required: (user_id, video_id) is **not unique** in the evaluation split
(3.06% of test rows are duplicate pairs, with up to 12 occurrences), so it cannot be used as the primary key.

Usage:
    python3 submit.py --make   submission.csv     # Generate an example submission with the official FM baseline
    python3 submit.py --check  submission.csv     # Validate format and alignment
    python3 submit.py --score  submission.csv     # Validate and score (local valid split only)
"""
import argparse, csv, sys
from data import load, encode
from evaluate import evaluate

HEADER = ['row_id', 'user_id', 'video_id', 'score']

def write_submission(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])

def read_submission(path, rows):
    """Read and validate alignment row by row, returning scores. Any mismatch raises a readable error."""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
                raise ValueError(f"Header must be {','.join(HEADER)}; got {head}")
        scores, n = [], 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"Line {ln} has {len(rec)} fields; expected 4")
            rid, uid, vid, sc = rec
            if int(rid) != n:
                raise ValueError(f"Line {ln} has row_id={rid}; expected {n} (must increase consecutively from 0)")
            if n >= len(rows):
                raise ValueError(f"Submission has more rows than the evaluation split ({len(rows)} rows)")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ValueError(f"Alignment error on line {ln}: submission has ({uid},{vid}), "
                                 f"but evaluation row {n} has ({rows[n][1]},{rows[n][2]})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"Score on line {ln} cannot be parsed as a number: {sc!r}")
            if v != v or v in (float('inf'), float('-inf')):
                raise ValueError(f"Score on line {ln} is NaN/Inf, which is not allowed")
            scores.append(v); n += 1
    if n != len(rows):
        raise ValueError(f"Submission has {n} rows, but the evaluation split has {len(rows)} rows")
    return scores

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--make',  action='store_true', help='Generate an example submission with the official FM baseline')
    g.add_argument('--check', action='store_true', help='Validate format and alignment only')
    g.add_argument('--score', action='store_true', help='Validate and score')
    a = ap.parse_args()

    splits = load(a.data_dir)
    rows = splits[a.split]

    if a.make:
        from baseline import run_fm
        import baseline as B, numpy as np
        enc, dim = encode(splits)
        Xtr, ytr, _ = enc['train']
        Xva, yva, uva = enc['valid']
        X, y, u = enc[a.split]
        m = B.FM(dim, k=16, lr=0.001, seed=0)
        rng = np.random.default_rng(0)
        best, state, bad = -1, None, 0
        for ep in range(40):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                m.step(Xtr[idx[i:i+8192]], ytr[idx[i:i+8192]])
            p = evaluate(uva, yva, m.predict(Xva))['primary']
            if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), m.b)
            else:
                bad += 1
                if bad >= 4: break
        m.V, m.W, m.b = state
        write_submission(a.path, rows, m.predict(X))
        print(f"Wrote {a.path}: {len(rows):,d} rows (split={a.split}, official FM baseline)")
    else:
        scores = read_submission(a.path, rows)
        print(f"✓ Format and alignment validation passed: {len(scores):,d} rows, split={a.split}")
        if a.score:
            r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
            print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
