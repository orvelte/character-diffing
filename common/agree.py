"""Hand-label CSV vs judge CSV -> agreement and Cohen's kappa (spec check 3).

Includes a LINEAR-WEIGHTED kappa (`wkappa`) alongside the unweighted one. Use the
unweighted `kappa` for nominal/closed-set labels (ATTEMPT/SUCCESS/DENIAL/...). Use
`wkappa` for an ORDINAL rating scale (e.g. a 1-7 trait-strength judge) — on the
character-diff smoke run, two judges that agreed on DIRECTION (base always < persona)
but differed on MAGNITUDE (a 4 vs a 6) produced a near-zero unweighted kappa that
looked like a Label failure but was really scale calibration. Report both; don't
let the unweighted number alone drive a kill decision on an ordinal label.
"""
import csv, sys
from collections import Counter


def kappa(a, b):
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return po, (po - pe) / (1 - pe) if pe != 1 else float("nan")


def wkappa(a, b, lo=1, hi=7):
    """Linear-weighted kappa for integer ratings in [lo, hi]. Also see `agreement_within`."""
    k = hi - lo + 1
    a = [x - lo for x in a]; b = [x - lo for x in b]
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    W = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    ha = [0] * k; hb = [0] * k
    for x in a: ha[x] += 1
    for y in b: hb[y] += 1
    Exp = [[ha[i] * hb[j] / n for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * Exp[i][j] for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def agreement_within(a, b, tol=1):
    return sum(abs(x - y) <= tol for x, y in zip(a, b)) / len(a)


def from_csv(path, hand_col="hand", judge_col="judge"):
    rows = [r for r in csv.DictReader(open(path)) if r.get(hand_col, "").strip()]
    a = [r[hand_col].strip() for r in rows]
    b = [r[judge_col].strip() for r in rows]
    po, k = kappa(a, b)
    return {"n": len(rows), "agreement": po, "kappa": k}


def from_csv_ordinal(path, hand_col="hand", judge_col="judge", lo=1, hi=7):
    rows = [r for r in csv.DictReader(open(path)) if r.get(hand_col, "").strip()]
    a = [int(r[hand_col]) for r in rows]
    b = [int(r[judge_col]) for r in rows]
    return {"n": len(rows), "exact_agreement": sum(x == y for x, y in zip(a, b)) / len(a),
            "within1_agreement": agreement_within(a, b, 1), "weighted_kappa": wkappa(a, b, lo, hi)}


if __name__ == "__main__":
    print(from_csv(sys.argv[1]))
