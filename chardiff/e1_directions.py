"""E1 cross-persona geometry: build a diff-of-means direction for every released
persona, then PCA them (spec section 4, E1).

Activations only -- no generation -- so this is minutes per persona, not the hour a
generation pass costs. The adapter is the only thing that changes between personas, so
the base-model activations are computed ONCE and reused for every diff rather than
recomputed ten times.

    python -m chardiff.e1_directions build      # GPU, ~10 personas
    python -m chardiff.e1_directions pca        # CPU, reads what build wrote
"""
import json, pathlib
import torch

from common.localmodel import LocalModel
from . import directions as D
from .constitutions import PERSONAS

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "maius/llama-3.1-8b-it-personas"


def build(personas=PERSONAS, position="prompt_end", base=BASE, adapter=ADAPTER):
    prompts = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))
    got = {}
    for i, p in enumerate(personas):
        lm = LocalModel(base, adapter=adapter, subfolder=p)
        layers = D.layers_for(lm.n_layers)
        v, meta = D.build(lm, prompts, layers, position=position)
        D.save(f"llama_{p}_{position}", v, meta)
        got[p] = {L: float(meta["norms"][L]["rel"]) for L in layers}
        print(f"  [{i+1}/{len(personas)}] {p:14s} " +
              "  ".join(f"L{L}:{meta['norms'][L]['rel']:.2f}" for L in layers), flush=True)
        del lm
        torch.cuda.empty_cache()
    (RES / f"e1_build_{position}.json").write_text(json.dumps(got, indent=1))
    return got


def pca(personas=PERSONAS, position="prompt_end", layer=None):
    """PC1-PC3 variance explained and the full cosine matrix, at one layer.

    Vectors are NOT normalised before PCA: the diff norms differ substantially between
    personas (D-012 shows ||v||/||base|| spanning 0.5-1.4 across layers), and unit-
    normalising would throw away 'how far training moved this persona', which is part
    of the geometry the spec asks about. The mean is removed, as PCA requires.
    """
    vecs, metas = {}, {}
    for p in personas:
        v, m = D.load(f"llama_{p}_{position}")
        vecs[p], metas[p] = v, m
    layers = sorted(next(iter(vecs.values())).keys())
    layer = layer if layer is not None else layers[len(layers) // 2]

    X = torch.stack([vecs[p][layer].float() for p in personas])      # (n_personas, d)
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2)
    frac = (var / var.sum()).tolist()

    labels, M = D.cosine_matrix(vecs, layer)
    out = {"position": position, "layer": layer, "personas": personas,
           "pc_variance_explained": frac,
           "pc1": frac[0], "pc1_3": sum(frac[:3]),
           "cosine_labels": labels, "cosine_matrix": M.tolist(),
           "mean_offdiag_cosine": float(
               (M.sum() - M.trace()) / (len(labels) ** 2 - len(labels)))}
    (RES / f"e1_pca_{position}_L{layer}.json").write_text(json.dumps(out, indent=1))

    print(f"\nPCA over {len(personas)} persona directions, {position}, block {layer}")
    print("  variance explained: " +
          "  ".join(f"PC{i+1} {f:.1%}" for i, f in enumerate(frac[:4])))
    print(f"  PC1-3 cumulative:   {sum(frac[:3]):.1%}   "
          f"(spec E1 prediction: PC1 explains 40-60%)")
    print(f"  mean off-diagonal cosine between personas: {out['mean_offdiag_cosine']:.3f}")
    print("\n  cosine matrix:")
    print("      " + " ".join(f"{p[:6]:>6s}" for p in labels))
    for i, a in enumerate(labels):
        print(f"  {a[:5]:5s} " + " ".join(f"{M[i][j]:6.2f}" for j in range(len(labels))))
    return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    pos = sys.argv[2] if len(sys.argv) > 2 else "prompt_end"
    (build if cmd == "build" else pca)(position=pos)
