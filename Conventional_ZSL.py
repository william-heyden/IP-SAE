# Conventional_ZSL.py
# CZSL runner + IP-SAE reconstructed visual space ("after IP-SAE")
# Supports AWA2/CUB/SUN/AWA1 via load_data.py
#
# IMPORTANT:
# - Returns y_pred (and sim) so the GUI can enable and plot a confusion matrix.
# - CZSL evaluates UNSEEN classes only (by design).

import numpy as np
from scipy import spatial

import load_data as ld
from sae_helper import SAE, acc_zsl, NormalizeFea


def _get_loader(dataset: str):
    dataset = dataset.strip().lower()
    mapping = {
        "awa2": ld.awa2,
        "cub": ld.cub,
        "sun": ld.sun,
        "awa1": ld.awa1,
    }
    if dataset not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose one of: {list(mapping.keys())}")
    return mapping[dataset]


def _ip_sae_reconstructed_visual_space(
    X: np.ndarray,
    W: np.ndarray,
    attr_dim: int | None,
    int_proj: bool,
) -> np.ndarray:
    """
    Compute reconstructed ("after IP-SAE") representation in the visual space:

        Z = X W
        X_hat = Z W^T = X W W^T

    If int_proj=True, X contains [visual, semantic] concatenation.
    In that case, we return only the reconstructed visual part (first D_visual dims).

    Returns:
        X_hat_visual: [N x D_visual]   (if int_proj=True)
        or [N x D_in]                  (if int_proj=False)
    """
    X = np.asarray(X)

    # Reconstruct in the model's input space
    X_hat = (X @ W) @ W.T  # [N x D_in]

    if not int_proj:
        return X_hat

    if attr_dim is None:
        raise ValueError("attr_dim is required when int_proj=True")

    D_in = X.shape[1]
    D_visual = D_in - attr_dim
    if D_visual <= 0:
        raise ValueError(f"Bad dimension split: D_in={D_in}, attr_dim={attr_dim}")

    return X_hat[:, :D_visual]


def run_conventional(
    dataset: str = "AWA2",
    int_proj: bool = True,
    paired_sample: bool = True,
    lamb: int = 500000,
    print_ip_space: bool = True,
    ip_preview_dims: int = 30,
    ip_preview_rows: int = 1,
):
    """
    Conventional ZSL:
      - Train SAE on seen data
      - Classify unseen test instances using unseen class prototypes
      - Compute CZSL accuracy (unseen-only)
      - ALSO compute IP-SAE reconstructed visual space for X_te (after model)

    Returns dict (for GUI):
      setting, dataset, accuracy,
      X_te, Y_te,
      sim, y_pred,
      S_proto, proj_proto,
      X_te_ip_visual
    """
    loader = _get_loader(dataset)

    # Loader returns:
    # X_tr (seen train instances), X_te (unseen test instances),
    # S_tr (attrs per seen instance),
    # S_te_pro (unseen class prototypes),
    # Y_te (unseen labels), te_cl_id,
    # Y_tr, S_te_all, att
    X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, Y_tr, S_te_all, att = loader(int_proj=int_proj, paired_sample=paired_sample)

    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    S_tr = np.asarray(S_tr)
    S_te_pro = np.asarray(S_te_pro)
    Y_te = np.asarray(Y_te).astype(int)
    te_cl_id = np.asarray(te_cl_id).astype(int)

    # Train SAE mapping
    W = SAE(X_tr.T, S_tr.T, lamb).T  # shape: [A x D_in]

    # Prototype classification (semantic -> visual)
    S_proto = NormalizeFea(S_te_pro.T, 2).T   # [C_unseen x A]
    proj_proto = S_proto.dot(W.T)             # [C_unseen x D_in]

    # Similarity matrix (higher is better)
    sim = 1.0 - spatial.distance.cdist(X_te, proj_proto, "cosine")  # [N_unseen x C_unseen]

    # Predicted class index in the *unseen prototype list*
    # In this repo, te_cl_id are the unseen class IDs (0-based original label space)
    # We map prototype index -> class id for confusion matrix.
    pred_proto_idx = np.argmax(sim, axis=1).astype(int)
    y_pred = te_cl_id[pred_proto_idx]  # y_pred is in the same label space as Y_te

    # Accuracy using the repo helper (expects similarity-like matrix per its sorting)
    acc = acc_zsl(sim, te_cl_id, Y_te) * 100.0

    # ---- IP-SAE reconstructed visual space ("after IP-SAE") ----
    attr_dim = S_tr.shape[1] if S_tr.ndim == 2 else None
    X_te_ip_visual = _ip_sae_reconstructed_visual_space(
        X=X_te,
        W=W,
        attr_dim=attr_dim,
        int_proj=int_proj,
    )

    if print_ip_space:
        np.set_printoptions(precision=4, suppress=True)
        print(f"[{dataset.upper()}] X_te shape (input to SAE): {X_te.shape}")
        print(f"[{dataset.upper()}] X_te_ip_visual shape (after IP-SAE): {X_te_ip_visual.shape}")

        rows = min(ip_preview_rows, X_te_ip_visual.shape[0])
        cols = min(ip_preview_dims, X_te_ip_visual.shape[1])
        print(f"[{dataset.upper()}] Preview of reconstructed visual space (first {rows} row(s), first {cols} dims):")
        print(X_te_ip_visual[:rows, :cols])

    return {
        "setting": "CZSL",
        "dataset": dataset.upper(),
        "accuracy": acc,
        "X_te": X_te,
        "Y_te": Y_te,
        "sim": sim,                 # allows GUI to derive y_pred if needed
        "y_pred": y_pred,           # enables confusion matrix directly
        "S_proto": S_proto,
        "proj_proto": proj_proto,
        "X_te_ip_visual": X_te_ip_visual,
        "te_cl_id": te_cl_id,       # useful for debugging/mapping
        "W": W,                     # optional; remove if you don't want to expose it
    }


if __name__ == "__main__":
    out = run_conventional(dataset="AWA2", int_proj=True, print_ip_space=True)
    print(f"CZSL accuracy: {out['accuracy']:.3f}%")
