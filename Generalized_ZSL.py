# Generalized_ZSL.py
# GZSL runner with CORRECT harmonic mean and IP-SAE reconstructed visual space
# Supports AWA2/CUB/SUN/AWA1 via load_data.py

import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split

import load_data as ld
from sae_helper import SAE, NormalizeFea


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


def _macro_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> float:
    """
    Macro-average per-class accuracy.
    Returns accuracy in [0,1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    class_ids = np.asarray(class_ids).astype(int)

    accs = []
    for c in class_ids:
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            continue
        accs.append(np.mean(y_pred[idx] == c))
    return float(np.mean(accs)) if accs else 0.0


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
        X_hat_visual: [N x D_visual]
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


def run_gzsl(
    dataset: str = "AWA2",
    int_proj: bool = True,
    paired_sample: bool = True,
    lamb: int = 500000,
    seen_test_frac: float = 0.2,
    seed: int = 42,
    print_ip_space: bool = True,
    ip_preview_dims: int = 30,
    ip_preview_rows: int = 1,
):
    """
    Runs a demo-style GZSL:
      - Train SAE on seen data
      - Test on (unseen test) + (held-out seen test)
      - Compute seen/unseen macro accuracies and harmonic mean correctly
      - ALSO compute the IP-SAE reconstructed visual space for X_te (after model)

    Returns dict:
      setting, dataset, harmonic, seen, unseen,
      X_te, Y_te, S_all, proj_all,
      X_te_ip_visual (visual space after IP-SAE)
    """
    loader = _get_loader(dataset)

    # Loader returns:
    # X_tr (seen instances), X_unseen (unseen instances), S_tr,
    # S_unseen_proto, Y_unseen, unseen_cls_ids, Y_tr, S_unseen_all, att (all class attributes)
    X_tr, X_unseen, S_tr, _S_unseen_proto, Y_unseen, unseen_cls_ids, Y_tr, _S_unseen_all, att = loader(int_proj=int_proj, paired_sample=paired_sample)

    X_tr = np.asarray(X_tr)
    Y_tr = np.asarray(Y_tr).astype(int)
    X_unseen = np.asarray(X_unseen)
    Y_unseen = np.asarray(Y_unseen).astype(int)
    att = np.asarray(att)
    S_tr = np.asarray(S_tr)

    # Hold out some seen instances for GZSL testing
    try:
        _X_seen_train, X_seen_test, _y_seen_train, y_seen_test = train_test_split(
            X_tr,
            Y_tr,
            test_size=seen_test_frac,
            random_state=seed,
            stratify=Y_tr,
        )
    except ValueError:
        _X_seen_train, X_seen_test, _y_seen_train, y_seen_test = train_test_split(
            X_tr,
            Y_tr,
            test_size=seen_test_frac,
            random_state=seed,
            stratify=None,
        )

    # Test set: unseen test + held-out seen test
    X_te = np.concatenate([X_unseen, X_seen_test], axis=0)
    Y_te = np.concatenate([Y_unseen, y_seen_test], axis=0)

    # Train SAE on seen training data (use full X_tr/S_tr like baseline)
    W = SAE(X_tr.T, S_tr.T, lamb).T

    # All-class prototypes for GZSL classification
    S_all = NormalizeFea(att.T, 2).T  # [C_all x A]
    proj_all = S_all.dot(W.T)         # [C_all x D_in]

    # Predict
    sim = 1.0 - spatial.distance.cdist(X_te, proj_all, "cosine")
    y_pred = np.argmax(sim, axis=1).astype(int)

    # Seen/unseen class sets
    seen_cls_ids = np.unique(Y_tr).astype(int)
    unseen_cls_ids = np.unique(unseen_cls_ids).astype(int)

    seen_acc = _macro_class_accuracy(Y_te, y_pred, seen_cls_ids)
    unseen_acc = _macro_class_accuracy(Y_te, y_pred, unseen_cls_ids)
    harm = (2.0 * seen_acc * unseen_acc) / (seen_acc + unseen_acc + 1e-12)

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
        "setting": "GZSL",
        "dataset": dataset.upper(),
        "harmonic": harm * 100.0,
        "seen": seen_acc * 100.0,
        "unseen": unseen_acc * 100.0,
        "X_te": X_te,
        "Y_te": Y_te,
        "S_all": S_all,
        "proj_all": proj_all,
        "X_te_ip_visual": X_te_ip_visual,  # <- use this for t-SNE in GUI
        "y_pred": y_pred,
        "seen_cls_ids": seen_cls_ids,
        "unseen_cls_ids": unseen_cls_ids,
    }


if __name__ == "__main__":
    out = run_gzsl(dataset="AWA1", int_proj=True, print_ip_space=True)
    print(
        "GZSL harmonic: {:.3f}% | seen: {:.3f}% | unseen: {:.3f}%".format(
            out["harmonic"], out["seen"], out["unseen"]
        )
    )
