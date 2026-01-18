# gui_app.py
# GUI for IP-SAE demo:
# - Select dataset root or dataset folder
# - Choose dataset (AWA2/CUB/SUN/AWA1)
# - Choose mode (CZSL/GZSL)
# - Run model
# - t-SNE plots (visual / semantic / projected / IP-SAE reconstructed visual)
# - Confusion matrix:
#     * CZSL: unseen-only
#     * GZSL: can show All / Seen-only / Unseen-only
#   Includes safe Top-K handling (Top-K auto-clamped)
# - NEW: checkbox "Save confusion matrix PNG" (only saves if checked)

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import load_data as ld
from Conventional_ZSL import run_conventional
from Generalized_ZSL import run_gzsl


DATASETS = ["AWA2", "CUB", "SUN", "AWA1"]
REQUIRED = ["res101.mat", "att_splits.mat"]


def normalize_selection_to_data_root(selection: str, dataset: str) -> str:
    selection = os.path.abspath(selection)
    dataset = dataset.upper()
    if os.path.basename(selection).upper() == dataset:
        return os.path.dirname(selection)
    return selection


def dataset_dir(data_root: str, dataset: str) -> str:
    return os.path.join(os.path.abspath(data_root), dataset.upper())


def check_dataset_files(data_root: str, dataset: str) -> (bool, str):
    ds_dir = dataset_dir(data_root, dataset)
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(ds_dir, f))]
    if missing:
        return False, f"Missing in {ds_dir}: {', '.join(missing)}"
    return True, f"OK: found files in {ds_dir}"


def tsne_2d(X: np.ndarray, seed: int = 42, perplexity: int = 30) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("t-SNE input must be a 2D array.")
    p = min(perplexity, max(5, (X.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=p)
    return tsne.fit_transform(X)


def plot_tsne(points_2d: np.ndarray, labels: np.ndarray, title: str, max_classes: int = 25):
    labels = np.asarray(labels).astype(int)
    classes, counts = np.unique(labels, return_counts=True)

    order = np.argsort(-counts)
    keep = set(classes[order[:max_classes]])

    mask = np.array([c in keep for c in labels])
    pts = points_2d[mask]
    labs = labels[mask]

    plt.figure()
    for c in np.unique(labs):
        idx = labs == c
        plt.scatter(pts[idx, 0], pts[idx, 1], s=10, label=str(c), alpha=0.8)
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def _top_k_classes(y_true: np.ndarray, k: int) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    classes, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    k = max(1, min(int(k), len(classes)))
    return classes[order[:k]]


def _can_compute_predictions(result: dict) -> bool:
    if result is None:
        return False
    if "Y_te" not in result:
        return False
    if "y_pred" in result:
        return True
    if "sim" in result:
        return True
    return False


def _ensure_y_pred(result: dict) -> np.ndarray:
    if "y_pred" in result:
        return np.asarray(result["y_pred"]).astype(int)
    if "sim" in result:
        sim = np.asarray(result["sim"])
        return np.argmax(sim, axis=1).astype(int)
    raise KeyError("No y_pred or sim found to compute predictions.")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IP-SAE Demo — Run + t-SNE + Confusion Matrix")
        self.geometry("1040x700")

        self.data_root = ""
        self.last_result = None

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Dataset dropdown
        ds_row = ttk.Frame(frm)
        ds_row.pack(fill="x")

        ttk.Label(ds_row, text="Dataset:").pack(side="left")
        self.dataset_var = tk.StringVar(value="AWA2")
        self.dataset_combo = ttk.Combobox(
            ds_row, textvariable=self.dataset_var, values=DATASETS, state="readonly", width=8
        )
        self.dataset_combo.pack(side="left", padx=8)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda e: self.revalidate())

        # Path row
        self.status_var = tk.StringVar(value="Select your dataset folder or the data root.")
        self.path_var = tk.StringVar(value="(none)")

        ttk.Label(frm, text="Selected folder (data root OR dataset folder):").pack(anchor="w", pady=(10, 0))
        ttk.Label(frm, textvariable=self.path_var, foreground="gray").pack(anchor="w", pady=(0, 8))
        ttk.Button(frm, text="Browse…", command=self.browse).pack(anchor="w")

        ttk.Separator(frm).pack(fill="x", pady=10)

        # Mode radio
        self.mode_var = tk.StringVar(value="CZSL")
        ttk.Label(frm, text="Mode:").pack(anchor="w")
        ttk.Radiobutton(frm, text="Conventional ZSL (unseen only)", variable=self.mode_var, value="CZSL").pack(anchor="w")
        ttk.Radiobutton(frm, text="Generalized ZSL (seen + unseen)", variable=self.mode_var, value="GZSL").pack(anchor="w")

        ttk.Separator(frm).pack(fill="x", pady=10)

        # Buttons
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x")

        self.run_btn = ttk.Button(btn_row, text="Run", command=self.run_clicked, state="disabled")
        self.run_btn.pack(side="left")

        self.paired_sample_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_row, text="Paired Samples", variable=self.paired_sample_var).pack(side="left", padx=10)

        self.viz_visual_btn = ttk.Button(btn_row, text="t-SNE: Visual (X_te)", command=self.viz_visual, state="disabled")
        self.viz_visual_btn.pack(side="left", padx=6)

        self.viz_sem_btn = ttk.Button(btn_row, text="t-SNE: Semantic Prototypes", command=self.viz_semantic, state="disabled")
        self.viz_sem_btn.pack(side="left", padx=6)

        self.viz_proj_btn = ttk.Button(btn_row, text="t-SNE: Projected Semantic→Visual", command=self.viz_projected, state="disabled")
        self.viz_proj_btn.pack(side="left", padx=6)

        self.viz_ip_btn = ttk.Button(
            btn_row, text="t-SNE: IP-SAE Visual (after model)", command=self.viz_ip_visual, state="disabled"
        )
        self.viz_ip_btn.pack(side="left", padx=6)

        self.cm_btn = ttk.Button(btn_row, text="Confusion Matrix", command=self.show_confusion_matrix, state="disabled")
        self.cm_btn.pack(side="left", padx=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        # Confusion matrix options
        cm_opts = ttk.Frame(frm)
        cm_opts.pack(fill="x", pady=(0, 10))

        ttk.Label(cm_opts, text="Confusion matrix options:").pack(side="left")

        self.cm_topk_var = tk.IntVar(value=25)
        ttk.Label(cm_opts, text="Top-K classes:").pack(side="left", padx=(10, 4))
        ttk.Spinbox(cm_opts, from_=2, to=500, textvariable=self.cm_topk_var, width=6).pack(side="left")

        self.cm_norm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cm_opts, text="Normalize rows", variable=self.cm_norm_var).pack(side="left", padx=10)

        self.cm_subset_var = tk.StringVar(value="AUTO")
        ttk.Label(cm_opts, text="Subset:").pack(side="left", padx=(10, 4))
        self.cm_subset_combo = ttk.Combobox(
            cm_opts, textvariable=self.cm_subset_var, values=["AUTO", "ALL", "SEEN", "UNSEEN"], state="readonly", width=8
        )
        self.cm_subset_combo.pack(side="left")

        # NEW: Save checkbox
        self.cm_save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cm_opts, text="Save confusion matrix PNG", variable=self.cm_save_var).pack(side="left", padx=10)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text="Log:").pack(anchor="w")
        self.log = tk.Text(frm, height=18)
        self.log.pack(fill="both", expand=True)

        ttk.Label(frm, textvariable=self.status_var, foreground="blue").pack(anchor="w", pady=(8, 0))

    def append_log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def disable_viz(self):
        self.viz_visual_btn.config(state="disabled")
        self.viz_sem_btn.config(state="disabled")
        self.viz_proj_btn.config(state="disabled")
        self.viz_ip_btn.config(state="disabled")
        self.cm_btn.config(state="disabled")

    def enable_viz(self):
        self.viz_visual_btn.config(state="normal")
        self.viz_sem_btn.config(state="normal")
        self.viz_proj_btn.config(state="normal")

        self.viz_ip_btn.config(state="normal" if (self.last_result and "X_te_ip_visual" in self.last_result) else "disabled")
        self.cm_btn.config(state="normal" if _can_compute_predictions(self.last_result) else "disabled")

    def browse(self):
        folder = filedialog.askdirectory(title="Select data root OR dataset folder (e.g., .../data or .../data/CUB)")
        if not folder:
            return
        self.path_var.set(os.path.abspath(folder))
        ds = self.dataset_var.get()
        self.data_root = normalize_selection_to_data_root(folder, ds)
        self.revalidate()

    def revalidate(self):
        ds = self.dataset_var.get()
        if not self.data_root and self.path_var.get() != "(none)":
            self.data_root = normalize_selection_to_data_root(self.path_var.get(), ds)

        if not self.data_root:
            self.status_var.set("Select your dataset folder or the data root.")
            self.run_btn.config(state="disabled")
            self.disable_viz()
            self.last_result = None
            return

        ok, msg = check_dataset_files(self.data_root, ds)
        self.status_var.set(msg)
        self.run_btn.config(state=("normal" if ok else "disabled"))
        if not ok:
            self.disable_viz()
            self.last_result = None

    def run_clicked(self):
        ds = self.dataset_var.get()
        if not self.data_root:
            messagebox.showerror("Error", "Please browse and select a folder first.")
            return

        ok, msg = check_dataset_files(self.data_root, ds)
        if not ok:
            messagebox.showerror("Missing files", msg)
            return

        self.disable_viz()
        self.last_result = None

        ld.set_data_root(self.data_root)

        mode = self.mode_var.get()
        self.append_log(f"Data root: {self.data_root}")
        self.append_log(f"Dataset: {ds}")
        self.append_log(f"Mode: {mode}")

        def worker():
            try:
                if mode == "CZSL":
                    out = run_conventional(dataset=ds, paired_sample=self.paired_sample_var.get(), print_ip_space=False) \
                        if "print_ip_space" in run_conventional.__code__.co_varnames else run_conventional(dataset=ds, paired_sample=self.paired_sample_var.get())
                    self.last_result = out
                    self.append_log(f"CZSL accuracy: {out.get('accuracy', float('nan')):.3f}%")
                    self.append_log("CZSL note: confusion matrix is UNSEEN-only by definition.")
                else:
                    out = run_gzsl(dataset=ds, paired_sample=self.paired_sample_var.get(), print_ip_space=False) \
                        if "print_ip_space" in run_gzsl.__code__.co_varnames else run_gzsl(dataset=ds, paired_sample=self.paired_sample_var.get())
                    self.last_result = out
                    if all(k in out for k in ("harmonic", "seen", "unseen")):
                        self.append_log(
                            "GZSL harmonic: {:.3f}% | seen: {:.3f}% | unseen: {:.3f}%".format(
                                out["harmonic"], out["seen"], out["unseen"]
                            )
                        )
                    else:
                        self.append_log("GZSL finished (metrics missing from output dict).")

                # Helpful debug: show keys
                self.append_log("Result keys: " + ", ".join(sorted(self.last_result.keys())))

                self.enable_viz()
                self.status_var.set("Run complete. You can now visualize plots / confusion matrix.")
            except Exception as e:
                self.last_result = None
                self.status_var.set("Error.")
                self.append_log("ERROR: " + str(e))
                messagebox.showerror("Run failed", str(e))

        threading.Thread(target=worker, daemon=True).start()

    # ---- Visualizations ----

    def viz_visual(self):
        if not self.last_result:
            return
        X_te = self.last_result["X_te"]
        Y_te = self.last_result["Y_te"]
        self.append_log("Computing t-SNE for visual features (X_te)...")
        pts2d = tsne_2d(X_te)
        plot_tsne(pts2d, Y_te, title=f"{self.last_result.get('setting','')} — t-SNE of Visual Features (X_te)")

    def viz_ip_visual(self):
        if not self.last_result or "X_te_ip_visual" not in self.last_result:
            messagebox.showinfo("Not available", "This run did not return IP-SAE reconstructed visual features.")
            return
        X_ip = self.last_result["X_te_ip_visual"]
        Y_te = self.last_result["Y_te"]
        self.append_log("Computing t-SNE for IP-SAE reconstructed visual space (X_te_ip_visual)...")
        pts2d = tsne_2d(X_ip)
        plot_tsne(pts2d, Y_te, title=f"{self.last_result.get('setting','')} — t-SNE of IP-SAE Visual Space (after model)")

    def viz_semantic(self):
        if not self.last_result:
            return
        setting = self.last_result.get("setting", "")

        if setting == "CZSL":
            S = self.last_result["S_proto"]
            labels = np.arange(S.shape[0])
            title = "CZSL — t-SNE of Semantic Prototypes (Unseen Classes)"
        else:
            if "S_all" not in self.last_result:
                messagebox.showerror("Missing", "GZSL output missing S_all.")
                return
            S = self.last_result["S_all"]
            labels = np.arange(S.shape[0])
            title = "GZSL — t-SNE of Semantic Prototypes (All Classes)"

        self.append_log("Computing t-SNE for semantic prototypes...")
        pts2d = tsne_2d(S)
        plot_tsne(pts2d, labels, title=title, max_classes=40)

    def viz_projected(self):
        if not self.last_result:
            return
        setting = self.last_result.get("setting", "")

        if setting == "CZSL":
            P = self.last_result["proj_proto"]
            labels = np.arange(P.shape[0])
            title = "CZSL — t-SNE of Projected Semantic→Visual Prototypes"
        else:
            if "proj_all" not in self.last_result:
                messagebox.showerror("Missing", "GZSL output missing proj_all.")
                return
            P = self.last_result["proj_all"]
            labels = np.arange(P.shape[0])
            title = "GZSL — t-SNE of Projected Semantic→Visual Prototypes"

        self.append_log("Computing t-SNE for projected prototypes (semantic→visual)...")
        pts2d = tsne_2d(P)
        plot_tsne(pts2d, labels, title=title, max_classes=40)

    # ---- Confusion Matrix ----

    def show_confusion_matrix(self):
        if not self.last_result or "Y_te" not in self.last_result:
            messagebox.showinfo("Not available", "No run results available.")
            return

        try:
            y_pred = _ensure_y_pred(self.last_result)
        except Exception as e:
            messagebox.showinfo("Not available", f"Cannot compute predictions for confusion matrix.\n{e}")
            return

        y_true = np.asarray(self.last_result["Y_te"]).astype(int)

        setting = self.last_result.get("setting", "RUN")
        dataset = self.last_result.get("dataset", self.dataset_var.get())

        subset_choice = self.cm_subset_var.get().upper()

        # CZSL always unseen-only
        if setting == "CZSL":
            subset = "UNSEEN"
        else:
            subset = "ALL" if subset_choice == "AUTO" else subset_choice
            if subset not in ("ALL", "SEEN", "UNSEEN"):
                subset = "ALL"

        seen_ids = np.asarray(self.last_result.get("seen_cls_ids", []), dtype=int)
        unseen_ids = np.asarray(self.last_result.get("unseen_cls_ids", []), dtype=int)

        if setting == "GZSL" and (seen_ids.size == 0 or unseen_ids.size == 0) and subset in ("SEEN", "UNSEEN"):
            messagebox.showinfo(
                "Subset not available",
                "This GZSL run did not provide seen/unseen class id sets.\nShowing ALL classes instead.",
            )
            subset = "ALL"

        if subset == "SEEN":
            mask = np.isin(y_true, seen_ids)
            y_true_f, y_pred_f = y_true[mask], y_pred[mask]
            subset_title = "SEEN-only"
        elif subset == "UNSEEN":
            if setting == "GZSL":
                mask = np.isin(y_true, unseen_ids)
                y_true_f, y_pred_f = y_true[mask], y_pred[mask]
            else:
                y_true_f, y_pred_f = y_true, y_pred
            subset_title = "UNSEEN-only"
        else:
            y_true_f, y_pred_f = y_true, y_pred
            subset_title = "ALL"

        if y_true_f.size == 0:
            messagebox.showinfo("No data", f"No samples found for subset: {subset_title}")
            return

        requested_topk = int(self.cm_topk_var.get())
        labels = _top_k_classes(y_true_f, requested_topk)
        actual_k = len(labels)
        if requested_topk > actual_k:
            self.append_log(f"Confusion matrix: requested Top-K={requested_topk}, clamped to K={actual_k} (available classes).")

        topk_mask = np.isin(y_true_f, labels)
        y_true_k = y_true_f[topk_mask]
        y_pred_k = y_pred_f[topk_mask]

        cm = confusion_matrix(y_true_k, y_pred_k, labels=labels)

        normalize_rows = bool(self.cm_norm_var.get())
        values_format = "d"
        if normalize_rows:
            cm = cm.astype(float)
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
            values_format = ".2f"

        fig_w = max(8, min(16, 0.35 * actual_k + 6))
        fig_h = max(6, min(14, 0.35 * actual_k + 4))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format=values_format, xticks_rotation=90, colorbar=True)
        ax.set_title(
            f"{setting} Confusion Matrix ({dataset}) — {subset_title} — Top {actual_k}"
            + (" (row-normalized)" if normalize_rows else "")
        )
        plt.tight_layout()
        plt.show()

        # NEW: save only if checkbox is checked
        if self.cm_save_var.get():
            out_name = f"confusion_{setting}_{dataset}_{subset_title.lower()}_top{actual_k}" + ("_norm" if normalize_rows else "") + ".png"
            fig.savefig(out_name, dpi=200)
            self.append_log(f"Saved confusion matrix: {out_name}")
        else:
            self.append_log("Confusion matrix not saved (checkbox off).")


if __name__ == "__main__":
    app = App()
    app.mainloop()
