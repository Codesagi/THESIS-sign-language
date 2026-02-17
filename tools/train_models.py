#!/usr/bin/env python3
# ============================================================
# tools/train_models.py  —  Training pipeline
#
# Responsibilities (ONLY this file owns these):
#   - Load raw data from DB          (with disk cache)
#   - Augment with mirror samples
#   - Stratified train/val/test split
#   - RF hyperparameter search
#   - Train CNN + RF
#   - Evaluate with confusion matrix + per-class metrics
#   - Save artefacts
#
# core/recognition.py owns normalization.
# core/models.py      owns architecture + save/load.
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.db import get_client
from core.models import (
    CNNModel, RandomForestModel,
    build_rf, save_cnn, save_rf,
)
from core.recognition import normalize_landmarks, mirror_landmarks


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING  (with disk cache)
# ──────────────────────────────────────────────────────────────

CACHE_DIR = Path(".cache")


def _cache_path(language: str) -> Path:
    return CACHE_DIR / f"dataset_{language.lower()}.pkl"


def load_dataset(language: str, force_reload: bool = False):
    """
    Load all landmark samples for *language* from Supabase.

    Returns
    -------
    X : (N, 21, 3)  float32  — normalized, NOT yet augmented
    y : (N,)        str      — label strings

    Disk cache
    ----------
    After the first fetch the raw rows are pickled to .cache/.
    Subsequent runs skip the network round-trip unless
    --force-reload is passed or the DB row-count changes.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = _cache_path(language)

    client = get_client()

    # Quick count to detect stale cache
    count_resp = (
        client.table("landmark_samples")
        .select("id", count="exact")
        .eq("language", language)
        .execute()
    )
    db_count = count_resp.count or 0

    if not force_reload and cache.exists():
        meta = pickle.loads(cache.read_bytes())
        if meta.get("db_count") == db_count:
            print(f"[data] Cache hit ({db_count} rows) ← {cache}")
            return meta["X"], meta["y"]
        print(f"[data] Cache stale (db={db_count}, cache={meta.get('db_count')}), reloading…")

    print(f"[data] Fetching {db_count} rows from DB for {language}…")
    t0 = time.time()

    all_rows, page, PAGE = [], 0, 1000
    while True:
        resp = (
            client.table("landmark_samples")
            .select("label, landmarks_json")
            .eq("language", language)
            .range(page, page + PAGE - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  …{len(all_rows)}/{db_count}", end="\r")
        if len(rows) < PAGE:
            break
        page += PAGE

    print(f"\n[data] Fetched {len(all_rows)} rows in {time.time()-t0:.1f}s")

    if not all_rows:
        raise ValueError(f"No data for language='{language}'")

    # Normalise using the canonical function from core/recognition
    X_list, y_list = [], []
    for row in all_rows:
        try:
            arr = normalize_landmarks(row["landmarks_json"])
            X_list.append(arr)
            y_list.append(row["label"])
        except Exception as e:
            print(f"  [skip] bad row: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # Persist cache
    cache.write_bytes(pickle.dumps({"X": X, "y": y, "db_count": db_count}))
    print(f"[data] Cached to {cache}")

    return X, y


# ──────────────────────────────────────────────────────────────
# 2. AUGMENTATION
# ──────────────────────────────────────────────────────────────

def augment_with_mirrors(X: np.ndarray, y: np.ndarray):
    """
    Double the dataset by appending a horizontally mirrored copy
    of every sample.

    WHY this fixes handedness bias
    --------------------------------
    If the dataset was collected from predominantly right-handed
    signers, the RF/CNN never see left-handed variants during
    training, so live left-hand input falls into a distribution
    gap → lower confidence or wrong predictions.  Mirroring costs
    nothing (pure numpy), doubles sample count, and makes both
    models hand-agnostic.
    """
    X_mir = np.array([mirror_landmarks(x) for x in X], dtype=np.float32)
    X_aug = np.concatenate([X, X_mir], axis=0)
    y_aug = np.concatenate([y, y],     axis=0)
    print(f"[aug]  {len(X)} → {len(X_aug)} samples after mirror augmentation")
    return X_aug, y_aug


# ──────────────────────────────────────────────────────────────
# 3. SPLITTING
# ──────────────────────────────────────────────────────────────

def split(X, y, train=0.70, val=0.15):
    """Stratified 70 / 15 / 15 split."""
    from sklearn.model_selection import train_test_split

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, train_size=train, stratify=y, random_state=42
    )
    val_frac = val / (1 - train)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, stratify=y_tmp, random_state=42
    )

    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"[split] train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")
    print(f"[split] classes={len(unique)}  min/max per class: "
          f"{counts.min()}/{counts.max()}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ──────────────────────────────────────────────────────────────
# 4. RF HYPERPARAMETER TUNING
# ──────────────────────────────────────────────────────────────

def tune_rf(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 20):
    """
    RandomizedSearchCV over the RF parameter space.
    Prints best params — paste them into models.RF_BEST_PARAMS.
    Returns the best estimator (already fit on full train set).
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    enc  = LabelEncoder()
    y_e  = enc.fit_transform(y_train)
    X_fl = X_train.reshape(len(X_train), -1)

    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 20, 30, 40],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf":  [1, 2, 3],
        "max_features":      ["sqrt", "log2"],
        "class_weight":      ["balanced", None],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_iter, cv=3,
        scoring="f1_weighted",
        n_jobs=-1, verbose=1,
        random_state=42,
    )
    print(f"[tune] Running {n_iter} RF configurations (3-fold CV)…")
    search.fit(X_fl, y_e)
    print(f"[tune] Best CV F1: {search.best_score_:.4f}")
    print(f"[tune] Best params:\n{search.best_params_}")
    print("\n[tune] → Paste these into core/models.RF_BEST_PARAMS to persist.\n")
    return search.best_estimator_, enc


# ──────────────────────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────────────────────

def train_cnn(X_tr, y_tr, X_va, y_va, language: str, epochs: int = 60):
    print(f"\n[CNN] Training…  (epochs={epochs})")
    mdl = CNNModel()
    mdl.train(X_tr, y_tr, X_va, y_va, epochs=epochs)

    out = Path("models")
    out.mkdir(exist_ok=True)
    path = str(out / f"cnn_{language.lower()}.h5")
    mdl.save(path)
    return mdl


def train_rf(X_tr, y_tr, language: str, tuned_estimator=None, tuned_enc=None):
    from sklearn.preprocessing import LabelEncoder

    print(f"\n[RF]  Training…")
    mdl = RandomForestModel()

    if tuned_estimator is not None:
        # Use the already-fit estimator from tuning
        mdl.model         = tuned_estimator
        mdl.label_encoder = tuned_enc
    else:
        mdl.train(X_tr, y_tr)

    out = Path("models")
    out.mkdir(exist_ok=True)
    path = str(out / f"rf_{language.lower()}.pkl")
    mdl.save(path)
    return mdl


# ──────────────────────────────────────────────────────────────
# 6. EVALUATION  (confusion matrix + per-class report)
# ──────────────────────────────────────────────────────────────

def evaluate(cnn_mdl, rf_mdl, X_te, y_te, language: str):
    """
    Evaluate whichever models are available.
    Works with CNN-only, RF-only, or both.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report,
        ConfusionMatrixDisplay, confusion_matrix,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use whichever encoder is available
    enc     = (cnn_mdl.label_encoder if cnn_mdl else None) or rf_mdl.label_encoder
    classes = enc.classes_
    y_enc   = enc.transform(y_te)

    results   = {}
    best_preds = None   # used for confusion matrix

    # ── CNN ───────────────────────────────────────────────────
    if cnn_mdl is not None:
        _, cnn_probs = cnn_mdl.predict(X_te)
        cnn_preds    = np.argmax(cnn_probs, axis=1)
    else:
        cnn_probs = None
        cnn_preds = None

    # ── RF ────────────────────────────────────────────────────
    if rf_mdl is not None:
        _, rf_probs = rf_mdl.predict(X_te)
        rf_preds    = np.argmax(rf_probs, axis=1)
    else:
        rf_probs = None
        rf_preds = None

    # ── Ensemble (only when both available) ───────────────────
    if cnn_probs is not None and rf_probs is not None:
        ens_probs = cnn_probs * 0.6 + rf_probs * 0.4
        ens_preds = np.argmax(ens_probs, axis=1)
    else:
        ens_probs = None
        ens_preds = None

    # ── Print metrics per available model ─────────────────────
    candidates = []
    if cnn_preds  is not None: candidates.append(("CNN",      cnn_preds))
    if rf_preds   is not None: candidates.append(("RF",       rf_preds))
    if ens_preds  is not None: candidates.append(("Ensemble", ens_preds))

    for name, preds in candidates:
        acc = accuracy_score(y_enc, preds)
        rep = classification_report(y_enc, preds,
                                    target_names=classes,
                                    zero_division=0,
                                    output_dict=True)
        wa  = rep["weighted avg"]
        results[name] = dict(accuracy=acc,
                             precision=wa["precision"],
                             recall=wa["recall"],
                             f1=wa["f1-score"])
        sep = "─" * 60
        print(f"\n{sep}\n  {name}\n{sep}")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  Precision: {wa['precision']*100:.2f}%")
        print(f"  Recall   : {wa['recall']*100:.2f}%")
        print(f"  F1       : {wa['f1-score']*100:.2f}%")
        print()
        print(classification_report(y_enc, preds,
                                    target_names=classes, zero_division=0))
        best_preds = preds   # keep last (highest priority: ensemble > rf)

    # ── Confusion matrix on best available predictions ────────
    if best_preds is not None:
        cm   = confusion_matrix(y_enc, best_preds)
        sz   = max(8, len(classes))
        fig, ax = plt.subplots(figsize=(sz, sz))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        title_suffix = "Ensemble" if ens_preds is not None else ("CNN" if cnn_preds is not None else "RF")
        ax.set_title(f"{title_suffix} Confusion Matrix — {language}")
        plt.tight_layout()
        cm_path = Path("models") / f"confusion_matrix_{language.lower()}.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        print(f"\n[eval] Confusion matrix saved → {cm_path}")

        # Lowest F1 classes
        worst = sorted(
            [(cls, rep[cls]["f1-score"])
             for cls in classes if cls in rep],
            key=lambda x: x[1]
        )[:5]
        print("\n[eval] Lowest F1 signs (collect more data for these):")
        for cls, f1 in worst:
            print(f"  {cls:>4}  F1={f1*100:.1f}%")

    return results


# ──────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Production training pipeline — CNN + Random Forest"
    )
    ap.add_argument("--language",      required=True, choices=["ASL", "FSL"])
    ap.add_argument("--epochs",        type=int, default=60)
    ap.add_argument("--cnn-only",      action="store_true")
    ap.add_argument("--rf-only",       action="store_true")
    ap.add_argument("--tune",          action="store_true",
                    help="Run RF hyperparameter search before training")
    ap.add_argument("--tune-iter",     type=int, default=20,
                    help="Number of RF param combinations to try (default 20)")
    ap.add_argument("--no-eval",       action="store_true")
    ap.add_argument("--force-reload",  action="store_true",
                    help="Ignore disk cache and reload from DB")
    args = ap.parse_args()

    print("=" * 64)
    print(f"  TRAINING  {args.language}  —  CNN + Random Forest")
    print("=" * 64)

    # ── 1. Load ───────────────────────────────────────────────
    X, y = load_dataset(args.language, force_reload=args.force_reload)

    # ── 2. Augment ────────────────────────────────────────────
    X, y = augment_with_mirrors(X, y)

    # ── 3. Split ──────────────────────────────────────────────
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)

    # ── 4. (optional) RF tuning ───────────────────────────────
    tuned_est = tuned_enc = None
    if args.tune and not args.cnn_only:
        tuned_est, tuned_enc = tune_rf(X_tr, y_tr, n_iter=args.tune_iter)

    # ── 5. Train (auto-skip CNN if TensorFlow not installed) ──
    from core.models import HAS_TF, HAS_SK
    cnn_mdl = rf_mdl = None

    want_cnn = not args.rf_only
    want_rf  = not args.cnn_only

    if want_cnn and not HAS_TF:
        print("\n[WARN] TensorFlow not installed — skipping CNN.")
        print("[WARN] Install with:  pip install tensorflow")
        print("[WARN] RF will still be trained and usable alone.\n")
        want_cnn = False

    if want_rf and not HAS_SK:
        print("\n[WARN] scikit-learn not installed — skipping RF.")
        want_rf = False

    if not want_cnn and not want_rf:
        print("[ERROR] Nothing to train. Install tensorflow and/or scikit-learn.")
        sys.exit(1)

    if want_cnn:
        cnn_mdl = train_cnn(X_tr, y_tr, X_va, y_va, args.language, args.epochs)

    if want_rf:
        rf_mdl = train_rf(X_tr, y_tr, args.language, tuned_est, tuned_enc)

    # ── 6. Evaluate (works with RF-only or CNN-only) ──────────
    if not args.no_eval and (cnn_mdl or rf_mdl):
        evaluate(cnn_mdl, rf_mdl, X_te, y_te, args.language)

    print("\n" + "=" * 64)
    print("  DONE")
    print("=" * 64)
    if cnn_mdl: print(f"  models/cnn_{args.language.lower()}.h5")
    if rf_mdl:  print(f"  models/rf_{args.language.lower()}.pkl")
    print(f"  models/confusion_matrix_{args.language.lower()}.png")
    if not HAS_TF:
        print()
        print("  NOTE: CNN was skipped (TensorFlow not installed).")
        print("  The app will use Random Forest only — still works great!")
        print("  For best accuracy install TF and retrain:")
        print("    pip install tensorflow")
    print()


if __name__ == "__main__":
    main()