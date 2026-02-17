# ============================================================
# core/models.py  —  Model DEFINITIONS only
#
# This file owns:
#   - CNN architecture
#   - RF hyperparameter config
#   - save / load helpers
#
# This file does NOT own:
#   - Data loading        → tools/train_models.py
#   - Normalization       → core/recognition.py
#   - Live inference      → core/recognition.InferenceEngine
#   - Training loop       → tools/train_models.py
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError as _tf_err:
    HAS_TF = False
    print(f"[models] TensorFlow import failed: {_tf_err}")
except Exception as _tf_err:
    # Catches DLL load failures on Windows (VC++ runtime missing)
    HAS_TF = False
    print(f"[models] TensorFlow load error (likely missing VC++ runtime): {_tf_err}")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[models] scikit-learn not found — RF disabled.")


# ──────────────────────────────────────────────────────────────
# CNN
# ──────────────────────────────────────────────────────────────

def build_cnn(num_classes: int, input_shape: Tuple[int, int] = (21, 3)):
    """
    Conv1D network for (21, 3) landmark sequences.

    Architecture notes
    ------------------
    • Three Conv1D blocks capture local joint relationships
      (adjacent landmarks in the MediaPipe chain are topologically
      connected, so a kernel_size=3 sees one full bone segment).
    • GlobalAveragePooling instead of Flatten keeps the parameter
      count low and reduces overfitting on small datasets.
    • Batch normalisation before every activation stabilises
      training when mixing augmented (mirrored) samples.
    """
    if not HAS_TF:
        raise ImportError("pip install tensorflow")

    inp = keras.Input(shape=input_shape, name="landmarks")

    x = keras.layers.Conv1D(64,  3, padding="same", name="conv1")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(128, 3, padding="same", name="conv2")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv1D(256, 3, padding="same", name="conv3")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    out = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="HandCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_cnn(model, encoder: "LabelEncoder", path: str):
    model.save(path)
    joblib.dump(encoder, path.replace(".h5", "_labels.pkl"))
    print(f"[CNN] saved → {path}")


def load_cnn(path: str):
    if not HAS_TF:
        raise ImportError("pip install tensorflow")
    model   = tf.keras.models.load_model(path)
    enc_p   = path.replace(".h5", "_labels.pkl")
    encoder = joblib.load(enc_p) if os.path.exists(enc_p) else None
    return model, encoder


# ──────────────────────────────────────────────────────────────
# RANDOM FOREST
# ──────────────────────────────────────────────────────────────

# Best hyperparameters found by RandomizedSearchCV on 8 k samples.
# Re-run tools/train_models.py --tune to update these.
RF_BEST_PARAMS = {
    "n_estimators":    300,
    "max_depth":       None,       # grow until pure (better than fixed 25)
    "min_samples_split": 3,
    "min_samples_leaf":  1,
    "max_features":    "sqrt",
    "class_weight":    "balanced", # fixes class-imbalance bias
    "random_state":    42,
    "n_jobs":          -1,
}


def build_rf(**overrides) -> "RandomForestClassifier":
    if not HAS_SK:
        raise ImportError("pip install scikit-learn")
    params = {**RF_BEST_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def save_rf(model, encoder: "LabelEncoder", path: str):
    joblib.dump({"model": model, "label_encoder": encoder}, path)
    print(f"[RF]  saved → {path}")


def load_rf(path: str):
    data = joblib.load(path)
    return data["model"], data["label_encoder"]


# ──────────────────────────────────────────────────────────────
# LEGACY WRAPPER CLASSES (kept for backward compatibility)
# New code should use build_*/save_*/load_* functions directly.
# ──────────────────────────────────────────────────────────────

class CNNModel:
    """Thin wrapper — delegates to module-level functions."""

    def __init__(self, model_path: Optional[str] = None):
        self.model         = None
        self.label_encoder = None
        if model_path and Path(model_path).exists():
            self.model, self.label_encoder = load_cnn(model_path)

    def train(self, X_train, y_train, X_val, y_val,
              epochs: int = 60, batch_size: int = 32):
        if not HAS_SK:
            raise ImportError("pip install scikit-learn")
        self.label_encoder = LabelEncoder()
        y_tr = self.label_encoder.fit_transform(y_train)
        y_va = self.label_encoder.transform(y_val)

        self.model = build_cnn(len(self.label_encoder.classes_))

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10,
                restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6),
        ]
        return self.model.fit(
            X_train, y_tr,
            validation_data=(X_val, y_va),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1,
        )

    def predict(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1), probs

    def save(self, path: str):
        save_cnn(self.model, self.label_encoder, path)

    def load(self, path: str):
        self.model, self.label_encoder = load_cnn(path)


class RandomForestModel:
    """Thin wrapper — delegates to module-level functions."""

    def __init__(self, model_path: Optional[str] = None):
        self.model         = None
        self.label_encoder = None
        if model_path and Path(model_path).exists():
            self.model, self.label_encoder = load_rf(model_path)

    def train(self, X_train, y_train, **rf_overrides):
        if not HAS_SK:
            raise ImportError("pip install scikit-learn")
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_train)
        X_flat = X_train.reshape(len(X_train), -1) if X_train.ndim == 3 else X_train
        self.model = build_rf(**rf_overrides)
        self.model.fit(X_flat, y_enc)
        return self.model

    def predict(self, X):
        X_flat = X.reshape(len(X), -1) if X.ndim == 3 else X
        return self.model.predict(X_flat), self.model.predict_proba(X_flat)

    def save(self, path: str):
        save_rf(self.model, self.label_encoder, path)

    def load(self, path: str):
        self.model, self.label_encoder = load_rf(path)


# ── EnsembleModel kept for backward compat with sign_to_word.py ──

class EnsembleModel:
    """Legacy shim.  New code should use InferenceEngine."""

    def __init__(self, cnn_path=None, rf_path=None,
                 cnn_weight=0.6, rf_weight=0.4):
        self.cnn_weight = cnn_weight
        self.rf_weight  = rf_weight
        self._cnn   = CNNModel(cnn_path)    if cnn_path else None
        self._rf    = RandomForestModel(rf_path) if rf_path  else None

        enc = None
        if self._cnn and self._cnn.label_encoder is not None:
            enc = self._cnn.label_encoder
        elif self._rf and self._rf.label_encoder is not None:
            enc = self._rf.label_encoder
        self._classes = enc.classes_ if enc else None

    def predict_top_k(self, X, k: int = 5):
        if self._classes is None:
            return []
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]          # (1,21,3)

        w = np.zeros(len(self._classes), dtype=np.float64)
        if self._cnn and self._cnn.model:
            _, p = self._cnn.predict(arr)
            w += p[0] * self.cnn_weight
        if self._rf and self._rf.model:
            _, p = self._rf.predict(arr)
            w += p[0] * self.rf_weight

        idx = np.argsort(w)[-k:][::-1]
        return [(str(self._classes[i]), round(w[i] * 100, 2)) for i in idx]