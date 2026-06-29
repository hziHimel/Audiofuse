"""
AudioFuse — TF/Keras Training Script (local adaptation of original Colab notebook)
Trains the dual-branch ViT + 1D-CNN model on PhysioNet 2016 and evaluates on val set.

Usage:
    python train_keras.py --train_csv data/train.csv --val_csv data/val.csv \
                          --seed 42 --output_dir outputs/keras/
"""

import os
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
)
import librosa
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────
class Config:
    IMG_SIZE = 224
    IN_CHANS = 1
    PATCH_SIZE = 16
    PROJECTION_DIM = 192
    NUM_HEADS = 8
    TRANSFORMER_LAYERS = 6
    MLP_UNITS = [192 * 2, 192]
    DROPOUT_RATE = 0.2

    WAVEFORM_LENGTH_SECONDS = 5
    SAMPLE_RATE = 22050
    WAVEFORM_MAX_LEN = WAVEFORM_LENGTH_SECONDS * SAMPLE_RATE  # 110 250

    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15

CONFIG = Config()
NUM_CLASSES = 2


# ── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Data pipeline ───────────────────────────────────────────────────────────
def load_spec_and_waveform(npy_path, wav_path, label):
    def _load_npy(path):
        full = np.load(path.numpy().decode())
        return full[:, :, 0:1].astype(np.float32)   # channel 0 = spectrogram

    def _load_wav(path):
        wav, _ = librosa.load(path.numpy().decode(), sr=CONFIG.SAMPLE_RATE, mono=True)
        if len(wav) > CONFIG.WAVEFORM_MAX_LEN:
            wav = wav[:CONFIG.WAVEFORM_MAX_LEN]
        else:
            wav = np.pad(wav, (0, CONFIG.WAVEFORM_MAX_LEN - len(wav)), "constant")
        return wav.astype(np.float32)

    [spec] = tf.py_function(_load_npy, [npy_path], [tf.float32])
    spec.set_shape((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, CONFIG.IN_CHANS))

    [wave] = tf.py_function(_load_wav, [wav_path], [tf.float32])
    wave.set_shape([CONFIG.WAVEFORM_MAX_LEN])

    label.set_shape([])
    return {"spec_input": spec, "wave_input": wave}, label


def make_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["npy_filepath"].values, df["filepath"].values, df["label"].values)
    )
    ds = ds.map(load_spec_and_waveform, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(CONFIG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model ───────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class PatchEmbed(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.proj = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )

    def call(self, images):
        x = self.proj(images)
        b, h, w, c = x.shape
        return tf.reshape(x, (-1, h * w, c))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size, "projection_dim": self.projection_dim})
        return cfg


def build_audiofuse():
    spec_input = layers.Input(
        shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, CONFIG.IN_CHANS), name="spec_input"
    )
    wave_input = layers.Input(shape=(CONFIG.WAVEFORM_MAX_LEN,), name="wave_input")

    # ── Spectrogram ViT branch ──────────────────────────────────────────────
    num_patches = (CONFIG.IMG_SIZE // CONFIG.PATCH_SIZE) ** 2   # 196

    x = PatchEmbed(CONFIG.PATCH_SIZE, CONFIG.PROJECTION_DIM, name="spec_patch_embed")(spec_input)
    pos_emb = layers.Embedding(num_patches, CONFIG.PROJECTION_DIM, name="spec_pos_emb")
    positions = tf.range(num_patches)
    x = x + pos_emb(positions)

    for i in range(CONFIG.TRANSFORMER_LAYERS):
        # Pre-norm + MHA
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"spec_ln1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=CONFIG.NUM_HEADS,
            key_dim=CONFIG.PROJECTION_DIM // CONFIG.NUM_HEADS,
            dropout=CONFIG.DROPOUT_RATE,
            name=f"spec_mha_{i}",
        )(x1, x1)
        x = layers.Add(name=f"spec_add1_{i}")([x, attn])

        # Pre-norm + MLP
        x2 = layers.LayerNormalization(epsilon=1e-6, name=f"spec_ln2_{i}")(x)
        for j, units in enumerate(CONFIG.MLP_UNITS):
            x2 = layers.Dense(units, activation="gelu", name=f"spec_mlp_{i}_{j}")(x2)
            x2 = layers.Dropout(CONFIG.DROPOUT_RATE, name=f"spec_drop_{i}_{j}")(x2)
        x = layers.Add(name=f"spec_add2_{i}")([x, x2])

    x = layers.LayerNormalization(epsilon=1e-6, name="spec_final_ln")(x)
    spec_feat = layers.GlobalAveragePooling1D(name="spec_gap")(x)   # (B, 192)

    # ── Waveform 1D-CNN branch ──────────────────────────────────────────────
    w = layers.Reshape((CONFIG.WAVEFORM_MAX_LEN, 1), name="wave_reshape")(wave_input)

    for i, filters in enumerate([64, 128, 256]):
        w = layers.Conv1D(filters, kernel_size=16, strides=4, padding="same",
                          name=f"wave_conv_{i}")(w)
        w = layers.BatchNormalization(name=f"wave_bn_{i}")(w)
        w = layers.Activation("relu", name=f"wave_relu_{i}")(w)
        w = layers.MaxPooling1D(pool_size=4, name=f"wave_pool_{i}")(w)

    wave_feat = layers.GlobalAveragePooling1D(name="wave_gap")(w)   # (B, 256)
    wave_feat = layers.Dense(64, activation="relu", name="wave_proj")(wave_feat)  # (B, 64)

    # ── Fusion head ─────────────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([spec_feat, wave_feat])  # (B, 256)
    fused = layers.Dense(192, activation="relu", name="head_dense")(fused)
    fused = layers.Dropout(0.5, name="head_drop")(fused)
    out = layers.Dense(1, activation="sigmoid", name="output")(fused)

    model = keras.Model(inputs=[spec_input, wave_input], outputs=out, name="AudioFuse")
    return model


# ── Training & evaluation ───────────────────────────────────────────────────
def train(args, seed: int):
    set_seed(seed)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Class weights
    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(train_df["label"]), y=train_df["label"].values
    )
    cw_dict = dict(enumerate(cw))
    print(f"Class weights: {cw_dict}")

    train_ds = make_dataset(train_df, shuffle=True)
    val_ds = make_dataset(val_df, shuffle=False)

    model = build_audiofuse()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=CONFIG.LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"best_seed{seed}.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=CONFIG.PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=CONFIG.EPOCHS,
        validation_data=val_ds,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\nRunning final evaluation on validation set...")
    val_ds_eval = make_dataset(val_df, shuffle=False)
    y_prob = model.predict(val_ds_eval).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = val_df["label"].values

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Seed {seed} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"{'='*50}")
    return {"seed": seed, "accuracy": acc, "f1": f1, "roc_auc": auc, "mcc": mcc}


# ── Entry point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--output_dir", default="outputs/keras/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"# Training with seed={seed}")
        print(f"{'#'*60}")
        results = train(args, seed)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    print("\n\nFinal Summary (mean ± std over seeds):")
    for col in ["accuracy", "f1", "roc_auc", "mcc"]:
        print(f"  {col:12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/results.csv")


if __name__ == "__main__":
    main()
