#!/usr/bin/env python3
"""
train_autoencoder.py
Train an audio autoencoder on a folder of wav/flac files.

Usage:
    python train_autoencoder.py samples/ folk
    python train_autoencoder.py /data/metal metal --epochs 100 --k 32 --frame 256
    python train_autoencoder.py samples/ pop --epochs 50 --batch 512
"""

import argparse
import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

def build_autoencoder(frame_size, bottleneck):
    from tensorflow import keras
    from tensorflow.keras import layers
    inner = frame_size // 8
    inp = keras.Input(shape=(frame_size, 1))
    x   = layers.Conv1D(16, 8, strides=2, padding="same", activation="relu")(inp)
    x   = layers.Conv1D(32, 8, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv1D(64, 8, strides=2, padding="same", activation="relu")(x)
    x   = layers.Flatten()(x)
    z   = layers.Dense(bottleneck, name="bottleneck")(x)
    x   = layers.Dense(64 * inner, activation="relu")(z)
    x   = layers.Reshape((inner, 64))(x)
    x   = layers.Conv1DTranspose(32, 8, strides=2, padding="same", activation="relu")(x)
    x   = layers.Conv1DTranspose(16, 8, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv1DTranspose( 1, 8, strides=2, padding="same", activation="tanh")(x)
    return keras.Model(inp, out, name="autoencoder")

def collect_frames(data_path, frame_size, hop, extensions=("*.wav","*.flac","*.mp3")):
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(data_path, "**", ext), recursive=True)
        files += glob.glob(os.path.join(data_path, ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No audio files found in {data_path}")

    print(f"Found {len(files)} audio files")
    all_frames = []
    for i, fpath in enumerate(files):
        try:
            s, _ = librosa.load(fpath, sr=44100, mono=True)
            s    = s / (np.max(np.abs(s)) + 1e-12)
            for start in range(0, len(s) - frame_size, hop):
                all_frames.append(s[start:start + frame_size])
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{len(files)}] {len(all_frames)} frames so far")
        except Exception as e:
            print(f"  Skipping {fpath}: {e}")

    return np.array(all_frames, dtype=np.float32)[..., np.newaxis], files

def main():
    parser = argparse.ArgumentParser(description="Train audio autoencoder")
    parser.add_argument("data_path",   help="Folder with audio files (searched recursively)")
    parser.add_argument("model_name",  help="Output model name, saved to models/<name>.keras")
    parser.add_argument("--frame",     type=int,   default=256,   help="Frame size (default 256)")
    parser.add_argument("--ratio",     type=float, default=0.25,  help="Bottleneck ratio k/frame (default 0.25)")
    parser.add_argument("--k",         type=int,   default=None,  help="Bottleneck size (overrides --ratio)")
    parser.add_argument("--epochs",    type=int,   default=50,    help="Training epochs (default 50)")
    parser.add_argument("--batch",     type=int,   default=256,   help="Batch size (default 256)")
    parser.add_argument("--lr",        type=float, default=1e-3,  help="Learning rate (default 0.001)")
    parser.add_argument("--val-split", type=float, default=0.1,   help="Validation split (default 0.1)")
    parser.add_argument("--out-dir",   default="models",          help="Output directory (default models/)")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow {tf.__version__}")

    frame_size = args.frame
    bottleneck = args.k if args.k else int(frame_size * args.ratio)
    hop        = frame_size // 2

    print(f"\nConfig:")
    print(f"  data_path  = {args.data_path}")
    print(f"  model_name = {args.model_name}")
    print(f"  frame_size = {frame_size}")
    print(f"  bottleneck = {bottleneck} ({bottleneck/frame_size*100:.1f}%)")
    print(f"  epochs     = {args.epochs}")
    print(f"  batch_size = {args.batch}")
    print(f"  lr         = {args.lr}")

    print(f"\nCollecting frames from {args.data_path}...")
    X, files = collect_frames(args.data_path, frame_size, hop)
    print(f"Training data: {X.shape}  ({X.shape[0]*frame_size/44100:.0f}s of audio total)\n")

    model = build_autoencoder(frame_size, bottleneck)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="mse")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ]

    history = model.fit(
        X, X,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.val_split,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"{args.model_name}.keras")
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(history.history["loss"],     color='#1a7abf', lw=1.5, label="train")
    ax.plot(history.history["val_loss"], color='#e74c3c', lw=1.5, label="val", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title(f"{args.model_name}  frame={frame_size}  k={bottleneck}  "
                 f"files={len(files)}  frames={X.shape[0]}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f"{args.model_name}_training.png")
    plt.savefig(plot_path, dpi=120)
    plt.show()
    print(f"Training plot saved: {plot_path}")

    meta_path = os.path.join(args.out_dir, f"{args.model_name}_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"model_name={args.model_name}\n")
        f.write(f"frame_size={frame_size}\n")
        f.write(f"bottleneck={bottleneck}\n")
        f.write(f"epochs_trained={len(history.history['loss'])}\n")
        f.write(f"final_train_loss={history.history['loss'][-1]:.6f}\n")
        f.write(f"final_val_loss={history.history['val_loss'][-1]:.6f}\n")
        f.write(f"n_files={len(files)}\n")
        f.write(f"n_frames={X.shape[0]}\n")
        f.write(f"data_path={args.data_path}\n")
        f.write("files_used=\n")
        for fp in files:
            f.write(f"  {fp}\n")
    print(f"Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()