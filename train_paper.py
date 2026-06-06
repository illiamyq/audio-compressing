#!/usr/bin/env python3
"""
train_dae_stft.py
Implementacja zgodna z Roche et al. SMC 2019:
- Wejście: magnitude spectrum z STFT (513 binów)
- Architektura: [513, 128, enc, 128, 513]
- Loss: MSE na log-magnitude spektrogramach
- Optymizer: Adam, lr=1e-3, 600 epok, early stopping patience=30
"""

import argparse, os, glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

N_FFT      = 1024
HOP_LENGTH = 512
SR         = 22050
THRESHOLD  = -100.0

def extract_stft_frames(fpath, sr, n_fft, hop_length, threshold_db):
    s, _ = librosa.load(fpath, sr=sr, mono=True)
    S    = np.abs(librosa.stft(s, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    S_db = np.maximum(S_db, threshold_db)
    energy = np.max(S_db, axis=0)
    S_db   = S_db[:, energy > threshold_db + 10]
    S_norm = (S_db - threshold_db) / (-threshold_db)
    S_norm = np.clip(S_norm, 0, 1) * 2 - 1
    return S_norm.T.astype(np.float32)

def collect_frames(data_path, sr, n_fft, hop_length, threshold_db,
                   extensions=("*.wav", "*.flac", "*.mp3")):
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(data_path, "**", ext), recursive=True)
        files += glob.glob(os.path.join(data_path, ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No audio in {data_path}")
    print(f"Found {len(files)} files")
    all_frames = []
    for i, fpath in enumerate(files):
        try:
            frames = extract_stft_frames(fpath, sr, n_fft, hop_length, threshold_db)
            all_frames.append(frames)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(files)}] {sum(len(f) for f in all_frames)} frames")
        except Exception as e:
            print(f"  Skipping {fpath}: {e}")
    X = np.concatenate(all_frames, axis=0)
    np.random.shuffle(X)
    return X, files

def build_dae(n_bins, enc_dim, hidden=128):
    inp = keras.Input(shape=(n_bins,))
    x   = layers.Dense(hidden, activation="tanh")(inp)
    z   = layers.Dense(enc_dim, name="bottleneck", activation="tanh")(x)
    x   = layers.Dense(hidden, activation="tanh")(z)
    out = layers.Dense(n_bins, activation="linear")(x)
    return keras.Model(inp, out, name=f"dae_enc{enc_dim}")

def reconstruct_audio(model, fpath, sr, n_fft, hop_length, threshold_db):
    s, _ = librosa.load(fpath, sr=sr, mono=True)
    S     = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    mag   = np.abs(S)
    phase = np.angle(S)
    S_db  = librosa.amplitude_to_db(mag, ref=np.max)
    S_db  = np.maximum(S_db, threshold_db)
    S_norm = (S_db - threshold_db) / (-threshold_db)
    S_norm = np.clip(S_norm, 0, 1) * 2 - 1
    X_in   = S_norm.T.astype(np.float32)
    X_rec  = model.predict(X_in, batch_size=512, verbose=0)
    S_rec_norm = X_rec.T
    S_rec_norm = np.clip(S_rec_norm, -1, 1)
    S_rec_db   = (S_rec_norm + 1) / 2 * (-threshold_db) + threshold_db
    S_rec_mag  = librosa.db_to_amplitude(S_rec_db)
    S_rec_complex = S_rec_mag * np.exp(1j * phase)
    audio_rec = librosa.istft(S_rec_complex, hop_length=hop_length)
    n = len(s)
    if len(audio_rec) < n:
        audio_rec = np.pad(audio_rec, (0, n - len(audio_rec)))
    return s, audio_rec[:n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_name")
    parser.add_argument("--enc",        type=int,   default=16,    help="Bottleneck dim (paper: 4-100)")
    parser.add_argument("--hidden",     type=int,   default=128,   help="Hidden layer size (paper: 128)")
    parser.add_argument("--epochs",     type=int,   default=200,   help="Max epochs (paper: 600)")
    parser.add_argument("--batch",      type=int,   default=512,   help="Batch size (paper: 512)")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--threshold",  type=float, default=-100.0)
    parser.add_argument("--sr",         type=int,   default=SR)
    parser.add_argument("--test-file",  default=None)
    parser.add_argument("--out-dir",    default="models")
    args = parser.parse_args()

    n_bins = N_FFT // 2 + 1

    print(f"\nConfig (Roche et al. SMC 2019):")
    print(f"  architecture = [513, {args.hidden}, {args.enc}, {args.hidden}, 513]")
    print(f"  n_fft        = {N_FFT}")
    print(f"  hop_length   = {HOP_LENGTH}")
    print(f"  threshold    = {args.threshold} dB")
    print(f"  batch        = {args.batch}")
    print(f"  epochs       = {args.epochs}")

    print(f"\nCollecting STFT frames...")
    X, files = collect_frames(args.data_path, args.sr, N_FFT, HOP_LENGTH, args.threshold)
    print(f"Dataset: {X.shape}  ({X.shape[0]} frames × {X.shape[1]} bins)")

    model = build_dae(n_bins, args.enc, args.hidden)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="mse")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15, verbose=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, f"{args.model_name}_best.keras"),
            save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        X, X,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"{args.model_name}.keras")
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    if args.test_file and os.path.exists(args.test_file):
        import soundfile as sf
        from dct_utils import compute_snr, log_spectral_distance
        print(f"\nTest reconstruction: {args.test_file}")
        orig, rec = reconstruct_audio(model, args.test_file, args.sr, N_FFT, HOP_LENGTH, args.threshold)
        n = min(len(orig), len(rec))
        snr = compute_snr(orig[:n], rec[:n])
        lsd = log_spectral_distance(orig[:n], rec[:n], args.sr)
        print(f"  SNR = {snr:.1f} dB")
        print(f"  LSD = {lsd:.1f}")
        sf.write(os.path.join(args.out_dir, f"{args.model_name}_reconstruction.wav"), rec, args.sr)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(history.history["loss"],     color="#1a7abf", lw=1.5, label="train")
    ax.plot(history.history["val_loss"], color="#e74c3c", lw=1.5, label="val", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title(f"{args.model_name}  arch=[513,{args.hidden},{args.enc},{args.hidden},513]")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.model_name}_training.png"), dpi=120)
    plt.show()

    meta_path = os.path.join(args.out_dir, f"{args.model_name}_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"type=DAE_STFT\n")
        f.write(f"architecture=[513,{args.hidden},{args.enc},{args.hidden},513]\n")
        f.write(f"n_fft={N_FFT}\nhop_length={HOP_LENGTH}\n")
        f.write(f"enc_dim={args.enc}\nhidden={args.hidden}\n")
        f.write(f"threshold_db={args.threshold}\nsr={args.sr}\n")
        f.write(f"epochs_trained={len(history.history['loss'])}\n")
        f.write(f"n_files={len(files)}\nn_frames={X.shape[0]}\n")
    print(f"Metadata: {meta_path}")

if __name__ == "__main__":
    main()