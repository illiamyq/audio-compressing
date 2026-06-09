#!/usr/bin/env python3
"""
train_vae_stft.py
VAE na spektrogramach STFT — identyczny pipeline jak DAE
ale z regularyzacją KL która umożliwia generację z N(0,I)
"""

import argparse, os, glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_FFT      = 1024
HOP_LENGTH = 512
SR         = 22050
THRESHOLD  = -100.0

def extract_frames(fpath, sr, n_fft, hop_length, threshold):
    s, _   = librosa.load(fpath, sr=sr, mono=True)
    S      = np.abs(librosa.stft(s, n_fft=n_fft, hop_length=hop_length))
    S_db   = librosa.amplitude_to_db(S, ref=np.max)
    S_db   = np.maximum(S_db, threshold)
    S_norm = (S_db - threshold) / (-threshold)
    S_norm = np.clip(S_norm, 0, 1) * 2 - 1
    return S_norm.T.astype(np.float32)

def collect(data_path, sr, n_fft, hop_length, threshold):
    files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        files += glob.glob(os.path.join(data_path, "**", ext), recursive=True)
        files += glob.glob(os.path.join(data_path, ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No audio in {data_path}")
    print(f"Found {len(files)} files")
    all_frames = []
    for i, f in enumerate(files):
        try:
            frames = extract_frames(f, sr, n_fft, hop_length, threshold)
            all_frames.append(frames)
            if (i+1) % 10 == 0:
                print(f"  [{i+1}/{len(files)}] {sum(len(x) for x in all_frames)} frames")
        except Exception as e:
            print(f"  skip {f}: {e}")
    X = np.concatenate(all_frames, axis=0)
    np.random.shuffle(X)
    return X, files

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(n_bins, enc_dim, hidden=128):
    inp       = keras.Input(shape=(n_bins,))
    x         = layers.Dense(hidden, activation="tanh")(inp)
    z_mean    = layers.Dense(enc_dim, name="z_mean")(x)
    z_log_var = layers.Dense(enc_dim, name="z_log_var")(x)
    z         = Sampling(name="bottleneck")([z_mean, z_log_var])
    return keras.Model(inp, [z_mean, z_log_var, z], name="encoder")

def build_decoder(n_bins, enc_dim, hidden=128):
    inp = keras.Input(shape=(enc_dim,))
    x   = layers.Dense(hidden, activation="tanh")(inp)
    out = layers.Dense(n_bins, activation="linear")(x)
    return keras.Model(inp, out, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.kl_weight     = kl_weight
        self.loss_tracker  = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.kl_tracker    = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

    def _compute(self, x, training):
        z_mean, z_log_var, z = self.encoder(x, training=training)
        x_rec  = self.decoder(z, training=training)
        recon  = tf.reduce_mean(tf.square(x - x_rec))
        kl     = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return recon + self.kl_weight * kl, recon, kl

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            loss, recon, kl = self._compute(x, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        loss, recon, kl = self._compute(x, training=False)
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

def generate_audio(decoder, n_frames=1292, enc_dim=16,
                   n_fft=N_FFT, hop_length=HOP_LENGTH,
                   sr=SR, threshold=THRESHOLD, smoothing=0.92):
    z = np.random.randn(1, enc_dim).astype(np.float32)
    all_z = [z[0]]
    for _ in range(n_frames - 1):
        noise = np.random.randn(enc_dim).astype(np.float32)
        z_new = smoothing * all_z[-1] + (1 - smoothing) * noise
        all_z.append(z_new)
    Z       = np.array(all_z, dtype=np.float32)
    mag_rec = decoder.predict(Z, batch_size=512, verbose=0)
    S_norm  = np.clip(mag_rec.T, -1, 1)
    S_db    = (S_norm + 1) / 2 * (-threshold) + threshold
    S_mag   = librosa.db_to_amplitude(S_db)
    audio   = librosa.griffinlim(S_mag, n_iter=64, hop_length=hop_length)
    audio   = audio / (np.max(np.abs(audio)) + 1e-12)
    return audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_name")
    parser.add_argument("--enc",       type=int,   default=16)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--out-dir",   default="new models")
    args = parser.parse_args()

    n_bins = N_FFT // 2 + 1
    print(f"\nConfig: arch=[{n_bins},128,{args.enc},128,{n_bins}]  kl={args.kl_weight}")

    X, files = collect(args.data_path, SR, N_FFT, HOP_LENGTH, THRESHOLD)
    print(f"Dataset: {X.shape}")

    encoder = build_encoder(n_bins, args.enc)
    decoder = build_decoder(n_bins, args.enc)
    vae     = VAE(encoder, decoder, kl_weight=args.kl_weight)
    vae.compile(optimizer=keras.optimizers.Adam(args.lr))
    encoder.summary()
    decoder.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
    ]

    history = vae.fit(X, X, epochs=args.epochs, batch_size=args.batch,
                      validation_split=0.1, callbacks=callbacks, verbose=1)

    os.makedirs(args.out_dir, exist_ok=True)
    encoder.save(os.path.join(args.out_dir, f"{args.model_name}_encoder.keras"))
    decoder.save(os.path.join(args.out_dir, f"{args.model_name}_decoder.keras"))
    print(f"Saved encoder/decoder to {args.out_dir}/")

    import soundfile as sf
    print("Generating 30s sample...")
    audio = generate_audio(decoder, n_frames=1292, enc_dim=args.enc)
    sf.write(os.path.join(args.out_dir, f"{args.model_name}_generated.wav"), audio, SR)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    for ax, key in zip(axes, ["loss", "recon", "kl"]):
        ax.plot(history.history[key],         color="#1a7abf", lw=1.5, label="train")
        ax.plot(history.history[f"val_{key}"], color="#e74c3c", lw=1.5,
                label="val", linestyle="--")
        ax.set_title(key); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f"{args.model_name}  enc={args.enc}  kl={args.kl_weight}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.model_name}_training.png"), dpi=120)
    plt.show()

if __name__ == "__main__":
    main()