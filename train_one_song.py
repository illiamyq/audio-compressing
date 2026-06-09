#!/usr/bin/env python3
"""
train_one_song.py
Trenuje VAE na jednej piosence — celowy overfitting.
Model zapamiętuje strukturę spektralną jednego utworu.
Generacja = wariacje na temat tej piosenki.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_FFT      = 1024
HOP_LENGTH = 512
SR         = 22050
THRESHOLD  = -100.0

def load_song(fpath, sr, n_fft, hop_length, threshold):
    s, _   = librosa.load(fpath, sr=sr, mono=True)
    s      = s / (np.max(np.abs(s)) + 1e-12)
    S      = np.abs(librosa.stft(s, n_fft=n_fft, hop_length=hop_length))
    S_db   = librosa.amplitude_to_db(S, ref=np.max)
    S_db   = np.maximum(S_db, threshold)
    S_norm = (S_db - threshold) / (-threshold)
    S_norm = np.clip(S_norm, 0, 1) * 2 - 1
    print(f"Loaded: {fpath}  {len(s)/sr:.1f}s  {S_norm.T.shape[0]} frames")
    return S_norm.T.astype(np.float32), len(s)

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
    def __init__(self, encoder, decoder, kl_weight=1e-5, **kwargs):
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
        kl     = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return recon + self.kl_weight * kl, recon, kl

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            loss, recon, kl = self._compute(x, True)
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        loss, recon, kl = self._compute(x, False)
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

def generate(decoder, n_frames, enc_dim, smoothing=0.95,
             n_fft=N_FFT, hop_length=HOP_LENGTH,
             threshold=THRESHOLD, n_iter=64):
    z = np.random.randn(1, enc_dim).astype(np.float32) * 0.5
    all_z = [z[0]]
    for _ in range(n_frames - 1):
        noise = np.random.randn(enc_dim).astype(np.float32) * 0.5
        all_z.append(smoothing * all_z[-1] + (1 - smoothing) * noise)
    Z       = np.array(all_z, dtype=np.float32)
    mag_rec = decoder.predict(Z, batch_size=512, verbose=0)
    S_norm  = np.clip(mag_rec.T, -1, 1)
    S_db    = (S_norm + 1) / 2 * (-threshold) + threshold
    S_mag   = librosa.db_to_amplitude(S_db)
    audio   = librosa.griffinlim(S_mag, n_iter=n_iter, hop_length=hop_length)
    audio   = audio / (np.max(np.abs(audio)) + 1e-12)
    return audio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("song_path",              help="ścieżka do pliku WAV")
    parser.add_argument("model_name")
    parser.add_argument("--enc",       type=int,   default=32)
    parser.add_argument("--kl-weight", type=float, default=1e-5)
    parser.add_argument("--epochs",    type=int,   default=500)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--out-dir",   default="new models")
    args = parser.parse_args()

    n_bins = N_FFT // 2 + 1
    X, song_len = load_song(args.song_path, SR, N_FFT, HOP_LENGTH, THRESHOLD)
    n_frames    = len(X)
    print(f"Frames: {n_frames}  enc_dim={args.enc}  kl={args.kl_weight}")

    encoder = build_encoder(n_bins, args.enc)
    decoder = build_decoder(n_bins, args.enc)
    vae     = VAE(encoder, decoder, kl_weight=args.kl_weight)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))

    # bez val_split — celowy overfitting na jednym pliku
    vae.fit(X, X, epochs=args.epochs, batch_size=args.batch,
            verbose=1,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, verbose=1),
            ])

    os.makedirs(args.out_dir, exist_ok=True)
    encoder.save(os.path.join(args.out_dir, f"{args.model_name}_encoder.keras"))
    decoder.save(os.path.join(args.out_dir, f"{args.model_name}_decoder.keras"))

    # rekonstrukcja oryginalnej piosenki
    z_mean, _, _ = encoder.predict(X, batch_size=512, verbose=0)
    mag_rec      = decoder.predict(z_mean, batch_size=512, verbose=0)
    S_norm       = np.clip(mag_rec.T, -1, 1)
    S_db         = (S_norm + 1) / 2 * (-THRESHOLD) + THRESHOLD
    S_mag        = librosa.db_to_amplitude(S_db)
    _, phase     = load_song(args.song_path, SR, N_FFT, HOP_LENGTH, THRESHOLD)
    rec_audio    = librosa.griffinlim(S_mag, n_iter=64, hop_length=HOP_LENGTH)
    rec_audio    = rec_audio / (np.max(np.abs(rec_audio)) + 1e-12)
    sf.write(os.path.join(args.out_dir, f"{args.model_name}_reconstruction.wav"),
             rec_audio, SR)

    # generacja nowej wariacji — tyle samo ramek co oryginał
    gen_audio = generate(decoder, n_frames=n_frames, enc_dim=args.enc)
    sf.write(os.path.join(args.out_dir, f"{args.model_name}_generated.wav"),
             gen_audio, SR)

    print(f"\nSaved to {args.out_dir}/:")
    print(f"  {args.model_name}_reconstruction.wav  (rekonstrukcja oryginału)")
    print(f"  {args.model_name}_generated.wav        (nowa wariacja)")