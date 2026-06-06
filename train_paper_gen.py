#!/usr/bin/env python3

import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_FFT = 1024
HOP = 512
SR = 22050
ENC = 16
NOISE_STD = 0.05


def load_frames(folder):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)

    all_X = []

    for f in files:
        s, _ = librosa.load(f, sr=SR, mono=True)

        S = np.abs(librosa.stft(s, n_fft=N_FFT, hop_length=HOP))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        S_db = np.clip(S_db, -80, 0)
        S = (S_db + 80) / 80  # 0..1

        frames = S.T.astype(np.float32)
        all_X.append(frames)

    X = np.concatenate(all_X, axis=0)
    np.random.shuffle(X)

    return X


def build_model(n_bins):
    inp = keras.Input(shape=(n_bins,))

    x = layers.Dense(128, activation="relu")(inp)
    z = layers.Dense(ENC, activation="relu")(x)

    x = layers.Dense(128, activation="relu")(z)
    out = layers.Dense(n_bins, activation="sigmoid")(x)

    return keras.Model(inp, out)


def main():
    X = load_frames("data/pop")

    print("Dataset:", X.shape)

    model = build_model(X.shape[1])
    model.compile(optimizer="adam", loss="mse")

    def noisy(x):
        return np.clip(x + NOISE_STD * np.random.randn(*x.shape), 0, 1)

    model.fit(
        noisy(X), X,
        epochs=50,
        batch_size=256,
        validation_split=0.1
    )

    model.save("pop_guessssss.keras")

if __name__ == "__main__":
    main()