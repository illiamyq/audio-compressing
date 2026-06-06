import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import soundfile as sf

def stft_to_input(s, n_fft, hop_length, threshold_db):
    S = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)
    S_db = librosa.amplitude_to_db(mag, ref=np.max)
    S_db = np.maximum(S_db, threshold_db)
    S_norm = (S_db - threshold_db) / (-threshold_db)
    S_norm = np.clip(S_norm, 0, 1) * 2 - 1
    return S_norm.T.astype(np.float32), mag, phase, S_db

def output_to_audio(X_rec, phase, threshold_db, hop_length):
    S_rec_norm = np.clip(X_rec.T, -1, 1)
    S_rec_db = (S_rec_norm + 1) / 2 * (-threshold_db) + threshold_db
    S_rec_mag = librosa.db_to_amplitude(S_rec_db)
    S_complex = S_rec_mag * np.exp(1j * phase)
    return librosa.istft(S_complex, hop_length=hop_length)

def reconstruct(model, sig, n_fft, hop_length, threshold_db):
    X_in, mag, phase, S_db = stft_to_input(sig, n_fft, hop_length, threshold_db)
    X_rec = model.predict(X_in, batch_size=512, verbose=0)
    rec = output_to_audio(X_rec, phase, threshold_db, hop_length)
    
    n = len(sig)
    if len(rec) < n:
        rec = np.pad(rec, (0, n - len(rec)))
    return rec[:n]

def run_experiment(dae_model_paths, test_file, compare_sr, n_fft, hop_length, threshold_db, out_dir, compute_snr, log_spectral_distance):
    """reconstruction, logs metrics, spectrograms, saves audio files."""
    import tensorflow.keras as keras

    dae_models = {}
    for name, path in dae_model_paths.items():
        if os.path.exists(path):
            dae_models[name] = keras.models.load_model(path)
            print(f"Loaded {name}")
        else:
            print(f"Missing {path}")
    sig, sr = librosa.load(test_file, sr=compare_sr, mono=True)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    print(f"\nSignal: {test_file}  {len(sig)/sr:.1f}s  {sr}Hz")

    results = {}
    for name, model in dae_models.items():
        rec = reconstruct(model, sig, n_fft, hop_length, threshold_db)
        snr = compute_snr(sig, rec)
        lsd = log_spectral_distance(sig, rec, sr)
        results[name] = {"rec": rec, "snr": snr, "lsd": lsd}
        print(f"  {name}: SNR={snr:.1f}dB  LSD={lsd:.1f}")

    n_models = len(results)
    fig, axes = plt.subplots(1 + n_models, 1, figsize=(14, 4 * (1 + n_models)))
    if n_models == 0:
        axes = [axes]  

    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=n_fft)), ref=np.max)
    img = librosa.display.specshow(D_orig, sr=sr, x_axis="time", y_axis="log", ax=axes[0])
    axes[0].set_title("Oryginał")
    plt.colorbar(img, ax=axes[0], format="%+2.0f dB")

    for i, (name, r) in enumerate(results.items()):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(r["rec"], n_fft=n_fft)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axes[i+1])
        axes[i+1].set_title(f"DAE STFT — {name}  SNR={r['snr']:.1f}dB  LSD={r['lsd']:.1f}")
        plt.colorbar(img, ax=axes[i+1], format="%+2.0f dB")

    plt.tight_layout()
    plot_name = f"dae_stft_comparison_{os.path.basename(test_file).split('.')[0]}.png"
    plt.savefig(os.path.join(out_dir, plot_name), dpi=120, bbox_inches="tight")
    plt.show()

    print("Oryginał:")
    display(Audio(sig, rate=sr))
    for name, r in results.items():
        sf.write(os.path.join(out_dir, f"{name}_rec.wav"), r["rec"], sr)
        print(f"{name}  SNR={r['snr']:.1f}dB:")
        display(Audio(r["rec"], rate=sr))

import numpy as np
import librosa

def get_stft_frames_count(n_samples, hop_length):
    """Accurately mimics librosa's padding behavior for frame counting."""
    return int(np.ceil(n_samples / hop_length))

def get_dct_topk_size_bytes(n_samples, frame_size, k, dtype_bytes=4):
    hop_length = frame_size // 2
    n_frames = get_stft_frames_count(n_samples, hop_length)
    index_bytes = 1 if frame_size <= 256 else 2
    return n_frames * k * (dtype_bytes + index_bytes)

def get_ae_size_bytes(n_samples, hop_length, bottleneck_dim, dtype_bytes=4):
    """Works for both old AE and DAE STFT since both store unindexed dense vectors."""
    n_frames = get_stft_frames_count(n_samples, hop_length)
    return n_frames * bottleneck_dim * dtype_bytes