# dct_utils.py
import librosa
import numpy as np
from scipy.fft import dct, idct
from scipy.signal import spectrogram
import os

def compress_dct_frames(signal, frame_size, k):
    frames = signal[:len(signal) - len(signal) % frame_size].reshape(-1, frame_size)
    rec = []
    for frame in frames:
        C = dct(frame, norm="ortho")
        C[k:] = 0
        rec.append(idct(C, norm="ortho"))
    return np.array(rec).flatten()

def compress_dct_overlap(signal, frame_size, k, hop=None):
    if hop is None:
        hop = frame_size // 2
    output = np.zeros(len(signal))
    counts = np.zeros(len(signal))
    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size]
        C = dct(frame, norm="ortho")
        C[k:] = 0
        rec = idct(C, norm="ortho")
        output[start:start + frame_size] += rec
        counts[start:start + frame_size] += 1
    counts[counts == 0] = 1
    return output / counts


def compress_mdct(signal, frame_size, k, hop=None):
    if hop is None:
        hop = frame_size // 2

    window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))

    output = np.zeros(len(signal))
    weights = np.zeros(len(signal))

    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C = dct(frame, norm="ortho")
        C[k:] = 0
        rec = idct(C, norm="ortho")
        output[start:start + frame_size] += rec * window
        weights[start:start + frame_size] += window ** 2

    weights[weights == 0] = 1
    return output / weights

def compress_dct_overlap_hann(signal, frame_size, k, hop=None):
    if hop is None:
        hop = frame_size // 2
    window = np.hanning(frame_size)
    output = np.zeros(len(signal))
    weights = np.zeros(len(signal))
    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C = dct(frame, norm="ortho")
        C[k:] = 0
        rec = idct(C, norm="ortho")
        output[start:start + frame_size] += rec * window
        weights[start:start + frame_size] += window ** 2
    weights[weights == 0] = 1
    return output / weights

def compress_dct_topk(signal, frame_size, k):
    frames = signal[:len(signal) - len(signal) % frame_size].reshape(-1, frame_size)
    rec = []
    for frame in frames:
        C = dct(frame, norm="ortho")
        idx = np.argsort(np.abs(C))[::-1]
        C_red = np.zeros_like(C)
        C_red[idx[:k]] = C[idx[:k]]
        rec.append(idct(C_red, norm="ortho"))
    return np.array(rec).flatten()

def compute_snr(x, y):
    n = min(len(x), len(y))
    noise = x[:n] - y[:n]
    return 10 * np.log10(np.sum(x[:n] ** 2) / (np.sum(noise ** 2) + 1e-12))

def log_spectral_distance(x, y, sr, n_fft=2048):
    n = min(len(x), len(y))
    X = np.abs(librosa.stft(x[:n], n_fft=n_fft)) + 1e-8
    Y = np.abs(librosa.stft(y[:n], n_fft=n_fft)) + 1e-8
    T = min(X.shape[1], Y.shape[1])
    return np.mean(np.sqrt(np.mean((20 * np.log10(X[:, :T]) - 20 * np.log10(Y[:, :T])) ** 2, axis=0)))

def aac_roundtrip(y, sr, bitrate="128k"):
    import subprocess, tempfile, os
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        wav_in  = os.path.join(tmp, "input.wav")
        aac_out = os.path.join(tmp, "compressed.aac")
        wav_out = os.path.join(tmp, "decoded.wav")
        sf.write(wav_in, y, sr)
        subprocess.run(["ffmpeg", "-y", "-i", wav_in, "-c:a", "aac", "-b:a", bitrate, aac_out], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", aac_out, wav_out], capture_output=True, check=True)
        y_aac, _ = sf.read(wav_out)
    y_aac = np.array(y_aac, dtype=np.float32)
    if y_aac.ndim > 1:
        y_aac = y_aac[:, 0]
    delay = np.argmax(np.correlate(y_aac[:sr], y[:sr], mode='full')) - (sr - 1)
    if delay > 0:
        y_aac = y_aac[delay:]
    n = min(len(y), len(y_aac))
    return y_aac[:n]

def mp3_roundtrip(y, sr, bitrate="128k"):
    import subprocess, tempfile, os
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        wav_in  = os.path.join(tmp, "input.wav")
        mp3_out = os.path.join(tmp, "compressed.mp3")
        wav_out = os.path.join(tmp, "decoded.wav")
        sf.write(wav_in, y, sr)
        subprocess.run(["ffmpeg", "-y", "-i", wav_in, "-c:a", "libmp3lame", "-b:a", bitrate, mp3_out], capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", mp3_out, wav_out], capture_output=True, check=True)
        y_mp3, _ = sf.read(wav_out)
    y_mp3 = np.array(y_mp3, dtype=np.float32)
    if y_mp3.ndim > 1:
        y_mp3 = y_mp3[:, 0]
    delay = np.argmax(np.correlate(y_mp3[:sr], y[:sr], mode='full')) - (sr - 1)
    if delay > 0:
        y_mp3 = y_mp3[delay:]
    n = min(len(y), len(y_mp3))
    return y_mp3[:n]

def bark_bands(frame_size, sr):
    """Zwraca indeksy granic pasm Barksa dla danej ramki DCT"""
    freqs = np.arange(frame_size) * sr / (2 * frame_size)
    bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    bands = []
    for b in range(1, 25):
        idx = np.where((bark >= b - 1) & (bark < b))[0]
        if len(idx) > 0:
            bands.append(idx)
    return bands

def compress_dct_psycho(signal, sr, frame_size, k_total, spreading_db=12):
    """
    DCT top-k z modelem psychoakustycznym opartym na maskowaniu Barksa.
    k_total - globalna liczba wspolczynnikow do zachowania (jak poprzednie k)
    spreading_db - ile dB ponizej lokalnego maksimum maskujemy
    """
    hop    = frame_size // 2
    window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))
    output  = np.zeros(len(signal))
    weights = np.zeros(len(signal))
    bands   = bark_bands(frame_size, sr)

    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C     = dct(frame, norm="ortho")
        power = C ** 2

        # prog maskowania per pasmo Barksa
        mask = np.zeros(frame_size)
        for band_idx in bands:
            band_power = power[band_idx]
            if len(band_power) == 0:
                continue
            peak_db = 10 * np.log10(np.max(band_power) + 1e-12)
            threshold_db = peak_db - spreading_db
            threshold_linear = 10 ** (threshold_db / 10)
            mask[band_idx] = threshold_linear

        C_masked = C.copy()
        C_masked[power < mask] = 0

        surviving = np.where(power >= mask)[0]
        if len(surviving) > k_total:
            surviving_power = power[surviving]
            top_idx = surviving[np.argsort(surviving_power)[::-1][:k_total]]
            C_red = np.zeros_like(C)
            C_red[top_idx] = C[top_idx]
        else:
            C_red = C_masked

        rec = idct(C_red, norm="ortho")
        output[start:start + frame_size]  += rec * window
        weights[start:start + frame_size] += window ** 2

    weights[weights == 0] = 1
    return output / weights

def file_size_kb(k, frame_size, signal_len, bits=32):
    hop = frame_size // 2
    n_frames = (signal_len - frame_size) // hop
    return n_frames * k * bits / 8 / 1024

def compress_dct_psycho_quantized(signal, sr, frame_size, k_total, spreading_db=36, bits=16):
    hop    = frame_size // 2
    window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))
    output  = np.zeros(len(signal))
    weights = np.zeros(len(signal))
    bands   = bark_bands(frame_size, sr)
    levels  = 2 ** (bits - 1) - 1
    dtype   = np.int8 if bits == 8 else np.int16

    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C     = dct(frame, norm="ortho")
        power = C ** 2

        mask = np.zeros(frame_size)
        for band_idx in bands:
            band_power = power[band_idx]
            if len(band_power) == 0:
                continue
            peak_db          = 10 * np.log10(np.max(band_power) + 1e-12)
            threshold_linear = 10 ** ((peak_db - spreading_db) / 10)
            mask[band_idx]   = threshold_linear

        C_masked  = C.copy()
        C_masked[power < mask] = 0
        surviving = np.where(power >= mask)[0]

        if len(surviving) > k_total:
            top_idx = surviving[np.argsort(power[surviving])[::-1][:k_total]]
            C_red   = np.zeros_like(C)
            C_red[top_idx] = C[top_idx]
        else:
            C_red = C_masked
        nonzero = np.where(C_red != 0)[0]
        if len(nonzero) > 0:
            max_val           = np.max(np.abs(C_red[nonzero])) + 1e-12
            C_quant           = np.round(C_red[nonzero] / max_val * levels).astype(dtype)
            C_red[nonzero]    = C_quant.astype(np.float32) / levels * max_val

        rec = idct(C_red, norm="ortho")
        output[start:start + frame_size]  += rec * window
        weights[start:start + frame_size] += window ** 2

    weights[weights == 0] = 1
    return output / weights

# dct_utils.py — dodaj na końcu pliku
import heapq
import struct
import os
from collections import Counter
from scipy.fft import dct, idct

def huffman_build(data):
    freq = Counter(data)
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return {sym: code for sym, code in heap[0][1:]}

def huffman_encode_bits(data, codebook):
    bits = "".join(codebook[s] for s in data)
    pad  = (8 - len(bits) % 8) % 8
    bits += "0" * pad
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8)), pad

def huffman_decode_bits(encoded_bytes, pad, codebook, n_symbols):
    bits    = "".join(f"{b:08b}" for b in encoded_bytes)
    bits    = bits[:len(bits) - pad] if pad else bits
    reverse = {v: k for k, v in codebook.items()}
    result, cur = [], ""
    for b in bits:
        cur += b
        if cur in reverse:
            result.append(reverse[cur])
            cur = ""
            if len(result) == n_symbols:
                break
    return result

def encode_psycho_huffman(signal, sr, frame_size, k, spreading_db=36, bits=8, path="out.bin"):
    hop    = frame_size // 2
    window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))
    bands  = bark_bands(frame_size, sr)
    levels = 2 ** (bits - 1) - 1
    dtype  = np.int16 if bits == 16 else np.int8

    frames_encoded = []
    all_quant_vals = []

    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C     = dct(frame, norm="ortho")
        power = C ** 2

        mask = np.zeros(frame_size)
        for band_idx in bands:
            if len(band_idx) == 0:
                continue
            peak_db        = 10 * np.log10(np.max(power[band_idx]) + 1e-12)
            mask[band_idx] = 10 ** ((peak_db - spreading_db) / 10)

        surviving = np.where(power >= mask)[0]
        top_idx   = surviving[np.argsort(power[surviving])[::-1][:k]] \
                    if len(surviving) > k else surviving

        if len(top_idx) == 0:
            frames_encoded.append((0.0, np.array([], dtype=np.uint16),
                                   np.array([], dtype=dtype)))
            continue

        max_val = np.max(np.abs(C[top_idx])) + 1e-12
        quant   = np.round(C[top_idx] / max_val * levels).astype(dtype)
        frames_encoded.append((max_val, top_idx.astype(np.uint16), quant))
        all_quant_vals.extend(quant.tolist())

    codebook = huffman_build(all_quant_vals) if all_quant_vals else {}

    with open(path, "wb") as f:
        f.write(struct.pack("iiii", frame_size, k, bits, len(signal)))
        f.write(struct.pack("i", len(codebook)))
        for sym, code in codebook.items():
            code_bytes = code.encode()
            f.write(struct.pack("hH", sym, len(code_bytes)))
            f.write(code_bytes)
        for max_val, idx, quant in frames_encoded:
            f.write(struct.pack("fH", max_val, len(idx)))
            if len(idx) == 0:
                continue
            f.write(idx.tobytes())
            encoded_bytes, pad = huffman_encode_bits(quant.tolist(), codebook)
            f.write(struct.pack("H", pad))
            f.write(struct.pack("I", len(encoded_bytes)))
            f.write(encoded_bytes)

    return os.path.getsize(path) / 1024

def encode_psycho_huffman_plus(signal, sr, frame_size, k, spreading_db=36, bits=8, path="out.bin"):
    hop    = frame_size // 2
    window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))
    bands  = bark_bands(frame_size, sr)
    levels = min(2 ** (bits - 1) - 1, 63)
    dtype  = np.int8

    frames_encoded = []
    all_quant_vals = []

    for start in range(0, len(signal) - frame_size, hop):
        frame = signal[start:start + frame_size] * window
        C     = dct(frame, norm="ortho")
        power = C ** 2

        mask = np.zeros(frame_size)
        for band_idx in bands:
            if len(band_idx) == 0:
                continue
            peak_db        = 10 * np.log10(np.max(power[band_idx]) + 1e-12)
            mask[band_idx] = 10 ** ((peak_db - spreading_db) / 10)

        surviving = np.where(power >= mask)[0]
        top_idx   = surviving[np.argsort(power[surviving])[::-1][:k]] \
                    if len(surviving) > k else surviving

        if len(top_idx) == 0:
            frames_encoded.append((0.0, np.array([], dtype=np.uint16),
                                   np.array([], dtype=dtype)))
            continue

        max_val = np.max(np.abs(C[top_idx])) + 1e-12
        quant   = np.round(C[top_idx] / max_val * levels).astype(dtype)
        quant[np.abs(quant) < 2] = 0

        frames_encoded.append((max_val, top_idx.astype(np.uint16), quant))
        all_quant_vals.extend(quant.tolist())

    codebook = huffman_build(all_quant_vals) if all_quant_vals else {}

    with open(path, "wb") as f:
        f.write(struct.pack("iiii", frame_size, k, bits, len(signal)))
        f.write(struct.pack("i", len(codebook)))
        for sym, code in codebook.items():
            code_bytes = code.encode()
            f.write(struct.pack("hH", sym, len(code_bytes)))
            f.write(code_bytes)
        for max_val, idx, quant in frames_encoded:
            f.write(struct.pack("fH", max_val, len(idx)))
            if len(idx) == 0:
                continue
            f.write(idx.tobytes())
            encoded_bytes, pad = huffman_encode_bits(quant.tolist(), codebook)
            f.write(struct.pack("H", pad))
            f.write(struct.pack("I", len(encoded_bytes)))
            f.write(encoded_bytes)

    return os.path.getsize(path) / 1024

def decode_psycho_huffman(path):
    with open(path, "rb") as f:
        frame_size, k, bits, signal_len = struct.unpack("iiii", f.read(16))
        levels = 2 ** (bits - 1) - 1
        hop    = frame_size // 2
        window = np.sin(np.pi / frame_size * (np.arange(frame_size) + 0.5))

        n_syms   = struct.unpack("i", f.read(4))[0]
        codebook = {}
        for _ in range(n_syms):
            sym, code_len = struct.unpack("hH", f.read(4))
            code = f.read(code_len).decode()
            codebook[sym] = code

        output  = np.zeros(signal_len + frame_size)
        weights = np.zeros(signal_len + frame_size)

        start = 0
        while start < signal_len - frame_size:
            max_val, n_idx = struct.unpack("fH", f.read(6))
            if n_idx == 0:
                start += hop
                continue
            idx           = np.frombuffer(f.read(n_idx * 2), dtype=np.uint16)
            pad           = struct.unpack("H", f.read(2))[0]
            n_enc_bytes   = struct.unpack("I", f.read(4))[0]
            encoded_bytes = f.read(n_enc_bytes)
            quant         = huffman_decode_bits(encoded_bytes, pad, codebook, n_idx)

            C_red       = np.zeros(frame_size)
            C_red[idx]  = np.array(quant, dtype=np.float32) / levels * max_val
            rec         = idct(C_red, norm="ortho")
            output[start:start + frame_size]  += rec * window
            weights[start:start + frame_size] += window ** 2
            start += hop

    weights[weights == 0] = 1
    return (output / weights)[:signal_len]

def generate_note(freq, sr=48000, duration=1.5):
    t   = np.linspace(0, duration, int(sr * duration))
    sig = (np.sin(2*np.pi*freq*t) +
           0.5*np.sin(4*np.pi*freq*t) +
           0.25*np.sin(6*np.pi*freq*t))
    env = np.exp(-t * 1.5)
    return (sig * env / np.max(np.abs(sig * env))).astype(np.float32)