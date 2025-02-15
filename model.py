import numpy as np
import librosa
import soundfile as sf
from scipy.signal import lfilter, butter
from scipy import signal
import os
import hashlib


def anonymize(input_audio_path):
    """
    Enhanced anonymization algorithm with dynamic parameter adaptation
    Targets: WER < 0.3, EER > 0.25
    """
    # Base parameters
    privacy_level = 0.25  # Increased for better anonymity
    intelligibility_boost = 0.7  # For WER improvement
    sr = None  # Preserve original sample rate

    y, sr = librosa.load(input_audio_path, sr=sr)

    # Create content-based seed for consistent randomization
    audio_hash = hashlib.sha256(y.tobytes()).hexdigest()
    seed = int(audio_hash[:8], 16) % (2**32)
    rng = np.random.default_rng(seed)

    # Dynamic parameter calculation
    pitch_shift_range = (-4.5 * privacy_level, 4.5 * privacy_level)
    formant_shift_range = (1 - 0.35 * privacy_level, 1 + 0.35 * privacy_level)
    noise_level = 0.012 * privacy_level
    reverb_level = 0.35 * privacy_level

    # Stage 1: Adaptive spectral preservation
    S = librosa.stft(y)
    S_mag = np.abs(S)

    # Noise estimation with percentile-based floor
    noise_floor = np.percentile(S_mag, 25, axis=1, keepdims=True)
    S_denoised = np.maximum(S_mag - 1.2 * noise_floor, 0.1 * noise_floor)
    y = librosa.istft(S_denoised * np.exp(1j * np.angle(S)), length=len(y))

    # Stage 2: Multi-step pitch manipulation
    pitch_shift1 = rng.uniform(*pitch_shift_range)
    pitch_shift2 = rng.uniform(-1.5, 1.5)  # Smaller secondary shift
    y_shifted = librosa.effects.pitch_shift(
        librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift1),
        sr=sr,
        n_steps=pitch_shift2,
    )

    # Stage 3: Non-linear formant shifting
    formant_shift = rng.uniform(*formant_shift_range)
    formant_shift += 0.1 * np.sin(2 * np.pi * rng.random())  # Add modulated variation

    y_resampled = librosa.resample(
        y_shifted, orig_sr=sr, target_sr=int(sr * formant_shift)
    )
    y_formant = librosa.resample(
        y_resampled, orig_sr=int(sr * formant_shift), target_sr=sr
    )
    y_formant = librosa.util.fix_length(y_formant, size=len(y))

    # Stage 4: Time-domain modulation
    alpha = 0.98 + 0.04 * rng.random()
    y_time = librosa.effects.time_stretch(y_formant, rate=alpha)
    y_time = librosa.util.fix_length(y_time, size=len(y))

    # Stage 5: Adaptive bandpass filtering
    nyq = 0.5 * sr
    low = max(80, 150 * (1 - 0.6 * privacy_level))
    high = min(5000, 3800 * (1 + 0.6 * privacy_level))
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    y_filtered = signal.filtfilt(b, a, y_time)

    # Stage 6: Dynamic noise injection
    if noise_level > 0:
        # Create modulated noise profile
        times = np.linspace(0, len(y_filtered) / sr, len(y_filtered))
        modulated = 0.5 * (
            np.sin(2 * np.pi * 3 * times) + 0.5 * rng.random(len(y_filtered))
        )
        noise = modulated * rng.normal(0, 1, len(y_filtered))
        noise = librosa.util.normalize(noise) * noise_level
        y_noisy = y_filtered + noise
    else:
        y_noisy = y_filtered

    # Stage 7: Randomized reverberation
    if reverb_level > 0:
        rev_time = 0.1 + 0.3 * privacy_level
        ir_len = int(sr * rev_time)
        impulse = np.exp(-np.linspace(0, 10, ir_len)) * (
            np.sin(2 * np.pi * np.linspace(0.1, 0.5, ir_len)) + 0.5 * rng.random(ir_len)
        )
        impulse /= np.sum(impulse)
        y_reverb = signal.fftconvolve(y_noisy, impulse, mode="same")
    else:
        y_reverb = y_noisy

    # Stage 8: Spectral similarity masking
    S_orig = librosa.stft(y)
    S_anon = librosa.stft(y_reverb)

    # Create frequency-dependent masking
    freq_mask = 0.5 + 0.5 * np.tanh(
        10 * (np.mean(S_anon, axis=1) - np.mean(S_orig, axis=1))
    )
    S_masked = S_anon * freq_mask[:, np.newaxis]

    y_final = librosa.istft(S_masked, length=len(y_reverb))

    # Stage 9: Intelligibility enhancement
    if intelligibility_boost > 0:
        # Emphasize 1-4 kHz frequency band
        sos = signal.butter(4, [1000, 4000], btype="bandpass", fs=sr, output="sos")
        y_enhanced = signal.sosfilt(sos, y_final)
        y_final = (
            1 - intelligibility_boost
        ) * y_final + intelligibility_boost * y_enhanced

    # Final adaptive normalization
    y_normalized = librosa.util.normalize(y_final) * 1.0  # Prevent clipping
    return y_normalized.astype(np.float32), sr
