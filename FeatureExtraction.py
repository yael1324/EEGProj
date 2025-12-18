# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from Preprocessing import preprocess_raw, make_epochs, safe_set_montage, EPOCH_LEN, LO_HZ, HI_HZ, NOTCH

# ====== input/output ======
CNT_FILES = [
    r"C:\_Davidson\projectFiles\eeg files\cnt\1009_1009_2023-04-21_07-40-21.cnt",
    r"C:\_Davidson\projectFiles\eeg files\cnt\1040_1040_2023-06-23_17-15-30.cnt",
]

OUTPUT_DIR = r"C:\_Davidson\projectFiles\eeg files\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== band and region definitions ======
BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

FRONTAL_CANDS = ["1Z", "2Z", "3Z", "1L", "2L", "1R", "2R", "1LA", "2LA", "1RA", "2RA"]
CENTRAL_CANDS = ["4Z", "6Z", "3L", "4L", "3R", "4R", "2LB", "3LB", "2RB", "3RB"]
OCCIPITAL_CANDS = ["7Z", "8Z", "9Z", "7L", "8L", "9L", "7R", "8R", "9R", "3LD", "4LD", "3RD", "4RD"]


def psd_by_epoch(epochs, fmin=1, fmax=45):
    """
    Compute PSD (Power Spectral Density) for each epoch using Welch's method.
    """
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=None)
    psd_data = psd.get_data()
    freqs = psd.freqs
    ch_names = epochs.ch_names
    return freqs, psd_data, ch_names


def band_power(psd_data, freqs, fmin, fmax, axis=-1):
    """
    Integrate power within a specific frequency range.
    """
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psd_data[..., idx].sum(axis=axis)


def summarize_features(epochs, freqs, psd_data, ch_names):
    """
    Return a DataFrame with average band powers, relative values,
    regional means, and cognitive indices (Theta/Alpha, Engagement).
    """
    n_epochs, n_ch, _ = psd_data.shape
    times = epochs.events[:, 0] / epochs.info["sfreq"]
    time_mid = times + (EPOCH_LEN / 2.0)

    total_power = psd_data.sum(axis=-1)
    band_abs = {}
    for band, (f1, f2) in BANDS.items():
        band_abs[band] = band_power(psd_data, freqs, f1, f2)

    df = pd.DataFrame({"time_s": time_mid})
    for band in BANDS.keys():
        df[f"{band}_abs_mean"] = band_abs[band].mean(axis=1)
        df[f"{band}_rel_mean"] = (band_abs[band] / (total_power + 1e-12)).mean(axis=1)

    for region_name, region_list in {
        "frontal": FRONTAL_CANDS,
        "central": CENTRAL_CANDS,
        "occipital": OCCIPITAL_CANDS,
    }.items():
        region_idx = [i for i, ch in enumerate(ch_names) if ch in region_list]
        if len(region_idx) >= 1:
            for band in BANDS.keys():
                df[f"{band}_{region_name}_abs_mean"] = band_abs[band][:, region_idx].mean(axis=1)
                df[f"{band}_{region_name}_rel_mean"] = (
                    band_abs[band][:, region_idx] / (total_power[:, region_idx] + 1e-12)
                ).mean(axis=1)

    theta = df["theta_rel_mean"]
    alpha = df["alpha_rel_mean"]
    beta = df["beta_rel_mean"]
    df["theta_over_alpha"] = theta / (alpha + 1e-12)
    df["engagement_index"] = beta / (alpha + theta + 1e-12)

    return df


def plot_quick(df, base_name):
    """
    Plot relative band powers and indices over time and save to PNG files.
    """
    plt.figure()
    for col in ["theta_rel_mean", "alpha_rel_mean", "beta_rel_mean", "gamma_rel_mean"]:
        plt.plot(df["time_s"], df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Power")
    plt.title("Relative Band Power over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_rel_bands.png"), dpi=150)
    plt.close()

    plt.figure()
    for col in ["theta_over_alpha", "engagement_index"]:
        plt.plot(df["time_s"], df[col], label=col)
    plt.xlabel("Time (s)")
    plt.ylabel("Index value")
    plt.title("Cognitive Load Proxies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_indices.png"), dpi=150)
    plt.close()


def process_cnt_file(fname):
    """
    Load, preprocess, extract features and save CSV + plots.
    """
    print(f"\n=== Processing: {os.path.basename(fname)} ===")
    raw = mne.io.read_raw_ant(fname, preload=True, verbose=True)

    safe_set_montage(raw)
    raw = preprocess_raw(raw)
    epochs = make_epochs(raw)
    freqs, psd_data, ch_names = psd_by_epoch(epochs, fmin=LO_HZ, fmax=HI_HZ)
    df = summarize_features(epochs, freqs, psd_data, ch_names)

    base = os.path.splitext(os.path.basename(fname))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{base}_band_features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved features: {csv_path}")

    plot_quick(df, base)
    print("✓ Plots saved to:", OUTPUT_DIR)
    return df


def main():
    all_runs = []
    for f in CNT_FILES:
        if os.path.isfile(f):
            df = process_cnt_file(f)
            df["recording"] = os.path.basename(f)
            all_runs.append(df)
        else:
            print(f"[Warning] File not found: {f}")

    if all_runs:
        big = pd.concat(all_runs, ignore_index=True)
        big_path = os.path.join(OUTPUT_DIR, "ALL_recordings_band_features.csv")
        big.to_csv(big_path, index=False, encoding="utf-8")
        print(f"\n✓ Combined file saved: {big_path}")


if __name__ == "__main__":
    main()
