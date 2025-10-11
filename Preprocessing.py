# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

# ====== input ======
CNT_FILES = [
    r"C:\_Davidson\projectFiles\eeg files\cnt\1009_1009_2023-04-21_07-40-21.cnt",
    r"C:\_Davidson\projectFiles\eeg files\cnt\1040_1040_2023-06-23_17-15-30.cnt",
]

# ====== output ======
OUTPUT_DIR = r"C:\_Davidson\projectFiles\eeg files\results" #output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# processing settings
LO_HZ, HI_HZ = 1.0, 45.0
NOTCH = 50.0  # israel electricity frequency
EPOCH_LEN = 2.0   # seconds
EPOCH_OVERLAP = 1.0  # 50% overlap

# bands definition by frequencies
BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# brain areas definition (groups of channels)
FRONTAL_CANDS = ["1Z", "2Z", "3Z", "1L", "2L", "1R", "2R", "1LA", "2LA", "1RA", "2RA"]
CENTRAL_CANDS = ["4Z", "6Z", "3L", "4L", "3R", "4R", "2LB", "3LB", "2RB", "3RB"]
OCCIPITAL_CANDS = ["7Z", "8Z", "9Z", "7L", "8L", "9L", "7R", "8R", "9R", "3LD", "4LD", "3RD", "4RD"]


def safe_set_montage(raw):
    """adding montage from supplied elc file."""
    pos = read_ant_elc(r"C:\_Davidson\projectFiles\eeg files\elc\NA-261.elc")
    montage = mne.channels.make_dig_montage(
        ch_pos=pos,
        coord_frame="head"
        # , nasion=[0,1,0], lpa=[-1,0,0], rpa=[1,0,0]
        )

    # verify the channels are aligned
    raw_chs = {c.upper() for c in raw.info['ch_names']}
    mon_chs = {c.upper() for c in montage.ch_names}
    missing_in_montage = sorted(raw_chs - mon_chs)
    extra_in_montage = sorted(mon_chs - raw_chs)
    print("missing in montage:", missing_in_montage[:20])
    print("extra in montage:", extra_in_montage[:20])

    # set the montage from the .elc file
    raw.set_montage(montage)

    raw.plot_sensors(show_names=True, kind='topomap') # 2D. replace with '3d' for 3D plot

"""data filtering and normalization"""
def preprocess_raw(raw):
    """Loads the EEG signal data into memory.
    In MNE, data can be stored on disk (lazy loading), 
    and this command ensures the data is fully available for filtering and processing."""
    raw.load_data()
    """
    Sets the EEG reference to the average of all electrodes.
    That means the signal at each electrode is recalculated relative to the mean potential across all electrodes.
    This helps reduce overall noise and ensures all channels are measured against a common reference.
    """
    # avg reference
    raw.set_eeg_reference("average", projection=False)
    # # Z-Score normalization for each channel
    data = raw.get_data(picks="eeg")
    data_z = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-12)
    raw._data = data_z
    # notch 50Hz
    raw.notch_filter(freqs=[NOTCH], picks="eeg")
    # band-pass 1-45Hz
    raw.filter(LO_HZ, HI_HZ, picks="eeg", fir_design="firwin")
    return raw

def make_epochs(raw):
    """split to time windows (epochs) of 2secs and 50% overlap"""
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LEN,
        overlap=EPOCH_LEN - EPOCH_OVERLAP,
        preload=True
    )
    return epochs

def psd_by_epoch(epochs, fmin=1, fmax=45):
    """PSD בשיטת Welch לכל חלון (Epoch). מחזיר freqs, psd בצורה [n_epochs, n_channels, n_freqs]."""
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=None)
    # MNE>=1.6: use .get_data(return_freqs=True)
    psd_data = psd.get_data()  # [n_epochs, n_channels, n_freqs]
    freqs = psd.freqs
    ch_names = epochs.ch_names
    return freqs, psd_data, ch_names

def band_power(psd_data, freqs, fmin, fmax, axis=-1):
    """אינטגרציה של עוצמה בתחום תדר."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    # סכימת הספק על פני תדרים בתחום
    return psd_data[..., idx].sum(axis=axis)

def summarize_features(epochs, freqs, psd_data, ch_names):
    """
    מחזיר DataFrame עם time_mid (שניות), ממוצעי פסי-תדר לכל האלקטרודות,
    יחסיים מול הסה״כ, ועוד אינדקסים (Theta/Alpha, Engagement).
    """
    n_epochs, n_ch, _ = psd_data.shape
    # נבדוק כמה ערוצים זוהו בכל אזור
    print(f"ערוצים מזוהים:")
    for region_name, region_list in {
        "frontal": FRONTAL_CANDS,
        "central": CENTRAL_CANDS,
        "occipital": OCCIPITAL_CANDS,
    }.items():
        region_idx = [i for i, ch in enumerate(ch_names) if ch in region_list]
        print(f"  {region_name}: {len(region_idx)} channels found ({[ch_names[i] for i in region_idx]})")


    # זמן אמצע לכל חלון - Compute each epoch’s time midpoint
    times = epochs.events[:, 0] / epochs.info["sfreq"]  # תחילת כל חלון
    time_mid = times + (EPOCH_LEN / 2.0)

    # This sums all frequencies together → gives the total spectral power for each channel and epoch.
    total_power = psd_data.sum(axis=-1)  # [n_epochs, n_ch]

    # עוצמות לכל פס לכל ערוץ
    band_abs = {}
    for band, (f1, f2) in BANDS.items():
        band_abs[band] = band_power(psd_data, freqs, f1, f2)  # [n_epochs, n_ch]

    # ממוצע על פני ערוצים (כל-המוח)
    #Average power across all channels
    df = pd.DataFrame({"time_s": time_mid})
    for band in BANDS.keys():
        df[f"{band}_abs_mean"] = band_abs[band].mean(axis=1)
        df[f"{band}_rel_mean"] = (band_abs[band] / (total_power + 1e-12)).mean(axis=1)

    # ערוצים פרונטליים (אם קיימים)
    # חישוב ממוצעים לכל אזורי המוח (קדמי, מרכזי, עורפי)
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

    # אינדקסים קוגניטיביים בסיסיים
    # Theta/Alpha (גבוה -> יותר עומס לרוב), Engagement = Beta/(Alpha+Theta)
    theta = df["theta_rel_mean"]
    alpha = df["alpha_rel_mean"]
    beta  = df["beta_rel_mean"]
    df["theta_over_alpha"] = theta / (alpha + 1e-12)
    df["engagement_index"] = beta / (alpha + theta + 1e-12)

    return df

def plot_quick(df, base_name):
    """שרטוט מהיר של פסי-תדר יחסיים ואינדקסים לאורך זמן."""
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
    print(f"\n=== מעבדת: {os.path.basename(fname)} ===")
    # שימוש בקורא הנכון ל-ANT Neuro
    raw = mne.io.read_raw_ant(fname, preload=True, verbose=True)

    safe_set_montage(raw)
    raw = preprocess_raw(raw)
    epochs = make_epochs(raw)
    freqs, psd_data, ch_names = psd_by_epoch(epochs, fmin=LO_HZ, fmax=HI_HZ)
    df = summarize_features(epochs, freqs, psd_data, ch_names)

    base = os.path.splitext(os.path.basename(fname))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{base}_band_features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"שמרתי פיצ’רים: {csv_path}")

    # שמירת גרפים בסיסיים
    plot_quick(df, base)
    print("✓ גרפים נשמרו לתקייה:", OUTPUT_DIR)
    return df

def read_ant_elc(fname):
    ch_names = []
    coords = []

    with open(fname, 'r') as f:
        lines = f.readlines()

    in_positions = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Positions"):
            in_positions = True
            continue
        if line.startswith("Labels"):
            break
        if in_positions:
            parts = line.replace(":", "").split()
            if len(parts) >= 4:
                name = parts[0]
                x, y, z = map(float, parts[1:4])
                ch_names.append(name)
                coords.append([x, y, z])

    coords = np.array(coords, dtype=float)
    coords /= 1000.0  # convert mm → meters (MNE expects SI units)

    pos = dict(zip(ch_names, coords))
    return pos

def main():
    all_runs = []
    for f in CNT_FILES:
        if os.path.isfile(f):
            df = process_cnt_file(f)
            df["recording"] = os.path.basename(f)
            all_runs.append(df)
        else:
            print(f"[אזהרה] לא נמצא קובץ: {f}")

    if all_runs:
        big = pd.concat(all_runs, ignore_index=True)
        big_path = os.path.join(OUTPUT_DIR, "ALL_recordings_band_features.csv")
        big.to_csv(big_path, index=False, encoding="utf-8")
        print(f"\n✓ קובץ מאוחד לכל ההקלטות: {big_path}")

if __name__ == "__main__":
    main()
