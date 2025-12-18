# -*- coding: utf-8 -*-
import os
import numpy as np
import mne

# input/output settings
LO_HZ, HI_HZ = 1.0, 45.0
NOTCH = 50.0  # Israel electricity frequency
EPOCH_LEN = 2.0   # seconds
EPOCH_OVERLAP = 1.0  # 50% overlap


def read_ant_elc(fname):
    """
    Read ANT .elc file and return a dictionary of channel names and positions (in meters).
    """
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

    # --- ROTATE COORDINATES BY 90 DEGREES + FIX LEFT/RIGHT ---
    coords = coords[:, [1, 0, 2]]  # swap x,y (הופך את ציר איקס עם ווי כדי לסובב ב90 מעלות)
    coords[:, 0] *= -1  # invert x-axis (הופך את כיוון ציר איקס כדי שיתן תמונה נכונה)

    pos = dict(zip(ch_names, coords))
    return pos


def safe_set_montage(raw):
    """
    Add montage from supplied .elc file and verify alignment.
    """
    pos = read_ant_elc(r"C:\_Davidson\projectFiles\eeg files\elc\NA-261.elc")
    montage = mne.channels.make_dig_montage(
        ch_pos=pos,
        coord_frame="head"
    )

    raw_chs = {c.upper() for c in raw.info['ch_names']}
    mon_chs = {c.upper() for c in montage.ch_names}
    missing_in_montage = sorted(raw_chs - mon_chs)
    extra_in_montage = sorted(mon_chs - raw_chs)
    print("missing in montage:", missing_in_montage[:20])
    print("extra in montage:", extra_in_montage[:20])

    raw.set_montage(montage)
    raw.plot_sensors(show_names=True, kind='topomap')  # 2D view
    return raw


def preprocess_raw(raw):
    """
    Load EEG data, apply referencing, normalization, and filtering.
    """
    raw.load_data()
    raw.set_eeg_reference("average", projection=False)

    data = raw.get_data(picks="eeg")
    data_z = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-12)
    raw._data = data_z

    raw.notch_filter(freqs=[NOTCH], picks="eeg")
    raw.filter(LO_HZ, HI_HZ, picks="eeg", fir_design="firwin")
    return raw


def make_epochs(raw):
    """
    Split EEG data into fixed-length epochs (2s) with 50% overlap.
    """
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LEN,
        overlap=EPOCH_LEN - EPOCH_OVERLAP,
        preload=True
    )
    return epochs

if __name__ == "__main__":
    # לקרוא את קובץ ה-EEG
    raw = mne.io.read_raw_ant(
        r"C:\_Davidson\projectFiles\eeg files\cnt\1009_1009_2023-04-21_07-40-21.cnt",
        preload=True
    )

    raw = safe_set_montage(raw) # להגדיר מונטאז' ולראות את האלקטרודות

    import matplotlib.pyplot as plt

    plt.show()  # משאיר את התמונה של האלקטרודות על הראש פתוחה במקום שתיסגר ישר

