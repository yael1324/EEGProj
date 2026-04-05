# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# input/output
# כאן ניתן לעדכן בכל פעם לקובץ ה-errors של ההקלטה הרצויה (1009 / 1040 וכו')
INPUT_CSV = r"C:\_Davidson\projectFiles\eeg files\results\1009_pattern_learning_with_errors.csv"

OUTPUT_DIR = r"C:\_Davidson\projectFiles\eeg files\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# יצירת שם בסיס לפי שם הקובץ כדי לא לדרוס תוצרים
# לדוגמה: 1040_..._pattern_learning_with_errors.csv -> base = 1040_...
base = os.path.splitext(os.path.basename(INPUT_CSV))[0]
base = base.replace("_pattern_learning_with_errors", "")
base = base.replace("pattern_learning_with_errors", "")  # אם זה השם הישן בלי prefix
base = base.strip("_") if base else "recording"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{base}_pattern_learning_with_probability.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, f"{base}_prob_understanding_over_time.png")


def errors_to_probability(errors: np.ndarray) -> np.ndarray:
    """
    Convert reconstruction errors to probability in [0, 1].
    Lower error -> higher probability.

    Uses min-max scaling + inversion:
        p = 1 - (e - min) / (max - min)

    - If all errors are identical, returns 0.5 for all rows to avoid division by zero.
    """
    e = np.asarray(errors, dtype=float)
    e_min = np.min(e)
    e_max = np.max(e)

    if np.isclose(e_max, e_min):
        return np.full_like(e, 0.5, dtype=float)

    p = 1.0 - (e - e_min) / (e_max - e_min)
    return np.clip(p, 0.0, 1.0)


def main():
    # 1) Load CSV (must contain: time_s, reconstruction_error)
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"time_s", "reconstruction_error"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    # 2) Convert errors to probability
    df["prob_understanding"] = errors_to_probability(df["reconstruction_error"].values)

    # Calculate average understanding
    print("Min:", df["prob_understanding"].min())
    print("Max:", df["prob_understanding"].max())
    print("Mean:", df["prob_understanding"].mean())

    # 3) Save new CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved probability file → {OUTPUT_CSV}")

    # 4) Plot probability over time
    plt.figure()
    plt.plot(df["time_s"], df["prob_understanding"])
    plt.xlabel("Time (s)")
    plt.ylabel("Probability of Understanding (0–1)")
    plt.title(f"Estimated Understanding Over Time ({base})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved plot → {OUTPUT_PLOT}")

    import os
    os.startfile(OUTPUT_PLOT)# פותח את התמונה של הגרף הבנה הסופי


if __name__ == "__main__":
    main()
