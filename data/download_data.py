"""
Download MIT-BIH Arrhythmia Database and split into hospital partitions.

Uses the wfdb library to download real ECG data from PhysioNet, extracts
5-second windows (1800 samples at 360Hz), maps annotations to 5 arrhythmia
classes, and partitions records across 3 simulated hospitals.
"""

import os
import sys
import json
import numpy as np
import wfdb
from collections import defaultdict, Counter
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.global_model import ANNOTATION_TO_LABEL, LABEL_MAP

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "mitdb")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
WINDOW_SIZE = 1800  # 5 seconds × 360 Hz
SAMPLING_FREQ = 360

# MIT-BIH record numbers
ALL_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234,
]

# Hospital partition splits (simulating data siloing)
HOSPITAL_SPLITS = {
    "hospital_1": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115],
    "hospital_2": [116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207],
    "hospital_3": [208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234],
}


def download_mitbih():
    """Download MIT-BIH Arrhythmia Database from PhysioNet."""
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 10:
        print(f"[INFO] MIT-BIH data already exists at {DATA_DIR}, skipping download.")
        return

    print("[INFO] Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
    os.makedirs(DATA_DIR, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=DATA_DIR)
    print("[INFO] Download complete.")


def extract_windows_from_record(record_num: int):
    """
    Extract 5-second ECG windows from a single MIT-BIH record.

    Returns:
        windows: list of np.ndarray (each shape (1800,))
        labels:  list of int (class index 0-4)
    """
    record_path = os.path.join(DATA_DIR, str(record_num))

    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
    except Exception as e:
        print(f"[WARN] Could not read record {record_num}: {e}")
        return [], []

    # Use first lead (MLII typically)
    signal = record.p_signal[:, 0]
    total_samples = len(signal)

    # Build sample → label mapping from annotations
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    windows = []
    labels = []

    # Iterate through annotations to create windows centered on each beat
    for i, (sample_idx, symbol) in enumerate(zip(ann_samples, ann_symbols)):
        # Map annotation symbol to our 5-class system
        if symbol not in ANNOTATION_TO_LABEL:
            continue

        label_str = ANNOTATION_TO_LABEL[symbol]
        label_idx = LABEL_MAP[label_str]

        # Extract window centered on the beat annotation
        start = sample_idx - WINDOW_SIZE // 2
        end = start + WINDOW_SIZE

        # Skip if window goes out of bounds
        if start < 0 or end > total_samples:
            continue

        window = signal[start:end].copy()

        # Normalize: zero mean, unit variance
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-6:
            window = (window - mean) / std
        else:
            window = window - mean

        windows.append(window.astype(np.float32))
        labels.append(label_idx)

    return windows, labels


def process_and_split():
    """
    Process all MIT-BIH records and split into hospital partitions.
    Also creates a server-side validation set (10% from each hospital).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    stats = {}

    for hospital_name, record_list in HOSPITAL_SPLITS.items():
        print(f"\n[INFO] Processing {hospital_name} (records: {record_list})")

        all_windows = []
        all_labels = []

        for rec_num in tqdm(record_list, desc=f"  {hospital_name}"):
            wins, labs = extract_windows_from_record(rec_num)
            all_windows.extend(wins)
            all_labels.extend(labs)

        if len(all_windows) == 0:
            print(f"[WARN] No windows extracted for {hospital_name}")
            continue

        all_windows = np.array(all_windows, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        # Shuffle
        indices = np.random.RandomState(42).permutation(len(all_windows))
        all_windows = all_windows[indices]
        all_labels = all_labels[indices]

        # Split: 80% train, 10% local test, 10% server validation
        n = len(all_windows)
        n_train = int(0.8 * n)
        n_local_test = int(0.1 * n)

        train_X = all_windows[:n_train]
        train_y = all_labels[:n_train]
        test_X = all_windows[n_train:n_train + n_local_test]
        test_y = all_labels[n_train:n_train + n_local_test]
        val_X = all_windows[n_train + n_local_test:]
        val_y = all_labels[n_train + n_local_test:]

        # Save hospital data partition
        hospital_dir = os.path.join(PROCESSED_DIR, hospital_name)
        os.makedirs(hospital_dir, exist_ok=True)

        np.save(os.path.join(hospital_dir, "train_X.npy"), train_X)
        np.save(os.path.join(hospital_dir, "train_y.npy"), train_y)
        np.save(os.path.join(hospital_dir, "test_X.npy"), test_X)
        np.save(os.path.join(hospital_dir, "test_y.npy"), test_y)
        np.save(os.path.join(hospital_dir, "val_X.npy"), val_X)
        np.save(os.path.join(hospital_dir, "val_y.npy"), val_y)

        label_dist = dict(Counter(all_labels.tolist()))
        stats[hospital_name] = {
            "total_windows": n,
            "train": len(train_X),
            "test": len(test_X),
            "val": len(val_X),
            "class_distribution": label_dist,
        }

        print(f"  Total windows: {n} | Train: {len(train_X)} | Test: {len(test_X)} | Val: {len(val_X)}")
        print(f"  Class distribution: {label_dist}")

    # Create server validation set by combining val splits from all hospitals
    server_val_X = []
    server_val_y = []
    for hospital_name in HOSPITAL_SPLITS:
        hospital_dir = os.path.join(PROCESSED_DIR, hospital_name)
        val_path_X = os.path.join(hospital_dir, "val_X.npy")
        val_path_y = os.path.join(hospital_dir, "val_y.npy")
        if os.path.exists(val_path_X):
            server_val_X.append(np.load(val_path_X))
            server_val_y.append(np.load(val_path_y))

    if server_val_X:
        server_val_X = np.concatenate(server_val_X)
        server_val_y = np.concatenate(server_val_y)
        np.save(os.path.join(PROCESSED_DIR, "server_val_X.npy"), server_val_X)
        np.save(os.path.join(PROCESSED_DIR, "server_val_y.npy"), server_val_y)
        print(f"\n[INFO] Server validation set: {len(server_val_X)} samples")

    # Create a small seed dataset for pretraining (10% from all data, balanced classes)
    _create_seed_dataset()

    # Save stats
    with open(os.path.join(PROCESSED_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n[INFO] Data processing and splitting complete!")
    return stats


def _create_seed_dataset():
    """Create a small balanced seed dataset (10% of total) for pretraining."""
    all_X = []
    all_y = []

    for hospital_name in HOSPITAL_SPLITS:
        hospital_dir = os.path.join(PROCESSED_DIR, hospital_name)
        train_X_path = os.path.join(hospital_dir, "train_X.npy")
        train_y_path = os.path.join(hospital_dir, "train_y.npy")
        if os.path.exists(train_X_path):
            all_X.append(np.load(train_X_path))
            all_y.append(np.load(train_y_path))

    if not all_X:
        return

    all_X = np.concatenate(all_X)
    all_y = np.concatenate(all_y)

    # Sample 10% equally from each class
    seed_X = []
    seed_y = []
    rng = np.random.RandomState(42)

    for class_idx in range(5):
        class_mask = all_y == class_idx
        class_X = all_X[class_mask]
        class_y = all_y[class_mask]

        if len(class_X) == 0:
            continue

        n_samples = max(1, int(0.1 * len(class_X)))
        indices = rng.choice(len(class_X), size=n_samples, replace=False)
        seed_X.append(class_X[indices])
        seed_y.append(class_y[indices])

    seed_X = np.concatenate(seed_X)
    seed_y = np.concatenate(seed_y)

    # Shuffle
    indices = rng.permutation(len(seed_X))
    seed_X = seed_X[indices]
    seed_y = seed_y[indices]

    np.save(os.path.join(PROCESSED_DIR, "seed_X.npy"), seed_X)
    np.save(os.path.join(PROCESSED_DIR, "seed_y.npy"), seed_y)
    print(f"[INFO] Seed dataset created: {len(seed_X)} samples")


def check_data_ready() -> bool:
    """Check if processed data already exists."""
    if not os.path.exists(PROCESSED_DIR):
        return False
    for hospital_name in HOSPITAL_SPLITS:
        hospital_dir = os.path.join(PROCESSED_DIR, hospital_name)
        if not os.path.exists(os.path.join(hospital_dir, "train_X.npy")):
            return False
    return True


def prepare_data():
    """Full pipeline: download + process + split."""
    if check_data_ready():
        print("[INFO] Processed data already exists. Skipping preparation.")
        return
    download_mitbih()
    process_and_split()


if __name__ == "__main__":
    prepare_data()
