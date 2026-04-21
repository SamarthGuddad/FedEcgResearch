"""
Hospital ECG Monitor — Live Arrhythmia Detection Dashboard.

Auto-detects hospital partitions from the data directory, streams real
MIT-BIH ECG data through the trained federated model, and displays
real-time beat-by-beat arrhythmia predictions with ground truth comparison.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from server.xai import compute_saliency
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.global_model import get_model, CLASS_NAMES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
WINDOW_SIZE = 1800
DEVICE = "cpu"

# ── Aesthetic pools (assigned dynamically to detected hospitals) ───────────────
NAME_POOL = [
    ("City General Hospital", "New York, NY"),
    ("Metro Heart Center", "Chicago, IL"),
    ("Pacific Medical Institute", "San Francisco, CA"),
    ("Regional Cardiac Unit", "Houston, TX"),
    ("University Medical Center", "Boston, MA"),
    ("Sunrise Health Campus", "Phoenix, AZ"),
]
COLOR_POOL = ["#00ff41", "#00d4ff", "#ff6b9d", "#ffaa00", "#7c3aed", "#22d3ee"]

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
CORS(app)

model = None
hospital_registry = {}   # id -> { meta + runtime data }
_model_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)
_saliency_cache = {}


def detect_hospitals():
    """Auto-detect hospital partitions from data/processed directory."""
    found = {}
    if not os.path.exists(PROCESSED_DIR):
        return found

    # Load stats if available
    stats_path = os.path.join(PROCESSED_DIR, "data_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    dirs = sorted(
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
        and os.path.exists(os.path.join(PROCESSED_DIR, d, "test_X.npy"))
    )

    for i, dirname in enumerate(dirs):
        display_name, location = NAME_POOL[i % len(NAME_POOL)]
        color = COLOR_POOL[i % len(COLOR_POOL)]
        class_dist = stats.get(dirname, {}).get("class_distribution", {})

        found[dirname] = {
            "name": display_name,
            "location": location,
            "color": color,
            "total_records": stats.get(dirname, {}).get("total_windows", 0),
            "class_distribution": class_dist,
        }
    return found


def load_resources():
    """Load trained model + hospital data (auto-detected)."""
    global model, hospital_registry
    model = get_model()

    for path in [
        os.path.join(CHECKPOINT_DIR, "best_global_model.pt"),
        os.path.join(CHECKPOINT_DIR, "global_model_pretrained.pt"),
    ]:
        if os.path.exists(path):
            print("[DEBUG] Loading model from:", path)
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"[MONITOR] Model loaded from {os.path.basename(path)}")
            break
    model.eval()

    detected = detect_hospitals()
    for hid, meta in detected.items():
        hdir = os.path.join(PROCESSED_DIR, hid)
        test_X = np.load(os.path.join(hdir, "test_X.npy"))
        test_y = np.load(os.path.join(hdir, "test_y.npy"))

        hospital_registry[hid] = {
            **meta,
            "signal": test_X.flatten(),
            "windows": test_X,
            "labels": test_y,
            "cursor": 0,
            "window_idx": 0,
            "stats": {"total": 0, "normal": 0, "abnormal": 0, "by_class": [0] * 5},
            "alerts": [],
        }
        print(f"[MONITOR] {hid} ({meta['name']}): {len(test_X)} test windows")

    print(f"[MONITOR] Auto-detected {len(hospital_registry)} hospitals")


def _compute_and_cache(hid, model, x):
    with _model_lock:
        s = compute_saliency(model, x)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    _saliency_cache[hid] = s.tolist()

def estimate_bpm(window):
    """Estimate heart rate from simple R-peak detection in the window."""
    threshold = np.mean(window) + 0.8 * np.std(window)
    peaks = []
    for i in range(2, len(window) - 2):
        if (window[i] > threshold
            and window[i] > window[i-1] and window[i] > window[i+1]
            and window[i] > window[i-2] and window[i] > window[i+2]):
            if not peaks or (i - peaks[-1]) > 120:
                peaks.append(i)
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / 360.0
        bpm = int(60.0 / np.mean(intervals))
        return max(40, min(200, bpm))
    return 72


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("monitor.html")


@app.route("/api/hospitals")
def hospital_list():
    """Return all detected hospitals with metadata (used by frontend to build UI)."""
    result = {}
    for hid, h in hospital_registry.items():
        result[hid] = {
            "name": h["name"],
            "location": h["location"],
            "color": h["color"],
            "total_records": h["total_records"],
            "class_distribution": h["class_distribution"],
            "test_windows": len(h["windows"]),
        }
    return jsonify(result)


@app.route("/api/ecg/<hospital_id>")
def ecg_data(hospital_id):
    """Stream the next chunk of real ECG samples."""
    if hospital_id not in hospital_registry:
        return jsonify({"error": "Unknown hospital"}), 404

    h = hospital_registry[hospital_id]
    chunk_size = 72
    signal = h["signal"]
    c = h["cursor"]
    end = c + chunk_size

    if end <= len(signal):
        chunk = signal[c:end].tolist()
    else:
        chunk = np.concatenate([signal[c:], signal[:end % len(signal)]]).tolist()

    h["cursor"] = end % len(signal)
    return jsonify({"points": chunk})


@app.route("/api/predict/<hospital_id>")
def predict(hospital_id):
    if hospital_id not in hospital_registry:
        return jsonify({"error": "Unknown hospital"}), 404

    h = hospital_registry[hospital_id]
    idx = h["window_idx"] % len(h["windows"])
    window = h["windows"][idx]
    true_label = int(h["labels"][idx])
    h["window_idx"] = idx + 1

    x = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0)

    # Lock model during forward pass
    with _model_lock:
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0].numpy()
            pred = int(np.argmax(probs))

    # Submit saliency computation every 5th beat (non-blocking)
    if h["stats"]["total"] % 5 == 0:
        _executor.submit(_compute_and_cache, hospital_id, model, x.clone())

    # Use cached saliency or zeros if not ready yet
    saliency = _saliency_cache.get(hospital_id, [0] * len(window))

    bpm = estimate_bpm(window)
    is_abnormal = pred != 0
    correct = pred == true_label

    h["stats"]["total"] += 1
    if is_abnormal:
        h["stats"]["abnormal"] += 1
    else:
        h["stats"]["normal"] += 1
    h["stats"]["by_class"][pred] += 1

    if is_abnormal:
        h["alerts"].append({
            "type": CLASS_NAMES[pred],
            "true_type": CLASS_NAMES[true_label],
            "confidence": round(float(probs[pred]) * 100, 1),
            "beat": h["stats"]["total"],
            "correct": correct,
            "timestamp": time.strftime("%H:%M:%S"),
        })
        h["alerts"] = h["alerts"][-20:]

    return jsonify({
        "prediction": pred,
        "name": CLASS_NAMES[pred],
        "true_label": true_label,
        "true_name": CLASS_NAMES[true_label],
        "correct": correct,
        "probabilities": [round(float(p) * 100, 1) for p in probs],
        "is_abnormal": is_abnormal,
        "confidence": round(float(probs[pred]) * 100, 1),
        "bpm": bpm,
        "stats": h["stats"],
        "alerts": h["alerts"][-5:],
        "window_index": idx,
        "source": "MIT-BIH Arrhythmia Database (real ECG)",
        "saliency": saliency,   # already a list, no .tolist() needed
        "signal": window.tolist(),
    })


def create_monitor_app():
    load_resources()
    return app


def run_monitor(host="127.0.0.1", port=8051):
    load_resources()
    print(f"\n{'='*60}")
    print(f"  Hospital ECG Monitor — http://{host}:{port}")
    print(f"  Detected {len(hospital_registry)} hospitals")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_monitor()
