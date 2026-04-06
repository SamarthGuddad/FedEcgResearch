"""
Central Federation Server — Flask REST API.

Manages the global model, receives client weight submissions, performs
FedAvg aggregation, and serves metrics for the dashboard.
"""

import os
import sys
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.global_model import (
    get_model, serialize_state_dict, deserialize_state_dict, CLASS_NAMES
)
from server.aggregator import FedAvgAggregator

# ── Configuration ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "global_model_pretrained.pt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_global_model.pt")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEVICE = "cpu"  # Server runs on CPU for portability

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Global State ───────────────────────────────────────────────────────────────
global_model = get_model().to(DEVICE)
aggregator = FedAvgAggregator()

# Metrics history
metrics = {
    "current_round": 0,
    "global_accuracy": [],
    "global_loss": [],
    "client_losses": {},      # client_id → [loss_per_round]
    "client_accuracies": {},  # client_id → [acc_per_round]
    "weight_divergences": {},  # client_id → [divergence_per_round]
    "best_accuracy": 0.0,
    "total_samples_trained": 0,
    "data_distribution": {},  # hospital → {class_idx: count}
}


def load_pretrained_model():
    """Load the pretrained model checkpoint at server startup."""
    global global_model
    if os.path.exists(PRETRAINED_PATH):
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
        global_model.load_state_dict(state_dict)
        print(f"[SERVER] Loaded pretrained model from {PRETRAINED_PATH}")
    else:
        print("[SERVER] No pretrained model found. Starting with random weights.")


def load_data_distribution():
    """Load class distribution stats for dashboard."""
    stats_path = os.path.join(PROCESSED_DIR, "data_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        for hospital_name, info in stats.items():
            metrics["data_distribution"][hospital_name] = info.get("class_distribution", {})


def evaluate_global_model():
    """Evaluate the global model on the server's held-out validation set."""
    val_X_path = os.path.join(PROCESSED_DIR, "server_val_X.npy")
    val_y_path = os.path.join(PROCESSED_DIR, "server_val_y.npy")

    if not os.path.exists(val_X_path):
        print("[SERVER] No validation set found. Skipping evaluation.")
        return 0.0, 0.0

    val_X = np.load(val_X_path)
    val_y = np.load(val_y_path)

    X_tensor = torch.from_numpy(val_X).float().unsqueeze(1).to(DEVICE)
    y_tensor = torch.from_numpy(val_y).long().to(DEVICE)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    global_model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = global_model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0

    return accuracy, avg_loss


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/get_model", methods=["GET"])
def get_model_endpoint():
    """Return current global model weights as JSON."""
    global_model.eval()
    state_dict = global_model.state_dict()
    serialized = serialize_state_dict(state_dict)
    return jsonify({
        "status": "ok",
        "round": metrics["current_round"],
        "weights": serialized,
    })


@app.route("/submit_weights", methods=["POST"])
def submit_weights_endpoint():
    """
    Receive client's updated weights after local training.

    Expects JSON body:
    {
        "client_id": str,
        "weights": dict,
        "num_samples": int,
        "local_loss": float,
        "local_accuracy": float
    }
    """
    data = request.get_json()
    client_id = data["client_id"]
    weights = deserialize_state_dict(data["weights"])
    num_samples = data["num_samples"]
    local_loss = data.get("local_loss", 0.0)
    local_accuracy = data.get("local_accuracy", 0.0)

    # Submit to aggregator
    aggregator.submit(client_id, weights, num_samples)

    # Track per-client metrics
    if client_id not in metrics["client_losses"]:
        metrics["client_losses"][client_id] = []
        metrics["client_accuracies"][client_id] = []
    metrics["client_losses"][client_id].append(local_loss)
    metrics["client_accuracies"][client_id].append(local_accuracy)
    metrics["total_samples_trained"] += num_samples

    print(f"[SERVER] Received weights from {client_id} "
          f"(samples={num_samples}, loss={local_loss:.4f}, acc={local_accuracy:.1f}%)")

    return jsonify({
        "status": "ok",
        "message": f"Weights from {client_id} received.",
        "submissions": aggregator.get_num_submissions(),
    })


@app.route("/aggregate", methods=["POST"])
def aggregate_endpoint():
    """
    Trigger FedAvg aggregation across all submitted client weights.
    Evaluates the aggregated model and saves checkpoint if best.
    """
    global global_model

    num_submissions = aggregator.get_num_submissions()
    if num_submissions == 0:
        return jsonify({"status": "error", "message": "No weights submitted."}), 400

    print(f"\n[SERVER] ═══ Starting aggregation (Round {metrics['current_round'] + 1}) ═══")
    print(f"[SERVER] Aggregating weights from {num_submissions} clients: "
          f"{aggregator.get_submitted_clients()}")

    # Compute weight divergence before aggregation
    divergences = aggregator.compute_weight_divergence(global_model.state_dict())
    for client_id, div in divergences.items():
        if client_id not in metrics["weight_divergences"]:
            metrics["weight_divergences"][client_id] = []
        metrics["weight_divergences"][client_id].append(div)

    # Perform FedAvg
    aggregated_weights = aggregator.aggregate(global_model.state_dict())
    global_model.load_state_dict(aggregated_weights)

    # Increment round counter
    metrics["current_round"] += 1

    # Evaluate global model
    accuracy, loss = evaluate_global_model()
    metrics["global_accuracy"].append(accuracy)
    metrics["global_loss"].append(loss)

    # Save best model
    if accuracy > metrics["best_accuracy"]:
        metrics["best_accuracy"] = accuracy
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(global_model.state_dict(), BEST_MODEL_PATH)
        print(f"[SERVER] ★ New best model saved! Accuracy: {accuracy:.2f}%")

    print(f"[SERVER] Round {metrics['current_round']} complete | "
          f"Global Acc: {accuracy:.2f}% | Loss: {loss:.4f}")
    print(f"[SERVER] Weight divergences: {divergences}")

    return jsonify({
        "status": "ok",
        "round": metrics["current_round"],
        "global_accuracy": accuracy,
        "global_loss": loss,
        "best_accuracy": metrics["best_accuracy"],
        "weight_divergences": divergences,
    })


@app.route("/global_metrics", methods=["GET"])
def global_metrics_endpoint():
    """Return full metrics history for the dashboard."""
    return jsonify({
        "status": "ok",
        "current_round": metrics["current_round"],
        "global_accuracy": metrics["global_accuracy"],
        "global_loss": metrics["global_loss"],
        "client_losses": metrics["client_losses"],
        "client_accuracies": metrics["client_accuracies"],
        "weight_divergences": metrics["weight_divergences"],
        "best_accuracy": metrics["best_accuracy"],
        "total_samples_trained": metrics["total_samples_trained"],
        "data_distribution": metrics["data_distribution"],
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Federation server is running."})


@app.route("/", methods=["GET"])
def server_dashboard():
    """Serve a live HTML dashboard for the federation server."""
    from flask import render_template_string
    return render_template_string(SERVER_DASHBOARD_HTML, metrics=metrics)


def create_app():
    """Factory function to create and configure the Flask app."""
    load_pretrained_model()
    load_data_distribution()
    return app


def run_server(host="127.0.0.1", port=5000):
    """Start the federation server."""
    create_app()
    print(f"\n[SERVER] Federation Server starting on http://{host}:{port}")
    print(f"[SERVER] Endpoints: /, /get_model, /submit_weights, /aggregate, /global_metrics")
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ── Server Dashboard HTML ──────────────────────────────────────────────────────
SERVER_DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedECG — Federation Server</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#050508;color:#e0e8f0;font-family:'Inter',sans-serif;min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:20px}
h1{font-size:1.5rem;font-weight:800;background:linear-gradient(90deg,#00ff41,#00d4ff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:4px}
.sub{text-align:center;color:#6b7b8d;font-size:.78rem;margin-bottom:24px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.card{background:#0a0f14;border:1px solid #141c24;border-radius:12px;padding:16px;text-align:center}
.card .label{font-size:.65rem;color:#6b7b8d;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px}
.card .value{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.panel{background:#0a0f14;border:1px solid #141c24;border-radius:12px;padding:16px}
.panel h2{font-size:.85rem;font-weight:700;margin-bottom:12px;color:#94a3b8}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{text-align:left;color:#6b7b8d;padding:6px 8px;border-bottom:1px solid #141c24;font-weight:600;text-transform:uppercase;font-size:.65rem;letter-spacing:1px}
td{padding:6px 8px;border-bottom:1px solid #141c2444;font-family:'JetBrains Mono',monospace}
.bar-track{height:6px;background:#141c24;border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .5s}
.acc-list{list-style:none;display:flex;flex-wrap:wrap;gap:6px}
.acc-list li{font-family:'JetBrains Mono',monospace;font-size:.72rem;padding:3px 10px;background:#141c24;border-radius:6px;border:1px solid #1e2a36}
.endpoint{font-family:'JetBrains Mono',monospace;font-size:.72rem;padding:4px 0;display:flex;justify-content:space-between;border-bottom:1px solid #141c2444}
.endpoint:last-child{border-bottom:none}
.method{padding:1px 6px;border-radius:4px;font-size:.6rem;font-weight:700}
.get{background:#00ff4120;color:#00ff41}.post{background:#ffaa0020;color:#ffaa00}
.refresh-note{text-align:center;color:#2a3a4a;font-size:.65rem;margin-top:12px}
</style>
</head>
<body>
<div class="wrap">
<h1>🖥️ FedECG — Federation Server</h1>
<p class="sub">Central aggregation server · FedAvg · Global model management</p>

<div class="cards" id="cards"></div>

<div class="row">
<div class="panel">
<h2>📈 Accuracy History</h2>
<ul class="acc-list" id="accList"><li style="color:#6b7b8d">Waiting for rounds...</li></ul>
</div>
<div class="panel">
<h2>📉 Client Submissions</h2>
<table><thead><tr><th>Client</th><th>Last Loss</th><th>Last Acc</th><th>Rounds</th></tr></thead>
<tbody id="clientTable"><tr><td colspan="4" style="color:#6b7b8d">No submissions yet</td></tr></tbody>
</table>
</div>
</div>

<div class="row">
<div class="panel">
<h2>📊 Data Distribution (per hospital)</h2>
<div id="distTable"></div>
</div>
<div class="panel">
<h2>🔗 API Endpoints</h2>
<div class="endpoint"><span>GET /</span><span class="method get">GET</span><span>This dashboard</span></div>
<div class="endpoint"><span>GET /get_model</span><span class="method get">GET</span><span>Download global weights</span></div>
<div class="endpoint"><span>POST /submit_weights</span><span class="method post">POST</span><span>Upload client weights</span></div>
<div class="endpoint"><span>POST /aggregate</span><span class="method post">POST</span><span>Trigger FedAvg</span></div>
<div class="endpoint"><span>GET /global_metrics</span><span class="method get">GET</span><span>Full metrics JSON</span></div>
<div class="endpoint"><span>GET /health</span><span class="method get">GET</span><span>Health check</span></div>
</div>
</div>

<p class="refresh-note">Auto-refreshes every 3 seconds</p>
</div>

<script>
const CLASS_NAMES = ['Normal','Atrial Fib','PVC','BBB','Pacemaker'];
async function refresh() {
    try {
        const r = await fetch('/global_metrics');
        const d = await r.json();
        document.getElementById('cards').innerHTML = [
            card('Current Round', d.current_round, '#00d4ff'),
            card('Best Accuracy', d.best_accuracy.toFixed(1)+'%', '#00ff41'),
            card('Total Samples', d.total_samples_trained.toLocaleString(), '#7c3aed'),
            card('Clients', Object.keys(d.client_losses).length, '#ffaa00'),
        ].join('');

        // Accuracy history
        const al = document.getElementById('accList');
        if (d.global_accuracy.length > 0) {
            al.innerHTML = d.global_accuracy.map((a,i) =>
                `<li style="color:${a>80?'#00ff41':a>50?'#ffaa00':'#ff3355'}">R${i+1}: ${a.toFixed(1)}%</li>`
            ).join('');
        }

        // Client table
        const ct = document.getElementById('clientTable');
        const clients = Object.entries(d.client_losses);
        if (clients.length > 0) {
            ct.innerHTML = clients.map(([cid, losses]) => {
                const accs = d.client_accuracies[cid] || [];
                return `<tr><td>${cid.replace(/_/g,' ')}</td><td>${losses[losses.length-1].toFixed(4)}</td>` +
                       `<td>${accs.length?accs[accs.length-1].toFixed(1)+'%':'—'}</td><td>${losses.length}</td></tr>`;
            }).join('');
        }

        // Data distribution
        const dd = document.getElementById('distTable');
        const dist = d.data_distribution;
        if (Object.keys(dist).length > 0) {
            dd.innerHTML = '<table><thead><tr><th>Hospital</th>' +
                CLASS_NAMES.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
                Object.entries(dist).map(([h, cd]) =>
                    `<tr><td>${h.replace(/_/g,' ')}</td>` +
                    [0,1,2,3,4].map(i => `<td>${(cd[String(i)]||0).toLocaleString()}</td>`).join('') + '</tr>'
                ).join('') + '</tbody></table>';
        }
    } catch(e) {}
}
function card(label, value, color) {
    return `<div class="card"><div class="label">${label}</div><div class="value" style="color:${color}">${value}</div></div>`;
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    run_server()
