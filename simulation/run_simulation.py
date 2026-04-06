"""
Federation Simulation Orchestrator.

Orchestrates the complete federated learning pipeline:
1. Downloads and prepares MIT-BIH data
2. Pretrains the global model on seed data
3. Starts the Flask federation server
4. Runs 10 federation rounds with 3 parallel hospital clients
5. Launches the live dashboard
6. Reports final results
"""

import os
import sys
import time
import threading
import requests
import webbrowser

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def wait_for_server(url: str = "http://127.0.0.1:5000", timeout: int = 30):
    """Wait until the Flask server is responding."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                print("[SIM] Server is ready!")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    print("[SIM] ERROR: Server didn't start within timeout.")
    return False


def run_simulation():
    """Main simulation entry point."""
    print("=" * 70)
    print("   FEDERATED LEARNING FOR ECG ARRHYTHMIA DETECTION")
    print("   Simulation Orchestrator")
    print("=" * 70)

    # ── Step 1: Prepare Data ───────────────────────────────────────────────
    print("\n[SIM] ══ STEP 1: Data Preparation ══")
    from data.download_data import prepare_data
    prepare_data()

    # ── Step 2: Pretrain Global Model ──────────────────────────────────────
    print("\n[SIM] ══ STEP 2: Pretraining Global Model ══")
    from server.pretrain import pretrain
    pretrain()

    # ── Step 3: Start Flask Server ─────────────────────────────────────────
    print("\n[SIM] ══ STEP 3: Starting Federation Server ══")
    from server.server import create_app, app

    # Create and configure the app (loads pretrained model)
    create_app()

    # Run the server in a background daemon thread
    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    if not wait_for_server():
        print("[SIM] Failed to start server. Exiting.")
        return

    # ── Step 4: Start Dashboard ────────────────────────────────────────────
    print("\n[SIM] ══ STEP 4: Starting Dashboard ══")
    from dashboard.dashboard import create_dashboard

    dash_app = create_dashboard()
    dashboard_thread = threading.Thread(
        target=lambda: dash_app.run(host="127.0.0.1", port=8050, debug=False),
        daemon=True,
    )
    dashboard_thread.start()
    time.sleep(2)

    # Open dashboard in browser
    try:
        webbrowser.open("http://127.0.0.1:8050")
        print("[SIM] Dashboard opened at http://127.0.0.1:8050")
    except Exception:
        print("[SIM] Dashboard running at http://127.0.0.1:8050 (open manually)")

    # ── Step 4b: Start Hospital ECG Monitor ────────────────────────────────
    print("\n[SIM] ══ STEP 4b: Starting Hospital ECG Monitor ══")
    from dashboard.hospital_monitor import create_monitor_app

    monitor_app = create_monitor_app()
    monitor_thread = threading.Thread(
        target=lambda: monitor_app.run(host="127.0.0.1", port=8051, debug=False, use_reloader=False),
        daemon=True,
    )
    monitor_thread.start()
    time.sleep(2)

    try:
        webbrowser.open("http://127.0.0.1:8051")
        print("[SIM] Hospital Monitor opened at http://127.0.0.1:8051")
    except Exception:
        print("[SIM] Hospital Monitor running at http://127.0.0.1:8051 (open manually)")

    # ── Step 5: Run Federation Rounds ──────────────────────────────────────
    print("\n[SIM] ══ STEP 5: Running Federated Learning ══")

    from client.client import FederatedClient

    SERVER_URL = "http://127.0.0.1:5000"
    NUM_ROUNDS = 10
    NUM_CLIENTS = 3

    # Define hospital-client mapping
    client_configs = [
        {"client_id": "Client_1_Hospital_A", "hospital_id": "hospital_1"},
        {"client_id": "Client_2_Hospital_B", "hospital_id": "hospital_2"},
        {"client_id": "Client_3_Hospital_C", "hospital_id": "hospital_3"},
    ]

    # Initialize clients (each loads their private data partition)
    clients = []
    for cfg in client_configs:
        client = FederatedClient(
            client_id=cfg["client_id"],
            hospital_id=cfg["hospital_id"],
            server_url=SERVER_URL,
            local_epochs=1,  # Reduced for faster simulation
            batch_size=256,
            learning_rate=1e-3,
        )
        clients.append(client)

    print(f"\n[SIM] Starting {NUM_ROUNDS} federation rounds with {NUM_CLIENTS} clients...")
    print("-" * 70)

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"  FEDERATION ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'='*70}")

        # Run all clients in parallel threads
        results = [None] * NUM_CLIENTS
        threads = []

        def client_round(idx, client):
            results[idx] = client.run_round()

        for i, client in enumerate(clients):
            t = threading.Thread(target=client_round, args=(i, client))
            threads.append(t)
            t.start()

        # Wait for all clients to finish
        for t in threads:
            t.join()

        # Check results
        all_ok = all(r and r.get("status") == "ok" for r in results)
        if not all_ok:
            print(f"[SIM] WARNING: Some clients failed in round {round_num}")
            for r in results:
                if r and r.get("status") != "ok":
                    print(f"  {r}")

        # Trigger server-side aggregation
        try:
            agg_response = requests.post(f"{SERVER_URL}/aggregate", timeout=120)
            agg_data = agg_response.json()

            global_acc = agg_data.get("global_accuracy", 0)
            global_loss = agg_data.get("global_loss", 0)
            best_acc = agg_data.get("best_accuracy", 0)

            # Print round summary
            loss_strs = []
            for r in results:
                if r and r.get("status") == "ok":
                    loss_strs.append(f"{r['client_id'].split('_')[1]}: {r['loss']:.4f}")

            print(f"\n[Round {round_num}] Global Acc: {global_acc:.1f}% | "
                  f"{'  |  '.join(loss_strs)}")
            print(f"           Global Loss: {global_loss:.4f} | Best Acc: {best_acc:.1f}%")

        except Exception as e:
            print(f"[SIM] Aggregation failed: {e}")

    # ── Step 6: Final Evaluation ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FEDERATION COMPLETE — FINAL RESULTS")
    print(f"{'='*70}")

    try:
        final_metrics = requests.get(f"{SERVER_URL}/global_metrics", timeout=30).json()

        print(f"\n  Total Rounds:          {final_metrics['current_round']}")
        print(f"  Best Global Accuracy:  {final_metrics['best_accuracy']:.2f}%")
        print(f"  Final Global Accuracy: {final_metrics['global_accuracy'][-1]:.2f}%")
        print(f"  Total Samples Trained: {final_metrics['total_samples_trained']:,}")

        print(f"\n  Accuracy progression: ", end="")
        for i, acc in enumerate(final_metrics["global_accuracy"]):
            print(f"R{i+1}:{acc:.1f}%", end="  ")
        print()

        # Per-client final losses
        print("\n  Per-client final losses:")
        for client_id, losses in final_metrics["client_losses"].items():
            short_name = client_id.split("_")[1]
            print(f"    {short_name}: {losses[-1]:.4f} (started at {losses[0]:.4f})")

        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        best_model_path = os.path.join(checkpoint_dir, "best_global_model.pt")
        if os.path.exists(best_model_path):
            size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"\n  Best model saved to: {best_model_path} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"[SIM] Could not fetch final metrics: {e}")

    print(f"\n{'='*70}")
    print("  Federation Dashboard:   http://127.0.0.1:8050")
    print("  Hospital ECG Monitor:   http://127.0.0.1:8051")
    print("  Press Ctrl+C to stop.")
    print(f"{'='*70}")

    # Keep main thread alive for dashboard
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SIM] Shutting down...")


if __name__ == "__main__":
    run_simulation()
