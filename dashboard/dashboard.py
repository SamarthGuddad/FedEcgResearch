"""
Live Dashboard for Federated Learning Monitoring.

Plotly Dash application that polls the federation server's /global_metrics
endpoint every 5 seconds and visualizes:
- Global accuracy per federation round
- Per-client local loss per round
- Data distribution per hospital (class balance)
- Current round, best accuracy, total samples trained
"""

import os
import sys
import json
import requests
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px


SERVER_URL = "http://127.0.0.1:5000"

# Class names for display
CLASS_NAMES = ["Normal (N)", "Atrial Fib (A)", "PVC (V)", "BBB (L/R)", "Pacemaker (P)"]

# Color palette
COLORS = {
    "bg": "#0f0f1a",
    "card": "#1a1a2e",
    "accent": "#00d4ff",
    "accent2": "#7c3aed",
    "accent3": "#f43f5e",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "grid": "#1e293b",
    "client1": "#00d4ff",
    "client2": "#7c3aed",
    "client3": "#f43f5e",
}

HOSPITAL_COLORS = {
    "hospital_1": "#00ff41",  # Match monitor colors
    "hospital_2": "#00d4ff",
    "hospital_3": "#ff6b9d",
}

HOSPITAL_NAMES = {
    "hospital_1": "City General Hospital",
    "hospital_2": "Metro Heart Center",
    "hospital_3": "Pacific Medical Institute",
    "Client_1_Hospital_A": "City General Hospital",
    "Client_2_Hospital_B": "Metro Heart Center",
    "Client_3_Hospital_C": "Pacific Medical Institute",
}



def create_dashboard():
    """Create and configure the Dash dashboard application."""

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="FedECG — Federated Learning Dashboard",
        suppress_callback_exceptions=True,
    )

    # ── Layout ─────────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "fontFamily": "'Inter', 'Segoe UI', sans-serif",
            "color": COLORS["text"],
            "padding": "24px",
        },
        children=[
            # Header
            html.Div(
                style={
                    "textAlign": "center",
                    "marginBottom": "32px",
                    "padding": "24px",
                    "background": f"linear-gradient(135deg, {COLORS['card']} 0%, #16213e 100%)",
                    "borderRadius": "16px",
                    "border": f"1px solid {COLORS['accent']}33",
                    "boxShadow": f"0 0 30px {COLORS['accent']}15",
                },
                children=[
                    html.H1(
                        "🫀 FedECG — Federated Learning Dashboard",
                        style={
                            "background": f"linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent2']})",
                            "WebkitBackgroundClip": "text",
                            "WebkitTextFillColor": "transparent",
                            "fontSize": "2rem",
                            "fontWeight": "800",
                            "marginBottom": "8px",
                        },
                    ),
                    html.P(
                        "Real-time monitoring of ECG arrhythmia detection across federated hospital nodes",
                        style={"color": COLORS["text_muted"], "fontSize": "1rem", "margin": "0"},
                    ),
                ],
            ),

            # Status Cards Row
            html.Div(
                id="status-cards",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "16px",
                    "marginBottom": "24px",
                },
                children=[
                    _status_card("Current Round", "0", "🔄", COLORS["accent"]),
                    _status_card("Best Accuracy", "— %", "🏆", COLORS["success"]),
                    _status_card("Total Samples", "0", "📊", COLORS["accent2"]),
                    _status_card("Active Clients", "0", "🏥", COLORS["warning"]),
                ],
            ),

            # Charts Row 1: Accuracy + Loss
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
                children=[
                    _chart_card("accuracy-chart", "Global Accuracy per Round"),
                    _chart_card("loss-chart", "Per-Client Loss per Round"),
                ],
            ),

            # Charts Row 2: Distribution + Divergence
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
                children=[
                    _chart_card("distribution-chart", "Data Distribution per Hospital"),
                    _chart_card("divergence-chart", "Weight Divergence per Client"),
                ],
            ),

            # Auto-refresh interval
            dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
        ],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────

    @app.callback(
        [
            Output("status-cards", "children"),
            Output("accuracy-chart", "figure"),
            Output("loss-chart", "figure"),
            Output("distribution-chart", "figure"),
            Output("divergence-chart", "figure"),
        ],
        [Input("refresh-interval", "n_intervals")],
    )
    def update_dashboard(n_intervals):
        """Poll server metrics and update all dashboard components."""
        try:
            response = requests.get(f"{SERVER_URL}/global_metrics", timeout=5)
            data = response.json()
        except Exception:
            # Return empty/default state if server isn't available
            return (
                _default_status_cards(),
                _empty_figure("Global Accuracy"),
                _empty_figure("Client Loss"),
                _empty_figure("Data Distribution"),
                _empty_figure("Weight Divergence"),
            )

        current_round = data.get("current_round", 0)
        global_accuracy = data.get("global_accuracy", [])
        global_loss = data.get("global_loss", [])
        client_losses = data.get("client_losses", {})
        client_accuracies = data.get("client_accuracies", {})
        weight_divergences = data.get("weight_divergences", {})
        best_accuracy = data.get("best_accuracy", 0.0)
        total_samples = data.get("total_samples_trained", 0)
        data_distribution = data.get("data_distribution", {})

        # ── Status Cards ───────────────────────────────────────────────
        status_cards = [
            _status_card("Current Round", f"{current_round}", "🔄", COLORS["accent"]),
            _status_card("Best Accuracy", f"{best_accuracy:.1f}%", "🏆", COLORS["success"]),
            _status_card("Total Samples", f"{total_samples:,}", "📊", COLORS["accent2"]),
            _status_card("Active Clients", f"{len(client_losses)}", "🏥", COLORS["warning"]),
        ]

        # ── Accuracy Chart ─────────────────────────────────────────────
        acc_fig = go.Figure()
        if global_accuracy:
            rounds = list(range(1, len(global_accuracy) + 1))
            acc_fig.add_trace(go.Scatter(
                x=rounds,
                y=global_accuracy,
                mode="lines+markers",
                name="Global Accuracy",
                line=dict(color=COLORS["accent"], width=3),
                marker=dict(size=8, color=COLORS["accent"]),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.08)",
            ))
        acc_fig.update_layout(
            **_chart_layout("Global Accuracy (%)"),
            yaxis=dict(
                title="Accuracy (%)",
                range=[0, 100],
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
            ),
            xaxis=dict(
                title="Federation Round",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
                dtick=1,
            ),
        )

        # ── Loss Chart ─────────────────────────────────────────────────
        loss_fig = go.Figure()
        client_colors = [COLORS["client1"], COLORS["client2"], COLORS["client3"]]
        for i, (client_id, losses) in enumerate(client_losses.items()):
            short_name = HOSPITAL_NAMES.get(client_id, client_id.replace("_", " "))
            rounds = list(range(1, len(losses) + 1))
            color = list(HOSPITAL_COLORS.values())[i % len(HOSPITAL_COLORS)]
            loss_fig.add_trace(go.Scatter(
                x=rounds,
                y=losses,
                mode="lines+markers",
                name=short_name,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
            ))
        loss_fig.update_layout(
            **_chart_layout("Training Loss"),
            yaxis=dict(
                title="Loss",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
            ),
            xaxis=dict(
                title="Federation Round",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
                dtick=1,
            ),
        )

        # ── Distribution Chart ─────────────────────────────────────────
        dist_fig = go.Figure()
        if data_distribution:
            for hospital, class_dist in data_distribution.items():
                counts = [class_dist.get(str(i), 0) for i in range(5)]
                color = HOSPITAL_COLORS.get(hospital, "#ffffff")
                display_name = HOSPITAL_NAMES.get(hospital, hospital.replace("_", " ").title())
                dist_fig.add_trace(go.Bar(
                    x=CLASS_NAMES,
                    y=counts,
                    name=display_name,
                    marker_color=color,
                    opacity=0.85,
                ))
        dist_fig.update_layout(
            **_chart_layout("Samples per Class"),
            barmode="group",
            yaxis=dict(
                title="Count",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
            ),
            xaxis=dict(color=COLORS["text_muted"]),
        )

        # ── Divergence Chart ───────────────────────────────────────────
        div_fig = go.Figure()
        for i, (client_id, divs) in enumerate(weight_divergences.items()):
            short_name = HOSPITAL_NAMES.get(client_id, client_id.replace("_", " "))
            rounds = list(range(1, len(divs) + 1))
            color = list(HOSPITAL_COLORS.values())[i % len(HOSPITAL_COLORS)]
            div_fig.add_trace(go.Scatter(
                x=rounds,
                y=divs,
                mode="lines+markers",
                name=short_name,
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=6, color=color),
            ))
        div_fig.update_layout(
            **_chart_layout("Weight Divergence"),
            yaxis=dict(
                title="L2 Divergence",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
            ),
            xaxis=dict(
                title="Federation Round",
                gridcolor=COLORS["grid"],
                color=COLORS["text_muted"],
                dtick=1,
            ),
        )

        return status_cards, acc_fig, loss_fig, dist_fig, div_fig

    return app


# ── Helper Functions ───────────────────────────────────────────────────────────

def _status_card(title: str, value: str, icon: str, color: str):
    """Create a status metric card."""
    return html.Div(
        style={
            "background": f"linear-gradient(135deg, {COLORS['card']} 0%, #16213e 100%)",
            "borderRadius": "12px",
            "padding": "20px",
            "border": f"1px solid {color}33",
            "boxShadow": f"0 4px 16px {color}10",
            "transition": "transform 0.2s ease",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Div([
                        html.P(title, style={
                            "color": COLORS["text_muted"],
                            "fontSize": "0.85rem",
                            "margin": "0 0 4px 0",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                        }),
                        html.H2(value, style={
                            "color": color,
                            "fontSize": "1.5rem",
                            "fontWeight": "700",
                            "margin": "0",
                        }),
                    ]),
                    html.Span(icon, style={"fontSize": "2rem"}),
                ],
            ),
        ],
    )


def _chart_card(chart_id: str, title: str):
    """Create a card wrapper for a chart."""
    return html.Div(
        style={
            "background": f"linear-gradient(135deg, {COLORS['card']} 0%, #16213e 100%)",
            "borderRadius": "12px",
            "padding": "20px",
            "border": f"1px solid {COLORS['accent']}22",
        },
        children=[
            html.H3(title, style={
                "color": COLORS["text"],
                "fontSize": "1rem",
                "fontWeight": "600",
                "marginBottom": "12px",
            }),
            dcc.Graph(
                id=chart_id,
                style={"height": "350px"},
                config={"displayModeBar": False},
            ),
        ],
    )


def _chart_layout(title: str) -> dict:
    """Common chart layout configuration."""
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color=COLORS["text_muted"], family="Inter, sans-serif"),
        "margin": dict(l=50, r=20, t=30, b=50),
        "legend": dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_muted"], size=11),
        ),
    }


def _empty_figure(title: str) -> go.Figure:
    """Create an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text="Waiting for data...",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=COLORS["text_muted"], size=14),
    )
    fig.update_layout(**_chart_layout(title))
    return fig


def _default_status_cards():
    """Return default status cards when server is unavailable."""
    return [
        _status_card("Current Round", "—", "🔄", COLORS["accent"]),
        _status_card("Best Accuracy", "— %", "🏆", COLORS["success"]),
        _status_card("Total Samples", "—", "📊", COLORS["accent2"]),
        _status_card("Active Clients", "—", "🏥", COLORS["warning"]),
    ]


if __name__ == "__main__":
    app = create_dashboard()
    print("[DASHBOARD] Starting at http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=8050, debug=True)
