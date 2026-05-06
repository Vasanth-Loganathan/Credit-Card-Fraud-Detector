"""
FraudGuard — Credit Card Fraud Detection Dashboard
Fixed version: scalar extraction bug, missing artifact fallback, dark-mode-safe CSS.
Run: streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="FraudGuard | Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DARK-MODE-SAFE CSS
# The key fix: every color that must stay visible uses !important so
# Streamlit's dark-theme override can't touch it.
# =============================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* Global font */
  html, body, [class*="css"] {
      font-family: 'IBM Plex Sans', sans-serif !important;
  }

  /* Hide sidebar */
  section[data-testid="stSidebar"] { display: none !important; }

  /* ── Header ── */
  .fg-header {
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
      padding: 18px 0 16px !important;
      border-bottom: 1px solid #d1d5db !important;
      margin-bottom: 24px !important;
  }
  .fg-logo {
      font-size: 22px !important;
      font-weight: 700 !important;
      letter-spacing: -0.5px !important;
      color: #0f172a !important;
  }
  .fg-logo span { color: #2563eb !important; }
  .fg-subtitle {
      font-size: 12px !important;
      color: #6b7280 !important;
      margin-top: 3px !important;
      font-family: 'IBM Plex Mono', monospace !important;
  }
  .fg-badge {
      background: #dcfce7 !important;
      color: #15803d !important;
      font-size: 11px !important;
      font-weight: 600 !important;
      padding: 5px 12px !important;
      border-radius: 20px !important;
      font-family: 'IBM Plex Mono', monospace !important;
      border: 1px solid #bbf7d0 !important;
  }

  /* ── Section title ── */
  .section-title {
      font-size: 11px !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.09em !important;
      color: #6b7280 !important;
      margin: 18px 0 10px !important;
      border-bottom: 1px solid #e5e7eb !important;
      padding-bottom: 6px !important;
  }

  /* ── Divider ── */
  .fg-divider {
      height: 1px !important;
      background: #e5e7eb !important;
      margin: 18px 0 !important;
  }

  /* ── Risk result cards ── */
  .risk-card {
      border-radius: 10px !important;
      padding: 18px 20px !important;
      margin-bottom: 14px !important;
  }
  .risk-high   { background: #fef2f2 !important; border: 1.5px solid #fca5a5 !important; }
  .risk-medium { background: #fffbeb !important; border: 1.5px solid #fcd34d !important; }
  .risk-low    { background: #f0fdf4 !important; border: 1.5px solid #86efac !important; }

  .risk-title-high   { font-size: 15px !important; font-weight: 700 !important; color: #991b1b !important; }
  .risk-title-medium { font-size: 15px !important; font-weight: 700 !important; color: #92400e !important; }
  .risk-title-low    { font-size: 15px !important; font-weight: 700 !important; color: #166534 !important; }

  .risk-prob-high   { font-size: 34px !important; font-weight: 700 !important; color: #dc2626 !important; font-family: 'IBM Plex Mono', monospace !important; margin: 6px 0 !important; }
  .risk-prob-medium { font-size: 34px !important; font-weight: 700 !important; color: #d97706 !important; font-family: 'IBM Plex Mono', monospace !important; margin: 6px 0 !important; }
  .risk-prob-low    { font-size: 34px !important; font-weight: 700 !important; color: #16a34a !important; font-family: 'IBM Plex Mono', monospace !important; margin: 6px 0 !important; }

  .risk-desc-high   { font-size: 12px !important; color: #b91c1c !important; }
  .risk-desc-medium { font-size: 12px !important; color: #b45309 !important; }
  .risk-desc-low    { font-size: 12px !important; color: #15803d !important; }

  /* ── Decision pill ── */
  .pill-block  { display:inline-block !important; padding:4px 12px !important; border-radius:20px !important; font-size:12px !important; font-weight:600 !important; margin-top:6px !important; }
  .pill-block  { background:#fee2e2 !important; color:#991b1b !important; }
  .pill-approve{ background:#dcfce7 !important; color:#166534 !important; }

  /* ── Signal rows ── */
  .signal-row {
      display: flex !important;
      align-items: center !important;
      gap: 10px !important;
      padding: 7px 0 !important;
      border-bottom: 1px solid #f3f4f6 !important;
      font-size: 13px !important;
  }
  .signal-label { color: #6b7280 !important; width: 80px !important; font-size: 12px !important; flex-shrink:0 !important; }
  .signal-val   { color: #111827 !important; flex:1 !important; font-weight:500 !important; }
  .signal-flag  { font-size: 14px !important; }

  /* ── Leaderboard table ── */
  .lb-table { width:100% !important; border-collapse:collapse !important; font-size:13px !important; }
  .lb-table th {
      background: #f8fafc !important;
      color: #475569 !important;
      font-size: 11px !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.06em !important;
      padding: 9px 14px !important;
      text-align: left !important;
      border-bottom: 2px solid #e2e8f0 !important;
  }
  .lb-table td {
      padding: 9px 14px !important;
      border-bottom: 1px solid #f1f5f9 !important;
      color: #1e293b !important;
  }
  .lb-table tr:hover td { background: #f8fafc !important; }
  .lb-rank1 td { color: #1d4ed8 !important; font-weight: 700 !important; }

  /* ── EDA metric cards ── */
  .eda-cards { display:flex !important; gap:12px !important; flex-wrap:wrap !important; margin-bottom:16px !important; }
  .eda-card {
      flex:1 !important;
      min-width:140px !important;
      background:#f8fafc !important;
      border:1px solid #e2e8f0 !important;
      border-radius:10px !important;
      padding:14px 16px !important;
  }
  .eda-label { font-size:11px !important; color:#6b7280 !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
  .eda-value { font-size:24px !important; font-weight:700 !important; color:#0f172a !important; margin-top:4px !important; font-family:'IBM Plex Mono',monospace !important; }
  .eda-sub   { font-size:11px !important; color:#16a34a !important; margin-top:2px !important; }

  /* ── Footer ── */
  .fg-footer {
      margin-top: 48px !important;
      padding: 14px 0 !important;
      border-top: 1px solid #e5e7eb !important;
      display: flex !important;
      justify-content: space-between !important;
      font-size: 11px !important;
      color: #9ca3af !important;
  }

  /* ── Streamlit tab strip ── */
  .stTabs [data-baseweb="tab-list"] { gap: 4px !important; }
  .stTabs [data-baseweb="tab"] {
      font-size: 13px !important;
      font-weight: 500 !important;
      padding: 10px 18px !important;
  }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

ARTIFACTS_DIR = "Models"   # folder where your .pkl files live

CATEGORY_OPTIONS = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel",
]

CATEGORY_DISPLAY = {
    "entertainment":  "🎭 Entertainment",
    "food_dining":    "🍽️  Food & Dining",
    "gas_transport":  "⛽ Gas & Transport",
    "grocery_net":    "🛒 Grocery (Online)",
    "grocery_pos":    "🛒 Grocery (In-store)",
    "health_fitness": "💪 Health & Fitness",
    "home":           "🏠 Home",
    "kids_pets":      "🐾 Kids & Pets",
    "misc_net":       "💻 Misc (Online)",
    "misc_pos":       "🏪 Misc (In-store)",
    "personal_care":  "✂️  Personal Care",
    "shopping_net":   "🛍️  Shopping (Online)",
    "shopping_pos":   "🛍️  Shopping (In-store)",
    "travel":         "✈️  Travel",
}

RISK_HIGH   = 0.65
RISK_MEDIUM = 0.35

# =============================================================================
# LOAD ARTIFACTS  (cached — loads once per session)
# =============================================================================

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    base = ARTIFACTS_DIR
    try:
        data = {
            "xgb":    joblib.load(f"{base}/xgb_model.pkl"),
            "scaler": joblib.load(f"{base}/scaler.pkl"),
            "meta":   joblib.load(f"{base}/metadata.pkl"),
            "dashboard_metrics": joblib.load(f"{base}/dashboard_metrics.pkl"),
            "shap_data":  joblib.load(f"{base}/shap_data.pkl"),
            "fraud_rules": pd.read_csv(f"{base}/fraud_rules.csv"),
        }
        # ── Optional models — loaded only if the file exists ──
        for key, fname in [
            ("lgbm",     "lgbm_model.pkl"),
            ("rf",       "rf_model.pkl"),
            ("dt",       "dt_model.pkl"),
            ("nn",       "nn_model.pkl"),
            ("ensemble", "ensemble_model.pkl"),
        ]:
            path = f"{base}/{fname}"
            if os.path.exists(path):
                data[key] = joblib.load(path)
        return data
    except Exception as e:
        return {"error": str(e)}


artifacts = load_artifacts()
artifacts_ok = "error" not in artifacts

# =============================================================================
# HELPER — build feature vector
# =============================================================================

def build_feature_vector(amount, category, hour, day_of_week, month,
                         age, distance_km, gender, city_pop, meta):
    """Reproduce the exact preprocessing from the training notebook."""
    freq = meta["freq_medians"]
    feature_names = meta["feature_names"]

    row = {
        "city_pop":      float(city_pop),
        "hour":          int(hour),
        "day_of_week":   int(day_of_week),
        "month":         int(month),
        "age":           int(age),
        "distance_km":   float(distance_km),
        "amt_log":       float(np.log1p(amount)),
        "merchant_freq": float(freq["merchant_freq"]),
        "job_freq":      float(freq["job_freq"]),
        "city_freq":     float(freq["city_freq"]),
    }

    # One-hot category  (drop_first=True in training → 'entertainment' = all zeros)
    for col in meta["category_columns"]:
        row[col] = bool(category == col.replace("category_", ""))

    row["gender_M"] = bool(gender == "Male")

    df = pd.DataFrame([row])[feature_names]
    scaled = artifacts["scaler"].transform(df)
    return pd.DataFrame(scaled, columns=feature_names)


# =============================================================================
# HELPER — predict  (BUG FIX: scalar extraction with [0])
# =============================================================================

def predict_fraud(feature_df, model_name="XGBoost"):
    """
    Returns (prob: float, pred: int).

    THE FIX for 'only 0-dimensional arrays can be converted to Python scalars':
      predict_proba(...) shape is (n_samples, 2).
      [:, 1]  gives a 1-element 1-D array  e.g.  array([0.87])
      float() cannot convert a 1-D array — you need   [:, 1][0]   first.
    """
    model_map = {
        "XGBoost":        artifacts.get("xgb"),
        "LightGBM":       artifacts.get("lgbm"),
        "Random Forest":  artifacts.get("rf"),
        "Decision Tree":  artifacts.get("dt"),
        "Neural Network": artifacts.get("nn"),
        "Ensemble":       artifacts.get("ensemble"),
    }

    model = model_map.get(model_name) or artifacts.get("xgb")   # fallback to XGBoost

    if model is None:
        st.error(f"Model '{model_name}' not found in artifacts. Falling back to XGBoost.")
        model = artifacts["xgb"]

    if model_name == "Neural Network" and hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        # Keras / TF model — returns shape (n, 1)
        raw = model.predict(feature_df)          # shape (1, 1)
        prob = float(np.ravel(raw)[0])           # ← scalar
    else:
        # scikit-learn / XGBoost / LightGBM
        proba = model.predict_proba(feature_df)  # shape (1, 2)
        prob  = float(proba[:, 1][0])            # ← scalar  (THE FIX)

    threshold = float(artifacts["meta"]["optimal_threshold"])
    pred = int(prob >= threshold)
    return prob, pred


# =============================================================================
# HELPER — SHAP for a single transaction
# =============================================================================

def get_shap_explanation(feature_df):
    explainer  = shap.TreeExplainer(artifacts["xgb"])
    shap_vals  = explainer.shap_values(feature_df)

    # shap_vals shape varies between SHAP versions:
    # older  → 2-D array (n_samples, n_features)   → shap_vals[0]
    # newer  → 1-D array (n_features,)             → shap_vals  already flat
    if shap_vals.ndim == 2:
        sv = shap_vals[0]
    else:
        sv = shap_vals

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[0] if hasattr(base, '__len__') else base)
    return sv, float(base)


# =============================================================================
# HELPER — risk label
# =============================================================================

def risk_label(prob):
    if prob >= RISK_HIGH:
        return "HIGH",   "high"
    elif prob >= RISK_MEDIUM:
        return "MEDIUM", "medium"
    else:
        return "LOW",    "low"


def risk_card_html(prob, pred, threshold):
    label, level = risk_label(prob)
    icons = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
    icon  = icons[label]
    action = "⛔ BLOCKED" if pred == 1 else "✅ APPROVED"
    pill_cls = "pill-block" if pred == 1 else "pill-approve"

    return f"""
    <div class="risk-card risk-{level.lower()}">
      <div class="risk-title-{level.lower()}">{icon}&nbsp; Fraud Risk: {label}</div>
      <div class="risk-prob-{level.lower()}">{prob*100:.1f}%</div>
      <div class="risk-desc-{level.lower()}">Threshold: {threshold:.3f}</div>
      <span class="{pill_cls}">{action}</span>
    </div>
    """


# =============================================================================
# PLOTLY theme helper  (white background, consistent font)
# =============================================================================

PLOT_LAYOUT = dict(
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    font          = dict(family="IBM Plex Sans", color="#1e293b"),
    margin        = dict(l=10, r=10, t=40, b=10),
)

GRID = dict(showgrid=True, gridcolor="#f1f5f9")


# =============================================================================
# ─── HEADER ───────────────────────────────────────────────────────────────────
# =============================================================================

st.markdown("""
<div class="fg-header">
  <div>
    <div class="fg-logo">Fraud<span>Guard</span></div>
    <div class="fg-subtitle">Credit Card Fraud Detection · Sparkov Dataset · XGBoost AUPRC 0.9506</div>
  </div>
  <div class="fg-badge">● LIVE MODEL</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ─── TABS ─────────────────────────────────────────────────────────────────────
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Transaction Checker",
    "📊  Model Comparison",
    "🔗  Fraud Patterns",
    "📈  Dataset EDA",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE TRANSACTION CHECKER
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if not artifacts_ok:
        st.error(f"Could not load model artifacts: {artifacts.get('error')}\n\n"
                 "Run the Phase 2 notebook cells first to generate the `Models/` folder.")
        st.stop()

    meta = artifacts["meta"]

    # Determine which models are available for the dropdown
    available_models = ["XGBoost"]
    if artifacts.get("lgbm"):     available_models.append("LightGBM")
    if artifacts.get("rf"):       available_models.append("Random Forest")
    if artifacts.get("dt"):       available_models.append("Decision Tree")
    if artifacts.get("nn"):       available_models.append("Neural Network")
    if artifacts.get("ensemble"): available_models.append("Ensemble")

    st.markdown('<div class="section-title">Transaction details</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    # ── INPUT FORM ──────────────────────────────────────────────────────────
    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input(
                "Transaction amount ($)",
                min_value=0.01, max_value=50000.0,
                value=249.99, step=10.0, format="%.2f",
            )
        with c2:
            category_disp = st.selectbox(
                "Merchant category",
                options=CATEGORY_OPTIONS,
                index=CATEGORY_OPTIONS.index("shopping_net"),
                format_func=lambda x: CATEGORY_DISPLAY.get(x, x),
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            hour = st.slider("Hour of day", 0, 23, 21,
                             help="0 = midnight · 12 = noon · 21 = 9 PM")
        with c4:
            day_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_of_week = st.selectbox(
                "Day of week", options=list(range(7)),
                index=4, format_func=lambda x: day_map[x],
            )
        with c5:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            month = st.selectbox(
                "Month", options=list(range(1, 13)),
                index=0, format_func=lambda x: month_names[x-1],
            )

        c6, c7, c8 = st.columns(3)
        with c6:
            age = st.number_input("Cardholder age", min_value=18, max_value=100, value=67)
        with c7:
            distance_km = st.number_input(
                "Distance to merchant (km)",
                min_value=0.0, max_value=5000.0, value=312.5, step=10.0,
            )
        with c8:
            city_pop = st.number_input(
                "City population", min_value=100,
                max_value=5_000_000, value=45000, step=1000,
            )

        c9, c10 = st.columns(2)
        with c9:
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        with c10:
            model_choice = st.selectbox(
                "Model to use", options=available_models, index=0,
                help="XGBoost has the highest AUPRC (0.9506) on the test set.",
            )

        st.markdown('<div class="fg-divider"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Transaction →",
                                use_container_width=True, type="primary")

    # ── RESULT PANEL ────────────────────────────────────────────────────────
    with col_result:
        if analyze_btn:
            with st.spinner("Running inference…"):
                feat_df = build_feature_vector(
                    amount, category_disp, hour, day_of_week, month,
                    age, distance_km, gender, city_pop, meta,
                )
                prob, pred = predict_fraud(feat_df, model_choice)

            # Risk card
            st.markdown(
                risk_card_html(prob, pred, meta["optimal_threshold"]),
                unsafe_allow_html=True,
            )

            # Key risk signals
            st.markdown('<div class="section-title">Key risk signals</div>',
                        unsafe_allow_html=True)

            signals = [
                ("Amount",   f"${amount:,.2f}",
                 "🔴" if amount > 500 else "🟢"),
                ("Category", CATEGORY_DISPLAY.get(category_disp, category_disp),
                 "🔴" if category_disp in ["shopping_net", "misc_net", "travel"] else "🟢"),
                ("Hour",     f"{hour:02d}:00",
                 "🔴" if (hour >= 20 or hour <= 5) else "🟢"),
                ("Distance", f"{distance_km:.0f} km",
                 "🔴" if distance_km > 200 else "🟢"),
                ("Age",      f"{age} yrs",
                 "🔴" if age >= 65 else "🟢"),
            ]

            rows_html = "".join(
                f'<div class="signal-row">'
                f'<span class="signal-label">{n}</span>'
                f'<span class="signal-val">{v}</span>'
                f'<span class="signal-flag">{f}</span>'
                f'</div>'
                for n, v, f in signals
            )
            st.markdown(rows_html, unsafe_allow_html=True)

            # SHAP waterfall
            st.markdown('<div class="section-title">Why? — SHAP explanation</div>',
                        unsafe_allow_html=True)
            with st.spinner("Computing SHAP values…"):
                sv, base_val = get_shap_explanation(feat_df)

            fig_shap, _ = plt.subplots(figsize=(6, 4.5))
            plt.style.use("default")
            shap.waterfall_plot(
                shap.Explanation(
                    values        = sv,
                    base_values   = base_val,
                    data          = feat_df.iloc[0].values,
                    feature_names = list(feat_df.columns),
                ),
                max_display = 10,
                show        = False,
            )
            plt.gca().set_facecolor("white")
            fig_shap.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

        else:
            st.info("Fill in the transaction details and click **Analyze Transaction →**")
            st.markdown("""
**How this works:**
- Transaction is preprocessed identically to training data (log-amount, Haversine distance, frequency encoding, one-hot, StandardScaler)
- Model outputs a fraud probability score
- The optimized decision threshold (from PR curve) determines **BLOCK / APPROVE**
- SHAP explains which features pushed the score up or down
            """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    if not artifacts_ok:
        st.error("Artifacts not loaded.")
        st.stop()

    metrics = artifacts["dashboard_metrics"]
    meta    = artifacts["meta"]
    colors  = px.colors.qualitative.Safe

    st.markdown('<div class="section-title">Model leaderboard — test set</div>',
                unsafe_allow_html=True)

    sorted_m = sorted(metrics, key=lambda x: x["auprc"], reverse=True)

    # ── HTML leaderboard table ──
    rows_html = ""
    for i, m in enumerate(sorted_m):
        tr_cls = 'class="lb-rank1"' if i == 0 else ""
        rows_html += (
            f'<tr {tr_cls}>'
            f'<td>{i+1}</td>'
            f'<td>{m["name"]}</td>'
            f'<td>{m["auprc"]:.4f}</td>'
            f'<td>{m["recall"]:.4f}</td>'
            f'<td>{m["precision"]:.4f}</td>'
            f'<td>{m["f1"]:.4f}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <table class="lb-table">
      <thead><tr>
        <th>#</th><th>Model</th><th>AUPRC ↓</th>
        <th>Recall</th><th>Precision</th><th>F1</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.caption(f"XGBoost optimal threshold: **{meta['optimal_threshold']:.4f}** (PR-curve optimized)")
    st.markdown('<div class="fg-divider"></div>', unsafe_allow_html=True)

    # ── Charts row ──
    col_bar, col_scat = st.columns(2)

    with col_bar:
        asc = sorted(metrics, key=lambda x: x["auprc"])
        fig_bar = go.Figure(go.Bar(
            x=[m["auprc"] for m in asc],
            y=[m["name"]  for m in asc],
            orientation="h",
            marker=dict(
                color=[m["auprc"] for m in asc],
                colorscale=[[0,"#bfdbfe"],[1,"#1d4ed8"]],
                showscale=False,
            ),
            text=[f'{m["auprc"]:.4f}' for m in asc],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="AUPRC comparison",
            xaxis=dict(range=[0,1.12], **GRID),
            yaxis=dict(showgrid=False),
            height=320, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scat:
        fig_sc = go.Figure()
        for i, m in enumerate(metrics):
            fig_sc.add_trace(go.Scatter(
                x=[m["recall"]], y=[m["precision"]],
                mode="markers+text",
                name=m["name"],
                text=[m["name"]],
                textposition="top center",
                textfont=dict(size=10, color="#1e293b"),
                marker=dict(size=14, color=colors[i % len(colors)]),
            ))
        fig_sc.update_layout(
            title="Precision vs Recall",
            xaxis=dict(title="Recall",    range=[0,1.1], **GRID),
            yaxis=dict(title="Precision", range=[0,1.1], **GRID),
            showlegend=False, height=320, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Precision-Recall curves ──
    st.markdown('<div class="section-title">Precision-Recall curves</div>',
                unsafe_allow_html=True)

    fig_pr = go.Figure()
    for i, m in enumerate(metrics):
        if m.get("pr_precisions") and m.get("pr_recalls"):
            fig_pr.add_trace(go.Scatter(
                x=m["pr_recalls"], y=m["pr_precisions"],
                mode="lines",
                name=f'{m["name"]} ({m["auprc"]:.4f})',
                line=dict(color=colors[i % len(colors)], width=2),
            ))
    fig_pr.update_layout(
        xaxis=dict(title="Recall",    range=[0,1], **GRID),
        yaxis=dict(title="Precision", range=[0,1], **GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        height=380, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── SHAP global bar ──
    st.markdown('<div class="section-title">Global feature importance — SHAP (XGBoost)</div>',
                unsafe_allow_html=True)

    shap_data = artifacts["shap_data"]
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    plt.style.use("default")
    shap.summary_plot(
        shap_data["shap_values"],
        shap_data["X_sample"],
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.gca().set_facecolor("white")
    fig_s.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FRAUD PATTERN EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    if not artifacts_ok:
        st.error("Artifacts not loaded.")
        st.stop()

    fraud_rules = artifacts["fraud_rules"]

    st.markdown("""
    Association rules discovered by **FP-Growth** on fraud transactions.  
    *"When these conditions hold → fraud is X times more likely than the baseline rate."*
    """)

    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_lift = st.slider("Min lift", 1.0, 12.0, 3.0, 0.5,
                             help="Lift > 1 = rule beats random guessing")
    with fc2:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.30, 0.05,
                             help="P(fraud | antecedents)")
    with fc3:
        min_sup  = st.slider("Min support", 0.005, 0.10, 0.01, 0.005,
                             help="Fraction of training rows matching the rule")

    filtered = (
        fraud_rules[
            (fraud_rules["lift"]       >= min_lift) &
            (fraud_rules["confidence"] >= min_conf) &
            (fraud_rules["support"]    >= min_sup)
        ]
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
    )

    st.caption(f"**{len(filtered)}** rules shown (of {len(fraud_rules)} total)")

    if filtered.empty:
        st.warning("No rules match. Try lowering the filter sliders.")
    else:
        # Bubble chart
        fig_bub = go.Figure(go.Scatter(
            x=filtered["confidence"],
            y=filtered["lift"],
            mode="markers",
            marker=dict(
                size      = (filtered["support"] * 800).clip(upper=60),
                color     = filtered["lift"],
                colorscale= [[0,"#bfdbfe"],[0.5,"#3b82f6"],[1,"#1e3a8a"]],
                showscale = True,
                colorbar  = dict(title="Lift", thickness=12),
                sizemode  = "area",
                sizemin   = 6,
            ),
            text=[
                f"Lift: {r['lift']:.2f}<br>"
                f"Conf: {r['confidence']:.2f}<br>"
                f"Sup: {r['support']:.3f}<br>"
                f"{str(r['antecedents'])[:70]}"
                for _, r in filtered.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_bub.update_layout(
            title="Fraud rules — Confidence vs Lift (bubble = support)",
            xaxis=dict(title="Confidence", **GRID),
            yaxis=dict(title="Lift",       **GRID),
            height=360, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_bub, use_container_width=True)

        # Table
        st.markdown('<div class="section-title">Rule details</div>', unsafe_allow_html=True)
        disp = filtered[["antecedents","support","confidence","lift"]].copy()
        disp["support"]    = disp["support"].map(lambda x: f"{x:.4f}")
        disp["confidence"] = disp["confidence"].map(lambda x: f"{x:.4f}")
        disp["lift"]       = disp["lift"].map(lambda x: f"{x:.2f}")
        disp.columns       = ["Antecedents (conditions)", "Support", "Confidence", "Lift"]
        disp.index         = range(1, len(disp)+1)
        st.dataframe(disp, use_container_width=True, height=340)

        # Top rule callout
        top = filtered.iloc[0]
        st.info(
            f"**Top rule — Lift {float(top['lift']):.2f}×** · "
            f"When `{str(top['antecedents'])[:80]}` → "
            f"fraud confidence **{float(top['confidence'])*100:.1f}%**"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DATASET EDA
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("""
    Exploratory analysis of the **Sparkov** dataset.  
    1,852,394 transactions · 0.57% fraud rate · 23 raw features → 25 engineered features.
    """)

    # ── KPI cards ──
    st.markdown('<div class="section-title">Dataset at a glance</div>',
                unsafe_allow_html=True)

    eda_cards = [
        ("Total transactions", "1,852,394", ""),
        ("Fraud transactions",   "10,739",  "0.58% of total"),
        ("Engineered features",     "25",   "+2 from raw 23"),
        ("SMOTE oversampling",      "20%",  "fraud / legitimate ratio"),
    ]
    cards_html = "".join(
        f'<div class="eda-card">'
        f'<div class="eda-label">{lbl}</div>'
        f'<div class="eda-value">{val}</div>'
        f'{"<div class=eda-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
        for lbl, val, sub in eda_cards
    )
    st.markdown(f'<div class="eda-cards">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="fg-divider"></div>', unsafe_allow_html=True)

    # ── Class pie + hour bar ──
    col_pie, col_hour = st.columns(2)

    # --- Find this section in Tab 4 ---
with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=["Legitimate", "Fraud"],
        values=[1841655, 10739], 
        marker=dict(colors=["#bfdbfe", "#1e3a8a"]),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    
    # THE FIX: Ensure margin is only defined ONCE
    fig_pie.update_layout(
        title="Transaction distribution",
        height=280,
        font=dict(family="IBM Plex Sans"),
        paper_bgcolor="white",
        showlegend=False,
        # Only one 'margin' definition here:
        margin=dict(l=20, r=20, t=40, b=20) 
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    with col_hour:
        hours = list(range(24))
        fr_hour = [
            0.012,0.015,0.018,0.021,0.024,0.019,
            0.008,0.005,0.004,0.004,0.004,0.005,
            0.005,0.006,0.006,0.007,0.007,0.008,
            0.009,0.012,0.018,0.022,0.020,0.016,
        ]
        fig_hour = go.Figure(go.Bar(
            x=hours, y=[r*100 for r in fr_hour],
            marker=dict(
                color=[r*100 for r in fr_hour],
                colorscale=[[0,"#dbeafe"],[1,"#1e3a8a"]],
                showscale=False,
            ),
        ))
        fig_hour.update_layout(
            title="Fraud rate by hour of day (%)",
            xaxis=dict(title="Hour (0=midnight)", dtick=2, showgrid=False),
            yaxis=dict(title="Fraud rate (%)", **GRID),
            height=280, **PLOT_LAYOUT,
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # ── Category bar ──
    st.markdown('<div class="section-title">Fraud rate by merchant category</div>',
                unsafe_allow_html=True)

    cats     = ["shopping_net","misc_net","travel","grocery_net","shopping_pos",
                "entertainment","food_dining","gas_transport","health_fitness","grocery_pos"]
    fr_cats  = [0.041,0.034,0.021,0.018,0.012,0.010,0.007,0.006,0.005,0.004]

    fig_cat = go.Figure(go.Bar(
        y=cats, x=[r*100 for r in fr_cats],
        orientation="h",
        marker=dict(
            color=[r*100 for r in fr_cats],
            colorscale=[[0,"#dbeafe"],[1,"#1e3a8a"]],
            showscale=False,
        ),
        text=[f"{r*100:.1f}%" for r in fr_cats],
        textposition="outside",
    ))
    fig_cat.update_layout(
        title="Fraud rate by merchant category",
        xaxis=dict(title="Fraud rate (%)", **GRID),
        yaxis=dict(showgrid=False),
        height=320, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # ── Feature engineering table ──
    st.markdown('<div class="section-title">Feature engineering summary</div>',
                unsafe_allow_html=True)

    feat_df = pd.DataFrame({
        "Feature":    ["amt_log","distance_km","hour","age","day_of_week",
                       "merchant_freq","city_freq","job_freq","category_* (13)","gender_M"],
        "Type":       ["Numeric","Numeric","Temporal","Demographic","Temporal",
                       "Freq-encoded","Freq-encoded","Freq-encoded","One-hot","One-hot"],
        "Source":     ["log1p(amt)","Haversine(card, merchant)",
                       "trans_date_trans_time","dob + trans_date",
                       "trans_date_trans_time","merchant col","city col","job col",
                       "category col","gender col"],
        "Why useful": [
            "Reduces right skew; fraud spikes at high amounts",
            "Fraudsters transact far from the cardholder's home city",
            "Fraud spikes 10 PM – 3 AM",
            "Seniors (65+) disproportionately targeted",
            "Weekend vs weekday patterns differ",
            "Rare merchants correlate with fraud",
            "Small-town card used in large city = suspicious",
            "Certain job demographics have higher fraud exposure",
            "Online shopping & misc have highest fraud rates",
            "Male cardholders slightly higher fraud rate",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="fg-footer">
  <span>FraudGuard · Sparkov Dataset · Academic Project</span>
  <span>XGBoost 0.9506 AUPRC · LightGBM · Isolation Forest · SHAP</span>
</div>
""", unsafe_allow_html=True)