import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoPredict",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #141720;
    --border: #1e2330;
    --accent: #00e5a0;
    --accent2: #3b82f6;
    --accent3: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}

.stSelectbox > div > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

.stFileUploader {
    background-color: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b37a) !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em;
    width: 100%;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
}

.metric-card .value {
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
}

.metric-card .sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.prediction-banner {
    background: linear-gradient(135deg, #00e5a01a, #3b82f61a);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    text-align: center;
}

.prediction-banner .pred-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-family: 'Space Mono', monospace;
}

.prediction-banner .pred-value {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    line-height: 1.1;
    margin: 0.3rem 0;
}

.prediction-banner .pred-date {
    font-size: 0.85rem;
    color: var(--muted);
}

.change-up { color: #00e5a0 !important; }
.change-down { color: #f87171 !important; }

.info-box {
    background: var(--surface);
    border-left: 3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--muted);
}

hr { border-color: var(--border) !important; }

[data-testid="stMarkdownContainer"] p {
    color: var(--text);
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# 📈 CryptoPredict")
    st.markdown("<p style='color:#64748b; margin-top:-0.8rem; font-size:0.9rem;'>Prediksi harga H+1 berbasis model Machine Learning & Deep Learning</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:right; font-family:Space Mono,monospace; font-size:0.7rem; color:#64748b;'>
        ARIMA · XGBoost · CNN-BiLSTM<br>
        <span style='color:#00e5a0;'>● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("<div class='section-title'>Pilih Aset</div>", unsafe_allow_html=True)
    asset = st.selectbox("", ["BTC – Bitcoin", "ETH – Ethereum", "BNB – BNB"], label_visibility="collapsed")
    asset_key = asset.split(" ")[0].lower()
    asset_symbol = asset.split(" ")[0]

    st.markdown("<br><div class='section-title'>Pilih Model</div>", unsafe_allow_html=True)
    model_choice = st.selectbox("", ["ARIMA", "XGBoost", "CNN-BiLSTM"], label_visibility="collapsed")

    st.markdown("<br><div class='section-title'>Upload Data</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Upload file <b>.csv</b> dengan minimal <b>30 baris</b> data historis terbaru.<br><br>
        Kolom wajib: <code>timestamp, open, high, low, close, volume, open_interest, macd, macd_signal, macd_hist, stoch_k, stoch_d</code>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Prediksi H+1")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
        <b style='color:#64748b;'>Tugas Akhir</b><br>
        Richard Angelico Pudjohartono<br>
        220711747 · UAJY · 2025<br><br>
        Model dilatih pada data harian<br>
        BTC, ETH, BNB (2020–2024)
    </div>
    """, unsafe_allow_html=True)

# ── Feature column definitions ────────────────────────────────────────────────
MODEL_DIR = "."

# Columns as they appear in the uploaded CSV (with underscores)
CSV_FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "open_interest",
    "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"
]

# XGBoost: underscore names, open_interest at the END (matches training order)
XGB_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "open_interest"
]

# CNN-BiLSTM: no-underscore names, openinterest after volume (matches training)
CNN_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "openinterest", "macd", "macdsignal", "macdhist", "stochk", "stochd"
]

# ARIMA orders from notebook grid search 
ARIMA_ORDERS = {"btc": (3, 2, 4), "eth": (2, 1, 3), "bnb": (2, 1, 4)}

WINDOW_SIZE = 30

# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_xgboost(asset_key):
    model_path  = os.path.join(MODEL_DIR, f"xgboost_{asset_key}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{asset_key}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

import tensorflow as tf
class SumLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)


@st.cache_resource
def load_cnn(asset_key):
    import joblib
    import tensorflow.keras.backend as K
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional,
                                         LSTM, Dense, Dropout, Flatten,
                                         Concatenate, LayerNormalization,
                                         Permute, Multiply, Lambda)

    _W, _N = 30, 11
    _f, _u1, _u2, _do, _d = 128, 64, 48, 0.1, 64

    def _attention(inp):
        s = Dense(1, activation="tanh")(inp)
        s = Flatten()(s)
        w = keras.layers.Activation("softmax")(s)
        w = keras.layers.RepeatVector(inp.shape[-1])(w)
        w = Permute([2, 1])(w)
        a = Multiply()([inp, w])
        return SumLayer()(a)

    inputs = Input(shape=(_W, _N))
    cnn  = Conv1D(_f,   3, activation="relu", padding="same")(inputs)
    cnn  = Conv1D(_f*2, 3, activation="relu", padding="same")(cnn)
    cnn  = MaxPooling1D(2)(cnn)
    cnn  = Dropout(_do)(cnn)
    cnn  = Flatten()(cnn)
    bl   = Bidirectional(LSTM(_u1, return_sequences=True))(inputs)
    bl   = Dropout(_do)(bl)
    bl   = Bidirectional(LSTM(_u2, return_sequences=True))(bl)
    bl   = Dropout(_do)(bl)
    ctx  = _attention(bl)
    mg   = Concatenate()([cnn, ctx])
    mg   = LayerNormalization()(mg)
    out  = Dense(_d, activation="relu")(mg)
    out  = Dropout(_do)(out)
    out  = Dense(32, activation="relu")(out)
    out  = Dense(1)(out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    print("✅ Arsitektur selesai", flush=True)

    model_path  = os.path.join(MODEL_DIR, f"cnn_bilstm_{asset_key}.weights.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_cnn_{asset_key}.pkl")
    print(f"✅ Loading weights dari: {model_path}", flush=True)
    model.load_weights(model_path)
    print("✅ Weights loaded", flush=True)
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded", flush=True)
    return model, scaler

# ── Helper: inference ─────────────────────────────────────────────────────────
def predict_arima(asset_key, df):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")
    order = ARIMA_ORDERS.get(asset_key, (3, 2, 4))
    close = df.sort_values("timestamp")["close"].astype(float).reset_index(drop=True)
    fitted = ARIMA(close, order=order).fit()
    fc = fitted.forecast(steps=1)
    return float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])

def predict_xgboost(model, scaler, df):
    df   = df.copy().sort_values("timestamp").reset_index(drop=True)
    last = df.iloc[-1:]
    # Force numpy so XGBoost doesn't complain about feature names
    feat_scaled = np.array(scaler.transform(last[XGB_FEATURE_COLS]))
    log_ret_pred = model.predict(feat_scaled)[0]

    print("log_ret_pred:", log_ret_pred)          
    print("feat_scaled:", feat_scaled)

    return float(df["close"].iloc[-1] * np.exp(log_ret_pred))

def predict_cnn(model, scaler, df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = df.rename(columns={
        "open_interest": "openinterest",
        "macd_signal":   "macdsignal",
        "macd_hist":     "macdhist",
        "stoch_k":       "stochk",
        "stoch_d":       "stochd",
    })
    window = df.iloc[-WINDOW_SIZE:]
    feat_scaled = np.array(scaler.transform(window[CNN_FEATURE_COLS]))
    X = feat_scaled.reshape(1, WINDOW_SIZE, len(CNN_FEATURE_COLS))
    X = tf.constant(X, dtype=tf.float32)
    log_ret_pred = float(model(X, training=False).numpy()[0][0])
    print(f"log_ret_pred: {log_ret_pred:.8f}", flush=True)
    return float(df["close"].iloc[-1] * np.exp(log_ret_pred))

# ── Main content ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    st.markdown("<div class='section-title'>Panduan Penggunaan</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("01", "Upload CSV", "Siapkan file .csv minimal 30 baris data historis harian dengan kolom teknikal yang lengkap."),
        ("02", "Pilih Konfigurasi", "Pilih aset cryptocurrency (BTC/ETH/BNB) dan model prediksi yang diinginkan dari sidebar."),
        ("03", "Lihat Prediksi", "Klik tombol Prediksi H+1. Sistem akan menampilkan estimasi harga hari berikutnya beserta plot historis."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding:1.5rem;'>
                <div style='font-family:Space Mono,monospace; font-size:2rem; color:#1e2330; font-weight:700; margin-bottom:0.5rem;'>{num}</div>
                <div style='font-weight:600; margin-bottom:0.5rem; color:#e2e8f0;'>{title}</div>
                <div style='font-size:0.82rem; color:#64748b; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Overview</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    models_info = [
        ("ARIMA", "Statistical", "Model statistik klasik berbasis autoregresif. Cocok sebagai baseline. Performa terbaik pada data stasioner.", "#f59e0b"),
        ("XGBoost", "Machine Learning", "Gradient boosting berbasis pohon keputusan. Cepat, efisien, dan akurat dengan trade-off terbaik.", "#3b82f6"),
        ("CNN-BiLSTM", "Deep Learning", "Arsitektur hybrid CNN + Bidirectional LSTM dengan attention mechanism. Akurasi tertinggi namun komputasi lebih berat.", "#00e5a0"),
    ]
    for col, (name, tag, desc, color) in zip([m1, m2, m3], models_info):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding:1.5rem; border-top: 3px solid {color};'>
                <div style='font-family:Space Mono,monospace; font-size:1rem; font-weight:700; color:{color};'>{name}</div>
                <div style='font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin:0.3rem 0 0.7rem;'>{tag}</div>
                <div style='font-size:0.82rem; color:#94a3b8; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Data loaded
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Validate columns
    required = ["timestamp", "close"] + CSV_FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan: {missing}")
        st.stop()

    if len(df) < WINDOW_SIZE:
        st.warning(f"Data kurang dari {WINDOW_SIZE} baris. Harap upload minimal {WINDOW_SIZE} baris.")
        st.stop()

    # Data preview
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    change     = last_close - prev_close
    change_pct = change / prev_close * 100
    change_cls = "change-up" if change >= 0 else "change-down"
    change_sym = "▲" if change >= 0 else "▼"
    last_date  = pd.to_datetime(df["timestamp"].iloc[-1]).strftime("%d %b %Y")

    with col_info1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Aset</div>
            <div class='value' style='font-size:1.3rem;'>{asset_symbol}</div>
            <div class='sub'>Selected asset</div>
        </div>""", unsafe_allow_html=True)
    with col_info2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Harga Terakhir</div>
            <div class='value'>${last_close:,.2f}</div>
            <div class='sub'>{last_date}</div>
        </div>""", unsafe_allow_html=True)
    with col_info3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Perubahan (1D)</div>
            <div class='value {change_cls}' style='font-size:1.4rem;'>{change_sym} {abs(change_pct):.2f}%</div>
            <div class='sub'>${change:+,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col_info4:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Jumlah Data</div>
            <div class='value' style='font-size:1.4rem;'>{len(df)}</div>
            <div class='sub'>baris / hari</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Menjalankan {model_choice} untuk {asset_symbol}..."):
            pred_price = None
            error_msg  = None
            try:
                if model_choice == "ARIMA":
                    pred_price = predict_arima(asset_key, df)
                elif model_choice == "XGBoost":
                    xgb_model, xgb_scaler = load_xgboost(asset_key)
                    pred_price = predict_xgboost(xgb_model, xgb_scaler, df)
                else:
                    cnn_model, cnn_scaler = load_cnn(asset_key)
                    pred_price = predict_cnn(cnn_model, cnn_scaler, df)
            except FileNotFoundError as e:
                error_msg = f"File model tidak ditemukan: {e}"
            except Exception as e:
                error_msg = f"Error saat inferensi: {e}"

        if error_msg:
            st.error(error_msg)
        else:
            pred_date  = pd.to_datetime(df["timestamp"].iloc[-1]) + timedelta(days=1)
            pred_change     = pred_price - last_close
            pred_change_pct = pred_change / last_close * 100
            direction       = "▲" if pred_change >= 0 else "▼"
            dir_color       = "#00e5a0" if pred_change >= 0 else "#f87171"

            st.markdown(f"""
            <div class='prediction-banner'>
                <div class='pred-label'>Prediksi Harga H+1 — {model_choice} · {asset_symbol}</div>
                <div class='pred-value'>${pred_price:,.2f}</div>
                <div style='font-size:1rem; color:{dir_color}; font-family:Space Mono,monospace; margin:0.2rem 0;'>
                    {direction} ${abs(pred_change):,.2f} ({pred_change_pct:+.2f}%)
                </div>
                <div class='pred-date'>Estimasi untuk {pred_date.strftime("%A, %d %B %Y")}</div>
            </div>
            """, unsafe_allow_html=True)

            # Plot
            st.markdown("<div class='section-title'>Historis Harga & Prediksi H+1</div>", unsafe_allow_html=True)
            n_show  = min(60, len(df))
            df_plot = df.iloc[-n_show:].copy()
            dates   = pd.to_datetime(df_plot["timestamp"])
            closes  = df_plot["close"].values

            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor("#0d0f14")
            ax.set_facecolor("#0d0f14")

            ax.plot(dates, closes, color="#3b82f6", linewidth=1.8, label="Harga Aktual", zorder=3)
            ax.fill_between(dates, closes, alpha=0.08, color="#3b82f6")

            pred_color = "#00e5a0" if pred_change >= 0 else "#f87171"
            ax.plot([dates.iloc[-1], pred_date], [last_close, pred_price],
                    color=pred_color, linewidth=2, linestyle="--", zorder=4)
            ax.scatter([pred_date], [pred_price], color=pred_color, s=80, zorder=5,
                       label=f"Prediksi H+1: ${pred_price:,.2f}")

            ax.annotate(f"${pred_price:,.2f}",
                        xy=(pred_date, pred_price),
                        xytext=(10, 10), textcoords="offset points",
                        color=pred_color, fontsize=9,
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#141720", edgecolor=pred_color, alpha=0.9))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2330")
            ax.tick_params(colors="#64748b", labelsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.grid(color="#1e2330", linewidth=0.5, linestyle="-")
            ax.legend(facecolor="#141720", edgecolor="#1e2330", labelcolor="#e2e8f0", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        # Show chart preview without prediction
        st.markdown("<div class='section-title'>Historis Harga (60 Hari Terakhir)</div>", unsafe_allow_html=True)
        n_show  = min(60, len(df))
        df_plot = df.iloc[-n_show:].copy()
        dates   = pd.to_datetime(df_plot["timestamp"])
        closes  = df_plot["close"].values

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")
        ax.plot(dates, closes, color="#3b82f6", linewidth=1.8)
        ax.fill_between(dates, closes, alpha=0.08, color="#3b82f6")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2330")
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(color="#1e2330", linewidth=0.5, linestyle="-")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class='info-box'>
            ✅ Data berhasil dimuat. Klik <b>Prediksi H+1</b> di sidebar untuk menjalankan model.
        </div>
        """, unsafe_allow_html=True)