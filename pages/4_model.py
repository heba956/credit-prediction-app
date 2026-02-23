import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── COLOUR PALETTE ─────────────────────────────────────────────────────────────
GOOD   = '#00E5A0'
STD    = '#FFD166'
POOR   = '#FF4D6D'
ACCENT = '#BD93F9'
BLUE   = '#8BE9FD'
BG     = '#07070F'
PANEL  = '#0F0F1C'
GRID   = '#1C1C2E'
BORDER = '#2A2A4A'
TEXT   = '#E8E8F5'
MUTED  = '#555575'

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Report · Under the Hood",
    page_icon="🔬",
    layout="wide",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { background-color: #07070F !important; color: #E8E8F5; }
.main .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

section[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12122A 0%, #1A1A2E 100%);
    border: 1px solid #2A2A4A;
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color 0.3s ease;
}
section[data-testid="stMetric"]:hover { border-color: #BD93F9; }

[data-testid="stMetricLabel"]  { font-family:'Space Mono',monospace !important; font-size:0.6rem !important; letter-spacing:3px !important; color:#555575 !important; text-transform:uppercase; }
[data-testid="stMetricValue"]  { font-family:'Space Mono',monospace !important; font-size:1.3rem !important; color:#BD93F9 !important; }
[data-testid="stMetricDelta"]  { font-size:0.7rem !important; color:#555575 !important; }

hr { border-color: #1C1C2E !important; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
.glow-text {
    background: linear-gradient(135deg, #BD93F9, #8BE9FD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
</style>
""", unsafe_allow_html=True)

# ── SHARED AXIS STYLE ──────────────────────────────────────────────────────────
AXIS = dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=GRID)

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_columns():
    with open('columns.pkl', 'rb') as f:
        return pickle.load(f)

model   = load_model()
columns = load_columns()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px;padding-top:20px;margin-bottom:4px'>
  MODEL REPORT &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp; CREDIT RISK
</div>
<h1 class='glow-text' style='margin-top:0;font-size:2.6rem;font-weight:700;letter-spacing:-1px'>
  Under the Hood
</h1>
<p style='color:#555575;font-size:0.95rem;margin-top:-8px'>
  How the model works, what it learned, and where it struggles.
</p>
<hr style='border-color:#1C1C2E;margin:24px 0'/>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Model",         "Random Forest")
m2.metric("Test Accuracy", "77.0%")
m3.metric("F1 Score",      "77.1%",  delta="weighted avg")
m4.metric("Trees",         "100",    delta="n_estimators")
m5.metric("Max Depth",     "10")

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>WHAT MATTERS MOST</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Feature Importance</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Which signals did the model learn to rely on most?</p>
""", unsafe_allow_html=True)

rf          = model.named_steps['classifier']
importances = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=True).tail(20)
threshold   = float(importances.quantile(0.8))
imp_values  = importances.values.tolist()          # plain Python list — safe for Plotly
imp_index   = importances.index.tolist()
bar_colors  = [ACCENT if v > threshold else '#1E1E3A' for v in imp_values]

fig_imp = go.Figure(go.Bar(
    x=imp_values,
    y=imp_index,
    orientation='h',
    marker_color=bar_colors,
    opacity=0.92,
    text=[f'{v:.3f}' for v in imp_values],
    textposition='outside',
    textfont=dict(color=MUTED, size=9, family='Space Mono'),
))
fig_imp.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    height=590,
    showlegend=False,
    title=dict(text='Top 20 Feature Importances (Random Forest)', font=dict(size=13)),
    xaxis_title='Importance Score',
    margin=dict(t=55, b=45, l=160, r=40),
    xaxis=dict(**AXIS),
    yaxis=dict(**AXIS, tickfont=dict(size=10)),
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>WHERE IT FAILS</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Confusion Matrix</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Rows = actual class &nbsp;·&nbsp; Columns = predicted class &nbsp;·&nbsp; Diagonal = correct predictions</p>
""", unsafe_allow_html=True)

labels = ['Good', 'Standard', 'Poor']
cm = np.array([
    [1325,  459,   80],
    [ 327, 2094,  493],
    [  50,  512, 1023],
])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

# Plotly heatmaps render row-0 at the bottom; flip so "Good" is at the top
z_flip      = cm_pct[::-1].tolist()   # list of lists — maximally safe
labels_flip = labels[::-1]            # ['Poor', 'Standard', 'Good'] bottom→top

# Build annotations — use plain text only (no HTML, no \n)
# Instead, put count and pct as separate annotation layers via two passes
annotations = []
for i, actual in enumerate(labels_flip):
    orig_i = len(labels) - 1 - i      # row index into original cm
    for j, predicted in enumerate(labels):
        count   = int(cm[orig_i, j])
        pct     = float(cm_pct[orig_i, j])
        is_diag = (orig_i == j)
        bright  = pct > 38

        # Line 1: count (larger, bolder via size)
        annotations.append(dict(
            x=predicted, y=actual,
            text=f'{count:,}',
            showarrow=False,
            yshift=9,
            font=dict(
                color='#FFFFFF' if bright else TEXT,
                size=14 if is_diag else 12,
                family='Space Mono, monospace',
            ),
        ))
        # Line 2: percentage (smaller, muted)
        annotations.append(dict(
            x=predicted, y=actual,
            text=f'{pct:.1f}%',
            showarrow=False,
            yshift=-9,
            font=dict(
                color='#FFFFFF' if bright else MUTED,
                size=10,
                family='Space Mono, monospace',
            ),
        ))

colorscale_cm = [
    [0.00, '#080816'],
    [0.30, '#111130'],
    [0.60, '#1E1E4A'],
    [0.80, '#4A2A7A'],
    [1.00, '#BD93F9'],
]

fig_cm = go.Figure(go.Heatmap(
    z=z_flip,
    x=labels,
    y=labels_flip,
    colorscale=colorscale_cm,
    showscale=True,
    zmin=0,
    zmax=100,
    colorbar=dict(
        title=dict(text='Row %', font=dict(size=10, color=MUTED)),
        tickfont=dict(size=9, color=MUTED),
        thickness=12,
        len=0.85,
    ),
    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Row %%: %{z:.1f}<extra></extra>',
))
fig_cm.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    height=440,
    annotations=annotations,
    margin=dict(t=55, b=70, l=100, r=80),
    title=dict(text='Confusion Matrix — Test Set', font=dict(size=13)),
    xaxis=dict(
        **AXIS,
        title=dict(text='Predicted Class', font=dict(size=12, color=MUTED)),
        side='bottom',
        tickfont=dict(size=12, color=TEXT),
    ),
    yaxis=dict(
        **AXIS,
        title=dict(text='Actual Class', font=dict(size=12, color=MUTED)),
        tickfont=dict(size=12, color=TEXT),
    ),
)
st.plotly_chart(fig_cm, use_container_width=True)

# ── CLASS INSIGHT CARDS ────────────────────────────────────────────────────────
i1, i2, i3 = st.columns(3)
for col, tier, color, bg_card, recall, note in [
    (i1, 'GOOD TIER', GOOD, '#0A1A13', '70.1%', 'minority class &nbsp;·&nbsp; hardest to predict'),
    (i2, 'STANDARD',  STD,  '#1A1500', '72.2%', 'dominant class &nbsp;·&nbsp; acts as error magnet'),
    (i3, 'POOR TIER', POOR, '#1A0A0E', '64.0%', 'most confused with Standard'),
]:
    col.markdown(f"""
<div style='background:{bg_card};border:1px solid {color}22;border-left:3px solid {color};
            border-radius:10px;padding:18px 16px;margin-top:8px'>
  <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:{color};
              letter-spacing:3px;margin-bottom:6px'>{tier}</div>
  <div style='font-size:1.5rem;font-weight:700;color:{color};
              font-family:Space Mono,monospace;line-height:1'>{recall}</div>
  <div style='font-size:0.72rem;color:{color}88;margin-top:6px'>recall</div>
  <div style='color:#445566;font-size:0.78rem;margin-top:8px;line-height:1.5'>{note}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:36px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>ALL MODELS</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Model Comparison</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Every architecture tested — same data, same split.</p>
""", unsafe_allow_html=True)

# Use ASCII star to avoid encoding issues with the Unicode star character
model_names = ['Random Forest *', 'Gradient Boosting', 'Decision Tree',
               'Logistic Regression', 'Naive Bayes']
acc_vals    = [77.0, 75.6, 72.3, 65.0, 32.6]
f1_vals     = [77.1, 75.5, 72.3, 65.0, 16.0]

# Simple per-bar color lists — no colorscale, pure hex strings
acc_colors = [ACCENT, '#333360', '#2A2A50', '#222244', '#1A1A38']
f1_colors  = [BLUE,   '#1A3040', '#152535', '#10202E', '#0C1A26']

fig_comp = go.Figure()

fig_comp.add_trace(go.Bar(
    name='Accuracy',
    x=model_names,
    y=acc_vals,
    marker_color=acc_colors,
    marker_line_width=0,
    opacity=0.90,
    text=[f'{v:.1f}%' for v in acc_vals],
    textposition='outside',
    textfont=dict(size=10, color=MUTED, family='Space Mono'),
))

fig_comp.add_trace(go.Bar(
    name='F1 Score',
    x=model_names,
    y=f1_vals,
    marker_color=f1_colors,
    marker_line_width=0,
    opacity=0.90,
    text=[f'{v:.1f}%' for v in f1_vals],
    textposition='outside',
    textfont=dict(size=10, color=MUTED, family='Space Mono'),
))

fig_comp.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    barmode='group',
    bargap=0.25,
    bargroupgap=0.08,
    height=420,
    margin=dict(t=55, b=60, l=60, r=40),
    title=dict(text='Accuracy & F1 Score Across All Tested Models', font=dict(size=13)),
    xaxis=dict(**AXIS, tickfont=dict(size=10)),
    yaxis=dict(**AXIS, title='Score (%)', range=[0, 105]),
    legend=dict(
        bgcolor='rgba(12,12,28,0.85)',
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=10),
    ),
)
st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;gap:48px;flex-wrap:wrap'>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>PIPELINE</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      pandas &nbsp;·&nbsp; numpy &nbsp;·&nbsp; scikit-learn<br>plotly &nbsp;·&nbsp; streamlit
    </div>
  </div>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>DATASET</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      14,879 synthetic credit profiles<br>3 tiers &nbsp;·&nbsp; 80/20 stratified split
    </div>
  </div>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>MODEL</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      Random Forest &nbsp;·&nbsp; GridSearchCV tuned<br>Ordinal + one-hot encoding
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



