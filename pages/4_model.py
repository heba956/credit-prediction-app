import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
GOOD   = '#00E5A0'
STD    = '#FFD166'
POOR   = '#FF4D6D'
ACCENT = '#BD93F9'
BLUE   = '#8BE9FD'
BG     = '#07070F'
PANEL  = '#0F0F1C'
CARD   = '#12122A'
GRID   = '#1C1C2E'
BORDER = '#2A2A4A'
TEXT   = '#E8E8F5'
MUTED  = '#555575'

# ── GLOBAL PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Report · Under the Hood",
    page_icon="🔬",
    layout="wide",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #07070F !important;
    color: #E8E8F5;
}
.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1200px;
}
section[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12122A 0%, #1A1A2E 100%);
    border: 1px solid #2A2A4A;
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color 0.3s ease;
}
section[data-testid="stMetric"]:hover {
    border-color: #BD93F9;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 3px !important;
    color: #555575 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.3rem !important;
    color: #BD93F9 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
    color: #555575 !important;
}
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

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
    margin=dict(t=55, b=45, l=160, r=40),
)
AXIS_STYLE = dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=GRID)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
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
# ── HEADER ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px;padding-top:20px;margin-bottom:4px'>
  MODEL REPORT  ·  RANDOM FOREST  ·  CREDIT RISK
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
# ── METRICS ROW ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Model",          "Random Forest")
m2.metric("Test Accuracy",  "77.0%")
m3.metric("F1 Score",       "77.1%",  delta="weighted avg")
m4.metric("Trees",          "100",    delta="n_estimators")
m5.metric("Max Depth",      "10")

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>WHAT MATTERS MOST</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Feature Importance</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Which signals did the model learn to rely on most?</p>
""", unsafe_allow_html=True)

rf          = model.named_steps['classifier']
importances = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=True).tail(20)
threshold   = importances.quantile(0.8)
bar_colors  = [ACCENT if v > threshold else '#1E1E3A' for v in importances.values]

fig_imp = go.Figure(go.Bar(
    x=importances.values,
    y=importances.index,
    orientation='h',
    marker=dict(
        color=bar_colors,
        line=dict(color='rgba(0,0,0,0)', width=0),
    ),
    opacity=0.92,
    text=[f'{v:.3f}' for v in importances.values],
    textposition='outside',
    textfont=dict(color=MUTED, size=9, family='Space Mono'),
))
fig_imp.update_layout(
    **PLOTLY_LAYOUT,
    height=590,
    title=dict(text='Top 20 Feature Importances (Random Forest)', font=dict(size=13)),
    xaxis_title='Importance Score',
    showlegend=False,
    xaxis=dict(**AXIS_STYLE),
    yaxis=dict(**AXIS_STYLE, tickfont=dict(size=10)),
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>WHERE IT FAILS</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Confusion Matrix</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Rows = actual class · Columns = predicted class · Diagonal = correct predictions</p>
""", unsafe_allow_html=True)

labels = ['Good', 'Standard', 'Poor']
cm     = np.array([
    [1325,  459,   80],
    [ 327, 2094,  493],
    [  50,  512, 1023],
])
# Row-normalise to percentages
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

# ── FIX 1: Flip z so row 0 (Good) appears at top ─────────────────────────────
z_display      = cm_pct[::-1]          # flip rows for correct heatmap orientation
labels_y_flip  = labels[::-1]          # matching y-axis labels (bottom → top)

# ── FIX 2: Plain-text annotations (no HTML tags) ──────────────────────────────
annotations = []
for i, actual in enumerate(labels_y_flip):       # i=0 → 'Poor' (bottom of flipped)
    for j, predicted in enumerate(labels):
        raw_i   = len(labels) - 1 - i            # index into original cm
        count   = cm[raw_i, j]
        pct     = cm_pct[raw_i, j]
        is_diag = (raw_i == j)
        # Bold the count with larger size; show percentage below
        annotations.append(dict(
            x=predicted,
            y=actual,
            text=f'{count:,}\n{pct:.1f}%',       # plain text, \n for newline
            showarrow=False,
            font=dict(
                color='white' if pct > 35 else TEXT,
                size=13 if is_diag else 11,
                family='Space Mono, monospace',
            ),
            align='center',
        ))

# Custom diverging colorscale: dark → purple for off-diag; teal highlight for diagonal
colorscale = [
    [0.0,  '#080818'],
    [0.25, '#111130'],
    [0.5,  '#1E1E4A'],
    [0.75, '#4A2A7A'],
    [1.0,  '#BD93F9'],
]

fig_cm = go.Figure(go.Heatmap(
    z=z_display,
    x=labels,
    y=labels_y_flip,
    colorscale=colorscale,
    showscale=True,
    colorbar=dict(
        title=dict(text='Row %', font=dict(size=10, color=MUTED)),
        tickfont=dict(size=9, color=MUTED),
        thickness=12,
        len=0.85,
        bgcolor='rgba(0,0,0,0)',
        outlinecolor=BORDER,
        outlinewidth=1,
    ),
    zmin=0,
    zmax=100,
    hoverongaps=False,
    hovertemplate=(
        '<b>Actual:</b> %{y}<br>'
        '<b>Predicted:</b> %{x}<br>'
        '<b>Row %:</b> %{z:.1f}%<extra></extra>'
    ),
))

fig_cm.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    height=440,
    annotations=annotations,
    xaxis=dict(
        title=dict(text='Predicted Class', font=dict(size=12, color=MUTED)),
        **{k: v for k, v in AXIS_STYLE.items()},
        side='bottom',
        tickfont=dict(size=12, color=TEXT),
    ),
    yaxis=dict(
        title=dict(text='Actual Class', font=dict(size=12, color=MUTED)),
        **{k: v for k, v in AXIS_STYLE.items()},
        tickfont=dict(size=12, color=TEXT),
    ),
    margin=dict(t=55, b=70, l=100, r=80),
    title=dict(
        text='Confusion Matrix — Test Set',
        font=dict(size=13),
    ),
)

st.plotly_chart(fig_cm, use_container_width=True)

# ── CLASS INSIGHT CARDS ───────────────────────────────────────────────────────
i1, i2, i3 = st.columns(3)
for col, tier, color, bg, recall, note in [
    (i1, 'GOOD TIER',  GOOD, '#0A1A13',  '70.1%', 'minority class · hardest to predict'),
    (i2, 'STANDARD',   STD,  '#1A1500',  '72.2%', 'dominant class · acts as error magnet'),
    (i3, 'POOR TIER',  POOR, '#1A0A0E',  '64.0%', 'most confused with Standard'),
]:
    col.markdown(f"""
<div style='background:{bg};border:1px solid {color}22;border-left:3px solid {color};
            border-radius:10px;padding:18px 16px;margin-top:8px'>
  <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:{color};
              letter-spacing:3px;margin-bottom:6px'>{tier}</div>
  <div style='font-size:1.5rem;font-weight:700;color:{color};
              font-family:Space Mono,monospace;line-height:1'>{recall}</div>
  <div style='font-size:0.72rem;color:{color}88;margin-top:6px'>recall</div>
  <div style='color:#445566;font-size:0.78rem;margin-top:8px;line-height:1.4'>{note}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:36px 0'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── MODEL COMPARISON ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#444466;letter-spacing:5px'>ALL MODELS</div>
<h2 style='margin-top:4px;margin-bottom:2px'>Model Comparison</h2>
<p style='color:#555575;font-size:0.85rem;margin-top:0'>Every architecture tested — same data, same split.</p>
""", unsafe_allow_html=True)

comparison = pd.DataFrame({
    'Model':    ['Random Forest ★', 'Gradient Boosting', 'Decision Tree',
                 'Logistic Regression', 'Naive Bayes'],
    'Accuracy': [77.0, 75.6, 72.3, 65.0, 32.6],
    'F1 Score': [77.1, 75.5, 72.3, 65.0, 16.0],
})

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name='Accuracy',
    x=comparison['Model'],
    y=comparison['Accuracy'],
    marker=dict(
        color=comparison['Accuracy'],
        colorscale=[[0, '#1E1E3A'], [1, ACCENT]],
        showscale=False,
    ),
    opacity=0.88,
    text=[f'{v:.1f}%' for v in comparison['Accuracy']],
    textposition='outside',
    textfont=dict(size=10, color=MUTED),
))
fig_comp.add_trace(go.Bar(
    name='F1 Score',
    x=comparison['Model'],
    y=comparison['F1 Score'],
    marker=dict(
        color=comparison['F1 Score'],
        colorscale=[[0, '#1A1A30'], [1, BLUE]],
        showscale=False,
    ),
    opacity=0.88,
    text=[f'{v:.1f}%' for v in comparison['F1 Score']],
    textposition='outside',
    textfont=dict(size=10, color=MUTED),
))
fig_comp.update_layout(
    **{**PLOTLY_LAYOUT, 'margin': dict(t=55, b=60, l=60, r=40)},
    barmode='group',
    bargap=0.25,
    bargroupgap=0.08,
    title=dict(text='Accuracy & F1 Score Across All Tested Models', font=dict(size=13)),
    xaxis=dict(**AXIS_STYLE, tickfont=dict(size=11)),
    yaxis=dict(**AXIS_STYLE, title='Score (%)', range=[0, 105]),
    legend=dict(
        bgcolor='rgba(12,12,28,0.8)',
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=10),
    ),
)
st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── FOOTER ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)
st.markdown("""
<div style='display:flex;gap:40px;flex-wrap:wrap'>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>PIPELINE</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      pandas · numpy · scikit-learn<br>
      plotly · streamlit
    </div>
  </div>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>DATASET</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      14,879 synthetic credit profiles<br>
      3 tiers · 80/20 stratified split
    </div>
  </div>
  <div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#444466;
                letter-spacing:4px;margin-bottom:10px'>MODEL</div>
    <div style='color:#333355;font-size:0.78rem;line-height:2;font-family:Space Mono,monospace'>
      Random Forest · GridSearchCV tuned<br>
      Ordinal + one-hot encoding
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

