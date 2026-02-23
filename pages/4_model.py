import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

GOOD  = '#00E5A0'; STD = '#FFD166'; POOR = '#FF4D6D'; ACCENT = '#BD93F9'
BG = '#07070F'; PANEL = '#11111E'; GRID = '#1C1C2E'; TEXT = '#E8E8F5'; MUTED = '#444466'

# ── FIXED: no yaxis/xaxis in base layout — set per-chart instead ──────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
    margin=dict(t=50, b=40, l=160, r=40)
)

AXIS_STYLE = dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)

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

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px;padding-top:16px'>MODEL REPORT</div>
<h1 style='margin-top:4px'>Under the Hood</h1>
<p style='color:#666688'>How the model works, what it learned, and where it struggles.</p>
<hr style='border-color:#1C1C2E;margin:20px 0'/>
""", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Model",         "Random Forest")
m2.metric("Test Accuracy", "77.0%")
m3.metric("F1 Score",      "77.1%", delta="weighted")
m4.metric("Trees",         "100",   delta="n_estimators")
m5.metric("Max Depth",     "10")

st.markdown("<hr style='border-color:#1C1C2E;margin:28px 0'/>", unsafe_allow_html=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px'>WHAT MATTERS MOST</div>
<h2 style='margin-top:2px'>Feature Importance</h2>
<p style='color:#666688;font-size:0.85rem'>Which signals did the model learn to rely on most?</p>
""", unsafe_allow_html=True)

rf = model.named_steps['classifier']
importances = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=True).tail(20)
bar_colors = [ACCENT if v > importances.quantile(0.8) else '#2A2A4A' for v in importances.values]

fig_imp = go.Figure(go.Bar(
    x=importances.values,
    y=importances.index,
    orientation='h',
    marker_color=bar_colors,
    opacity=0.9,
    text=[f'{v:.3f}' for v in importances.values],
    textposition='outside',
    textfont=dict(color=MUTED, size=9)
))
fig_imp.update_layout(
    **PLOTLY_LAYOUT,
    height=580,
    title='Top 20 Feature Importances (Random Forest)',
    xaxis_title='Importance Score',
    showlegend=False,
    xaxis=dict(**AXIS_STYLE),
    yaxis=dict(**AXIS_STYLE, tickfont=dict(size=10))
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:28px 0'/>", unsafe_allow_html=True)

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px'>WHERE IT FAILS</div>
<h2 style='margin-top:2px'>Confusion Matrix</h2>
<p style='color:#666688;font-size:0.85rem'>Rows = actual class · Columns = predicted class · Diagonal = correct predictions</p>
""", unsafe_allow_html=True)

cm = np.array([
    [1325,  459,   80],
    [ 327, 2094,  493],
    [  50,  512, 1023],
])
labels = ['Good', 'Standard', 'Poor']
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

annotations = []
for i in range(3):
    for j in range(3):
        annotations.append(dict(
            x=labels[j], y=labels[i],
            text=f'<b>{cm[i,j]:,}</b><br><span style="font-size:10px">{cm_pct[i,j]:.1f}%</span>',
            showarrow=False,
            font=dict(color='white' if cm_pct[i,j] > 30 else TEXT, size=12, family='Space Mono')
        ))

fig_cm = go.Figure(go.Heatmap(
    z=cm_pct, x=labels, y=labels,
    colorscale=[[0,'#0D0D1C'],[0.5,'#1A1A3A'],[1,'#BD93F9']],
    showscale=False,
))
fig_cm.update_layout(
    **PLOTLY_LAYOUT,
    height=380,
    annotations=annotations,
    xaxis_title='Predicted',
    yaxis_title='Actual',
    margin=dict(t=50, b=60, l=80, r=20),
    title='Confusion Matrix — Test Set',
    xaxis=dict(**AXIS_STYLE),
    yaxis=dict(**AXIS_STYLE),
)
st.plotly_chart(fig_cm, use_container_width=True)

i1, i2, i3 = st.columns(3)
i1.markdown(f"""
<div style='background:#0A1A13;border:1px solid {GOOD};border-radius:8px;padding:16px'>
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:{GOOD};letter-spacing:3px'>GOOD TIER</div>
    <div style='font-size:1.4rem;font-weight:700;color:{GOOD};font-family:Space Mono,monospace'>70.1%</div>
    <div style='color:#446655;font-size:0.8rem'>recall — minority class is hardest to predict</div>
</div>
""", unsafe_allow_html=True)
i2.markdown(f"""
<div style='background:#1A1A0A;border:1px solid {STD};border-radius:8px;padding:16px'>
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:{STD};letter-spacing:3px'>STANDARD</div>
    <div style='font-size:1.4rem;font-weight:700;color:{STD};font-family:Space Mono,monospace'>72.2%</div>
    <div style='color:#665544;font-size:0.8rem'>recall — acts as a gravity well for errors</div>
</div>
""", unsafe_allow_html=True)
i3.markdown(f"""
<div style='background:#1A0A0E;border:1px solid {POOR};border-radius:8px;padding:16px'>
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:{POOR};letter-spacing:3px'>POOR TIER</div>
    <div style='font-size:1.4rem;font-weight:700;color:{POOR};font-family:Space Mono,monospace'>64.0%</div>
    <div style='color:#664455;font-size:0.8rem'>recall — often confused with Standard</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:28px 0'/>", unsafe_allow_html=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px'>ALL MODELS</div>
<h2 style='margin-top:2px'>Model Comparison</h2>
""", unsafe_allow_html=True)

comparison = pd.DataFrame({
    'Model':    ['Random Forest ★', 'Gradient Boosting', 'Decision Tree', 'Logistic Regression', 'Naive Bayes'],
    'Accuracy': [77.0, 75.6, 72.3, 65.0, 32.6],
    'F1 Score': [77.1, 75.5, 72.3, 65.0, 16.0],
})

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(name='Accuracy', x=comparison['Model'],
                          y=comparison['Accuracy'], marker_color=ACCENT, opacity=0.85))
fig_comp.add_trace(go.Bar(name='F1 Score', x=comparison['Model'],
                          y=comparison['F1 Score'], marker_color='#555580', opacity=0.85))
fig_comp.update_layout(
    **PLOTLY_LAYOUT,
    barmode='group',
    margin=dict(t=50, b=60, l=60, r=20),
    title='Accuracy & F1 Score Across All Tested Models',
    xaxis=dict(**AXIS_STYLE),
    yaxis=dict(**AXIS_STYLE, title='Score (%)', range=[0, 100]),
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:28px 0'/>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px;margin-bottom:12px'>ABOUT THIS PROJECT</div>
<div style='color:#555575;font-size:0.8rem;line-height:2;font-family:Space Mono,monospace'>
Pipeline: pandas · numpy · scikit-learn<br>
Visualisation: plotly · streamlit<br>
Model: Random Forest (GridSearchCV tuned)<br>
Dataset: 14,879 synthetic credit profiles · 3 tiers<br>
Split: 80/20 train/test · stratified by class<br>
Encoding: ordinal + one-hot · feature engineering applied
</div>
""", unsafe_allow_html=True)
