import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── THEME ─────────────────────────────────────────────────────────────────────
BG    = '#07070F'
PANEL = '#11111E'
GOOD  = '#00E5A0'
STD   = '#FFD166'
POOR  = '#FF4D6D'
MUTED = '#444466'
TEXT  = '#E8E8F5'
GRID  = '#1C1C2E'
SC    = {'Good': GOOD, 'Standard': STD, 'Poor': POOR}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
    margin=dict(t=50, b=40, l=50, r=20)
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df.drop(columns=['Name', 'Unnamed: 0'], inplace=True, errors='ignore')
    df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace('_',''), errors='coerce')
    df = df[(df['Age'] > 0) & (df['Age'] < 110)]
    for col in ['Monthly_Inhand_Salary','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                'Outstanding_Debt','Annual_Income','Interest_Rate','Credit_Utilization_Ratio',
                'Monthly_Balance','Amount_invested_monthly']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('_',''), errors='coerce')
    df['Annual_Income'] = df['Annual_Income'].clip(upper=df['Annual_Income'].quantile(0.99))
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(lower=0)
    df.dropna(subset=['Credit_Score'], inplace=True)

    def parse_age(s):
        try:
            y = int(str(s).split('Years')[0].strip())
            m = int(str(s).split('and')[1].split('Months')[0].strip())
            return y * 12 + m
        except: return np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(parse_age)
    return df

df = load_data()

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px;padding-top:16px'>
THE STORY
</div>
<h1 style='margin-top:4px'>The Anatomy of a<br>Credit Trap</h1>
<p style='color:#666688;max-width:600px'>
Five acts. One thesis: the difference between Good and Poor credit is not income — it is compounding behaviour.
</p>
<hr style='border-color:#1C1C2E;margin:24px 0'/>
""", unsafe_allow_html=True)

# ── ACT I — THE CLIFF ─────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FF4D6D;letter-spacing:4px'>ACT I</div>
<h2 style='margin-top:2px;color:#E8E8F5'>The 20-Payment Cliff</h2>
<p style='color:#666688;font-size:0.9rem'>Past 20 missed payments, the probability of a Good score collapses from <b style='color:#00E5A0'>44%</b> to <b style='color:#FF4D6D'>1.7%</b></p>
""", unsafe_allow_html=True)

bins = [0, 5, 10, 15, 20, 30, 50]
labels = ['0–5', '6–10', '11–15', '16–20', '21–30', '31–50']
df['Delay_Bin'] = pd.cut(df['Num_of_Delayed_Payment'].clip(0, 50), bins=bins, include_lowest=True)
ct = pd.crosstab(df['Delay_Bin'], df['Credit_Score'], normalize='index') * 100
ct.index = labels[:len(ct)]

fig1 = go.Figure()
for score, color in SC.items():
    if score in ct.columns:
        fig1.add_trace(go.Bar(name=score, x=ct.index, y=ct[score],
                              marker_color=color, opacity=0.85))

fig1.add_vrect(x0=3.5, x1=5.5, fillcolor=POOR, opacity=0.06, line_width=0)
fig1.add_vline(x=3.5, line_dash='dash', line_color=POOR, opacity=0.5)
fig1.add_annotation(x=4.2, y=60, text="← DANGER ZONE<br>past 20 missed payments",
                    font=dict(color=POOR, size=10, family='Space Mono'), showarrow=False)
fig1.update_layout(**PLOTLY_LAYOUT, barmode='group',
                   title='% Credit Score by Number of Delayed Payments',
                   xaxis_title='Delayed Payments', yaxis_title='% of People')
st.plotly_chart(fig1, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ── ACT II — DEBT MOUNTAIN ────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FF4D6D;letter-spacing:4px'>ACT II</div>
<h2 style='margin-top:2px;color:#E8E8F5'>The Debt Mountain</h2>
<p style='color:#666688;font-size:0.9rem'>Poor scorers carry <b style='color:#FF4D6D'>3×</b> more outstanding debt than Good scorers</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    fig2 = go.Figure()
    for score, color in SC.items():
        data = df[df['Credit_Score'] == score]['Outstanding_Debt'].dropna()
        data = data[data < 5000]
        fig2.add_trace(go.Histogram(x=data, name=score, marker_color=color,
                                    opacity=0.55, histnorm='probability density',
                                    nbinsx=50))
    fig2.update_layout(**PLOTLY_LAYOUT, barmode='overlay',
                       title='Outstanding Debt Distribution',
                       xaxis_title='Outstanding Debt ($)', yaxis_title='Density')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    for score, color in [('Good', GOOD), ('Standard', STD), ('Poor', POOR)]:
        med = df[df['Credit_Score'] == score]['Outstanding_Debt'].dropna().median()
        st.markdown(f"""
        <div style='background:#11111E;border-left:3px solid {color};padding:16px;border-radius:6px;margin-bottom:12px'>
            <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:{color};letter-spacing:3px'>{score.upper()}</div>
            <div style='font-size:1.5rem;font-weight:700;color:{color};font-family:Space Mono,monospace'>${med:,.0f}</div>
            <div style='font-size:0.75rem;color:#444466'>median debt</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ── ACT III — TIME IS EVERYTHING ─────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FF4D6D;letter-spacing:4px'>ACT III</div>
<h2 style='margin-top:2px;color:#E8E8F5'>Time is the Most Unfair Advantage</h2>
<p style='color:#666688;font-size:0.9rem'>Good scorers have <b style='color:#00E5A0'>nearly 9 extra years</b> of credit history</p>
""", unsafe_allow_html=True)

history_means = df.groupby('Credit_Score')['Credit_History_Months'].mean() / 12
fig3 = go.Figure()
order = ['Poor', 'Standard', 'Good']
colors_order = [POOR, STD, GOOD]
for score, color in zip(order, colors_order):
    if score in history_means:
        fig3.add_trace(go.Bar(
            x=[history_means[score]], y=[score], orientation='h',
            name=score, marker_color=color, opacity=0.85,
            text=f'{history_means[score]:.1f} yrs', textposition='outside',
            textfont=dict(color=color, family='Space Mono')
        ))
fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=200,
                   title='Average Credit History by Tier (Years)',
                   xaxis_title='Years', yaxis_title='')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ── ACT IV — INQUIRY TRAP ─────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FF4D6D;letter-spacing:4px'>ACT IV</div>
<h2 style='margin-top:2px;color:#E8E8F5'>The Inquiry Trap</h2>
<p style='color:#666688;font-size:0.9rem'>Every desperate credit application <b style='color:#FF4D6D'>hurts your score</b> — the trap punishes you for trying to escape it</p>
""", unsafe_allow_html=True)

inq_bins = [0, 5, 10, 15, 20, 100]
inq_labels = ['0–5', '6–10', '11–15', '16–20', '20+']
df['Inq_Bin'] = pd.cut(df['Num_Credit_Inquiries'].clip(0, 100), bins=inq_bins, include_lowest=True)
ct4 = pd.crosstab(df['Inq_Bin'], df['Credit_Score'], normalize='index') * 100
ct4.index = inq_labels[:len(ct4)]

fig4 = go.Figure()
for score, color in SC.items():
    if score in ct4.columns:
        fig4.add_trace(go.Scatter(x=ct4.index, y=ct4[score], name=score,
                                  line=dict(color=color, width=2.5),
                                  mode='lines+markers',
                                  marker=dict(size=8, color=color)))
fig4.update_layout(**PLOTLY_LAYOUT, title='% Credit Score by Number of Credit Inquiries',
                   xaxis_title='Credit Inquiries', yaxis_title='% of People')
st.plotly_chart(fig4, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ── THE VERDICT ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#BD93F9;letter-spacing:4px'>THE VERDICT</div>
<h2 style='margin-top:2px;color:#E8E8F5'>The Trap Is Not Income. It Is Behaviour.</h2>
""", unsafe_allow_html=True)

verdicts = [
    (GOOD,  "GOOD",     "<5 missed payments · $664 median debt · 21.9 yr history"),
    (STD,   "STANDARD", "~7 missed payments · $1,015 median debt · 17.3 yr history"),
    (POOR,  "POOR",     ">15 missed payments · $1,897 median debt · 13.2 yr history"),
]
for color, label, desc in verdicts:
    st.markdown(f"""
    <div style='background:linear-gradient(90deg,{"rgba(0,229,160,0.05)" if color==GOOD else "rgba(255,77,109,0.05)" if color==POOR else "rgba(255,209,102,0.05)"},transparent);
                border-left:3px solid {color};padding:16px 20px;border-radius:0 8px 8px 0;margin-bottom:10px;display:flex;align-items:center;gap:20px'>
        <div style='font-family:Space Mono,monospace;font-size:0.85rem;font-weight:700;color:{color};min-width:80px'>{label}</div>
        <div style='color:#888899;font-size:0.85rem'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)
