import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

GOOD  = '#00E5A0'; STD = '#FFD166'; POOR = '#FF4D6D'
BG = '#07070F'; PANEL = '#11111E'; GRID = '#1C1C2E'; TEXT = '#E8E8F5'
SC = {'Good': GOOD, 'Standard': STD, 'Poor': POOR}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family='Space Mono, monospace', color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
    margin=dict(t=50, b=40, l=50, r=20)
)

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df.drop(columns=['Name','Unnamed: 0'], inplace=True, errors='ignore')
    df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace('_',''), errors='coerce')
    df = df[(df['Age'] > 0) & (df['Age'] < 110)]
    for col in ['Monthly_Inhand_Salary','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                'Outstanding_Debt','Annual_Income','Interest_Rate','Credit_Utilization_Ratio',
                'Monthly_Balance','Amount_invested_monthly','Num_of_Loan',
                'Total_EMI_per_month','Delay_from_due_date']:
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
    # Feature engineering
    df['Debt_to_Income']    = df['Outstanding_Debt'] / (df['Annual_Income'] / 12)
    df['EMI_Burden']        = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary']
    df['Savings_Rate']      = df['Amount_invested_monthly'] / df['Monthly_Inhand_Salary']
    df['Inquiry_Pressure']  = df['Num_Credit_Inquiries'] / (df['Credit_History_Months'] / 12).replace(0, np.nan)
    df['Stress_Index']      = (df['Debt_to_Income'].clip(0,5)/5 + df['Num_of_Delayed_Payment'].clip(0,30)/30 + df['Num_Credit_Inquiries'].clip(0,50)/50) / 3
    return df

df = load_data()

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px;padding-top:16px'>EXPLORE</div>
<h1 style='margin-top:4px'>Explore the Data</h1>
<p style='color:#666688'>Interact with every feature. See how the three credit tiers separate.</p>
<hr style='border-color:#1C1C2E;margin:20px 0'/>
""", unsafe_allow_html=True)

# ── FEATURE SELECTOR ──────────────────────────────────────────────────────────
FEATURES = {
    '── RAW FEATURES ──': None,
    'Outstanding Debt':        ('Outstanding_Debt',       0,    5000),
    'Annual Income':           ('Annual_Income',           0,    200000),
    'Interest Rate (%)':       ('Interest_Rate',           0,    100),
    'Delayed Payments':        ('Num_of_Delayed_Payment',  0,    50),
    'Credit Inquiries':        ('Num_Credit_Inquiries',    0,    60),
    'Credit Utilization (%)':  ('Credit_Utilization_Ratio',0,   100),
    'Monthly Balance':         ('Monthly_Balance',         -2000,5000),
    'Credit History (months)': ('Credit_History_Months',   0,    600),
    'Age':                     ('Age',                     18,   80),
    '── ENGINEERED FEATURES ──': None,
    'Debt-to-Income Ratio ★':  ('Debt_to_Income',          0,    3),
    'EMI Burden ★':            ('EMI_Burden',              0,    0.4),
    'Savings Rate ★':          ('Savings_Rate',            0,    0.5),
    'Inquiry Pressure ★':      ('Inquiry_Pressure',        0,    5),
    'Stress Index ★':          ('Stress_Index',            0,    0.7),
}

valid_features = {k: v for k, v in FEATURES.items() if v is not None}

col_sel, col_chart = st.columns([1, 3])

with col_sel:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:8px'>SELECT FEATURE</div>", unsafe_allow_html=True)
    selected_label = st.radio("", list(valid_features.keys()), label_visibility='collapsed')
    col_name, xmin, xmax = valid_features[selected_label]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:8px'>CHART TYPE</div>", unsafe_allow_html=True)
    chart_type = st.radio("", ['Histogram', 'Box Plot', 'Violin'], label_visibility='collapsed')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:8px'>SHOW TIERS</div>", unsafe_allow_html=True)
    show_good = st.checkbox("Good",     value=True)
    show_std  = st.checkbox("Standard", value=True)
    show_poor = st.checkbox("Poor",     value=True)

with col_chart:
    active = {s: c for s, c in SC.items()
              if (s=='Good' and show_good) or (s=='Standard' and show_std) or (s=='Poor' and show_poor)}

    fig = go.Figure()

    if chart_type == 'Histogram':
        for score, color in active.items():
            data = df[df['Credit_Score']==score][col_name].dropna()
            data = data[(data >= xmin) & (data <= xmax)]
            fig.add_trace(go.Histogram(x=data, name=score, marker_color=color,
                                       opacity=0.55, histnorm='probability density', nbinsx=50))
        fig.update_layout(**PLOTLY_LAYOUT, barmode='overlay')

    elif chart_type == 'Box Plot':
        for score, color in active.items():
            data = df[df['Credit_Score']==score][col_name].dropna().clip(xmin, xmax)
            fig.add_trace(go.Box(y=data, name=score, marker_color=color,
                                  line_color=color, fillcolor=color,
                                  opacity=0.7, boxmean=True))
        fig.update_layout(**PLOTLY_LAYOUT)

    else:  # Violin
        for score, color in active.items():
            data = df[df['Credit_Score']==score][col_name].dropna().clip(xmin, xmax)
            fig.add_trace(go.Violin(y=data, name=score, line_color=color,
                                     fillcolor=color, opacity=0.5,
                                     meanline_visible=True, box_visible=True))
        fig.update_layout(**PLOTLY_LAYOUT)

    fig.update_layout(title=f'{selected_label} by Credit Score Tier',
                      height=430)
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    stats_rows = []
    for score in ['Good', 'Standard', 'Poor']:
        d = df[df['Credit_Score']==score][col_name].dropna()
        d = d[(d >= xmin) & (d <= xmax)]
        stats_rows.append({
            'Tier': score,
            'Median': f'{d.median():.2f}',
            'Mean':   f'{d.mean():.2f}',
            'Std':    f'{d.std():.2f}',
            'Min':    f'{d.min():.2f}',
            'Max':    f'{d.max():.2f}',
        })
    stats_df = pd.DataFrame(stats_rows).set_index('Tier')
    st.dataframe(stats_df, use_container_width=True)

st.markdown("<hr style='border-color:#1C1C2E;margin:32px 0'/>", unsafe_allow_html=True)

# ── BUBBLE CHART ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px'>THE CREDIT UNIVERSE</div>
<h2 style='margin-top:2px'>Income vs Debt vs Stress</h2>
<p style='color:#666688;font-size:0.85rem'>Bubble size = delayed payments · Color = credit tier</p>
""", unsafe_allow_html=True)

sample = df.dropna(subset=['Annual_Income','Outstanding_Debt','Stress_Index','Num_of_Delayed_Payment'])
sample = pd.concat([
    sample[sample['Credit_Score']=='Good'].sample(min(350,len(sample[sample['Credit_Score']=='Good'])), random_state=42),
    sample[sample['Credit_Score']=='Standard'].sample(min(450,len(sample[sample['Credit_Score']=='Standard'])), random_state=42),
    sample[sample['Credit_Score']=='Poor'].sample(min(300,len(sample[sample['Credit_Score']=='Poor'])), random_state=42),
])

fig_bub = go.Figure()
for score, color in [('Poor',POOR),('Standard',STD),('Good',GOOD)]:
    sub = sample[sample['Credit_Score']==score]
    sizes = (sub['Num_of_Delayed_Payment'].clip(0,35)/35)*60+5
    fig_bub.add_trace(go.Scatter(
        x=sub['Annual_Income'], y=sub['Outstanding_Debt'],
        mode='markers', name=score,
        marker=dict(size=sizes, color=color, opacity=0.55,
                    line=dict(color=color, width=0.5)),
        hovertemplate=f'<b>{score}</b><br>Income: $%{{x:,.0f}}<br>Debt: $%{{y:,.0f}}<br><extra></extra>'
    ))

fig_bub.update_layout(
    **PLOTLY_LAYOUT, height=420,
    xaxis_title='Annual Income ($)', yaxis_title='Outstanding Debt ($)',
    title='The Credit Universe — Income vs Debt',
    xaxis=dict(**PLOTLY_LAYOUT['xaxis'], tickformat='$,.0f'),
    yaxis=dict(**PLOTLY_LAYOUT['yaxis'], tickformat='$,.0f'),
)
st.plotly_chart(fig_bub, use_container_width=True)
