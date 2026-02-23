import streamlit as st

st.set_page_config(
    page_title="The Credit Trap",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07070F;
    color: #E8E8F5;
}
.stApp { background-color: #07070F; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #0D0D1C !important;
    border-right: 1px solid #1C1C2E;
}
section[data-testid="stSidebar"] * { color: #A0A0C0 !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #666688 !important; }

/* ── HEADINGS ── */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #F0F0FA !important; letter-spacing: -1px; }
h2 { color: #C0C0E0 !important; }
h3 { color: #9090C0 !important; font-size: 1rem !important; }

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: #11111E;
    border: 1px solid #1C1C2E;
    border-radius: 10px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #666688 !important; font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F0F0FA !important; font-family: 'Space Mono', monospace !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #1C1C2E, #252540) !important;
    color: #E8E8F5 !important;
    border: 1px solid #3A3A5C !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00E5A0 !important;
    color: #00E5A0 !important;
    box-shadow: 0 0 12px rgba(0,229,160,0.2) !important;
}

/* ── SLIDERS ── */
.stSlider > div > div > div > div { background: #00E5A0 !important; }

/* ── SELECT BOXES ── */
.stSelectbox > div > div {
    background-color: #11111E !important;
    border-color: #1C1C2E !important;
    color: #E8E8F5 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background-color: #0D0D1C; border-bottom: 1px solid #1C1C2E; }
.stTabs [data-baseweb="tab"] { color: #666688 !important; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.stTabs [aria-selected="true"] { color: #00E5A0 !important; border-bottom-color: #00E5A0 !important; }

/* ── DIVIDER ── */
hr { border-color: #1C1C2E !important; }

/* ── PLOTLY CONTAINER ── */
.js-plotly-plot { border-radius: 10px; }

/* ── CAPTION / SMALL TEXT ── */
.stCaption, small { color: #555575 !important; font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #07070F; }
::-webkit-scrollbar-thumb { background: #1C1C2E; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px 0'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#F0F0FA;font-weight:700;letter-spacing:-0.5px'>
            💳 THE CREDIT<br>&nbsp;&nbsp;&nbsp;TRAP
        </div>
        <div style='font-size:0.65rem;color:#444466;font-family:Space Mono,monospace;margin-top:6px'>
            A DATA STORY
        </div>
    </div>
    <hr style='border-color:#1C1C2E;margin-bottom:16px'/>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.65rem;color:#444466;font-family:Space Mono,monospace;margin-bottom:12px'>
    NAVIGATE
    </div>
    """, unsafe_allow_html=True)

    st.page_link("app.py",                 label="🏠  Home",           )
    st.page_link("pages/1_story.py",       label="📖  The Story",      )
    st.page_link("pages/2_explore.py",     label="🔍  Explore Data",   )
    st.page_link("pages/3_trap_finder.py", label="🎯  Trap Finder",    )
    st.page_link("pages/4_model.py",       label="🤖  Model Report",   )

    st.markdown("""
    <hr style='border-color:#1C1C2E;margin-top:24px'/>
    <div style='font-size:0.6rem;color:#333355;font-family:Space Mono,monospace;line-height:1.8'>
    14,879 credit profiles<br>
    Random Forest · sklearn<br>
    77.0% accuracy
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 60px 0 20px 0'>
    <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#444466;letter-spacing:4px;margin-bottom:12px'>
        A DATA SCIENCE PROJECT
    </div>
    <h1 style='font-size:3.2rem;margin:0;line-height:1.1'>
        THE<br>CREDIT<br>TRAP
    </h1>
    <div style='width:60px;height:3px;background:linear-gradient(90deg,#00E5A0,#FFD166,#FF4D6D);margin:20px 0'></div>
    <p style='color:#666688;max-width:500px;line-height:1.8;font-size:0.95rem'>
        How small financial habits compound into permanent credit tiers —
        told through 14,879 real profiles and a machine learning model
        that predicts your fate with 77% accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Profiles",    "14,879")
c2.metric("Good Scorers",      "17.8%",  delta="minority class")
c3.metric("Model Accuracy",    "77.0%",  delta="Random Forest")
c4.metric("Features Used",     "50",     delta="after encoding")

st.markdown("<br>", unsafe_allow_html=True)

# Credit tier cards
st.markdown("""
<div style='display:flex;gap:16px;margin-top:8px'>
    <div style='flex:1;background:#0A1A13;border:1px solid #00E5A0;border-radius:10px;padding:20px'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#00E5A0;letter-spacing:3px'>GOOD</div>
        <div style='font-size:1.8rem;font-weight:700;color:#00E5A0;font-family:Space Mono,monospace'>17.8%</div>
        <div style='color:#446655;font-size:0.8rem;margin-top:4px'>Median debt $664 · 21+ yr history</div>
    </div>
    <div style='flex:1;background:#1A1A0A;border:1px solid #FFD166;border-radius:10px;padding:20px'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FFD166;letter-spacing:3px'>STANDARD</div>
        <div style='font-size:1.8rem;font-weight:700;color:#FFD166;font-family:Space Mono,monospace'>52.5%</div>
        <div style='color:#665544;font-size:0.8rem;margin-top:4px'>Median debt $1,015 · 17 yr history</div>
    </div>
    <div style='flex:1;background:#1A0A0E;border:1px solid #FF4D6D;border-radius:10px;padding:20px'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#FF4D6D;letter-spacing:3px'>POOR</div>
        <div style='font-size:1.8rem;font-weight:700;color:#FF4D6D;font-family:Space Mono,monospace'>29.6%</div>
        <div style='color:#664455;font-size:0.8rem;margin-top:4px'>Median debt $1,897 · 13 yr history</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#333355;text-align:center'>
    USE THE SIDEBAR TO NAVIGATE · START WITH THE STORY →
</div>
""", unsafe_allow_html=True)
