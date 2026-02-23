import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

GOOD  = '#00E5A0'; STD = '#FFD166'; POOR = '#FF4D6D'
BG = '#07070F'; PANEL = '#11111E'; GRID = '#1C1C2E'

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

LABEL_MAP = {0: 'Poor', 1: 'Standard', 2: 'Good'}
COLOR_MAP  = {'Good': GOOD, 'Standard': STD, 'Poor': POOR}
EMOJI_MAP  = {'Good': '✅', 'Standard': '⚠️', 'Poor': '🚨'}
DESC_MAP   = {
    'Good':     'Your financial profile suggests strong credit habits. Low debt, consistent payments, long history.',
    'Standard': 'Your profile is average. There are clear areas to improve — especially payment consistency and debt load.',
    'Poor':     'Your profile shows significant financial stress. Missed payments and high debt are pulling your score down.',
}

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:4px;padding-top:16px'>TRAP FINDER</div>
<h1 style='margin-top:4px'>Where Do You Fall?</h1>
<p style='color:#666688;max-width:560px'>
Enter your financial details below. The model — trained on 14,879 real credit profiles —
will predict which credit tier you belong to.
</p>
<hr style='border-color:#1C1C2E;margin:20px 0'/>
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>PERSONAL</div>", unsafe_allow_html=True)
    age             = st.slider("Age",                          18,  75,  30)
    annual_income   = st.slider("Annual Income ($)",         5000, 180000, 50000, step=1000)
    monthly_salary  = st.slider("Monthly Take-Home ($)",      500,  15000,  4000, step=100)
    occupation      = st.selectbox("Occupation", [
        'Accountant','Architect','Developer','Doctor','Engineer',
        'Entrepreneur','Journalist','Lawyer','Manager','Mechanic',
        'Media_Manager','Musician','Scientist','Teacher','Writer','Other'
    ])

with c2:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>CREDIT PROFILE</div>", unsafe_allow_html=True)
    num_bank        = st.slider("Bank Accounts",               1,  10,   3)
    num_cc          = st.slider("Credit Cards",                1,  11,   4)
    interest_rate   = st.slider("Interest Rate (%)",           1, 100,  14)
    num_loan        = st.slider("Number of Loans",             0,  11,   3)
    credit_hist     = st.slider("Credit History (months)",     0, 400, 120)
    credit_util     = st.slider("Credit Utilization (%)",      0, 100,  30)

with c3:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>PAYMENT BEHAVIOUR</div>", unsafe_allow_html=True)
    delay_days      = st.slider("Avg Days Late on Payments",   0,  60,   5)
    num_delayed     = st.slider("Missed Payments (count)",     0,  30,   3)
    changed_limit   = st.slider("Credit Limit Change ($)",   -20,  30,   5)
    num_inquiries   = st.slider("Credit Inquiries",            0,  50,  10)
    outstanding     = st.slider("Outstanding Debt ($)",        0, 5000, 800, step=50)
    emi             = st.slider("Monthly EMI ($)",             0, 2000, 200, step=10)
    invested        = st.slider("Monthly Investment ($)",      0, 2000, 150, step=10)
    balance         = st.slider("Monthly Balance ($)",      -500, 5000, 300, step=50)

st.markdown("<br>", unsafe_allow_html=True)

# Loan type checkboxes
st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>LOAN TYPES</div>", unsafe_allow_html=True)
lc1, lc2, lc3, lc4 = st.columns(4)
has_personal  = lc1.checkbox("Personal Loan")
has_payday    = lc1.checkbox("Payday Loan")
has_builder   = lc2.checkbox("Credit-Builder")
has_student   = lc2.checkbox("Student Loan")
has_mortgage  = lc3.checkbox("Mortgage Loan")
has_equity    = lc3.checkbox("Home Equity Loan")
has_debt_con  = lc4.checkbox("Debt Consolidation")
has_auto      = lc4.checkbox("Auto Loan")

# Credit mix & payment behaviour
st.markdown("<br>", unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3)
credit_mix        = mc1.selectbox("Credit Mix",              ['Good', 'Standard', 'Bad', 'Unknown'])
payment_min       = mc2.selectbox("Pay Minimum Amount?",     ['Yes', 'No', 'NM'])
spending_level    = mc3.selectbox("Spending Level",          ['High_spent', 'Low_spent'])
payment_val       = mc3.selectbox("Payment Value Size",      ['Large_value_payments', 'Medium_value_payments', 'Small_value_payments'])

predict_btn = st.button("⚡  PREDICT MY CREDIT TIER", use_container_width=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
if predict_btn:
    # Build a zero-row dict matching all training columns
    input_dict = {col: 0 for col in columns}

    # Numeric assignments
    input_dict['Age']                       = age
    input_dict['Annual_Income']             = annual_income
    input_dict['Monthly_Inhand_Salary']     = monthly_salary
    input_dict['Num_Bank_Accounts']         = num_bank
    input_dict['Num_Credit_Card']           = num_cc
    input_dict['Interest_Rate']             = interest_rate
    input_dict['Num_of_Loan']               = num_loan
    input_dict['Delay_from_due_date']       = delay_days
    input_dict['Num_of_Delayed_Payment']    = num_delayed
    input_dict['Changed_Credit_Limit']      = changed_limit
    input_dict['Num_Credit_Inquiries']      = num_inquiries
    input_dict['Outstanding_Debt']          = outstanding
    input_dict['Credit_Utilization_Ratio']  = credit_util
    input_dict['Total_EMI_per_month']       = emi
    input_dict['Amount_invested_monthly']   = invested
    input_dict['Monthly_Balance']           = balance
    input_dict['Credit_History_Age_Months'] = credit_hist

    # Loan flags
    input_dict['has_Personal_Loan']             = int(has_personal)
    input_dict['has_Payday_Loan']               = int(has_payday)
    input_dict['has_Credit-Builder_Loan']       = int(has_builder)
    input_dict['has_Student_Loan']              = int(has_student)
    input_dict['has_Mortgage_Loan']             = int(has_mortgage)
    input_dict['has_Home_Equity_Loan']          = int(has_equity)
    input_dict['has_Debt_Consolidation_Loan']   = int(has_debt_con)
    input_dict['has_Auto_Loan']                 = int(has_auto)

    # Occupation dummies
    occ_col = f'Occupation_{occupation}'
    if occ_col in input_dict:
        input_dict[occ_col] = 1

    # Spending behaviour dummies
    spend_col = f'Spending_Behaviour_{spending_level}'
    if spend_col in input_dict:
        input_dict[spend_col] = 1

    pay_col = f'Payment_Value_{payment_val}'
    if pay_col in input_dict:
        input_dict[pay_col] = 1

    # Credit mix dummies
    if credit_mix in ['Good', 'Standard', '_']:
        mix_col = f'Credit_Mix_{credit_mix}'
        if mix_col in input_dict:
            input_dict[mix_col] = 1

    # Payment of min amount dummies
    pmin_col = f'Payment_of_Min_Amount_{payment_min}'
    if pmin_col in input_dict:
        input_dict[pmin_col] = 1

    # Build DataFrame and predict
    input_df  = pd.DataFrame([input_dict])[columns]
    pred_num  = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0]
    pred_label = LABEL_MAP[pred_num]
    color      = COLOR_MAP[pred_label]

    st.markdown("<hr style='border-color:#1C1C2E;margin:28px 0'/>", unsafe_allow_html=True)

    # Result card
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{"rgba(0,229,160,0.08)" if pred_label=="Good" else "rgba(255,77,109,0.08)" if pred_label=="Poor" else "rgba(255,209,102,0.08)"},#07070F);
                border:1px solid {color};border-radius:12px;padding:32px;text-align:center;margin-bottom:24px'>
        <div style='font-size:3rem;margin-bottom:8px'>{EMOJI_MAP[pred_label]}</div>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:{color};letter-spacing:6px;margin-bottom:8px'>PREDICTED TIER</div>
        <div style='font-family:Space Mono,monospace;font-size:3rem;font-weight:700;color:{color};line-height:1'>{pred_label.upper()}</div>
        <div style='color:#666688;margin-top:16px;max-width:480px;margin-left:auto;margin-right:auto;font-size:0.9rem;line-height:1.6'>{DESC_MAP[pred_label]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>CONFIDENCE BREAKDOWN</div>", unsafe_allow_html=True)

    prob_cols = st.columns(3)
    for i, (label, col, pc) in enumerate(zip(
        ['Poor', 'Standard', 'Good'],
        [POOR, STD, GOOD],
        [pred_prob[0], pred_prob[1], pred_prob[2]]
    )):
        with prob_cols[i]:
            st.markdown(f"""
            <div style='background:#11111E;border:1px solid #1C1C2E;border-radius:8px;padding:16px;text-align:center'>
                <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:{col};letter-spacing:3px'>{label.upper()}</div>
                <div style='font-size:1.8rem;font-weight:700;color:{col};font-family:Space Mono,monospace'>{pc*100:.1f}%</div>
                <div style='background:#1C1C2E;border-radius:4px;height:6px;margin-top:10px;overflow:hidden'>
                    <div style='background:{col};width:{pc*100:.0f}%;height:100%;border-radius:4px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Key risk factors
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#444466;letter-spacing:2px;margin-bottom:12px'>YOUR KEY SIGNALS</div>", unsafe_allow_html=True)

    dti = outstanding / max(annual_income / 12, 1)
    signals = [
        ("Debt-to-Income",     f"{dti:.2f}",  GOOD if dti < 0.3 else POOR if dti > 1.0 else STD),
        ("Missed Payments",    str(num_delayed), GOOD if num_delayed < 5 else POOR if num_delayed > 15 else STD),
        ("Credit Inquiries",   str(num_inquiries), GOOD if num_inquiries < 10 else POOR if num_inquiries > 25 else STD),
        ("Credit History",     f"{credit_hist} mo", GOOD if credit_hist > 200 else POOR if credit_hist < 60 else STD),
    ]

    sig_cols = st.columns(4)
    for (label, val, col), sc in zip(signals, sig_cols):
        sc.markdown(f"""
        <div style='background:#11111E;border-left:3px solid {col};padding:12px;border-radius:0 6px 6px 0'>
            <div style='font-size:0.6rem;color:#444466;font-family:Space Mono,monospace'>{label}</div>
            <div style='font-size:1.1rem;font-weight:700;color:{col};font-family:Space Mono,monospace'>{val}</div>
        </div>
        """, unsafe_allow_html=True)
