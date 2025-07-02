import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import json
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Optional imports
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="💰")

# --- App Title ---
st.title("💰 Job Profitability Dashboard")
st.markdown("Analyzes job data from Google Sheets to calculate profitability metrics.")

# --- Constants ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="
INSTALL_COST_PER_SQFT = 15.0

# --- Material Parsing ---
def parse_material(s: str):
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    return (
        brand_match.group(1).strip() if brand_match else "",
        color_match.group(1).strip() if color_match else ""
    )

# --- Load & Process Data ---
@st.cache_data(ttl=300)
def load_and_process_data(creds_dict):
    creds = Credentials.from_service_account_info(creds_dict, scopes=[
        'https://www.googleapis.com/auth/spreadsheets'
    ])
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())

    # Numeric cleanup
    num_map = {
        'Total Job Price $': 'Revenue',
        'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
        'Total Job SqFT': 'Total_Job_SqFt',
        'Job Throughput - Rework COGS': 'Rework_COGS',
        'Job Throughput - Rework Job Labor': 'Rework_Labor',
        'Job Throughput - Job GM (original)': 'Original_GM'
    }
    for orig, new in num_map.items():
        if orig in df.columns:
            cleaned = df[orig].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[new] = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        else:
            df[new] = 0.0

    # Date parse
    for col in ['Template - Date','Ready to Fab - Date','Ship-Blank - Date','Install - Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Profit calculations
    df['Install Cost'] = df['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT
    df['Total Rework Cost'] = df['Rework_COGS'] + df['Rework_Labor']
    df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost'] + df['Total Rework Cost']
    df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
    df['Branch Profit Margin %'] = df.apply(
        lambda r: (r['Branch Profit']/r['Revenue']*100) if r['Revenue'] else 0, axis=1
    )
    df['Profit Variance'] = df['Branch Profit'] - df['Original_GM']

    # Material columns
    if 'Job Material' in df.columns:
        mat = df['Job Material'].astype(str).apply(parse_material)
        df[['Material Brand','Material Color']] = pd.DataFrame(mat.tolist(), index=df.index)
    else:
        df['Material Brand'], df['Material Color'] = "", ""

    # Stage durations
    df['Days_Template_to_RTF'] = (df['Ready to Fab - Date'] - df['Template - Date']).dt.days
    df['Days_RTF_to_Ship']     = (df['Ship-Blank - Date'] - df['Ready to Fab - Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install - Date'] - df['Ship-Blank - Date']).dt.days
    for col in ['Days_Template_to_RTF','Days_RTF_to_Ship','Days_Ship_to_Install']:
        df.loc[df[col] < 0, col] = pd.NA

    # Job Link
    if 'Production #' in df.columns:
        df['Job Link'] = MORAWARE_SEARCH_URL + df['Production #'].astype(str)

    return df

# --- Credentials ---
st.sidebar.header("⚙️ Config")
creds = None
if "google_creds_json" in st.secrets:
    creds = json.loads(st.secrets["google_creds_json"])
else:
    up = st.sidebar.file_uploader("Upload JSON Key", type="json")
    if up:
        creds = json.load(up)
if not creds:
    st.sidebar.error("Google credentials required.")
    st.stop()

df_full = load_and_process_data(creds)

# --- Helper: In-Tab Filters ---
def filter_within_tab(df):
    with st.expander("Filters", expanded=True):
        # Date range
        if 'Template - Date' in df.columns:
            min_d = df['Template - Date'].min().date()
            max_d = df['Template - Date'].max().date()
            dr = st.date_input("Template Date Range", [min_d, max_d])
            df = df[(df['Template - Date'].dt.date >= dr[0]) & (df['Template - Date'].dt.date <= dr[1])]
        # Multi-filters
        for col, label in [
            ('Salesperson','Salesperson'),
            ('Customer Category','Category'),
            ('Material Brand','Material'),
            ('City','City')
        ]:
            if col in df.columns:
                opts = sorted(df[col].dropna().unique())
                sel = st.multiselect(f"Filter by {label}", opts, default=opts)
                df = df[df[col].isin(sel)]
    return df

# --- Tabs ---
tabs = st.tabs([
    "Overall","Detailed","Rework","Phase","Geo","Durations","Trends"
])

# Tab: Overall
with tabs[0]:
    st.header("Overall Performance")
    df = filter_within_tab(df_full)
    rev  = df['Revenue'].sum()
    prof = df['Branch Profit'].sum()
    marg = (prof/rev*100) if rev else 0
    c1,c2,c3 = st.columns(3)
    c1.metric("Revenue", f"${rev:,.0f}")
    c2.metric("Profit", f"${prof:,.0f}")
    c3.metric("Avg Margin", f"{marg:.1f}%")

    st.subheader("Low Profit Jobs")
    thr = st.slider("Margin threshold (%)", min_value=-50, max_value=50, value=10)
    low = df[df['Branch Profit Margin %'] < thr]
    st.dataframe(low[['Production #','Job Name','Branch Profit Margin %']])

# Tab: Detailed
with tabs[1]:
    st.header("Detailed Data")
    df = filter_within_tab(df_full)
    cols = ['Production #','Job Link','Job Name','Revenue','Branch Profit','Branch Profit Margin %']
    st.dataframe(df[cols], use_container_width=True)

# Tab: Rework
with tabs[2]:
    st.header("Rework & Variance")
    df = filter_within_tab(df_full)
    if 'Total Rework Cost' in df:
        agg = df.groupby('Rework - Stone Shop - Reason')['Total Rework Cost']\
                 .agg(['sum','count']).rename(columns={'sum':'Cost','count':'Jobs'})
        st.table(agg)
    st.subheader("Negative Profit Jobs")
    st.dataframe(df[df['Branch Profit'] < 0][['Production #','Job Name','Branch Profit']])

# Tab: Phase
with tabs[3]:
    st.header("Phase Drilldown")
    df = filter_within_tab(df_full)
    if 'Phase Throughput - Name' in df:
        phase = st.selectbox("Select Phase", sorted(df['Phase Throughput - Name'].unique()))
        sub = df[df['Phase Throughput - Name']==phase]
        st.metric("Total Rev", f"${sub['Phase Throughput - Phase Rev'].sum():,.0f}")
        st.metric("Avg Margin", f"{sub['Phase Throughput - Phase GM %'].mean():.1f}%")

# Tab: Geo
with tabs[4]:
    st.header("Geo & Clusters")
    df = filter_within_tab(df_full)
    st.bar_chart(df.groupby('City')['Branch Profit'].sum())

# Tab: Durations
with tabs[5]:
    st.header("Stage Durations")
    df = filter_within_tab(df_full)
    names = {'Days_Template_to_RTF':'Temp→RTF','Days_RTF_to_Ship':'RTF→Ship','Days_Ship_to_Install':'Ship→Inst'}
    avg = {names[c]: df[c].dropna().mean() for c in names if c in df}
    st.json(avg)
    if MATPLOTLIB_AVAILABLE:
        for col, label in names.items():
            fig, ax = plt.subplots()
            df[col].dropna().hist(ax=ax)
            ax.set_title(label)
            st.pyplot(fig)

# Tab: Trends
with tabs[6]:
    st.header("Forecasts & Trends")
    df = filter_within_tab(df_full)
    if 'Template - Date' in df:
        ts = df.set_index('Template - Date').resample('M').agg(
            Rev=('Revenue','sum')
        )
        st.line_chart(ts)
        if SKLEARN_AVAILABLE and len(ts) >= 6:
            last = ts.tail(6)['Rev'].reset_index(drop=True)
            model = LinearRegression().fit(last.index.values.reshape(-1,1), last.values)
            fut = pd.date_range(start=ts.index[-1]+relativedelta(months=1), periods=3, freq='M')
            preds = model.predict([[i] for i in range(len(last), len(last)+3)])
            fc = pd.Series(preds, index=fut, name='Forecast')
            st.line_chart(fc)
    else:
        st.info("Insufficient date data for trends.")
