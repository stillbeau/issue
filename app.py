import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import json
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Attempt to import optional libraries ---
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

# --- Constants & Configuration ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
MORAWARE_SEARCH_URL = "https://floformcountertops.moraware.net/sys/search?&search="
INSTALL_COST_PER_SQFT = 15.0

# --- Material Parsing Helper ---
def parse_material(s: str):
    brand_match = re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(', s)
    color_match = re.search(r'\)\s*([^()]+?)\s*\(', s)
    brand = brand_match.group(1).strip() if brand_match else ""
    color = color_match.group(1).strip() if color_match else ""
    return brand, color

# --- Load & Process Data ---
@st.cache_data(ttl=300)
def load_and_process_data(creds_dict):
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())

    # --- Numeric cols cleaning ---
    num_map = {
        'Total Job Price $': 'Revenue',
        'Job Throughput - Job Plant Invoice': 'Cost_From_Plant',
        'Job Throughput - Rework COGS': 'Rework_COGS',
        'Job Throughput - Rework Job Labor': 'Rework_Labor',
        'Job Throughput - Job GM (original)': 'Original_GM',
        'Job Throughput - Job SqFt': 'Total_Job_SqFt'
    }
    for o, n in num_map.items():
        if o in df:
            df[n] = (df[o].astype(str)
                     .str.replace(r'[\$,]', '', regex=True)
                     .astype(float).fillna(0))
        else:
            df[n] = 0.0
    
    # --- Dates parse ---
    date_cols = ['Template - Date', 'Ready to Fab - Date', 'Ship-Blank - Date', 'Install - Date']
    for c in date_cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    
    # --- Profitability ---
    df['Install Cost'] = df['Total_Job_SqFt'] * INSTALL_COST_PER_SQFT
    df['Total Rework Cost'] = df['Rework_COGS'] + df['Rework_Labor']
    df['Total Branch Cost'] = df['Cost_From_Plant'] + df['Install Cost'] + df['Total Rework Cost']
    df['Branch Profit'] = df['Revenue'] - df['Total Branch Cost']
    df['Branch Profit Margin %'] = df.apply(
        lambda r: (r['Branch Profit'] / r['Revenue'] * 100) if r['Revenue'] else 0, axis=1
    )
    df['Profit Variance'] = df['Branch Profit'] - df['Original_GM']

    # --- Material parse ---
    if 'Job Material' in df.columns:
        df[['Material Brand', 'Material Color']] = df['Job Material']\
            .apply(lambda x: pd.Series(parse_material(str(x))))
    else:
        df['Material Brand'] = ""
        df['Material Color'] = ""

    # --- Stage durations ---
    df['Days_Template_to_RTF'] = (df['Ready to Fab - Date'] - df['Template - Date']).dt.days
    df['Days_RTF_to_Ship'] = (df['Ship-Blank - Date'] - df['Ready to Fab - Date']).dt.days
    df['Days_Ship_to_Install'] = (df['Install - Date'] - df['Ship-Blank - Date']).dt.days

    # --- Job link ---
    df['Job Link'] = MORAWARE_SEARCH_URL + df['Production #'].astype(str)
    
    return df

# --- Credentials & Initial Load ---
st.sidebar.header("⚙️ Configuration")
creds = None
if "google_creds_json" in st.secrets:
    creds = json.loads(st.secrets["google_creds_json"])
else:
    up = st.sidebar.file_uploader("Upload Service Account JSON", type="json")
    if up:
        creds = json.load(up)
if not creds:
    st.sidebar.info("Please provide Google credentials.")
    st.stop()

df_full = load_and_process_data(creds)

# Date-range filter
if 'Template - Date' in df_full.columns and not df_full['Template - Date'].dropna().empty:
    min_d = df_full['Template - Date'].min().date()
    max_d = df_full['Template - Date'].max().date()
    start_date, end_date = st.sidebar.date_input("Template Date Range", [min_d, max_d])
    df = df_full[(df_full['Template - Date'].dt.date >= start_date) & (df_full['Template - Date'].dt.date <= end_date)].copy()
else:
    st.sidebar.warning("'Template - Date' column not found or is empty. Cannot apply date filter.")
    df = df_full.copy()


# --- Sidebar Filters ---
st.sidebar.header("Filters")
def get_opts(col): return sorted(df[col].dropna().unique()) if col in df else []

sales_opts = get_opts('Salesperson')
cat_opts = get_opts('Customer Category')
mat_opts = get_opts('Material Brand')
city_opts = get_opts('City')

sel_sales = st.sidebar.multiselect("Salesperson", sales_opts, default=sales_opts)
sel_cat = st.sidebar.multiselect("Customer Category", cat_opts, default=cat_opts)
sel_mat = st.sidebar.multiselect("Material Brand", mat_opts, default=mat_opts)
sel_city = st.sidebar.multiselect("City", city_opts, default=city_opts)

# Apply filters safely
if sel_sales and 'Salesperson' in df.columns: df = df[df['Salesperson'].isin(sel_sales)]
if sel_cat and 'Customer Category' in df.columns: df = df[df['Customer Category'].isin(sel_cat)]
if sel_mat and 'Material Brand' in df.columns: df = df[df['Material Brand'].isin(sel_mat)]
if sel_city and 'City' in df.columns: df = df[df['City'].isin(sel_city)]


# --- Tabs ---
tabs = st.tabs([
    "📈 Overall",
    "📋 Detailed Data",
    "🔬 Rework & Variance",
    "🔍 Phase Drilldown",
    "🌐 Geo & Clusters",
    "⏱️ Stage Durations",
    "📅 Forecasts & Trends"
])

# Helper to apply in-tab filters
def apply_filters(df_tab, key_prefix):
    with st.expander("Filter This Tab"):
        sel_sales = st.multiselect("Salesperson", sales_opts, default=sales_opts, key=f"{key_prefix}_sales")
        sel_cat = st.multiselect("Customer Category", cat_opts, default=cat_opts, key=f"{key_prefix}_cat")
        sel_mat = st.multiselect("Material Brand", mat_opts, default=mat_opts, key=f"{key_prefix}_mat")
        sel_city = st.multiselect("City", city_opts, default=city_opts, key=f"{key_prefix}_city")
    
    if sel_sales: df_tab = df_tab[df_tab['Salesperson'].isin(sel_sales)]
    if sel_cat: df_tab = df_tab[df_tab['Customer Category'].isin(sel_cat)]
    if sel_mat: df_tab = df_tab[df_tab['Material Brand'].isin(sel_mat)]
    if sel_city: df_tab = df_tab[df_tab['City'].isin(sel_city)]
    return df_tab

# Tab1: Overall
with tabs[0]:
    st.header("📈 Overall Performance")
    df_tab = apply_filters(df, "tab1")
    total_rev = df_tab['Revenue'].sum()
    total_prof = df_tab['Branch Profit'].sum()
    avg_marg = (total_prof / total_rev * 100) if total_rev else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Profit", f"${total_prof:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_marg:.1f}%")
    st.markdown("---")
    st.subheader("Profit by Salesperson")
    st.bar_chart(df_tab.groupby('Salesperson')['Branch Profit'].sum())
    st.subheader("Material Brand Leaderboard")
    mat_lb = df_tab.groupby('Material Brand').agg(
        Total_Profit=('Branch Profit', 'sum'),
        Avg_Margin=('Branch Profit Margin %', 'mean')
    ).sort_values('Total_Profit', ascending=False)
    st.dataframe(mat_lb)
    st.subheader("Low Profit Alerts")
    thresh = st.number_input("Margin below (%)", min_value=0.0, max_value=100.0, value=10.0, key="lp_thresh")
    low = df_tab[df_tab['Branch Profit Margin %'] < thresh]
    if not low.empty:
        st.markdown(f"Jobs with margin below {thresh}%:")
        st.dataframe(low[['Production #', 'Job Name', 'Branch Profit Margin %', 'Branch Profit']])
    else:
        st.write("No low-profit jobs.")

# Tab2: Detailed
with tabs[1]:
    st.header("📋 Detailed Data")
    df_tab = apply_filters(df, "tab2")
    cols = ['Production #', 'Job Link', 'Job Name', 'Revenue', 'Branch Profit', 'Branch Profit Margin %']
    df_disp = df_tab[[c for c in cols if c in df_tab]]
    st.dataframe(df_disp, use_container_width=True, column_config={"Job Link": st.column_config.LinkColumn("Job Link", "Open ↗")})

# Tab3: Rework
with tabs[2]:
    st.header("🔬 Rework Insights")
    df_tab = apply_filters(df, "tab3")
    if 'Rework_COGS' in df_tab:
        df_tab['Rework Cost'] = df_tab['Rework_COGS'] + df_tab['Rework_Labor']
        agg = df_tab.groupby('Rework - Stone Shop - Reason')['Rework Cost'].agg(['sum', 'count'])
        st.dataframe(agg)
    st.subheader("Low Profit in Rework")
    low_r = df_tab[df_tab['Branch Profit'] < 0]
    if not low_r.empty:
        st.dataframe(low_r[['Production #', 'Job Name', 'Branch Profit']])

# Tab4: Phase
with tabs[3]:
    st.header("🔍 Phase Drilldown")
    df_tab = apply_filters(df, "tab4")
    if 'Phase Throughput - Name' in df_tab:
        ph = st.selectbox("Phase", sorted(df_tab['Phase Throughput - Name'].unique()))
        sub = df_tab[df_tab['Phase Throughput - Name'] == ph]
        metrics = {
            'Total Rev': sub['Phase Throughput - Phase Rev'].sum(),
            'Avg Margin%': sub['Phase Throughput - Phase GM %'].mean()
        }
        st.json(metrics)

# Tab5: Geo
with tabs[4]:
    st.header("🌐 Geo & Clusters")
    df_tab = apply_filters(df, "tab5")
    st.dataframe(df_tab.groupby('City')['Branch Profit'].sum())

# Tab6: Durations
with tabs[5]:
    st.header("⏱️ Stage Durations")
    df_tab = apply_filters(df, "tab6")
    avg = {
        'Temp→RTF': df_tab['Days_Template_to_RTF'].mean(),
        'RTF→Ship': df_tab['Days_RTF_to_Ship'].mean(),
        'Ship→Inst': df_tab['Days_Ship_to_Install'].mean()
    }
    st.json(avg)
    st.subheader("Duration Distributions")
    if MATPLOTLIB_AVAILABLE:
        for col in avg.keys():
            fig, ax = plt.subplots()
            df_tab[col].dropna().hist(ax=ax)
            ax.set_title(col)
            st.pyplot(fig)
    else:
        st.warning("Duration charts require the 'matplotlib' library. Please add it to your requirements.txt file.")


# Tab7: Trends
with tabs[6]:
    st.header("📅 Forecasts & Trends")
    df_tab = apply_filters(df, "tab7")
    ts = df_tab.set_index('Template - Date').resample('M').agg(
        Rev=('Revenue', 'sum'), Jobs=('Production #', 'count')
    )
    st.line_chart(ts)
    
    st.subheader("Linear Forecast (Next 3 Months)")
    if SKLEARN_AVAILABLE:
        last = ts.tail(6)['Rev'].reset_index(drop=True)
        model = LinearRegression().fit(last.index.values.reshape(-1, 1), last.values)
        future_idx = pd.DataFrame({'x': range(6, 9)})
        preds = model.predict(future_idx[['x']])
        fc = pd.Series(preds, index=pd.date_range(start=ts.index[-1] + relativedelta(months=1), periods=3, freq='M'))
        st.line_chart(fc, height=200)
    else:
        st.warning("Forecasting requires the 'scikit-learn' library. Please add it to your requirements.txt file.")

