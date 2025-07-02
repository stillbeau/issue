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
    for orig, new in num_map.items():
        if orig in df:
            cleaned_series = df[orig].astype(str).str.replace(r'[$,]', '', regex=True)
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            df[new] = numeric_series.fillna(0)
        else:
            df[new] = 0.0
    
    # --- Dates parse ---
    date_cols = ['Template - Date', 'Ready to Fab - Date', 'Install - Date', 'Product Rcvd - Date']
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
    if 'Ready to Fab - Date' in df.columns and 'Template - Date' in df.columns:
        df['Days_Template_to_RTF'] = (df['Ready to Fab - Date'] - df['Template - Date']).dt.days
    if 'Install - Date' in df.columns and 'Template - Date' in df.columns:
        df['Days_Template_to_Install'] = (df['Install - Date'] - df['Template - Date']).dt.days
    if 'Product Rcvd - Date' in df.columns and 'Ready to Fab - Date' in df.columns:
        df['Days_RTF_to_Rcvd'] = (df['Product Rcvd - Date'] - df['Ready to Fab - Date']).dt.days
    
    # Handle illogical negative durations by converting them to NaN so they are ignored in calculations
    for col in ['Days_Template_to_RTF', 'Days_Template_to_Install', 'Days_RTF_to_Rcvd']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # --- Job link ---
    if 'Production #' in df.columns:
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

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Date-range filter
if 'Template - Date' in df_full.columns and not df_full['Template - Date'].dropna().empty:
    min_d = df_full['Template - Date'].min().date()
    max_d = df_full['Template - Date'].max().date()
    start_date, end_date = st.sidebar.date_input("Template Date Range", [min_d, max_d])
    df = df_full[(df_full['Template - Date'].dt.date >= start_date) & (df_full['Template - Date'].dt.date <= end_date)].copy()
else:
    st.sidebar.warning("'Template - Date' column not found or is empty. Cannot apply date filter.")
    df = df_full.copy()

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

# Tab1: Overall
with tabs[0]:
    st.header("📈 Overall Performance")
    total_rev = df['Revenue'].sum()
    total_prof = df['Branch Profit'].sum()
    avg_marg = (total_prof / total_rev * 100) if total_rev else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Profit", f"${total_prof:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_marg:.1f}%")
    st.markdown("---")
    st.subheader("Profit by Salesperson")
    if 'Salesperson' in df.columns:
        st.bar_chart(df.groupby('Salesperson')['Branch Profit'].sum())
    st.subheader("Material Brand Leaderboard")
    if 'Material Brand' in df.columns:
        mat_lb = df.groupby('Material Brand').agg(
            Total_Profit=('Branch Profit', 'sum'),
            Avg_Margin=('Branch Profit Margin %', 'mean')
        ).sort_values('Total_Profit', ascending=False)
        st.dataframe(mat_lb.style.format({'Total_Profit': '${:,.2f}', 'Avg_Margin': '{:.2f}%'}))
    st.subheader("Low Profit Alerts")
    thresh = st.number_input("Margin below (%)", min_value=-100.0, max_value=100.0, value=10.0, step=1.0, key="lp_thresh")
    low = df[df['Branch Profit Margin %'] < thresh]
    if not low.empty:
        st.markdown(f"Jobs with margin below {thresh}%:")
        st.dataframe(low[['Production #', 'Job Name', 'Branch Profit Margin %', 'Branch Profit']])
    else:
        st.write("No low-profit jobs in this selection.")

# Tab2: Detailed
with tabs[1]:
    st.header("📋 Detailed Data")
    cols = ['Production #', 'Job Link', 'Job Name', 'Revenue', 'Branch Profit', 'Branch Profit Margin %', 'Profit Variance']
    df_disp = df[[c for c in cols if c in df]]
    st.dataframe(
        df_disp, 
        use_container_width=True, 
        column_config={
            "Job Link": st.column_config.LinkColumn("Job Link", display_text="Open ↗"),
            "Revenue": st.column_config.NumberColumn(format='$%.2f'),
            "Branch Profit": st.column_config.NumberColumn(format='$%.2f'),
            "Profit Variance": st.column_config.NumberColumn(format='$%.2f'),
            "Branch Profit Margin %": st.column_config.NumberColumn(format='%.2f%%')
        }
    )

# Tab3: Rework
with tabs[2]:
    st.header("🔬 Rework Insights")
    if 'Rework_COGS' in df:
        rw = df[df['Total Rework Cost'] > 0]
        if not rw.empty and 'Rework - Stone Shop - Reason' in rw.columns:
            agg = rw.groupby('Rework - Stone Shop - Reason')['Total Rework Cost'].agg(['sum', 'count'])
            agg.columns = ['Total Rework Cost', 'Num Jobs']
            st.dataframe(agg.style.format({'Total Rework Cost': '${:,.2f}'}))
        else:
            st.info("No rework costs recorded in this selection.")
    else:
        st.info("Rework columns not found.")
    st.subheader("Low Profit in Rework")
    low_r = df[df['Branch Profit'] < 0]
    if not low_r.empty:
        st.dataframe(low_r[['Production #', 'Job Name', 'Branch Profit']])

# Tab4: Phase
with tabs[3]:
    st.header("🔍 Phase Drilldown")
    if 'Phase Throughput - Name' in df:
        ph = st.selectbox("Phase", sorted(df['Phase Throughput - Name'].dropna().unique()))
        sub = df[df['Phase Throughput - Name'] == ph]
        metrics = {
            'Total Revenue': sub['Phase Throughput - Phase Rev'].sum(),
            'Avg Margin %': sub['Phase Throughput - Phase GM %'].mean()
        }
        st.json(metrics)
    else:
        st.info("Phase throughput data not available.")

# Tab5: Geo
with tabs[4]:
    st.header("🌐 Geo & Clusters")
    if 'City' in df.columns:
        st.dataframe(df.groupby('City')['Branch Profit'].sum())

# Tab6: Durations
with tabs[5]:
    st.header("⏱️ Stage Durations")
    
    st.subheader("Jobs with Illogical Date Sequences")
    if 'Ready to Fab - Date' in df.columns and 'Template - Date' in df.columns:
        illogical_rtf = df[(df['Ready to Fab - Date'].notna()) & (df['Template - Date'].notna()) & (df['Ready to Fab - Date'] < df['Template - Date'])]
        if not illogical_rtf.empty:
            st.warning("Found jobs where 'Ready to Fab' date is BEFORE 'Template' date:")
            st.dataframe(illogical_rtf[['Job Name', 'Production #', 'Template - Date', 'Ready to Fab - Date']])
        else:
            st.success("No illogical RTF dates found.")
    st.markdown("---")

    duration_cols_map = {
        'Template → RTF': 'Days_Template_to_RTF',
        'Template → Install': 'Days_Template_to_Install',
        'RTF → Product Received': 'Days_RTF_to_Rcvd'
    }
    
    avg_durations = {}
    for friendly_name, actual_col in duration_cols_map.items():
        if actual_col in df.columns:
            avg_durations[friendly_name] = df[actual_col].mean()

    st.write("**Average Days in Each Stage**")
    st.json({k: f"{v:.1f}" if pd.notna(v) else "N/A" for k, v in avg_durations.items()})

    st.subheader("Duration Distributions")
    if MATPLOTLIB_AVAILABLE:
        for friendly_name, actual_col in duration_cols_map.items():
            if actual_col in df.columns and pd.notna(df[actual_col].mean()):
                fig, ax = plt.subplots()
                df[actual_col].dropna().hist(ax=ax)
                ax.set_title(friendly_name)
                st.pyplot(fig)
    else:
        st.warning("Duration charts require the 'matplotlib' library. Please add it to your requirements.txt file.")

# Tab7: Trends
with tabs[6]:
    st.header("📅 Forecasts & Trends")
    if 'Template - Date' in df.columns and not df['Template - Date'].dropna().empty:
        # Ensure the column is datetime before resampling
        df_trends = df.copy()
        df_trends['Template - Date'] = pd.to_datetime(df_trends['Template - Date'], errors='coerce')
        df_trends = df_trends.dropna(subset=['Template - Date'])
        
        if not df_trends.empty:
            ts = df_trends.set_index('Template - Date').resample('M').agg(
                Rev=('Revenue', 'sum'), Jobs=('Production #', 'count')
            )
            st.subheader("Monthly Trend: Revenue & Jobs")
            st.line_chart(ts)
            
            st.subheader("Linear Forecast (Next 3 Months)")
            if SKLEARN_AVAILABLE:
                last = ts.tail(6)['Rev'].reset_index(drop=True)
                if len(last) > 1:
                    model = LinearRegression().fit(last.index.values.reshape(-1, 1), last.values)
                    future_idx = pd.DataFrame({'x': range(len(last), len(last) + 3)})
                    preds = model.predict(future_idx[['x']])
                    fc = pd.Series(preds, index=pd.date_range(start=ts.index[-1] + relativedelta(months=1), periods=3, freq='M'))
                    st.line_chart(fc, height=200)
                else:
                    st.info("Not enough data for a forecast.")
            else:
                st.warning("Forecasting requires the 'scikit-learn' library. Please add it to your requirements.txt file.")
    else:
        st.info("Not enough date data for trend analysis.")
