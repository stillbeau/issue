import streamlit as st
import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import SpreadsheetNotFound, APIError
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from collections import Counter

# --- DEFAULT CONFIGURATION ---
DEFAULT_SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
DEFAULT_WORKSHEET_NAME = "jobs"
INSTALL_COST_PER_SQFT = 15.0
CACHE_TTL = 600

# Shared dataframe column configs
COLUMN_CONFIG = {
    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
    "Days_Behind": st.column_config.NumberColumn("Days Behind/Ahead", help="Positive: Behind. Negative: Ahead."),
    "Revenue": st.column_config.NumberColumn(format='$%.2f'),
    "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
    "Branch_Profit": st.column_config.NumberColumn(format='$%.2f'),
    "Branch_Profit_Margin_%": st.column_config.ProgressColumn("Branch Profit %", format='%.2f%%', min_value=-50, max_value=100)
}

# --- DATA LOADING & PROCESSING ---
@st.cache_data(ttl=CACHE_TTL)
def load_data(creds_dict, sheet_id, worksheet_name):
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    try:
        sheet = gc.open_by_key(sheet_id).worksheet(worksheet_name)
    except SpreadsheetNotFound:
        raise
    except APIError as e:
        st.error(f"Google Sheets API error: {e}")
        st.stop()
    df = pd.DataFrame(sheet.get_all_records())
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+','_',regex=True).str.replace(r'[^\w]','',regex=True)
    # parse dates
    for col in ['Template_Date','Ready_to_Fab_Date','Install_Date','Ship_Date','Job_Creation','Next_Sched_Date','Product_Rcvd_Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

@st.cache_data(ttl=CACHE_TTL)
def process_data(df):
    # durations
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days.clip(lower=0)
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days.clip(lower=0)
    df['Days_Behind'] = (datetime.now() - df['Next_Sched_Date']).dt.days if 'Next_Sched_Date' in df.columns else np.nan
    # numeric
    def to_num(col): return pd.to_numeric(df.get(col,0).astype(str).str.replace(r'[$,%]','',regex=True), errors='coerce').fillna(0)
    df['Revenue'] = to_num('Total_Job_Price_')
    df['Plant_Invoice'] = to_num('Job_Throughput_Job_Plant_Invoice')
    df['Install_Cost'] = to_num('Total_Job_SqFT') * INSTALL_COST_PER_SQFT
    df['Rework_Cost'] = to_num('Rework_Stone_Shop_Rework_Price')
    df['Branch_Cost'] = df['Plant_Invoice'] + df['Install_Cost'] + df['Rework_Cost']
    df['Branch_Profit'] = df['Revenue'] - df['Branch_Cost']
    df['Branch_Profit_Margin_%'] = np.where(df['Revenue']>0, df['Branch_Profit']/df['Revenue']*100, 0)
    return df

# parse free-text issues
@st.cache_data(ttl=CACHE_TTL)
def issue_keywords(series, top_n=10):
    words = Counter()
    for txt in series.dropna():
        for w in re.findall(r"\b\w{4,}\b", txt.lower()):
            words[w]+=1
    return pd.DataFrame(words.most_common(top_n), columns=['Keyword','Count'])

# --- APPLICATION ---
def main():
    st.set_page_config(layout="wide", page_title="Enhanced Profit Dashboard")
    st.title("💰 Job Profitability & Ops Dashboard")

    # Sidebar: connection settings
    with st.sidebar.expander("🔧 Connection Settings", expanded=True):
        sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SPREADSHEET_ID)
        ws_name = st.text_input("Worksheet Name", value=DEFAULT_WORKSHEET_NAME)
        st.caption("Ensure your service-account email is shared with this sheet.")
        creds = None
        if 'google_creds_json' in st.secrets:
            creds = json.loads(st.secrets['google_creds_json'])
        else:
            up = st.file_uploader("Service Account JSON", type='json')
            if up:
                creds = json.load(up)
    if not creds:
        st.sidebar.error("Credentials required to load data.")
        return
    try:
        raw_df = load_data(creds, sheet_id, ws_name)
    except SpreadsheetNotFound:
        st.sidebar.error("⚠️ Spreadsheet not found: check ID, worksheet name, and sharing settings.")
        return

    df = process_data(raw_df)

    # Sidebar: filters
    st.sidebar.header("Filters & View")
    if 'Job_Creation' in df.columns:
        mn, mx = df['Job_Creation'].min().date(), df['Job_Creation'].max().date()
        dr = st.sidebar.date_input("Job Creation Range", [mn, mx])
        df = df[df['Job_Creation'].dt.date.between(dr[0], dr[1])]
    jn = st.sidebar.text_input("Job Name Contains")
    pn = st.sidebar.text_input("Production # Contains")
    if jn:
        df = df[df['Job_Name'].str.contains(jn, case=False, na=False)]
    if pn:
        df = df[df['Production_'].astype(str).str.contains(pn)]
    div = st.sidebar.selectbox("Division View", ['All','Stone/Quartz','Laminate'])
    if div!='All':
        df = df[df['Division'].str.contains(div.split('/')[0], case=False, na=False)]

    # Tabs
tabs = st.tabs(["Overview","Cost Variance","Rework & Issues","Customer Scorecard","Workload Balance","Pipeline","Forecasting"])

    # Overview Tab
    with tabs[0]:
        st.header("🏁 Overview")
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
        c2.metric("Total Profit", f"${df['Branch_Profit'].sum():,.0f}")
        m = df['Branch_Profit'].sum()/df['Revenue'].sum()*100 if df['Revenue'].sum()>0 else 0
        c3.metric("Avg Margin", f"{m:.1f}%")

    # Cost Variance Tab
    with tabs[1]:
        st.header("📊 Cost Variance by Phase")
        if 'PhaseThroughputName' in df:
            pv = df.groupby('PhaseThroughputName').agg(
                Revenue=('PhaseThroughputPhaseRev','sum'),
                COGS=('PhaseThroughputPhaseCogs','sum'),
                PlantInv=('PhaseThroughputPhasePlantInvoice','sum'),
            )
            st.bar_chart(pv[['Revenue','COGS','PlantInv']])
            st.dataframe(pv.style.format({'Revenue':'${:,.0f}','COGS':'${:,.0f}','PlantInv':'${:,.0f}'}))
        else:
            st.info("No phase throughput data.")

    # Rework & Issues Tab
    with tabs[2]:
        st.header("🔧 Rework & Issue Patterns")
        if 'Rework_Stone_Shop_Rework_Price' in df:
            df['Rework_per_SqFt'] = df['Rework_Stone_Shop_Rework_Price']/df.get('Rework_Stone_Shop_Square_Feet',1)
            high = df[df['Rework_per_SqFt']>5]
            st.metric("> $5/sqft Rework Jobs", len(high))
            st.dataframe(high[['Job_Name','Rework_per_SqFt']])
        if 'Job_Issues' in df:
            ik = issue_keywords(df['Job_Issues'])
            st.bar_chart(ik.set_index('Keyword')['Count'])
            st.dataframe(ik)

    # Customer Scorecard Tab
    with tabs[3]:
        st.header("👥 Customer Profitability")
        if 'Customer_Category' in df:
            cc = df.groupby('Customer_Category').agg(
                AvgM=('Branch_Profit_Margin_%','mean'),
                TotP=('Branch_Profit','sum')
            )
            st.bar_chart(cc['TotP'])
            st.dataframe(cc.style.format({'AvgM':'{:.1f}%','TotP':'${:,.0f}'}))

    # Workload Balance Tab
    with tabs[4]:
        st.header("📈 Workload by City & Assignee")
        for phase, col in [('Template','Template_Assigned_To'),('Install','Install_Assigned_To')]:
            st.subheader(phase)
            tmp = df.dropna(subset=[col])
            wb = tmp.groupby(['City',col]).agg(Jobs=('Job_Name','count'), SqFt=('Total_Job_SqFt','sum')).reset_index()
            st.dataframe(wb)

    # Pipeline Tab
    with tabs[5]:
        st.header("🚧 Pipeline Durations")
        st.metric("Avg T→RTF", f"{df['Days_Template_to_RTF'].mean():.1f} days")
        st.metric("Avg T→Install", f"{df['Days_Template_to_Install'].mean():.1f} days")
        st.subheader("T→RTF Distribution")
        st.bar_chart(df['Days_Template_to_RTF'].value_counts(bins=[0,3,7,14,30,999]))
        st.subheader("T→Install Distribution")
        st.bar_chart(df['Days_Template_to_Install'].value_counts(bins=[0,7,14,30,60,999]))

    # Forecasting Tab
    with tabs[6]:
        st.header("🔮 Revenue & Jobs Forecast")
        if 'Job_Creation' in df:
            ts = df.set_index('Job_Creation').resample('M').agg({'Revenue':'sum','Job_Name':'count'}).rename(columns={'Job_Name':'Jobs'})
            st.line_chart(ts)
            if len(ts)>2:
                X = np.arange(len(ts)).reshape(-1,1)
                for metric in ['Revenue','Jobs']:
                    y = ts[metric].values
                    model = LinearRegression().fit(X,y)
                    future = model.predict(np.arange(len(ts), len(ts)+6).reshape(-1,1))
                    fc = pd.Series(future, index=pd.date_range(ts.index[-1]+pd.offsets.MonthBegin(), periods=6, freq='M'))
                    st.subheader(f"Forecasted {metric}")
                    st.line_chart(pd.concat([ts[metric], fc]))

if __name__ == '__main__':
    main()
