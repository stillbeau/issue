import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from collections import Counter

# --- CONSTANTS & CONFIGURATION ---
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1zj38"
WORKSHEET_NAME = "jobs"
INSTALL_COST_PER_SQFT = 15.0
CACHE_TTL = 600

# Shared dataframe column configurations
COLUMN_CONFIG = {
    "Link": st.column_config.LinkColumn("Prod #", display_text=r".*search=(.*)"),
    "Days_Behind": st.column_config.NumberColumn("Days Behind/Ahead", help="Positive: Behind. Negative: Ahead."),
    "Revenue": st.column_config.NumberColumn(format='$%.2f'),
    "Total_Job_SqFt": st.column_config.NumberColumn("SqFt", format='%.2f'),
    "Cost_From_Plant": st.column_config.NumberColumn("Production Cost", format='$%.2f'),
    "Material_Cost": st.column_config.NumberColumn("Material Cost", format='$%.2f'),
    "Shop_Cost": st.column_config.NumberColumn("Shop Cost", format='$%.2f'),
    "Total_Branch_Cost": st.column_config.NumberColumn(format='$%.2f'),
    "Branch_Profit": st.column_config.NumberColumn(format='$%.2f'),
    "Branch_Profit_Margin_%": st.column_config.ProgressColumn("Branch Profit %", format='%.2f%%', min_value=-50, max_value=100),
    "Shop_Profit_Margin_%": st.column_config.ProgressColumn("Shop Profit %", format='%.2f%%', min_value=-50, max_value=100),
}

# --- DATA LOADING & PROCESSING ---
@st.cache_data(ttl=CACHE_TTL)
def load_data(creds_dict):
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(sheet.get_all_records())
    # normalize columns
    df.columns = df.columns.str.strip().str.replace(r'[\s-]+','_',regex=True).str.replace(r'[^\w]','',regex=True)
    # parse dates
    date_cols = [
        'Template_Date','Ready_to_Fab_Date','Ship_Date','Install_Date',
        'Job_Creation','Next_Sched_Date','Product_Rcvd_Date'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

@st.cache_data(ttl=CACHE_TTL)
def process_data(df):
    # compute durations
    df['Days_Template_to_RTF'] = (df['Ready_to_Fab_Date'] - df['Template_Date']).dt.days.clip(lower=0)
    df['Days_Template_to_Install'] = (df['Install_Date'] - df['Template_Date']).dt.days.clip(lower=0)
    df['Days_RTF_to_Ship'] = (df['Ship_Date'] - df['Ready_to_Fab_Date']).dt.days.clip(lower=0)
    df['Days_Ship_to_Install'] = (df['Install_Date'] - df['Ship_Date']).dt.days.clip(lower=0)
    df['Days_Behind'] = (datetime.now() - df['Next_Sched_Date']).dt.days if 'Next_Sched_Date' in df else np.nan
    # material parsing
    if 'Job_Material' in df:
        df[['Material_Brand','Material_Color']] = df['Job_Material'].apply(lambda s: pd.Series(parse_material(str(s))))
    # tags for product type
    df['Product_Type'] = np.where(df['Division'].str.contains('laminate', case=False, na=False),'Laminate','Stone/Quartz')
    # rework cost per sqft
    df['Rework_Cost_per_SqFt'] = df['Rework_Stone_Shop_Rework_Price'].astype(float) / df['Rework_Stone_Shop_Square_Feet'].replace(0,np.nan)
    # handle numeric columns
    def to_num(col): return pd.to_numeric(df.get(col,0).astype(str).str.replace(r'[$,%]','',regex=True),errors='coerce').fillna(0)
    # compute branch profit etc.
    df['Revenue'] = to_num('Total_Job_Price_')
    df['Plant_Invoice'] = to_num('Job_Throughput_Job_Plant_Invoice')
    df['Total_COGS'] = to_num('Job_Throughput_Total_COGS')
    df['Install_Cost'] = to_num('Total_Job_SqFt') * INSTALL_COST_PER_SQFT
    df['Branch_Cost'] = df['Plant_Invoice'] + df['Install_Cost'] + df['Rework_Stone_Shop_Rework_Price']
    df['Branch_Profit'] = df['Revenue'] - df['Branch_Cost']
    df['Branch_Profit_Margin_%'] = np.where(df['Revenue']>0, df['Branch_Profit']/df['Revenue']*100, 0)
    return df

# simple material parser
parse_material = lambda s: (re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(',s).group(1).strip() if re.search(r'-\s*,\d+\s*-\s*([A-Za-z0-9 ]+?)\s*\(',s) else 'N/A',
                              re.search(r'\)\s*([^()]+?)\s*\(',s).group(1).strip() if re.search(r'\)\s*([^()]+?)\s*\(',s) else 'N/A')

# extract issue keywords
@st.cache_data(ttl=CACHE_TTL)
def issue_keywords(series, top_n=10):
    words = Counter()
    for txt in series.dropna():
        for w in re.findall(r"\b\w{4,}\b", txt.lower()):
            words[w]+=1
    return pd.DataFrame(words.most_common(top_n), columns=['Keyword','Count'])

# --- APP LAYOUT ---
def main():
    st.set_page_config(layout="wide", page_title="Enhanced Profit Dashboard")
    st.title("💰 Enhanced Job Profitability & Operations Dashboard")

    # --- SIDEBAR ---
    creds = None
    if 'google_creds_json' in st.secrets:
        creds = json.loads(st.secrets['google_creds_json'])
    else:
        up = st.sidebar.file_uploader("Service Account JSON", type='json')
        if up: creds = json.load(up)
    if not creds:
        st.sidebar.error("Credentials required to load data.")
        st.stop()

    raw = load_data(creds)
    df = process_data(raw)

    # filters
    st.sidebar.header("Filters & Settings")
    # date filter
    if 'Job_Creation' in df:
        mn,mx = df['Job_Creation'].min(), df['Job_Creation'].max()
        dr = st.sidebar.date_input("Job Creation Range", [mn, mx])
        df = df[df['Job_Creation'].between(*map(pd.to_datetime,dr))]
    # text filters
    jf = st.sidebar.text_input("Job Name Contains")
    pf = st.sidebar.text_input("Production # Contains")
    if jf: df = df[df['Job_Name'].str.contains(jf, case=False, na=False)]
    if pf: df = df[df['Production_'].astype(str).str.contains(pf)]
    # division select
    div = st.sidebar.selectbox("Division View", ['Company-Wide','Stone/Quartz','Laminate'])
    if div!='Company-Wide': df = df[df['Product_Type']==div]

    # --- TABS ---
    tabs = st.tabs(["Overview","Cost Variance","Rework & Issues","Customer Scorecard","Workload Balance","Pipeline & Aging","Forecasting"] )

    # --- Tab: Overview ---
    with tabs[0]:
        st.header("🏁 Overview KPIs")
        c1,c2,c3 = st.columns(3)
        total_rev = df['Revenue'].sum()
        total_pft = df['Branch_Profit'].sum()
        avg_mgn = total_pft/total_rev*100 if total_rev>0 else 0
        c1.metric("Total Revenue", f"${total_rev:,.0f}")
        c2.metric("Total Profit", f"${total_pft:,.0f}")
        c3.metric("Avg Margin", f"{avg_mgn:.1f}%")
        st.markdown("---")
        st.subheader("Profit by Salesperson")
        if 'Salesperson' in df:
            pb = df.groupby('Salesperson')['Branch_Profit'].sum().sort_values()
            st.bar_chart(pb)

    # --- Tab: Cost Variance ---
    with tabs[1]:
        st.header("📊 Phase-by-Phase Cost Variance")
        phases = [c for c in df.columns if 'PhaseThroughputPhase' in c or 'PhaseThroughput' in c]
        # group by actual phase name
        if 'PhaseThroughputName' in df:
            summary = df.groupby('PhaseThroughputName').agg(
                Revenue=('PhaseThroughputPhaseRev','sum'),
                COGS=('PhaseThroughputTotalCOGS','sum'),
                PlantInvoice=('PhaseThroughputPhasePlantInvoice','sum')
            )
            st.bar_chart(summary[['Revenue','COGS','PlantInvoice']])
            st.dataframe(summary.style.format({'Revenue':'${:,.0f}','COGS':'${:,.0f}','PlantInvoice':'${:,.0f}'}))
        else:
            st.info("Phase throughput data not available.")

    # --- Tab: Rework & Issues ---
    with tabs[2]:
        st.header("🔧 Rework Cost & Issue Heatmap")
        # rework cost per sqf
        r = df.dropna(subset=['Rework_Cost_per_SqFt'])
        if not r.empty:
            high = r[r['Rework_Cost_per_SqFt']>5]
            st.metric("Jobs > $5/sqft Rework", len(high))
            st.dataframe(high[['Job_Name','Rework_Cost_per_SqFt']].sort_values('Rework_Cost_per_SqFt', ascending=False))
        # keyword heatmap
        if 'Job_Issues' in df:
            kw = issue_keywords(df['Job_Issues'], top_n=10)
            st.subheader("Top Issue Keywords")
            st.bar_chart(kw.set_index('Keyword')['Count'])
            st.dataframe(kw)

    # --- Tab: Customer Scorecard ---
    with tabs[3]:
        st.header("👥 Customer & Account Profitability")
        # by Customer Category
        if 'Customer_Category' in df:
            cc = df.groupby('Customer_Category').agg(
                AvgMargin=('Branch_Profit_Margin_%','mean'),
                TotalProfit=('Branch_Profit','sum'),
                Count=('Job_Name','count')
            ).sort_values('TotalProfit', ascending=False)
            st.bar_chart(cc['TotalProfit'])
            st.dataframe(cc.style.format({'AvgMargin':'{:.1f}%','TotalProfit':'${:,.0f}'}))
        # by Account
        if 'Account' in df:
            acc = df.groupby('Account').agg(
                AvgMargin=('Branch_Profit_Margin_%','mean'),
                TotalProfit=('Branch_Profit','sum'),
                Jobs=('Job_Name','count')
            ).sort_values('TotalProfit', ascending=False)
            st.subheader("By Account")
            st.dataframe(acc.style.format({'AvgMargin':'{:.1f}%','TotalProfit':'${:,.0f}'}))

    # --- Tab: Workload Balance ---
    with tabs[4]:
        st.header("📈 Workload by City & Role")
        # template assignments
        ta = df.dropna(subset=['Template_Assigned_To'])
        if not ta.empty:
            by_city = ta.groupby(['City','Template_Assigned_To']).agg(
                Jobs=('Job_Name','count'),
                SqFt=('Total_Job_SqFt','sum')
            ).reset_index()
            st.subheader("Templates by City & Person")
            st.dataframe(by_city)
        # install assignments
        ia = df.dropna(subset=['Install_Assigned_To'])
        if not ia.empty:
            by_city_i = ia.groupby(['City','Install_Assigned_To']).agg(
                Jobs=('Job_Name','count'),
                SqFt=('Total_Job_SqFt','sum')
            ).reset_index()
            st.subheader("Installs by City & Person")
            st.dataframe(by_city_i)

    # --- Tab: Pipeline & Aging ---
    with tabs[5]:
        st.header("🚧 Pipeline & Durations")
        # summary durations
        st.metric("Avg Template→RTF (days)", f"{df['Days_Template_to_RTF'].mean():.1f}")
        st.metric("Avg Template→Install (days)", f"{df['Days_Template_to_Install'].mean():.1f}")
        # histogram
        st.subheader("Template→RTF Distribution")
        st.bar_chart(df['Days_Template_to_RTF'].value_counts(bins=[0,3,7,14,30,999]).sort_index())
        st.subheader("Template→Install Distribution")
        st.bar_chart(df['Days_Template_to_Install'].value_counts(bins=[0,7,14,30,60,999]).sort_index())

    # --- Tab: Forecasting & Trends ---
    with tabs[6]:
        st.header("🔮 Forecasting & Trends")
        if 'Job_Creation' in df:
            tr = df.set_index('Job_Creation').resample('M').agg({'Revenue':'sum','Job_Name':'count'}).rename(columns={'Job_Name':'Jobs'})
            st.line_chart(tr)
            if len(tr)>2:
                X = np.arange(len(tr)).reshape(-1,1)
                y = tr['Revenue'].values
                m = LinearRegression().fit(X,y)
                r2 = r2_score(y, m.predict(X))
                mae = mean_absolute_error(y, m.predict(X))
                st.write(f"R²: {r2:.2f}, MAE: ${mae:,.0f}")
                future_idx = pd.date_range(tr.index[-1] + pd.offsets.MonthBegin(), periods=6, freq='M')
                Xf = np.arange(len(tr), len(tr)+6).reshape(-1,1)
                yf = m.predict(Xf)
                df_f = pd.Series(yf, index=future_idx)
                combined = pd.concat([tr['Revenue'], df_f])
                st.line_chart(combined)

if __name__=='__main__':
    main()
