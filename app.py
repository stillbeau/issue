# -*- coding: utf-8 -*-
"""
A Streamlit dashboard for analyzing job profitability data from a Google Sheet.

This refactored version incorporates best practices such as centralized constants,
clean tables for pipeline metrics including invoice durations, and a count of
jobs marked Ready-to-Fab complete in the past 30 days.
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from datetime import datetime, timedelta

# Optional libs
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

# Page config
st.set_page_config(layout="wide", page_title="Profitability Dashboard", page_icon="💰")

# Constants
SPREADSHEET_ID = "1iToy3C-Bfn06bjuEM_flHNHwr2k1zMCV1wX9MNKzj38"
WORKSHEET_NAME = "jobs"
INSTALL_COST_PER_SQFT = 15.0

# Column constants
class COLS:
    JOB_NAME = 'Job_Name'
    RTF_STATUS = 'Ready_to_Fab_Status'
    TEMPLATE_DATE = 'Template_Date'
    RTF_DATE = 'Ready_to_Fab_Date'
    INVOICE_DATE = 'Invoice_Date'

# Helpers

def _clean(df):
    return df.rename(columns=lambda c: re.sub(r'[^0-9A-Za-z_]', '', c.strip().replace(' ', '_')))


def _parse_dates(df):
    for col in [COLS.TEMPLATE_DATE, COLS.RTF_DATE, COLS.INVOICE_DATE]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _calc_durations(df):
    if COLS.TEMPLATE_DATE in df and COLS.INVOICE_DATE in df:
        df['Days_Template_to_Invoice'] = (df[COLS.INVOICE_DATE] - df[COLS.TEMPLATE_DATE]).dt.days.clip(lower=0)
    if COLS.RTF_DATE in df and COLS.INVOICE_DATE in df:
        df['Days_RTF_to_Invoice'] = (df[COLS.INVOICE_DATE] - df[COLS.RTF_DATE]).dt.days.clip(lower=0)
    return df

@st.cache_data(ttl=300)
def load_data(creds):
    creds = Credentials.from_service_account_info(creds,
        scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())
    df = _clean(df)
    df = _parse_dates(df)
    df = _calc_durations(df)
    return df

# UI

def render_pipeline_tab(df):
    """Clean table of pipeline & invoice durations plus count of recent RTF completes"""
    st.header("🚧 Pipeline & Invoice Durations")

    # Count Ready-to-Fab completes in the past 30 days
    if COLS.RTF_STATUS in df.columns and COLS.RTF_DATE in df.columns:
        cutoff = pd.to_datetime(datetime.now() - timedelta(days=30))
        recent = df[
            df[COLS.RTF_STATUS].str.lower().eq('complete') &
            (df[COLS.RTF_DATE] >= cutoff)
        ]
        st.metric(
            label="RTF Completed (last 30 days)",
            value=len(recent)
        )

    # Display table
    cols = [COLS.JOB_NAME, COLS.TEMPLATE_DATE, COLS.RTF_DATE,
            COLS.INVOICE_DATE, 'Days_Template_to_Invoice', 'Days_RTF_to_Invoice']
    tbl = df[cols].copy()
    st.dataframe(tbl, use_container_width=True)

# Main

def main():
    st.title("💰 Profit Dashboard")
    creds = None
    if 'google_creds_json' in st.secrets:
        creds = json.loads(st.secrets['google_creds_json'])
    else:
        up = st.sidebar.file_uploader("Cred JSON", type='json')
        if up:
            creds = json.load(up)
    if not creds:
        st.stop()

    df = load_data(creds)
    render_pipeline_tab(df)

if __name__ == '__main__':
    main()
