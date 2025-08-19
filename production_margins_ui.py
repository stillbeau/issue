"""Production Margins Tab UI
---------------------------------
Renders Production Margins analytics including KPI summary,
filters, main table, detail view and CSV export.
"""

import streamlit as st
import pandas as pd

# Default target cost per square foot
DEFAULT_TARGET_COST = 26.37


def prepare_production_margin_data(df: pd.DataFrame, target_cost: float = DEFAULT_TARGET_COST) -> pd.DataFrame:
    """Create derived production margin metrics from raw Google Sheet export."""
    data = df.copy()

    # Base fields with fallbacks
    data['sqft'] = data.get('Total Job SqFT', pd.Series(dtype=float))
    if 'Orders - Total Sq. Ft.' in data.columns:
        data['sqft'] = data['sqft'].fillna(data['Orders - Total Sq. Ft.'])

    if 'Total Job Price $' in data.columns:
        data['sell_price'] = data['Total Job Price $']
    elif 'Orders - Total Price' in data.columns:
        data['sell_price'] = data['Orders - Total Price']
    else:
        data['sell_price'] = pd.NA

    data['plant_invoice'] = data.get('Plant INV $', pd.Series(dtype=float))
    if 'Job Throughput - Job Plant Invoice' in data.columns:
        data['plant_invoice'] = data['plant_invoice'].fillna(
            data['Job Throughput - Job Plant Invoice']
        )

    data['labor_cost'] = data.get('Job Throughput - Total Job Labor', pd.NA)
    data['total_cogs'] = data.get('Job Throughput - Total COGS', pd.NA)

    if 'Job Creation' in data.columns:
        data['Job Creation'] = pd.to_datetime(data['Job Creation'], errors='coerce')

    # Invoice date with fallback
    date_series = data.get('Plant INV - Date', pd.Series(dtype=str))
    if 'Invoice - Date' in data.columns:
        date_series = date_series.fillna(data['Invoice - Date'])
    data['Plant INV - Date'] = pd.to_datetime(date_series, errors='coerce')
    data['plant_inv_date'] = data['Plant INV - Date']

    # Ensure numeric types for calculations
    numeric_cols = [
        'sqft',
        'sell_price',
        'plant_invoice',
        'labor_cost',
        'total_cogs',
        'Rework - Stone Shop - Rework Price',
        'Job Throughput - Total Job Cost',
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Derived cost components
    data['plant_cost_per_sqft'] = data['plant_invoice'] / data['sqft']
    data['other_cogs'] = data['total_cogs'] - data['labor_cost'] - data['plant_invoice']
    data['job_profit'] = data['sell_price'] - (
        data['plant_invoice'] + data['labor_cost'] + data['other_cogs']
    )
    data['job_margin_pct'] = data['job_profit'] / data['sell_price']

    data['variance_per_sqft'] = data['plant_cost_per_sqft'] - target_cost
    data['overspend_total'] = data.apply(
        lambda r: r['variance_per_sqft'] * r['sqft'] if r['variance_per_sqft'] > 0 else 0,
        axis=1,
    )
    data['underspend_total'] = data.apply(
        lambda r: abs(r['variance_per_sqft']) * r['sqft'] if r['variance_per_sqft'] < 0 else 0,
        axis=1,
    )

    data['rework_pct_of_cost'] = data['Rework - Stone Shop - Rework Price'] / data['Job Throughput - Total Job Cost']

    today = pd.Timestamp.today().normalize()
    data['Invoice Health'] = data.apply(
        lambda r: 'Missing'
        if pd.isna(r['plant_invoice'])
        else (
            'Stale'
            if (
                str(r.get('Plant INV - Status', '')).lower() == 'pending'
                and r['plant_inv_date'] < today - pd.Timedelta(days=30)
            )
            else 'OK'
        ),
        axis=1,
    )

    data['Production Cost Health'] = data['plant_cost_per_sqft'].apply(
        lambda x: 'Green' if x <= target_cost + 1 else ('Amber' if x <= target_cost + 3 else 'Red')
    )
    data['Margin Health'] = data['job_margin_pct'].apply(
        lambda x: 'Green' if x >= 0.40 else ('Amber' if x >= 0.25 else 'Red')
    )

    data['Invoice Date'] = data['plant_inv_date']

    return data


def render_production_margins_tab(df_full: pd.DataFrame) -> None:
    """Render the Production Margins tab."""
    st.header("🏭 Production Margins")

    if df_full.empty:
        st.info("No data available for production margin analysis.")
        return

    # --- Controls ---
    target_cost = st.sidebar.number_input(
        "Target Cost / Sq Ft",
        min_value=0.0,
        value=DEFAULT_TARGET_COST,
        step=0.5,
    )

    # Prepare data early for date range defaults
    data = prepare_production_margin_data(df_full, target_cost)

    date_options = []
    if 'Job Creation' in data.columns:
        date_options.append('Job Creation')
    if 'Invoice Date' in data.columns:
        date_options.append('Invoice Date')

    date_field = st.sidebar.selectbox('Filter By Date', date_options)
    date_series = data['Job Creation'] if date_field == 'Job Creation' else data['Invoice Date']
    start_date, end_date = st.sidebar.date_input(
        "Date Range",
        value=(date_series.min(), date_series.max()),
    )

    # Filter selectors
    filter_cols = {
        'Account': 'Account',
        'Salesperson': 'Salesperson',
        'Division': 'Division',
        'City': 'City',
        'Job Status': 'Job Status',
        'Invoice Status': 'Plant INV - Status',
    }

    filters = {}
    for label, col in filter_cols.items():
        if col in df_full.columns:
            options = ['All'] + sorted(df_full[col].dropna().unique().tolist())
            filters[col] = st.sidebar.selectbox(label, options)
        else:
            filters[col] = 'All'

    show_overspend_only = st.sidebar.checkbox("Show Overspend Only", value=False)
    hide_missing = st.sidebar.checkbox("Hide Missing Data", value=False)

    # Apply date range filter
    if date_field == 'Job Creation' and 'Job Creation' in data.columns:
        data = data[
            (data['Job Creation'] >= pd.to_datetime(start_date))
            & (data['Job Creation'] <= pd.to_datetime(end_date))
        ]
    elif date_field == 'Invoice Date' and 'Invoice Date' in data.columns:
        data = data[
            (data['Invoice Date'] >= pd.to_datetime(start_date))
            & (data['Invoice Date'] <= pd.to_datetime(end_date))
        ]

    # Apply categorical filters
    for col, value in filters.items():
        if value != 'All' and col in data.columns:
            data = data[data[col] == value]

    if hide_missing:
        data = data[data['Invoice Health'] != 'Missing']

    if show_overspend_only:
        data = data[data['overspend_total'] > 0]

    # --- KPIs ---
    avg_cost = data['plant_cost_per_sqft'].mean()
    pct_over = (data['variance_per_sqft'] > 0).mean() * 100 if len(data) else 0
    overspend_total = data['overspend_total'].sum()
    underspend_total = data['underspend_total'].sum()
    missing_count = (data['Invoice Health'] == 'Missing').sum()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Avg Plant Cost / SqFt", f"${avg_cost:,.2f}")
    kpi_cols[1].metric("% Jobs Over Target", f"{pct_over:.1f}%")
    kpi_cols[2].metric("Total Overspend $", f"${overspend_total:,.0f}")
    kpi_cols[3].metric("Total Underspend $", f"${underspend_total:,.0f}")
    kpi_cols[4].metric("Missing Data Count", missing_count)

    # --- Main Table ---
    display_cols = [
        'Job Name', 'Account', 'Salesperson', 'sqft', 'sell_price', 'plant_invoice',
        'plant_cost_per_sqft', 'variance_per_sqft', 'overspend_total', 'underspend_total',
        'job_margin_pct', 'Production Cost Health', 'Invoice Health',
        'Invoice Date', 'Rework - Stone Shop - Rework Price', 'rework_pct_of_cost'
    ]
    existing_cols = [c for c in display_cols if c in data.columns]
    table = data[existing_cols]

    st.dataframe(table, use_container_width=True)

    # Detail view
    selected_job = st.selectbox("Select Job for Details", data['Job Name'] if 'Job Name' in data.columns else [])
    if selected_job:
        job = data[data['Job Name'] == selected_job].iloc[0]
        with st.expander("Job Details", expanded=True):
            st.write(job.to_dict())

    # --- Export ---
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📤 Export CSV", csv, "production_margins.csv", "text/csv")

