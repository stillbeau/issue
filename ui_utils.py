import pandas as pd
import streamlit as st


def display_po_links(
    df: pd.DataFrame,
    column_config: dict | None = None,
    use_container_width: bool = True,
    hide_index: bool = True,
    **kwargs,
):
    """Render a DataFrame with optional PO link column.

    If a column named ``Link`` or ``Moraware Link`` exists, it will be renamed to
    ``PO`` and displayed using ``st.column_config.LinkColumn`` with the purchase
    order number extracted from the link.
    """
    if df is None:
        st.dataframe(df, column_config=column_config, use_container_width=use_container_width, hide_index=hide_index, **kwargs)
        return

    display_df = df.copy()
    link_col = None
    for candidate in ["Link", "Moraware Link"]:
        if candidate in display_df.columns:
            display_df = display_df.rename(columns={candidate: "PO"})
            link_col = "PO"
            break

    if link_col:
        cols = [link_col] + [c for c in display_df.columns if c != link_col]
        display_df = display_df[cols]
        column_config = column_config.copy() if column_config else {}
        column_config[link_col] = st.column_config.LinkColumn(
            "PO", display_text=r".*search=(.*)"
        )

    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=use_container_width,
        hide_index=hide_index,
        **kwargs,
    )
