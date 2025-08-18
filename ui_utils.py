import pandas as pd
import streamlit as st


def display_po_links(
    df: pd.DataFrame | None,
    column_config: dict | None = None,
    use_container_width: bool = True,
    hide_index: bool = True,
    **kwargs,
):
    """Render a DataFrame with an optional "PO" link column.

    If a column resembling a job link (``Link``/``Moraware Link`` in any
    capitalization or with underscores) exists, it will be renamed to ``PO`` and
    displayed using :class:`st.column_config.LinkColumn` with the purchase order
    number extracted from the link.
    """

    if df is None:
        st.dataframe(
            pd.DataFrame(),
            column_config=column_config,
            use_container_width=use_container_width,
            hide_index=hide_index,
            **kwargs,
        )
        return

    display_df = df.copy()

    # Locate a link column in a case-insensitive manner and handle common variants
    link_col = None
    for col in display_df.columns:
        normalized = col.strip().lower().replace("_", " ")
        if normalized in {"link", "moraware link"}:
            display_df = display_df.rename(columns={col: "PO"})
            link_col = "PO"
            break

    if link_col:
        cols = [link_col] + [c for c in display_df.columns if c != link_col]
        display_df = display_df[cols]

        # Ensure any provided column configuration is preserved
        column_config = dict(column_config or {})
        column_config[link_col] = st.column_config.LinkColumn(
            "PO", display_text=r".*search=([^&]*)"
        )

    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=use_container_width,
        hide_index=hide_index,
        **kwargs,
    )
