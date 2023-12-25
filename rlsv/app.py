"""Reef Life Survey volunteer data validator: Streamlit app."""
import io
from typing import Sequence

import pandas as pd
import streamlit as st


def main() -> None:
    """Run the RLS volunteer data validator Streamlit app."""
    st.set_page_config(page_title="RLS data sheet validator")
    st.title("RLS data sheet validator")
    st.write(
        "Upload your spreadsheet for validation. Save XLS files as XLSX if you're "
        "working with an older template. Alternatively, export the data sheet to CSV."
    )
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is None:
        return
    try:
        data_df = _parse_uploaded_file(uploaded_file)
    except Exception as ex:
        st.error(f"Unable to parse the file. Error: {ex}")
        return

    # TODO: turn this block into a function
    info_messages = ["* Parsed file and found the data sheet with valid columns."]
    if data_df.iloc[0, :10].isna().all():
        info_messages.append("* Skipped first non-header row (assumed empty example).")
        data_df = data_df.iloc[1:].reset_index(drop=True)
    non_zero_cumsum = (data_df["Total"] != 0).cumsum()
    zero_streak_start_index = (
        non_zero_cumsum[non_zero_cumsum == non_zero_cumsum.iloc[-1]].index[0] + 1
    )
    if zero_streak_start_index < len(data_df):
        info_messages.append(
            f"* Ignored last {len(data_df) - zero_streak_start_index} rows with "
            "'Total' value of zero."
        )
        data_df = data_df.iloc[:zero_streak_start_index]
    info_messages.append(
        f"* Ran validations on the {len(data_df)} remaining data rows."
    )

    # TODO: apply other validations that rely on external data -- copy from chat
    # TODO: better to iterate over rows once? check
    size_columns = data_df.columns[data_df.columns.get_loc("Inverts") + 1 :]
    duplicate_check_columns = [
        "Diver",
        "Site No.",
        "Date",
        "Depth",
        "Method",
        "Block",
        "Species",
    ]
    validation_df = pd.DataFrame(
        {
            "Species present": data_df["Species"] != "NOT PRESENT",
            "Total > 0": data_df["Total"] > 0,
            "Only inverts or sized": (data_df["Inverts"] > 0)
            ^ (data_df[size_columns].sum(axis=1) > 0),
            "Unique": ~data_df[duplicate_check_columns].duplicated(keep=False),
        }
    )

    st.info("\n".join(info_messages))
    validation_with_data_df = pd.concat([validation_df, data_df], axis=1)
    failed_validation_mask = (~validation_df).sum(axis=1).astype(bool)
    if failed_validation_mask.any():
        st.error(
            f"Found {failed_validation_mask.sum()} suspicious rows. "
            "Please fix them and re-upload the file."
        )
        st.dataframe(validation_with_data_df[failed_validation_mask], hide_index=True)
    else:
        st.success("No suspicious rows found. :tada:")
    show_all_rows = st.checkbox("Show all the rows that got checked")
    if show_all_rows:
        st.dataframe(validation_with_data_df, hide_index=True)

    st.write(
        f"""
        ---
        ### Validations

        * `Species present`: The 'Species' column value isn't "NOT PRESENT".
        * `Total > 0`: The 'Total' column value is greater than zero.
        * `Only inverts or sized`: Either the 'Inverts' column contains a non-zero
          count, or the size columns (not both).
        * `Unique`: The row is a unique record when considering only these columns:
          {duplicate_check_columns}.
    """
    )


def _parse_uploaded_file(
    uploaded_file: io.BytesIO,
    sheet_name: str = "DATA",
    expected_columns: Sequence[str] = (
        "ID",
        "Diver",
        "Buddy",
        "Site No.",
        "Site Name",
        "Latitude",
        "Longitude",
        "Date",
        "vis",
        "Direction",
        "Time",
        "P-Qs",
        "Depth",
        "Method",
        "Block",
        "Code",
        "Species",
        "Common name",
        "Total",
        "Inverts",
        "2.5",
        "5",
        "7.5",
        "10",
        "12.5",
        "15",
        "20",
        "25",
        "30",
        "35",
        "40",
        "50",
        "62.5",
        "75",
        "87.5",
        "100",
        "112.5",
        "125",
        "137.5",
        "150",
        "162.5",
        "175",
        "187.5",
        "200",
        "250",
        "300",
        "350",
        "400",
    ),
) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        if sheet_name not in xls.sheet_names:
            raise ValueError(f"No sheet named '{sheet_name}' found.")
        df = xls.parse(sheet_name)
        df.columns = df.columns.astype(str)
    if df.columns.tolist() != list(expected_columns):
        raise ValueError(f"Columns don't match expected names: {expected_columns}")
    return df


if __name__ == "__main__":
    main()
