"""Reef Life Survey volunteer data validator: Streamlit app."""
import pandas as pd
import streamlit as st


def main() -> None:
    """Run the RLS volunteer data validator Streamlit app."""
    st.title("RLS data sheet validator")
    st.write("Upload your spreadsheet for validation")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is None:
        return
    try:
        xls = pd.ExcelFile(uploaded_file)
        if "DATA" in xls.sheet_names:
            st.success(
                "Validation succeeded: Spreadsheet parsed and 'DATA' sheet found."
            )
        else:
            st.error("Validation failed: No sheet named 'DATA' found.")
    except Exception as e:
        st.error(f"Validation failed: Unable to parse the file. Error: {e}")


if __name__ == "__main__":
    main()
