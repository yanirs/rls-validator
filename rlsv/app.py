"""Reef Life Survey volunteer data validator: Streamlit app."""
import dataclasses
import io
import json
from typing import Sequence

import pandas as pd
import streamlit as st

from rlsv.data import download_text_file


def main() -> None:
    """Run the RLS volunteer data validator Streamlit app."""
    st.set_page_config(page_title="RLS data sheet validator", layout="wide")
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

    # TODO: add checkboxes for disabling specific checks?
    info_messages = ["* Parsed file and found the data sheet with valid columns."]
    drop_ignored_columns = st.checkbox(
        "Drop ignored columns",
        value=True,
        help="This makes the output easier to scroll through",
    )

    # TODO: turn this block into a tested function
    if data_df.iloc[0, :10].isna().all():
        info_messages.append("* Skipped first non-header row (assumed empty example).")
        data_df = data_df.iloc[1:].reset_index(drop=True)
    if drop_ignored_columns:
        ignored_columns = [
            "Buddy",
            "Site Name",
            "Latitude",
            "Longitude",
            "vis",
            "Direction",
            "Time",
            "P-Qs",
            "Common name",
        ]
        data_df = data_df.drop(columns=ignored_columns)
        info_messages.append(f"* Dropped ignored columns: {ignored_columns}.")
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
    data_df["Method"] = data_df["Method"].astype(int)

    # TODO: apply other validations that rely on external data -- copy from chat
    # TODO: better to iterate over rows once? check
    size_columns = data_df.columns[data_df.columns.get_loc("Inverts") + 1 :]
    data_df["Sized"] = data_df[size_columns].sum(axis=1) > 0
    data_df["Max size"] = (
        data_df[size_columns]
        .apply(lambda row: row.last_valid_index(), axis=1)
        .astype(float)
    )
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
            "Only inverts or sized": (data_df["Inverts"] > 0) ^ data_df["Sized"],
            "Unique": ~data_df[duplicate_check_columns].duplicated(keep=False),
        }
    )

    # TODO: make this a separate function; consider vectorising
    all_species_info, families_and_genera = _load_species_info()
    species_known = []
    method_matches = []
    m1_sized = []
    max_size_ok = []
    expected_max_sizes: list[int | None] = []
    for _, row in data_df.iterrows():
        # TODO: warn on superseding?
        is_debris = row["Species"].startswith("Debris")
        is_spp = row["Species"].endswith("spp.")
        # TODO: also get valid spp. from surveys.json
        # TODO: consider sizing and checking methods for spp. (split is_debris/spp then)
        if is_debris or is_spp:
            species_known.append(
                is_debris or row["Species"].split()[0] in families_and_genera
            )
            method_matches.append(is_spp or row["Method"] in (0, 2))
            m1_sized.append(True)
            max_size_ok.append(True)
            expected_max_sizes.append(None)
            continue
        species_info = all_species_info.get(row["Species"])
        if not species_info:
            species_known.append(False)
            method_matches.append(False)
            m1_sized.append(False)
            max_size_ok.append(False)
            expected_max_sizes.append(None)
            continue
        species_known.append(True)
        method_matches.append(
            row["Method"] == 0 or row["Method"] in species_info.methods
        )
        if [2] == species_info.methods:
            m1_sized.append(True)
            max_size_ok.append(True)
            expected_max_sizes.append(None)
        else:
            m1_sized.append(1 in species_info.methods and row["Sized"])
            # TODO: consider getting observed min-max sizes (needs rls-data change:
            # TODO: sizes.json based on observations)
            max_size_ok.append(
                bool(
                    not species_info.max_length_cm
                    or row["Max size"] <= species_info.max_length_cm
                )
            )
            expected_max_sizes.append(species_info.max_length_cm)
    validation_df["Species known"] = species_known
    validation_df["Method matches"] = method_matches
    validation_df["M1 sized"] = m1_sized
    validation_df["Max size OK"] = max_size_ok
    validation_df["Expected max size"] = expected_max_sizes

    st.info("\n".join(info_messages))
    validation_with_data_df = pd.concat([validation_df, data_df], axis=1)
    failed_validation_mask = (
        (~validation_df.drop(columns=["Expected max size"])).sum(axis=1).astype(bool)
    )
    if failed_validation_mask.any():
        st.error(
            f"Found {failed_validation_mask.sum()} suspicious rows. "
            "Please fix them and re-upload the file."
        )
        st.dataframe(
            validation_with_data_df[failed_validation_mask],
            hide_index=True,
            column_config={"Date": st.column_config.DateColumn()},
        )
    else:
        st.success("No suspicious rows found. :tada:")
    show_all_rows = st.checkbox("Show all the rows that got checked")
    if show_all_rows:
        st.dataframe(
            validation_with_data_df,
            hide_index=True,
            column_config={"Date": st.column_config.DateColumn()},
        )

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
        * `Species known`: The species exists in the RLS database.
        * `Method matches`: One of the methods recorded for the species in the RLS
          database was used.
        * `M1 sized`: If the species is a Method 1 species, its size was entered (even
          if entered under Method 2).
        * `Max size OK`: The entered maximum size is less than or equal to the size in
          the RLS database. Note that this excludes species for which sizing isn't
          expected or for which there is no data. The expected max size is shown in the
          `Expected max size` column. It's not always right, so use your best judgement.
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


@dataclasses.dataclass
class SpeciesInfo:
    """The subset of species info relevant to this app."""

    name: str
    max_length_cm: int | None
    methods: Sequence[int]
    superseded_by_name: str | None


# TODO: add @st.cache_data or move to data.py and cache everything that's loaded.
def _load_species_info(
    url: str = "https://raw.githubusercontent.com/yanirs/rls-data/master/output/species.json",
) -> tuple[dict[str, SpeciesInfo], set[str]]:
    """Load species JSON data to SpeciesInfo and a set of valid families & genera."""
    with download_text_file(url).open() as json_file:
        raw_species_data = json.load(json_file)
    families_and_genera = set()
    all_species_info = {}
    for species in raw_species_data:
        if "family" in species:
            families_and_genera.add(species["family"])
        if "genus" in species:
            families_and_genera.add(species["genus"])
        # TODO: round up max_length_cm to a valid size class to reduce false negatives
        all_species_info[species["scientific_name"]] = SpeciesInfo(
            name=species["scientific_name"],
            max_length_cm=species.get("max_length_cm"),
            methods=species.get("methods") or [],
            superseded_by_name=None,
        )
        for superseded_name in species.get("superseded_names", ()):
            all_species_info[superseded_name] = SpeciesInfo(
                name=superseded_name,
                max_length_cm=species.get("max_length_cm"),
                methods=species.get("methods") or [],
                superseded_by_name=species["scientific_name"],
            )
    return all_species_info, families_and_genera


if __name__ == "__main__":
    main()
