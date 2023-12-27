"""Various constant values."""
from typing import Final, Sequence

# Columns and subset of columns of the uploaded data sheet.
ALL_COLUMNS: Final[Sequence[str]] = [
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
]
SIZE_COLUMNS: Final[Sequence[str]] = ALL_COLUMNS[20:]
IGNORED_COLUMNS: Final[Sequence[str]] = [
    "Buddy",
    "Site Name",
    "vis",
    "Direction",
    "Time",
    "P-Qs",
    "Common name",
]
DUPLICATE_CHECK_COLUMNS: Final[Sequence[str]] = [
    "Diver",
    "Site No.",
    "Date",
    "Depth",
    "Method",
    "Block",
    "Species",
]

VALIDATION_HELP_TEXT: Final[str] = f"""
    ---
    ### Validations

    * `Species present`: The 'Species' column value isn't "NOT PRESENT".
    * `Total > 0`: The 'Total' column value is greater than zero.
    * `Only inverts or sized`: Either the 'Inverts' column contains a non-zero
      count, or the size columns (not both).
    * `Unique`: The row is a unique record when considering only these columns:
      {DUPLICATE_CHECK_COLUMNS}.
    * `Species known`: The species exists in the RLS database.
    * `Species expected`: The species was recorded within the specified distance from
      the site. If it wasn't, provide a picture for verification when submitting the
      data.
    * `Method matches`: One of the methods recorded for the species in the RLS
      database was used.
    * `M1 sized`: If the species is a Method 1 species, its size was entered (even
      if entered under Method 2).
    * `Max size OK`: The entered maximum size (shown in `Row max size`) is less than or
      equal to the size in the RLS database (shown in `Expected max size`). Note that
      this excludes species for which sizing isn't expected or for which there is no
      data. It's not always right, so use your best judgement.
"""
