"""General utility functions."""

import numpy as np


def estimate_earth_distance(lon1, lat1, lon2, lat2, earth_radius_km=6371):  # type: ignore
    """
    Estimate the kilometre distance between two coordinates specified in radians.

    Adapted from https://stackoverflow.com/a/4913653 and ChatGPT output. Accepts scalars
    and arrays.

    This function is expected to be inaccurate for distant points, but should be good
    enough for our purposes. It's orders of magnitude faster than applying geopy's
    distance function, so it works well for a large number of calculations.
    """
    return (
        2
        * earth_radius_km
        * np.arcsin(
            np.sqrt(
                np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            )
        )
    )
