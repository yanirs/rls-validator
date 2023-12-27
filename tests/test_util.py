# type: ignore
import numpy as np
import pytest

from rlsv.util import estimate_earth_distance


@pytest.mark.parametrize(
    ("lon1", "lat1", "lon2", "lat2", "expected"),
    [
        pytest.param(
            np.deg2rad(0),
            np.deg2rad(0),
            np.deg2rad(90),
            np.deg2rad(0),
            6371 * np.pi / 2,
            id="scalars",
        ),
        pytest.param(
            np.deg2rad([0, 90]),
            np.deg2rad([0, 0]),
            np.deg2rad([90, 180]),
            np.deg2rad([0, 0]),
            np.array([6371 * np.pi / 2, 6371 * np.pi / 2]),
            id="vectors",
        ),
    ],
)
def test_estimate_earth_distance(lon1, lat1, lon2, lat2, expected):
    result = estimate_earth_distance(lon1, lat1, lon2, lat2)
    if isinstance(expected, float):
        assert abs(result - expected) < 0.1
    else:
        np.testing.assert_allclose(result, expected, rtol=0.01)
