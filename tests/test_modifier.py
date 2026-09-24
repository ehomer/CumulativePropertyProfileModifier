import numpy as np
import pytest
from ovito.data import DataCollection, Particles

from cumulative_property_profile import CumulativePropertyProfileModifier


def make_data(positions, property_name="Mass", values=None):
    positions = np.asarray(positions, dtype=float)
    if values is None:
        values = np.arange(1, len(positions) + 1, dtype=float)

    particles = Particles()
    particles.create_property("Position", data=positions)
    particles.create_property(property_name, data=values)

    data = DataCollection()
    data.objects.append(particles)
    return data


def test_profile_is_sorted_without_reordering_particles():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(axis="Z", input_property="Mass")
    )

    table = data.tables["cumulative-property-profile"]
    np.testing.assert_allclose(table.xy(), [[1, 2], [2, 7], [3, 17]])
    np.testing.assert_allclose(
        np.asarray(data.particles.positions),
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
    )
    assert table.axis_label_x == "Position (Z)"
    assert table.axis_label_y == "Cumulative Mass"


def test_profile_supports_other_axes():
    data = make_data(
        [[3, 0, 0], [1, 0, 0], [2, 0, 0]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(axis="X", input_property="Mass")
    )

    np.testing.assert_allclose(
        data.tables["cumulative-property-profile"].xy(),
        [[1, 2], [2, 7], [3, 17]],
    )


def test_property_is_required():
    data = make_data([[0, 0, 0]])

    with pytest.raises(ValueError, match="must be specified"):
        data.apply(CumulativePropertyProfileModifier())


def test_property_must_be_scalar():
    data = make_data([[0, 0, 0]])
    data.particles_.create_property("Vector", data=[[1, 2, 3]])

    with pytest.raises(ValueError, match="scalar"):
        data.apply(CumulativePropertyProfileModifier(input_property="Vector"))


def test_missing_property_is_reported():
    data = make_data([[0, 0, 0]])

    with pytest.raises(RuntimeError, match="was not found"):
        data.apply(CumulativePropertyProfileModifier(input_property="Energy"))
