import numpy as np
import pytest
from ovito.data import DataCollection, Particles
from ovito.traits import PropertyReference
from scipy.special import gammainc, gammaincinv

from cumulative_property_profile import CumulativePropertyProfileModifier


def test_property_parameter_uses_a_property_reference():
    trait = CumulativePropertyProfileModifier.class_traits()[
        "input_property"
    ].trait_type

    assert isinstance(trait, PropertyReference)
    assert CumulativePropertyProfileModifier().input_property == "Entropy averaged"


def make_data(positions, property_name="Mass", values=None, identifiers=None):
    positions = np.asarray(positions, dtype=float)
    if values is None:
        values = np.arange(1, len(positions) + 1, dtype=float)

    particles = Particles()
    particles.create_property("Position", data=positions)
    particles.create_property(property_name, data=values)
    if identifiers is not None:
        particles.create_property("Particle Identifier", data=identifiers)

    data = DataCollection()
    data.objects.append(particles)
    return data


def test_profile_is_sorted_without_reordering_particles():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            subtract_reference=False,
            normalize=False,
            fit_generalized_normal=False,
        )
    )

    table = data.tables["cumulative-property-profile"]
    np.testing.assert_allclose(table.xy(), [[1, 2], [2, 7], [3, 17]])
    np.testing.assert_allclose(
        np.asarray(data.particles.positions),
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
    )
    assert table.axis_label_x == "Position (Z)"
    assert table.axis_label_y == "Cumulative Mass"


def test_reference_is_subtracted_before_cumulative_sum():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            reference_value=3.0,
            normalize=False,
            fit_generalized_normal=False,
        )
    )

    np.testing.assert_allclose(
        data.tables["cumulative-property-profile"].xy(),
        [[1, -1], [2, 1], [3, 8]],
    )


def test_global_attribute_reference_is_used():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )
    data.attributes["Mean Mass"] = 3.0

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            reference_mode="Global attribute",
            reference_attribute="Mean Mass",
            normalize=False,
            fit_generalized_normal=False,
        )
    )

    np.testing.assert_allclose(
        data.tables["cumulative-property-profile"].xy(),
        [[1, -1], [2, 1], [3, 8]],
    )


def test_absolute_value_is_applied_per_particle_after_reference():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            reference_value=3.0,
            absolute_value=True,
            normalize=False,
            fit_generalized_normal=False,
        )
    )

    table = data.tables["cumulative-property-profile"]
    np.testing.assert_allclose(table.xy(), [[1, 1], [2, 3], [3, 10]])
    assert table.axis_label_y == "Cumulative |Mass - reference|"


def test_sorted_particle_identifiers_are_included_in_table():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
        identifiers=[300, 100, 200],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            subtract_reference=False,
            fit_generalized_normal=False,
        )
    )

    table = data.tables["cumulative-property-profile"]
    np.testing.assert_array_equal(
            np.asarray(table["Particle Identifier"]),
        [100, 200, 300],
    )


def test_profile_is_normalized_by_final_cumulative_value_by_default():
    data = make_data(
        [[0, 0, 3], [0, 0, 1], [0, 0, 2]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            subtract_reference=False,
            fit_generalized_normal=False,
        )
    )

    table = data.tables["cumulative-property-profile"]
    np.testing.assert_allclose(table.xy(), [[1, 2 / 17], [2, 7 / 17], [3, 1]])
    assert table.axis_label_y == "Normalized Cumulative Mass"


def test_normalization_rejects_zero_final_value():
    data = make_data(
        [[0, 0, 1], [0, 0, 2]],
        values=[1, -1],
    )

    with pytest.raises(ValueError, match="final value is zero"):
        data.apply(
            CumulativePropertyProfileModifier(
                axis="Z",
                input_property="Mass",
                subtract_reference=False,
                fit_generalized_normal=False,
            )
        )


def test_generalized_normal_fit_adds_curve_and_attributes(capsys):
    positions = np.linspace(-20.0, 20.0, 101)
    center = 0.7
    alpha = 2.0
    beta = 2.2
    cdf = np.where(
        positions < center,
        0.5
        * (
            1.0
            - gammainc(
                1.0 / beta,
                (np.abs(positions - center) / alpha) ** beta,
            )
        ),
        0.5
        * (
            1.0
            + gammainc(
                1.0 / beta,
                (np.abs(positions - center) / alpha) ** beta,
            )
        ),
    )
    increments = np.diff(np.r_[0.0, cdf])
    data = make_data(
        np.column_stack((np.zeros_like(positions), np.zeros_like(positions), positions[::-1])),
        values=increments[::-1],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="Z",
            input_property="Mass",
            subtract_reference=False,
            fit_generalized_normal=True,
            fix_beta=True,
            beta=beta,
        )
    )

    table = data.tables["cumulative-property-profile"]
    profile = table.xy()
    assert profile.shape == (len(positions), 3)
    np.testing.assert_allclose(profile[:, 1], cdf, atol=1e-10)
    np.testing.assert_allclose(profile[:, 2], cdf, atol=1e-8)

    prefix = "CumulativePropertyProfile.Mass"
    assert f"{prefix}.X.fit_center" not in data.attributes
    assert data.attributes[f"{prefix}.fit_center"] == pytest.approx(center, abs=0.05)
    assert data.attributes[f"{prefix}.fit_alpha"] == pytest.approx(alpha, abs=0.05)
    assert data.attributes[f"{prefix}.fit_beta"] == pytest.approx(beta)
    expected_width = 2.0 * alpha * gammaincinv(1.0 / beta, 0.95) ** (1.0 / beta)
    assert data.attributes[f"{prefix}.fit_width"] == pytest.approx(
        expected_width, abs=0.1
    )
    assert data.attributes[f"{prefix}.fit_r2"] > 0.99
    assert data.attributes[f"{prefix}.fit_message"] == "OK"

    output = capsys.readouterr().out
    assert "CumulativePropertyProfile fit for Mass:" in output
    assert "Width (w95):" in output
    assert "R^2:" in output and "(RMSE:" in output
    assert "Beta:" in output
    assert "Alpha:" in output


def test_profile_supports_other_axes():
    data = make_data(
        [[3, 0, 0], [1, 0, 0], [2, 0, 0]],
        values=[10, 2, 5],
    )

    data.apply(
        CumulativePropertyProfileModifier(
            axis="X",
            input_property="Mass",
            normalize=False,
            fit_generalized_normal=False,
        )
    )

    np.testing.assert_allclose(
        data.tables["cumulative-property-profile"].xy(),
        [[1, 2], [2, 7], [3, 17]],
    )


def test_property_is_required():
    data = make_data([[0, 0, 0]])

    with pytest.raises(ValueError, match="must be specified"):
        data.apply(
            CumulativePropertyProfileModifier(
                input_property="", fit_generalized_normal=False
            )
        )


def test_property_must_be_scalar():
    data = make_data([[0, 0, 0]])
    data.particles_.create_property("Vector", data=[[1, 2, 3]])

    with pytest.raises(ValueError, match="scalar"):
        data.apply(
            CumulativePropertyProfileModifier(
                input_property="Vector", fit_generalized_normal=False
            )
        )


def test_missing_property_is_reported():
    data = make_data([[0, 0, 0]])

    with pytest.raises(RuntimeError, match="was not found"):
        data.apply(
            CumulativePropertyProfileModifier(
                input_property="Energy", fit_generalized_normal=False
            )
        )
