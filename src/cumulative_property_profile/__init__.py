"""OVITO modifier for atom-resolved cumulative property profiles."""

from __future__ import annotations

import numpy as np
from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from ovito.traits import PropertyReference
from scipy.optimize import least_squares
from scipy.special import gammainc, gammaincinv
from traits.api import Bool, Enum, Float, String

__all__ = ["CumulativePropertyProfileModifier"]


_AXES = ["X", "Y", "Z"]
_AXIS_INDICES = {axis: index for index, axis in enumerate(_AXES)}
_NUMERIC_REFERENCE = "Numeric value"
_GLOBAL_ATTRIBUTE_REFERENCE = "Global attribute"
_REFERENCE_MODES = [_NUMERIC_REFERENCE, _GLOBAL_ATTRIBUTE_REFERENCE]
_R2_WARNING_THRESHOLD = 0.9


def _is_scalar_numeric_property(container, property_):
    """Return whether a particle property can be cumulatively summed."""

    return property_.component_count == 1 and np.issubdtype(
        np.dtype(property_.dtype), np.number
    )


def _generalized_normal_cdf(x, center, alpha, beta):
    """Evaluate the CDF corresponding to the paper's generalized normal PDF."""

    x = np.asarray(x, dtype=float)
    shape = 1.0 / beta
    scaled_distance = (np.abs(x - center) / alpha) ** beta
    probability = gammainc(shape, scaled_distance)
    return np.where(
        x < center,
        0.5 * (1.0 - probability),
        0.5 * (1.0 + probability),
    )


def _initial_fit_parameters(x, y, beta):
    """Estimate center and scale parameters for the nonlinear fit."""

    center = float(x[np.argmin(np.abs(y - 0.5))])
    low_position = float(x[np.argmin(np.abs(y - 0.05))])
    high_position = float(x[np.argmin(np.abs(y - 0.95))])
    profile_span = max(float(np.ptp(x)), np.finfo(float).eps)
    alpha = abs(high_position - low_position) / 2.62
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = profile_span / 4.0
    return center, max(alpha, np.finfo(float).eps), beta


def _compute_width_95(alpha, beta):
    """Compute the central 95-percent width of the generalized normal."""

    gamma_quantile = gammaincinv(1.0 / beta, 0.95)
    return 2.0 * alpha * gamma_quantile ** (1.0 / beta)


def _fit_quality(y, fitted):
    """Return RMSE and coefficient of determination for a fitted profile."""

    residuals = y - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total_sum_of_squares = float(np.sum((y - np.mean(y)) ** 2))
    if total_sum_of_squares == 0.0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(residuals**2) / total_sum_of_squares)
    return rmse, r2


def _fit_generalized_normal(x, y, *, fix_beta, beta_initial):
    """Fit the generalized-normal CDF to the normalized cumulative profile."""

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        raise ValueError("At least three finite profile points are required.")

    fit_x = np.asarray(x[finite], dtype=float)
    fit_y = np.asarray(y[finite], dtype=float)
    center, alpha, beta = _initial_fit_parameters(
        fit_x, fit_y, beta_initial
    )
    if beta <= 0.0:
        raise ValueError("Beta must be positive.")

    if fix_beta:
        initial = np.array([center, np.log(alpha)])

        def unpack(parameters):
            return parameters[0], np.exp(parameters[1]), beta

    else:
        initial = np.array([center, np.log(alpha), np.log(beta)])

        def unpack(parameters):
            return parameters[0], np.exp(parameters[1]), np.exp(parameters[2])

    def residuals(parameters):
        fit_center, fit_alpha, fit_beta = unpack(parameters)
        return _generalized_normal_cdf(
            fit_x, fit_center, fit_alpha, fit_beta
        ) - fit_y

    result = least_squares(
        residuals,
        initial,
        max_nfev=2000,
        x_scale="jac",
    )
    if not result.success:
        raise RuntimeError(result.message)

    fit_center, fit_alpha, fit_beta = unpack(result.x)
    fitted = _generalized_normal_cdf(
        fit_x, fit_center, fit_alpha, fit_beta
    )
    rmse, r2 = _fit_quality(fit_y, fitted)
    message = "OK"
    if not np.isfinite(r2) or r2 < _R2_WARNING_THRESHOLD:
        message = "Warning: R2 below 0.9"

    return {
        "center": fit_center,
        "alpha": fit_alpha,
        "beta": fit_beta,
        "width": _compute_width_95(fit_alpha, fit_beta),
        "rmse": rmse,
        "r2": r2,
        "message": message,
        "curve": _generalized_normal_cdf(
            np.asarray(x, dtype=float), fit_center, fit_alpha, fit_beta
        ),
    }


class CumulativePropertyProfileModifier(ModifierInterface):
    """Plot the cumulative sum of a scalar particle property along an axis.

    Particles are sorted by their coordinate along the selected axis for the
    calculation only. The particle order in the data pipeline is not changed.
    The result is stored in a line-plot ``DataTable`` with particle positions
    as x-values and the cumulative property sum as y-values.
    """

    axis = Enum("Z", _AXES, label="Axis")
    input_property = PropertyReference(
        default_value="Entropy averaged",
        mode=PropertyReference.Mode.Properties,
        filter=_is_scalar_numeric_property,
        label="Particle property",
    )
    subtract_reference = Bool(True, label="Subtract reference")
    reference_mode = Enum(
        _NUMERIC_REFERENCE,
        _REFERENCE_MODES,
        label="Reference source",
    )
    reference_value = Float(0.0, label="Reference value")
    reference_attribute = String("", label="Reference attribute")
    absolute_value = Bool(False, label="Absolute value")
    normalize = Bool(True, label="Normalize cumulative profile")
    fit_generalized_normal = Bool(True, label="Fit generalized normal")
    fix_beta = Bool(True, label="Fix beta")
    beta = Float(2.2, label="Beta")

    def modify(self, data: DataCollection, frame: int, **kwargs):
        """Create the cumulative profile data table for the current frame."""

        del frame, kwargs

        if data.particles is None or data.particles.count == 0:
            raise RuntimeError(
                "The cumulative property profile requires particles."
            )
        if not self.input_property.strip():
            raise ValueError("A particle property must be specified.")

        property_name = self.input_property.strip()
        try:
            positions = np.asarray(data.particles.positions, dtype=float)
            values = np.asarray(data.particles[property_name], dtype=float)
        except KeyError as error:
            raise RuntimeError(
                f"Input particle property {property_name!r} was not found."
            ) from error
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Input particle property {property_name!r} must be numeric."
            ) from error

        if values.ndim != 1:
            raise ValueError(
                "CumulativePropertyProfileModifier supports scalar particle "
                "properties only."
            )

        reference = self._reference_value(data)
        contributions = values - reference if self.subtract_reference else values
        if self.absolute_value:
            contributions = np.abs(contributions)

        axis_index = _AXIS_INDICES[self.axis]
        sort_order = np.argsort(positions[:, axis_index], kind="stable")
        sorted_positions = positions[sort_order, axis_index]
        cumulative_values = np.cumsum(contributions[sort_order], dtype=float)

        normalized_profile = self.normalize or self.fit_generalized_normal
        if normalized_profile:
            final_value = cumulative_values[-1]
            if not np.isfinite(final_value) or final_value == 0.0:
                raise ValueError(
                    "Cannot normalize the cumulative profile because its "
                    "final value is zero or non-finite."
                )
            cumulative_values = cumulative_values / final_value

        fit_result = None
        if self.fit_generalized_normal:
            try:
                fit_result = _fit_generalized_normal(
                    sorted_positions,
                    cumulative_values,
                    fix_beta=self.fix_beta,
                    beta_initial=self.beta,
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                fit_result = {
                    "center": float("nan"),
                    "alpha": float("nan"),
                    "beta": float("nan"),
                    "width": float("nan"),
                    "rmse": float("nan"),
                    "r2": float("nan"),
                    "message": f"Fit failed: {error}",
                    "curve": np.full_like(cumulative_values, np.nan),
                }

        particle_identifiers = data.particles.identifiers
        if particle_identifiers is None:
            identifier_name = "Particle Index"
            identifiers = np.arange(data.particles.count, dtype=np.int64)
        else:
            identifier_name = "Particle Identifier"
            identifiers = np.asarray(particle_identifiers)

        yield f"Calculating cumulative {property_name} profile"

        if self.subtract_reference:
            contribution_label = f"{property_name} - reference"
        else:
            contribution_label = property_name
        if self.absolute_value:
            contribution_label = f"|{contribution_label}|"
        y_label = f"Cumulative {contribution_label}"
        if normalized_profile:
            y_label = f"Normalized {y_label}"

        if fit_result is not None:
            prefix = f"CumulativePropertyProfile.{property_name}"
            data.attributes[f"{prefix}.fit_center"] = fit_result["center"]
            data.attributes[f"{prefix}.fit_alpha"] = fit_result["alpha"]
            data.attributes[f"{prefix}.fit_beta"] = fit_result["beta"]
            data.attributes[f"{prefix}.fit_width"] = fit_result["width"]
            data.attributes[f"{prefix}.fit_rmse"] = fit_result["rmse"]
            data.attributes[f"{prefix}.fit_r2"] = fit_result["r2"]
            data.attributes[f"{prefix}.fit_message"] = fit_result["message"]
            print(f"CumulativePropertyProfile fit for {property_name}:")
            print(f"  Width (w95): {fit_result['width']:.6g}")
            print(
                f"  R^2: {fit_result['r2']:.6g} "
                f"(RMSE: {fit_result['rmse']:.6g})"
            )
            print(f"  Beta: {fit_result['beta']:.6g}")
            print(f"  Alpha: {fit_result['alpha']:.6g}")
            if fit_result["message"] != "OK":
                print(f"  {fit_result['message']}")

        table = data.tables.create(
            identifier="cumulative-property-profile",
            title=f"{y_label} vs. {self.axis}",
            plot_mode=DataTable.PlotMode.Line,
        )
        table.create_property(identifier_name, data=identifiers[sort_order])
        table.x = table.create_property(
            f"Position {self.axis}", data=sorted_positions
        )
        if fit_result is None:
            table.y = table.create_property(y_label, data=cumulative_values)
        else:
            table.y = table.create_property(
                y_label,
                data=np.column_stack((cumulative_values, fit_result["curve"])),
                components=(y_label, "Generalized normal fit"),
            )
        table.axis_label_x = f"Position ({self.axis})"
        table.axis_label_y = y_label

    def _reference_value(self, data: DataCollection) -> float:
        """Resolve the selected numeric or upstream global-attribute reference."""

        if not self.subtract_reference:
            return 0.0
        if self.reference_mode == _NUMERIC_REFERENCE:
            return float(self.reference_value)

        attribute_name = self.reference_attribute.strip()
        if not attribute_name:
            raise ValueError(
                "A global attribute name is required when the reference "
                "source is set to Global attribute."
            )
        try:
            return float(data.attributes[attribute_name])
        except KeyError as error:
            raise RuntimeError(
                f"Reference global attribute {attribute_name!r} was not found."
            ) from error
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Reference global attribute {attribute_name!r} must be numeric."
            ) from error
