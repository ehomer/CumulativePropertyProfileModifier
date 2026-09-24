"""OVITO modifier for atom-resolved cumulative property profiles."""

from __future__ import annotations

import numpy as np
from ovito.data import DataCollection, DataTable
from ovito.pipeline import ModifierInterface
from traits.api import Enum, String

__all__ = ["CumulativePropertyProfileModifier"]


_AXES = ["X", "Y", "Z"]
_AXIS_INDICES = {axis: index for index, axis in enumerate(_AXES)}


class CumulativePropertyProfileModifier(ModifierInterface):
    """Plot the cumulative sum of a scalar particle property along an axis.

    Particles are sorted by their coordinate along the selected axis for the
    calculation only. The particle order in the data pipeline is not changed.
    The result is stored in a line-plot ``DataTable`` with particle positions
    as x-values and the cumulative property sum as y-values.
    """

    axis = Enum("Z", _AXES, label="Axis")
    input_property = String("", label="Particle property")

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

        axis_index = _AXIS_INDICES[self.axis]
        sort_order = np.argsort(positions[:, axis_index], kind="stable")
        sorted_positions = positions[sort_order, axis_index]
        cumulative_values = np.cumsum(values[sort_order], dtype=float)

        yield f"Calculating cumulative {property_name} profile"

        table = data.tables.create(
            identifier="cumulative-property-profile",
            title=f"Cumulative {property_name} vs. {self.axis}",
            plot_mode=DataTable.PlotMode.Line,
        )
        table.x = table.create_property(
            f"Position {self.axis}", data=sorted_positions
        )
        table.y = table.create_property(
            f"Cumulative {property_name}", data=cumulative_values
        )
        table.axis_label_x = f"Position ({self.axis})"
        table.axis_label_y = f"Cumulative {property_name}"
