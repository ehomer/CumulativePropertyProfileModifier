"""Create an entropy profile from the included aluminum structure.

This example requires the ``WeightedNeighborAverageModifier`` package to be
installed and importable. The ``CalculateLocalEntropyFunction`` is provided
by the OVITO Pro Python environment.
"""

import functools
from pathlib import Path

from ovito.io import export_file, import_file
from ovito.modifiers import *

from weighted_neighbor_average import *
from cumulative_property_profile import *


example_file = Path(__file__).with_name(
    "rstruct_S3_fcc_N0_n1_1_Al_M99_201228.249.out"
)
pipeline = import_file(str(example_file))

# Python modifier function 'Calculate local entropy':
pipeline.modifiers.append(functools.partial(CalculateLocalEntropyFunction,
    cutoff=4.5,
    sigma=0.2,
    use_local_density=False,
    compute_average=False,
    average_cutoff=5.0))

# Entropy Profile - Weighted Neighbor Average:
pipeline.modifiers.append(
    WeightedNeighborAverageModifier(
        input_property="Entropy",
        cutoff=4.5,
    )
)

# Entropy Profile - Expression selection:
pipeline.modifiers.append(
    ExpressionSelectionModifier(expression="Position.Z < -40 || Position.Z > 40")
)

# Entropy Profile - Delete selected:
pipeline.modifiers.append(DeleteSelectedModifier())

# Entropy Profile - Cumulative Property Profile:
pipeline.modifiers.append(
    CumulativePropertyProfileModifier(
        axis="Z",
        input_property="Entropy averaged",
        reference_value=-6.311,
    )
)

# Entropy Profile - Color coding:
pipeline.modifiers.append(ColorCodingModifier(property="Entropy averaged"))

data = pipeline.compute()

fit_prefix = "CumulativePropertyProfile.Entropy averaged"
print("Fit results:")
print(f"Width (w95): {data.attributes[f'{fit_prefix}.fit_width']:.6g}")
print(
    f"R^2: {data.attributes[f'{fit_prefix}.fit_r2']:.6g} "
    f"(RMSE: {data.attributes[f'{fit_prefix}.fit_rmse']:.6g})"
)
print(f"Beta: {data.attributes[f'{fit_prefix}.fit_beta']:.6g}")
print(f"Alpha: {data.attributes[f'{fit_prefix}.fit_alpha']:.6g}")

export_file(
    pipeline,
    "entropy-cumulative-property-profile.txt",
    "txt/table",
    key="cumulative-property-profile",
)
