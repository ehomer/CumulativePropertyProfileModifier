# Cumulative Property Profile

An OVITO Python modifier that creates an atom-resolved cumulative profile of
a scalar particle property along the X, Y, or Z axis.

## Description

The modifier sorts particles by their position along the selected axis for the
calculation, computes the cumulative sum of the selected scalar particle
property, and outputs the result as a line-plot `DataTable`. The input particle
order is not changed, and no spatial binning is performed.

## Parameters

- `axis`: Coordinate axis used to order the particles (`X`, `Y`, or `Z`).
- `input_property`: Name of the scalar particle property to accumulate.

## Example

```python
from ovito.io import import_file
from cumulative_property_profile import CumulativePropertyProfileModifier

pipeline = import_file("input.dump")
pipeline.modifiers.append(
    CumulativePropertyProfileModifier(
        axis="Z",
        input_property="Mass",
    )
)

data = pipeline.compute()
profile = data.tables["cumulative-property-profile"]
print(profile.xy())
```

The table appears as a line plot in OVITO Pro's data inspector and can also be
exported with:

```python
from ovito.io import export_file

export_file(
    pipeline,
    "cumulative-profile.txt",
    "txt/table",
    key="cumulative-property-profile",
)
```

## Installation

From a checked-out copy, install into OVITO Pro's integrated Python environment:

```text
ovitos -m pip install --user --editable /path/to/CumulativePropertyProfileModifier
```

For another Python interpreter or Conda environment, use:

```text
pip install --editable /path/to/CumulativePropertyProfileModifier
```

The `--editable` option makes the package importable while keeping the source
tree live for development.

```text
ovitos -m pip install --user --editable .
```

## Technical information

- Requires OVITO 3.9.1 or newer.
- Requires NumPy, supplied by OVITO's Python environment.

## License

MIT License.
