"""Example use of the Cumulative Property Profile modifier."""

from ovito.io import export_file, import_file

from cumulative_property_profile import CumulativePropertyProfileModifier


pipeline = import_file("input.dump")
pipeline.modifiers.append(
    CumulativePropertyProfileModifier(
        axis="Z",
        input_property="Mass",
    )
)

data = pipeline.compute()
print(data.tables["cumulative-property-profile"].xy())
export_file(
    pipeline,
    "cumulative-property-profile.txt",
    "txt/table",
    key="cumulative-property-profile",
)
