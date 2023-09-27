from __future__ import annotations

import h5py

import iotbx.phil
from dxtbx import flumpy

import dials.util
from dials.util.options import ArgumentParser, flatten_reflections

phil_scope = iotbx.phil.parse(
    """
output.filename = refls.h5
.type = str
"""
)
import numpy as np


@dials.util.show_mail_handle_errors()
def run(args=None):
    """Run symmetry analysis from the command-line."""
    usage = "dials.convert_refl_to_h5 [options] observations.refl"

    parser = ArgumentParser(
        usage=usage,
        phil=phil_scope,
        read_reflections=True,
        read_experiments=False,
        check_format=False,
        epilog="",
    )

    params, options, args = parser.parse_args(
        args=args, show_diff_phil=False, return_unhandled=True
    )

    reflections = flatten_reflections(params.input.reflections)
    assert len(reflections) == 1
    reflections = reflections[0]

    handle = h5py.File(params.output.filename, "w")

    group = handle.create_group("entry/data")
    group.attrs["num_reflections"] = len(reflections)
    identifiers_group = handle.create_group("entry/experiment_identifiers")
    identifiers = np.array(
        [str(i) for i in reflections.experiment_identifiers().values()], dtype="S"
    )
    identifiers_group.create_dataset(
        "identifiers", data=identifiers, dtype=identifiers.dtype
    )
    ids = np.array(list(reflections.experiment_identifiers().keys()), dtype=int)
    identifiers_group.create_dataset("ids", data=ids, dtype=ids.dtype)
    # group.attrs["experiment_identifier"] = reflections.experiment_identifiers()[0]

    for col in reflections.keys():
        if col == "shoebox" or col == "bbox":
            continue
        data = flumpy.to_numpy(reflections[col])
        group.create_dataset(col, data=data, shape=data.shape, dtype=data.dtype)
    handle.close()
    print(f"Written converted file to {params.output.filename}")


if __name__ == "__main__":
    run()
