from __future__ import annotations

import h5py

from dxtbx import flumpy

from dials.array_family import flex


class H5FlexTable(object):
    def __init__(
        self, file, handle, cumulative_selection, keys, flex_table, size, identifier
    ):
        self._file = file
        self._handle = handle
        self._cumulative_selection = cumulative_selection
        self._keys = list(keys)
        self._flex_table = flex_table
        self._initial_size = size
        self._identifier = identifier
        self._flex_table.experiment_identifiers()[0] = identifier

    @classmethod
    def from_file(cls, h5_file):
        handle = h5py.File(h5_file, "r")
        cumulative_selection = None
        keys = handle["entry"]["SWEEP1"].keys()
        flex_table = flex.reflection_table([])
        size = handle["entry"]["SWEEP1"].attrs["num_reflections"]
        identifier = handle["entry"]["SWEEP1"].attrs["experiment_identifier"]
        return cls(
            h5_file, handle, cumulative_selection, keys, flex_table, size, identifier
        )

    def select(self, sel):
        if self._cumulative_selection:
            cumulative_selection = self._cumulative_selection.select(sel)
        else:
            cumulative_selection = sel.iselection()
        flex_table = self._flex_table.select(sel)
        size = flex_table.size()
        keys = self._keys
        file = self._file
        handle = h5py.File(file)
        identifier = self._identifier
        return H5FlexTable(
            file, handle, cumulative_selection, keys, flex_table, size, identifier
        )

    def size(self):
        if self._cumulative_selection is None:
            return int(self._initial_size)
        return self._flex_table.size()

    def get_flags(self, flag, *args, **kwargs):
        if "flags" not in self._flex_table:
            self._flex_table["flags"] = self["flags"]
        return self._flex_table.get_flags(flag, *args, **kwargs)

    def unset_flags(self, sel, flags):
        if "flags" not in self._flex_table:
            self._flex_table["flags"] = self["flags"]
        self._flex_table.unset_flags(sel, flags)

    def set_flags(self, sel, flags):
        if "flags" not in self._flex_table:
            self._flex_table["flags"] = self["flags"]
        self._flex_table.set_flags(sel, flags)

    def __len__(self):
        return self.size()

    def __contains__(self, key):
        return key in self._keys

    def __delitem__(self, key):
        if key in self._flex_table:
            del self._flex_table[key]
            self._keys.remove(key)

    def del_selected(self, sel):
        if type(sel).__name__ == "bool":
            if self._cumulative_selection:
                self._cumulative_selection = self._cumulative_selection.select(~sel)
            else:
                self._cumulative_selection = sel.iselection()
        else:
            # if self._cumulative_selection:
            n = self.size()
            non_isel = flex.bool(n, True)
            non_isel.set_selected(sel, False)
            if self._cumulative_selection:
                self._cumulative_selection = self._cumulative_selection.select(non_isel)
            else:
                self._cumulative_selection = non_isel.iselection()
        self._flex_table.del_selected(sel)

    def __del__(self):
        print(len(self._keys))
        print(len(list(self._flex_table.keys())))
        print(list(self._flex_table.keys()))
        self._handle.close()

    def keys(self):
        return self._keys

    def __getitem__(self, k):
        assert k in self._keys
        if k not in self._flex_table:
            data = self._handle["entry"]["SWEEP1"][k][()]
            self._flex_table[k] = self._convert(k, data)
        return self._flex_table[k]

    def __setitem__(self, k, v):
        self._flex_table[k] = v
        if k not in self._keys:
            self._keys.append(k)

    def _convert(self, key, data):
        if key == "miller_index":  # special
            val = flumpy.vec_from_numpy(data)
            new = flex.miller_index(
                val.as_vec3_double().parts()[0].iround(),
                val.as_vec3_double().parts()[1].iround(),
                val.as_vec3_double().parts()[2].iround(),
            )
        elif len(data.shape) == 2:
            if data.shape[1] == 3 or data.shape[1] == 2:  # vec3 or vec2 double
                new = flumpy.vec_from_numpy(data)
            else:
                raise RuntimeError("Unrecognised data")
        else:
            new = flumpy.from_numpy(data)
        if self._cumulative_selection:
            new = new.select(self._cumulative_selection)
        return new

    def as_file(self, f):
        handle = h5py.File(f, "w")
        group = handle.create_group("entry/SWEEP1")
        group.attrs["num_reflections"] = self.size()
        group.attrs["experiment_identifier"] = self._identifier

        for k in self._keys:
            if k in self._flex_table:
                data = flumpy.to_numpy(self._flex_table[k])
                group.create_dataset(k, data=data, shape=data.shape, dtype=data.dtype)
            else:
                data = self._handle["entry"]["SWEEP1"][k][()]
                if self._cumulative_selection:
                    sel = flumpy.to_numpy(self._cumulative_selection)
                    data = data[sel]
                group.create_dataset(k, data=data, shape=data.shape, dtype=data.dtype)
        handle.close()

    def __getattr__(self, name):
        print(f"delegating call to {name}")
        return getattr(self._flex_table, name)
