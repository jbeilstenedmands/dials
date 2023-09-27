from __future__ import annotations

# have a multi-file H5FlexTable ?
# or, just extend flex table and then separate out when saving? or just save as combined for now.
import copy
import os
from pathlib import Path

import h5py
import numpy as np

from dxtbx import flumpy

from dials.array_family import flex


class H5FlexTable(object):
    def __init__(
        self,
        file_to_ids_map,
        file_to_handle_map,
        file_to_cumulative_selection,
        keys,
        flex_table,
        initial_size_per_file,
    ):
        # assert that can't have data from a single sweep in multiple files
        self._file_to_ids_map = file_to_ids_map
        self._file_to_handle_map = file_to_handle_map
        self._files = list(file_to_ids_map.keys())
        self._handles = list(file_to_handle_map.values())
        self._file_to_cumulative_selection = file_to_cumulative_selection
        self._keys = list(keys)  # must be same for all files in order to extend
        self._flex_table = (
            flex_table  # a single table, can contain data from multiple files/sweeps
        )
        self._initial_size_per_file = initial_size_per_file

    def __str__(self):

        print(self._file_to_ids_map)
        print(self._file_to_handle_map)
        print(self._file_to_cumulative_selection)
        print(list(f.size() for f in self._file_to_cumulative_selection.values()))
        print(self._keys)
        print(self._flex_table.size())
        print(self._initial_size_per_file)
        return ""

    def select_on_experiment_identifiers(self, list_of_identifiers):
        id_values = []
        for k, v in zip(
            self._flex_table.experiment_identifiers().keys(), self._flex_table.experiment_identifiers().values()
        ):
            if v in list_of_identifiers:
                id_values.append(k)
        '''if len(id_values) != len(list_of_identifiers):
            logger.warning(
                """Not all requested identifiers
found in the table's map, has the experiment_identifiers() map been created?
Requested %s:
Found %s"""
                % (list_of_identifiers, id_values)
            )'''
        # Build up a selection and use this
        sel = flex.bool(self.size(), False)
        for id_val in id_values:
            id_sel = self["id"] == id_val
            sel.set_selected(id_sel, True)
        new = self.select(sel)
        # Remove entries from the experiment_identifiers map
        for k in new._flex_table.experiment_identifiers().keys():
            if k not in id_values:
                del new._flex_table.experiment_identifiers()[k]
        return new
        

    @classmethod
    def from_file(cls, h5_file):
        print("here")
        h5_file = os.fspath(Path(h5_file).resolve())
        handle = h5py.File(h5_file, "r")
        keys = handle["entry"]["data"].keys()
        flex_table = flex.reflection_table([])
        size = handle["entry"]["data"].attrs["num_reflections"]
        identifiers = handle["entry"]["experiment_identifiers"]["identifiers"][()]
        ids = handle["entry"]["experiment_identifiers"]["ids"][()]
        for id_, identifier in zip(ids, identifiers):
            flex_table.experiment_identifiers()[int(id_)] = str(identifier.decode())
        file_to_ids_map = {h5_file: ids}
        file_to_handle_map = {h5_file: handle}
        file_to_cumulative_selection = {h5_file: None}
        initial_size_per_file = {h5_file: int(size)}
        return cls(
            file_to_ids_map,
            file_to_handle_map,
            file_to_cumulative_selection,
            keys,
            flex_table,
            initial_size_per_file,
        )

    def extend(self, other):
        assert set(self._keys) == set(other._keys)
        self._file_to_ids_map.update(other._file_to_ids_map)
        handle_map = {}
        for file_ in other._files:
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
        self._file_to_handle_map.update(handle_map)
        self._files.extend(other._files)
        self._handles.extend(handle_map.values())
        for f, v in other._file_to_cumulative_selection.items():
            if f in self._file_to_cumulative_selection:
                self._file_to_cumulative_selection[f].extend(v)
            else:
                self._file_to_cumulative_selection[f] = v
        #self._file_to_cumulative_selection.update(other._file_to_cumulative_selection)
        self._flex_table = flex.reflection_table.concat([self._flex_table, other._flex_table])
        self._initial_size_per_file.update(other._initial_size_per_file)
        

    def select(self, sel):
        # need sel to be a bool.
        if type(sel).__name__ == "size_t":
            bool_sel = flex.bool(self.size(), False)
            bool_sel.set_selected(sel, True)
            sel = bool_sel

        ftocs = copy.deepcopy(self._file_to_cumulative_selection)
        if all(v is None for v in self._file_to_cumulative_selection.values()):
            # cumulative_selection = sel.iselection()
            # partition by file
            n = 0
            for file_, initial_size in self._initial_size_per_file.items():
                this_sel = sel[n : n + initial_size]
                n += initial_size
                ftocs[file_] = this_sel.iselection()
        else:
            # we already have some selection, so need to apply this new sel on top
            n = 0
            # need current sel to be a bool.
            for file_, file_sel in self._file_to_cumulative_selection.items():
                n_this = file_sel.size()  # current size
                this_sel = sel[n : n + n_this]
                ftocs[file_] = file_sel.select(this_sel)
                n += n_this
            print(n)
            print(len(self._flex_table))
            print(self)
            assert n == len(self._flex_table)
        flex_table = self._flex_table.select(sel)
        handle_map = {}
        for file_ in self._files:
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle

        return H5FlexTable(
            self._file_to_ids_map,
            handle_map,
            ftocs,
            self._keys,
            flex_table,
            self._initial_size_per_file,
        )

    def split_by_experiment_id(self):
        tables = []
        for id_ in list(set(self["id"])):
            sel = (self["id"] == id_)
            tables.append(self.select(sel))
        return tables

    def del_selected(self, sel):
        if type(sel).__name__ == "size_t":
            bool_sel = flex.bool(self.size(), True)
            bool_sel.set_selected(sel, False)
        else:
            bool_sel = flex.bool(self.size(), True)
            bool_sel.set_selected(sel, False)

        if all(v is None for v in self._file_to_cumulative_selection.values()):
            n = 0
            for file_, initial_size in self._initial_size_per_file.items():
                this_sel = bool_sel[n : n + initial_size]
                n += initial_size
                self._file_to_cumulative_selection[file_] = this_sel.iselection()
        else:
            n = 0
            # need current sel to be a bool.
            for file_, file_sel in self._file_to_cumulative_selection.items():
                n_this = file_sel.size()  # current size
                this_sel = bool_sel[n : n + n_this]
                self._file_to_cumulative_selection[file_] = file_sel.select(this_sel)
                n += n_this
            assert n == len(self._flex_table)
        self._flex_table.del_selected(sel)

    def size(self):
        if all(v is None for v in self._file_to_cumulative_selection.values()):
            return sum(v for v in self._initial_size_per_file.values())
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

    def __del__(self):
        print(len(self._keys))
        print(len(list(self._flex_table.keys())))
        print(list(self._flex_table.keys()))
        for handle in self._handles:
            handle.close()

    def keys(self):
        return self._keys

    def __getitem__(self, k):
        assert k in self._keys
        if k not in self._flex_table:
            data = self._handles[0]["entry"]["data"][k][()]
            data = self._convert(k, data)
            if self._file_to_cumulative_selection[self._files[0]] is not None:
                data = data.select(self._file_to_cumulative_selection[self._files[0]])
            if len(self._handles) > 1:
                for file_ in self._files[1:]:
                    handle = self._file_to_handle_map[file_]
                    handle_data = handle["enty"]["data"][k][()]
                    handle_data = self._convert(k, handle_data)
                    if self._file_to_cumulative_selection[file_] is not None:
                        handle_data = handle_data.select(
                            self._file_to_cumulative_selection[file_]
                        )
                data.extend(handle_data)
            print(data.size())
            print(self._flex_table.size())
            self._flex_table[k] = data
        return self._flex_table[k]

    def __deepcopy__(self, memo):
        
        ftoids = copy.deepcopy(self._file_to_ids_map)
        handle_map = {}
        for file_ in ftoids.keys():
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
        table = copy.deepcopy(self._flex_table, memo)
        keys = copy.deepcopy(self._keys, memo)
        ftocs = copy.deepcopy(self._file_to_cumulative_selection, memo)
        initial = copy.deepcopy(self._initial_size_per_file)

        return H5FlexTable(
            ftoids,
            handle_map,
            ftocs,
            keys,
            table,
            initial,
        )

    def __setitem__(self, k, v):
        print(v.size())
        print(self._flex_table.size())
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
        return new

    def as_file(self, f):
        new_handle = h5py.File(f, "w")
        group = new_handle.create_group("entry/data")
        identifiers_group = new_handle.create_group("entry/experiment_identifiers")
        group.attrs["num_reflections"] = self.size()
        identifiers = np.array(
            [str(i) for i in self._flex_table.experiment_identifiers().values()],
            dtype="S",
        )
        identifiers_group.create_dataset(
            "identifiers", data=identifiers, dtype=identifiers.dtype
        )
        ids = np.array(
            list(self._flex_table.experiment_identifiers().keys()), dtype=int
        )
        identifiers_group.create_dataset("ids", data=ids, dtype=ids.dtype)

        for k in self._keys:
            if k in self._flex_table:
                data = flumpy.to_numpy(self._flex_table[k])
            else:
                data = None
                for file_, handle in self._file_to_handle_map.items():
                    this_data = handle["entry"]["data"][k][()]
                    if self._file_to_cumulative_selection[file_] is not None:
                        sel = flumpy.to_numpy(self._file_to_cumulative_selection[file_])
                        this_data = this_data[sel]
                    if data is None:
                        data = this_data
                    else:
                        data = np.concatenate([data, this_data])
            group.create_dataset(k, data=data, shape=data.shape, dtype=data.dtype)
        new_handle.close()

    def __getattr__(self, name):
        print(f"delegating call to {name}")
        return getattr(self._flex_table, name)
