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

class H5FlexTable2(object):
    def __init__(
        self,
        file_to_ids_map,
        file_to_handle_map,
        file_to_cumulative_selection,
        keys,
        file_to_flex_table,
        initial_size_per_file,
    ):
        # assert that can't have data from a single sweep in multiple files
        self._file_to_ids_map = file_to_ids_map
        self._file_to_handle_map = file_to_handle_map
        self._files = list(file_to_ids_map.keys())
        self._handles = list(file_to_handle_map.values())
        self._file_to_cumulative_selection = file_to_cumulative_selection
        self._keys = list(keys)  # must be same for all files in order to extend
        self._file_to_flex_table = file_to_flex_table
        self._initial_size_per_file = initial_size_per_file
        self._experiment_identifiers = {}
        for t in self._file_to_flex_table.values():
            self._experiment_identifiers.update(t.experiment_identifiers())

    def __str__(self):
        out = f"H5FlexTable at {id(self)}"
        files = ''.join(f"\n    {f}" for f in self._files)
        out += f"\n  Files: {files}"
        
        sizes = ", ".join(str(v) for v in self._initial_size_per_file.values())
        out += f"\n  Table sizes (n_rows) on disk: {sizes}"
        sizes = ", ".join(str(v.size()) for v in self._file_to_flex_table.values())
        out += f"\n  Current table sizes (n_rows) in memory: {sizes}"
        out += f"\n  N available keys {len(self._keys)}"#: {self._keys}"
        in_use = list(list(self._file_to_flex_table.values())[0].keys())
        out += f"\n  Keys in use (n={len(in_use)}): {in_use}"

        return out

    def experiment_identifiers(self):
        return self._experiment_identifiers


    @classmethod
    def from_file(cls, h5_file):
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
        file_to_flex_table = {h5_file : flex_table}
        return cls(
            file_to_ids_map,
            file_to_handle_map,
            file_to_cumulative_selection,
            keys,
            file_to_flex_table,
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
        for f, t in other._file_to_flex_table.items():
            if f in self._file_to_flex_table:
                self._file_to_flex_table[f].extend(t)
            else:
                self._file_to_flex_table[f] = t
        # FIXME check no clash?
        self._experiment_identifiers.update(other._experiment_identifiers)
        self._initial_size_per_file.update(other._initial_size_per_file)

    def size(self):
        n = 0
        for f in self._files:
            if self._file_to_cumulative_selection[f] is None:
                n += self._initial_size_per_file[f]
            else:
                n += self._file_to_cumulative_selection[f].size()
        return n

    def select(self, sel):
        # need sel to be a bool.
        if type(sel).__name__ == "size_t":
            bool_sel = flex.bool(self.size(), False)
            bool_sel.set_selected(sel, True)
            sel = bool_sel

        ftocs = copy.deepcopy(self._file_to_cumulative_selection)
        flex_table_map = {}
        if all(v is None for v in self._file_to_cumulative_selection.values()):
            n = 0
            for file_, initial_size in self._initial_size_per_file.items():
                this_sel = sel[n : n + initial_size]
                n += initial_size
                ftocs[file_] = this_sel.iselection()
                table = self._file_to_flex_table[file_]
                if table.size():
                    flex_table_map[file_] = self._file_to_flex_table[file_].select(this_sel)
                else:
                    flex_table_map[file_] = table
        else:
            # we already have some selection, so need to apply this new sel on top
            n = 0
            # need current sel to be a bool.
            for file_, file_sel in self._file_to_cumulative_selection.items():
                n_this = file_sel.size()  # current size
                this_sel = sel[n : n + n_this]
                ftocs[file_] = file_sel.select(this_sel)
                n += n_this
                flex_table_map[file_] = self._file_to_flex_table[file_].select(this_sel)
            assert n == self.size()
        handle_map = {}
        for file_ in self._files:
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle

        return H5FlexTable2(
            self._file_to_ids_map,
            handle_map,
            ftocs,
            self._keys,
            flex_table_map,
            self._initial_size_per_file,
        )

    def as_file(self, f):
        pass

    def __deepcopy__(self, memo):
        
        ftoids = copy.deepcopy(self._file_to_ids_map)
        handle_map = {}
        for file_ in ftoids.keys():
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
        #table = copy.deepcopy(self._flex_table, memo)
        keys = copy.deepcopy(self._keys, memo)
        ftocs = copy.deepcopy(self._file_to_cumulative_selection, memo)
        flex_table_map = copy.deepcopy(self._file_to_flex_table, memo)
        initial = copy.deepcopy(self._initial_size_per_file, memo)

        return H5FlexTable2(
            ftoids,
            handle_map,
            ftocs,
            keys,
            flex_table_map,
            initial,
        )

    def __len__(self):
        return self.size()

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
                self._file_to_flex_table[file_] = self._file_to_flex_table[file_].select(this_sel)
        else:
            n = 0
            # need current sel to be a bool.
            for file_, file_sel in self._file_to_cumulative_selection.items():
                n_this = file_sel.size()  # current size
                this_sel = bool_sel[n : n + n_this]
                self._file_to_cumulative_selection[file_] = file_sel.select(this_sel)
                n += n_this
                self._file_to_flex_table[file_] = self._file_to_flex_table[file_].select(this_sel)
            #assert n == len(self._flex_table)
        #self._flex_table.del_selected(sel)

    def __contains__(self, key):
        return key in self._keys

    def keys(self):
        return self._keys

    @property
    def flags(self):
        f = None
        for t in self._file_to_flex_table.values():
            ti = t.flags
            if f:
                f.extend(ti)
            else:
                f = ti
        return f

    def get_flags(self, *args, **kwargs):
        table = flex.reflection_table([])
        table["flags"] = self["flags"]
        return table.get_flags(*args, **kwargs)

    def unset_flags(self, sel, flags):
        if "flags" not in self._file_to_flex_table[self._files[0]].keys():
            self["flags"] = self["flags"]
        n = 0
        for file_, table in self._file_to_flex_table.items():
            n_this = table.size()
            this_sel = sel[n:n+n_this]
            #this_flags = flags[n:n+n_this]
            table.unset_flags(this_sel, flags)
            n += n_this

    def set_flags(self, sel, flags):
        if "flags" not in self._file_to_flex_table[self._files[0]].keys():
            self["flags"] = self["flags"]
        n = 0
        for file_, table in self._file_to_flex_table.items():
            n_this = table.size()
            this_sel = sel[n:n+n_this]
            #this_flags = flags[n:n+n_this]
            table.set_flags(this_sel, flags)
            n += n_this

    def __getitem__(self, k):
        assert k in self._keys
        all_data = None
        for f, t in self._file_to_flex_table.items():
            if k not in t:
                handle = self._file_to_handle_map[f]
                data = handle["entry"]["data"][k][()]
                data = self._convert(k, data)
                if self._file_to_cumulative_selection[f] is not None:
                    data = data.select(self._file_to_cumulative_selection[f])
                t[k] = data
            data = t[k]
            if not all_data:
                all_data = data
            else:
                print(f"joining for {k}")
                all_data = all_data.concatenate(data)
        return all_data

    def __setitem__(self, k, v):
        if sum(t.size() for t in self._file_to_flex_table.values()):
            assert v.size() == sum(t.size() for t in self._file_to_flex_table.values())
        if k not in self._keys:
            self._keys.append(k)
        if len(self._files) == 1:
            self._file_to_flex_table[self._files[0]][k] = v
            return
        n = 0
        for f, t in self._file_to_flex_table.items():
            sel = self._file_to_cumulative_selection[f]
            if sel is None:
                n_this = self._initial_size_per_file[f]
            else:
                n_this = sel.size()
            t[k] = v[n:n+n_this]
            n += n_this

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
        