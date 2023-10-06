from __future__ import annotations

# have a multi-file H5FlexTable ?
# or, just extend flex table and then separate out when saving? or just save as combined for now.
import copy
import os
from pathlib import Path

import h5py

from dxtbx import flumpy

from dials.array_family import flex


class H5FlexTable2(object):
    def __init__(
        self,
        identifier_to_file_map,
        file_to_handle_map,
        identifier_to_cumulative_selection_map,
        keys,
        identifier_to_table_map,
        identifier_to_initial_size_map,
        experiment_identifiers_map,
    ):
        # assert that can't have data from a single sweep in multiple files
        self._identifier_to_file_map = identifier_to_file_map
        self._file_to_handle_map = file_to_handle_map
        self._identifier_to_cumulative_selection = (
            identifier_to_cumulative_selection_map
        )
        self._keys = keys  # must be same for all files in order to extend
        self._identifier_to_table_map = identifier_to_table_map
        self._identifier_to_initial_size_map = identifier_to_initial_size_map
        self._experiment_identifiers = experiment_identifiers_map

    def __str__(self):
        out = f"H5FlexTable at {id(self)}"
        files = "".join(f"\n    {f}" for f in self._file_to_handle_map.keys())
        out += f"\n  Files: {files}"

        sizes = ", ".join(str(v) for v in self._identifier_to_initial_size_map.values())
        out += f"\n  Table sizes (n_rows) on disk: {sizes}"
        sizes = ", ".join(str(v.size()) for v in self._identifier_to_table_map.values())
        out += f"\n  Current table sizes (n_rows) in memory: {sizes}"
        out += f"\n  N available keys {len(self._keys)}"  #: {self._keys}"
        in_use = list(list(self._identifier_to_table_map.values())[0].keys())
        out += f"\n  Keys in use (n={len(in_use)}): {in_use}"

        return out

    def reset_ids(self):
        reverse_map = {v: k for k, v in self.experiment_identifiers().items()}
        for k in list(self.experiment_identifiers().keys()):
            del self.experiment_identifiers()[k]
        if not len(self):
            return
        orig_id = self["id"].deep_copy()
        for i_exp, exp_id in enumerate(reverse_map.keys()):
            sel_exp = orig_id == reverse_map[exp_id]
            self["id"].set_selected(sel_exp, i_exp)
            self.experiment_identifiers()[i_exp] = exp_id

    def experiment_identifiers(self):
        return self._experiment_identifiers

    @classmethod
    def from_file(cls, h5_file):
        h5_file = os.fspath(Path(h5_file).resolve())
        handle = h5py.File(h5_file, "r", track_order=True)
        data = handle["reflections"]

        identifiers = []
        column_keys = None
        identifiers_to_initial_size_map = {}

        for k in data.keys():
            if data[k].attrs["experiment_identifier"]:
                assert k == data[k].attrs["experiment_identifier"]
                identifiers.append(k)
                if column_keys is None:
                    column_keys = set(data[k].keys())
                else:
                    assert column_keys == set(data[k].keys())
                identifiers_to_initial_size_map[k] = int(
                    data[k].attrs["num_reflections"]
                )

        identifiers_to_file_map = {}
        identifiers_to_table_map = {}
        identifiers_to_cumulative_selection_map = {}
        experiment_identifiers_map = {}
        for n, identifier in enumerate(identifiers):
            identifiers_to_file_map[identifier] = h5_file
            identifiers_to_table_map[identifier] = flex.reflection_table([])
            experiment_identifiers_map[n] = identifier
            identifiers_to_cumulative_selection_map[identifier] = None

        file_to_handle_map = {h5_file: handle}

        return cls(
            identifiers_to_file_map,
            file_to_handle_map,
            identifiers_to_cumulative_selection_map,
            list(column_keys),
            identifiers_to_table_map,
            identifiers_to_initial_size_map,
            experiment_identifiers_map,
        )

    def extend(self, other):
        # a 'virtual' extend
        assert set(self._keys) == set(other._keys)
        self._identifier_to_file_map.update(other._identifier_to_file_map)
        handle_map = {}
        for file_ in other._identifier_to_file_map.values():
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
        self._file_to_handle_map.update(handle_map)
        # self._files.extend(other._files)
        for identifier, v in other._identifier_to_cumulative_selection.items():
            if identifier in self._identifier_to_cumulative_selection:
                self._identifier_to_cumulative_selection[identifier].extend(v)
            else:
                self._identifier_to_cumulative_selection[identifier] = v
        for identifier, t in other._identifier_to_table_map.items():
            if identifier in self._identifier_to_table_map:
                self._identifier_to_table_map[identifier].extend(t)
            else:
                self._identifier_to_table_map[identifier] = t
        # FIXME check no clash?
        self._experiment_identifiers.update(other._experiment_identifiers)
        self._identifier_to_initial_size_map.update(
            other._identifier_to_initial_size_map
        )

    def size(self):
        n = 0
        for identifier, sel in self._identifier_to_cumulative_selection.items():
            if sel is None:
                n += self._identifier_to_initial_size_map[identifier]
            else:
                n += sel.size()
        return n

    def select_on_experiment_identifiers(self, list_of_identifiers):
        for identifier in list(self._identifier_to_file_map.keys()):
            if identifier not in list_of_identifiers:
                """file = self._identifier_to_file_map[identifier]
                handle = self._file_to_handle_map[file]
                handle.close()
                del self._file_to_handle_map[file]"""
                del self._identifier_to_cumulative_selection[identifier]
                del self._identifier_to_table_map[identifier]
                del self._identifier_to_initial_size_map[identifier]
                for k, v in zip(
                    list(self._experiment_identifiers.keys()),
                    list(self._experiment_identifiers.values()),
                ):
                    if v == identifier:
                        del self._experiment_identifiers[k]
                del self._identifier_to_file_map[identifier]
                # now if no more identifiers point to that file - close the handle?
        # FIXME update experiment_identifiers?
        return self

    def split_by_experiment_id(self):
        tables = []
        for id_, identifier in enumerate(self._identifier_to_file_map.keys()):
            file_ = copy.deepcopy(self._identifier_to_file_map[identifier])
            idtocs = {
                identifier: copy.deepcopy(
                    self._identifier_to_cumulative_selection[identifier]
                )
            }
            handle_map = {}
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
            id_to_file_map = {identifier: file_}
            flex_table_map = {
                identifier: copy.deepcopy(self._identifier_to_table_map[identifier])
            }
            initial_size_map = {
                identifier: copy.deepcopy(
                    self._identifier_to_initial_size_map[identifier]
                )
            }
            experiment_identifiers = {id_: identifier}
            tables.append(
                H5FlexTable2(
                    id_to_file_map,
                    handle_map,
                    idtocs,
                    copy.deepcopy(self._keys),
                    flex_table_map,
                    initial_size_map,
                    experiment_identifiers,
                )
            )
        return tables

    def select(self, sel):
        # need sel to be a bool.
        if type(sel).__name__ == "size_t":
            bool_sel = flex.bool(self.size(), False)
            bool_sel.set_selected(sel, True)
            sel = bool_sel

        idtocs = copy.deepcopy(self._identifier_to_cumulative_selection)
        flex_table_map = {}
        experiment_identifiers_map = copy.deepcopy(self.experiment_identifiers())
        if all(v is None for v in self._identifier_to_cumulative_selection.values()):
            n = 0
            for i, (
                identifier,
                initial_size,
            ) in enumerate(self._identifier_to_initial_size_map.items()):
                this_sel = sel[n : n + initial_size]
                n += initial_size
                idtocs[identifier] = this_sel.iselection()
                table = self._identifier_to_table_map[identifier]
                if (
                    table.size()
                ):  # FIXME tables always have at least id? so unnecessary check?
                    flex_table_map[identifier] = table.select(this_sel)

                else:
                    flex_table_map[identifier] = table  # pass the empty table
        else:
            # we already have some selection, so need to apply this new sel on top
            n = 0
            # need current sel to be a bool.
            for (
                identifier,
                identifier_sel,
            ) in self._identifier_to_cumulative_selection.items():
                n_this = identifier_sel.size()  # current size
                this_sel = sel[n : n + n_this]
                idtocs[identifier] = identifier_sel.select(this_sel)
                n += n_this
                flex_table_map[identifier] = self._identifier_to_table_map[
                    identifier
                ].select(this_sel)
            assert n == self.size()
        handle_map = {}
        for file_ in self._file_to_handle_map.keys():
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle

        return H5FlexTable2(
            copy.deepcopy(self._identifier_to_file_map),
            handle_map,
            idtocs,
            copy.deepcopy(self._keys),
            flex_table_map,
            copy.deepcopy(self._identifier_to_initial_size_map),
            experiment_identifiers_map,
        )

    def as_file(self, f):
        new_handle = h5py.File(f, "w", track_order=True)
        for identifier, t in self._identifier_to_table_map.items():
            group = new_handle.create_group(f"reflections/{identifier}")
            group.attrs["num_reflections"] = t.size()
            group.attrs["experiment_identifier"] = identifier
            for k in self._keys:
                if k in t:
                    data = flumpy.to_numpy(t[k])
                else:  # pull from disk and write to new file (#FIXME proper h5 vds?)
                    file_ = self._identifier_to_file_map[identifier]
                    handle = self._file_to_handle_map[file_]
                    data = handle["reflections"][identifier][k][()]
                    sel = self._identifier_to_cumulative_selection[identifier]
                    if sel is not None:
                        sel = flumpy.to_numpy(sel)
                        data = data[sel]
                group.create_dataset(k, data=data, shape=data.shape, dtype=data.dtype)
        new_handle.close()

    def __delitem__(self, key):
        id_0 = list(self._identifier_to_table_map.keys())[0]
        if key in self._identifier_to_table_map[id_0].keys():
            for t in self._identifier_to_table_map.values():
                del t[key]
            # FIXME - should we remove from keys if on disk? probably not - check if key is in file.
            self._keys.remove(key)

    def __deepcopy__(self, memo):

        idtof = copy.deepcopy(self._identifier_to_file_map, memo)
        files = copy.deepcopy(list(self._file_to_handle_map.keys()), memo)
        handle_map = {}
        for file_ in files:
            handle = h5py.File(file_, "r")
            handle_map[file_] = handle
        # table = copy.deepcopy(self._flex_table, memo)
        keys = copy.deepcopy(self._keys, memo)
        idtocs = copy.deepcopy(self._identifier_to_cumulative_selection, memo)
        flex_table_map = copy.deepcopy(self._identifier_to_table_map, memo)
        initial = copy.deepcopy(self._identifier_to_initial_size_map, memo)
        experiment_identifiers_map = copy.deepcopy(self._experiment_identifiers, memo)
        return H5FlexTable2(
            idtof,
            handle_map,
            idtocs,
            keys,
            flex_table_map,
            initial,
            experiment_identifiers_map,
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

        if all(v is None for v in self._identifier_to_cumulative_selection.values()):
            n = 0
            for (
                identifier,
                initial_size,
            ) in self._identifier_to_initial_size_map.items():
                this_sel = bool_sel[n : n + initial_size]
                n += initial_size
                self._identifier_to_cumulative_selection[
                    identifier
                ] = this_sel.iselection()
                self._identifier_to_table_map[
                    identifier
                ] = self._identifier_to_table_map[identifier].select(this_sel)
        else:
            n = 0
            # need current sel to be a bool.
            for (
                identifier,
                identifier_sel,
            ) in self._identifier_to_cumulative_selection.items():
                n_this = identifier_sel.size()  # current size
                this_sel = bool_sel[n : n + n_this]
                self._identifier_to_cumulative_selection[
                    identifier
                ] = identifier_sel.select(this_sel)
                n += n_this
                self._identifier_to_table_map[
                    identifier
                ] = self._identifier_to_table_map[identifier].select(this_sel)

    def __contains__(self, key):
        return key in self._keys

    def keys(self):
        return self._keys

    @property
    def flags(self):
        id_0 = list(self._identifier_to_table_map.keys())[0]

        return self._identifier_to_table_map[id_0].flags

    def get_flags(self, *args, **kwargs):
        table = flex.reflection_table([])
        table["flags"] = self["flags"]
        res = table.get_flags(*args, **kwargs)
        return res

    def unset_flags(self, sel, flags):
        id_0 = list(self._identifier_to_table_map.keys())[0]
        if "flags" not in self._identifier_to_table_map[id_0].keys():
            self["flags"] = self["flags"]
        n = 0
        for identifier, table in self._identifier_to_table_map.items():
            n_this = table.size()
            this_sel = sel[n : n + n_this]
            # this_flags = flags[n:n+n_this]
            table.unset_flags(this_sel, flags)
            n += n_this

    def set_flags(self, sel, flags):
        id_0 = list(self._identifier_to_table_map.keys())[0]
        if "flags" not in self._identifier_to_table_map[id_0].keys():
            self["flags"] = self["flags"]
        n = 0
        for identifier, table in self._identifier_to_table_map.items():
            n_this = table.size()
            this_sel = sel[n : n + n_this]
            # this_flags = flags[n:n+n_this]
            table.set_flags(this_sel, flags)
            n += n_this

    def __getitem__(self, k):
        assert k in self._keys
        all_data = None
        for identifier, t in self._identifier_to_table_map.items():
            if k not in t:
                f = self._identifier_to_file_map[identifier]
                handle = self._file_to_handle_map[f]
                data = handle["reflections"][identifier][k][()]
                data = self._convert(k, data)
                if self._identifier_to_cumulative_selection[identifier] is not None:
                    data = data.select(
                        self._identifier_to_cumulative_selection[identifier]
                    )
                t[k] = data
            data = t[k]
            if not all_data:
                all_data = data
            else:
                all_data = all_data.concatenate(data)
        return all_data

    def __setitem__(self, k, v):
        if any(t.size() for t in self._identifier_to_table_map.values()):
            assert v.size() == sum(
                t.size() for t in self._identifier_to_table_map.values()
            )
        if k not in self._keys:
            self._keys.append(k)
        if len(self._identifier_to_table_map) == 1:
            list(self._identifier_to_table_map.values())[0][k] = v
            # self._file_to_flex_table[self._files[0]][k] = v
            return
        n = 0
        for identifier, t in self._identifier_to_table_map.items():
            sel = self._identifier_to_cumulative_selection[identifier]
            if sel is None:
                n_this = self._identifier_to_initial_size_map[identifier]
            else:
                n_this = sel.size()
            t[k] = v[n : n + n_this]
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
