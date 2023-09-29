from dials.array_family.h5_flex_table_2 import H5FlexTable2
from scitbx.array_family import flex

t1 = H5FlexTable2.from_file("/Users/whi10850/Documents/data/rotation/ins_8_3/data/SWEEP1.h5")
t2 = H5FlexTable2.from_file("/Users/whi10850/Documents/data/rotation/ins_8_3/data/SWEEP2.h5")
t3 = H5FlexTable2.from_file("/Users/whi10850/Documents/data/rotation/ins_8_3/data/SWEEP3.h5")

from dials.util.multi_dataset_handling import renumber_table_id_columns
t1,t2,t3 = renumber_table_id_columns([t1,t2,t3])


print("extending t1 with t2")
t1.extend(t2)
print("t1 experiment identifiers")
print(t1.experiment_identifiers())

sel = flex.bool(t1["id"].size(), False)
sel[0] = True
sel[-1] = True
sel[-2] = True
t4 = t1.select(sel)
print("t1 id column after selection")
print(list(t4["id"]))
print(t4.size())
t4["id"] = flex.int(t4.size(), 2)
print("t4 id after setting to 2.")
print(list(t4["id"]))
print(t4)
