import os
import pickle
import random
import pandas as pd

tbl_path = os.path.abspath("/scratch/trose/pruned_annotations_dict_300922.pickle")

tbl_list = pickle.load(open(tbl_path, 'rb')) 

rand = random.choice(list(tbl_list.values()))

print((rand.fdr.value_counts()[0.5]))

fdr_counts = rand.fdr.value_counts()

fdr_dict = {ds_id: tbl_list[ds_id].fdr.value_counts() for ds_id in tbl_list.keys()}


rand2 = random.choice(list(fdr_dict.values()))
print(rand2)
