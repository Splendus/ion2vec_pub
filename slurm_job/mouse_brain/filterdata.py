import os
import numpy as np, pandas as pd, pickle

ds_dir = 'data'

with open('ind_names.pickle', 'rb') as f:
    ind_names = pickle.load(f)

for f in os.listdir(ds_dir):
    pkl_df = pd.read_pickle(os.path.join(ds_dir,f))
    ionlist = pkl_df.drop(columns=['x', 'y']).columns.tolist()
    keep_ions = list(set(ind_names).intersection(set(ionlist)))
    keep_ions = ['y', 'x'] + keep_ions 
    pkl_df = pkl_df.loc[keep_ions]
    pkl_df.to_pickle(os.path.join(ds_dir,f))
