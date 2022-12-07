import numpy as np, pandas as pd
import pickle
import os

ds_dir = 'mouse_brain_datasets'
for f in os.listdir(ds_dir):
    if f.endswith('.pickle'):
        ds_df = pd.read_pickle(os.path.join(ds_dir,f))
        ion_names = ds_df.drop(columns=['x', 'y']).columns.tolist()
        ds_df[ion_names] = ds_df[ion_names] / (ds_df[ion_names].sum())
        ds_df.to_pickle(os.path.join(ds_dir,f))

