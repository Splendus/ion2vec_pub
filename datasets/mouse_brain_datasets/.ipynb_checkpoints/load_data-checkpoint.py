import numpy as np, pandas as pd
import metaspace # API to load data
from pixel_dataframes import create_pixel_df # create pixel dataframe from image
import argparse

parser = argparse.ArgumentParser(description = 'Process data loading')
parser.add_argument('-input', metavar='Input csv file with dataset IDs', type = str, default = 'datasets_filtered_weighted.csv', 
help = 'Dataset dataframe csv file from which to sample')

args = parser.parse_args()

ds_df = pd.read_csv(args.input)

for ds_id in ds_df.ds_id:
    try:
        ds, _  = create_pixel_df(ds_id)
    except (AssertionError, IndexError):
        continue
    # sparse_df = ds.astype(pd.SparseDtype('float', 0))
    # save_sparse = sparse_df.to_pickle(f'pixel_df_{ds_id}.pickle')
    # save = ds.to_csv(f'pixel_df_{ds_id}.csv', index=False)
    save = ds.to_pickle(f'pixel_df_{ds_id}.pickle')
