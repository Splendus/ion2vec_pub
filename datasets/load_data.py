import os
import pandas as pd
import metaspace # API to load data
from pixel_dataframes import create_pixel_df # create pixel dataframe from image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dsid", help="One or multiple ds_ids directly as input", nargs = "+", default=None) 
parser.add_argument("-csv", help="Reading a csv file containing ds_ids",  default=None)
parser.add_argument("-n", help="How many datasets of the csv to take", default=None)
parser.add_argument("-output", help="Specify output directory", default=None)
parser.add_argument("-fdr", help="fdr-threshold for annotations", default = 0.1)

args = parser.parse_args()
if args.dsid:
    for ds_id in args.dsid:
        ds, ion_names = create_pixel_df(ds_id, fdr=args.fdr)
        save = ds.to_pickle(os.path.join(args.output, f'pixel_df_{ds_id}.pickle'), protocol=4)

if args.csv:
    ds_df = pd.read_csv(f'{args.csv}')

    for ds_id in ds_df.ds_id[:args.n]:
        ds, ion_names  = create_pixel_df(ds_id, fdr=args.fdr)
        # sparse_df = ds.astype(pd.SparseDtype('float', 0))
        # save_sparse = sparse_df.to_pickle(f'pixel_df_{ds_id}.pickle')
        # save = ds.to_csv(f'pixel_df_{ds_id}.csv', index=False)
        save = ds.to_pickle(os.path.join(args.output, f'pixel_df_{ds_id}.pickle'), protocol=4)
