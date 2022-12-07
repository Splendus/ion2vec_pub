from CleaningMetadata import get_stratified_sample
import pandas as pd, numpy as np
import argparse 

ds_df_csv = 'datasets_filtered_weighted.csv'

parser = argparse.ArgumentParser(description = 'Process data filtering')
parser.add_argument('-ds_df', metavar='dataset dataframe', type = str, default = ds_df_csv, 
help = 'Dataset dataframe csv file from which to sample')
parser.add_argument('-org', metavar='organism', type = str, default = None, 
help = 'Organisms to filter for. Default are human, rat and mouse')
parser.add_argument('-org_part', metavar='organism_part', type = str, nargs='+', default = 'brain', 
help = 'Organism type to filter for. Default is brain.')
parser.add_argument('-cond', metavar='condition', type = str, nargs='+', default = None, 
help = 'Condition to filter for. Default is None.')
parser.add_argument('-n', metavar='Number of datasets', type=int, default = 100, 
help = 'Number of datasets to sample.')
parser.add_argument('-output', type = str, default = 'stratified_sample.csv', 
help = 'Name of output file')
parser.add_argument('-fdr_ann', metavar='Required minimal number of anns of up to 10$ fdr threshold', type = float, default = 50)

args = parser.parse_args()


ds_df = pd.read_csv(args.ds_df)
if args.org:
    ds_df = ds_df[ds_df['organism'] == str(args.org)] # filter for organism
else: 
    ds_df = ds_df[(ds_df['organism'] == 'human') | (ds_df['organism'] == 'mouse') | (ds_df['organism'] == 'rat') ] # filter for organism
if args.org_part:
    ds_df = ds_df[ds_df['organism_part'] == args.org_part] # filter for organism type
if args.cond:
    ds_df = ds_df[ds_df['condition'] == args.cond] # filter for condition
if args.fdr_ann:
    ds_df = ds_df[(ds_df['fdr5'] + ds_df['fdr10']) > args.fdr_ann]
    
sample_df = get_stratified_sample(ds_df, args.n)

sample_df.to_csv(args.output + '.csv')
