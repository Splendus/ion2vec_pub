import os
import pickle

import ast

import numpy as np, pandas as pd
import logging
import random

from numpy import percentile, nan as np_nan

import scipy
from sklearn.metrics.pairwise import pairwise_kernels

logger = logging.getLogger(__name__)

def random_coloc_walk(coloc_matrix, n=5):
    tmp = coloc_matrix.copy()
    np.fill_diagonal(tmp, 0)
    #transition_matrix = np.apply_along_axis(scipy.special.softmax, 0, tmp)
    sequence = [np.random.choice(range(tmp.shape[0]))]
    for i in range(n):
        try:
            sequence.append(random.choices(range(tmp.shape[0]), weights=tmp[:, sequence[-1]])[0])
        except ValueError:  # This is only a temporary fix! Normally the value error should not occur
            print("Weights sum up to zero")
            print(tmp)
            continue        
    return sequence[1:]

# implement ion to int
# implement window stride length

class PixelCorpus(object): #Replaced this one
    def __init__(self, fdr_thresh=0.1, pix_per=0.5, int_per=0.5, window=5, quan=99., ind_name=None, ds_dir=None, ds_ids=None, stride = 1):
        self.fdr = fdr_thresh
        self.ds_dir = ds_dir
        self.p = pix_per
        self.i = int_per
        self.w = window
        self.q = quan
        
        self.stride = stride
        self.ind_name = ind_name
        self.ds_ids = ds_ids #new: list of metaspace dataset ids used as training data

    def __iter__(self):

        ion2ids = {} #__iter__s own id dictionary, might be merged together later
        for f in os.listdir(self.ds_dir):
                try:
                    ds_df = pd.read_pickle(os.path.join(self.ds_dir,f))
                except IsADirectoryError:
                    continue
                all_ions = ds_df.drop(columns=['y','x']).columns.tolist()
                logger.info("ds_df pixel size %i ", len(ds_df[['x', 'y']].drop_duplicates().index))

                if self.ind_name != None:
                    ion_names = list(set(self.ind_name).intersection(all_ions)) # intersection between all ions in the ds and specified ions
                    pop_ions = list(set(all_ions).difference(set(ion_names)))
                    ds_df = ds_df.drop(columns=pop_ions) # drop ions not in ind_name

                else: ion_names = all_ions # either the specified ion names or all ions

                if not ion_names:
                    pass # skip empty iterations

                for ion in ion_names:
                    ion2ids[ion] = ion2ids.get(ion, len(list(ion2ids.values())) +1 )

                # fiter out rows based on intensity and quantile param
                int_thresh = percentile(ds_df, self.q) * self.i  # this now uses a general intensity threshold,
                                                               # could also use an ion specific one
                ds_df[ion_names] = ds_df[ion_names][ds_df[ion_names] > int_thresh]
                filt_df = ds_df.rename(columns=ion2ids)

                filt_df = filt_df.astype(pd.SparseDtype("int", np_nan)) # int should be replaced, when we care about the exact number of intensity

                # sample pixels
                if self.p != 1.0:
                    sampled_coord_df = filt_df.dropna(how='all').drop_duplicates().sample(frac=self.p)
                else:
                    sampled_coord_df = filt_df.dropna(how='all').drop_duplicates()
                #logging.info("%i pixels selected for %s", len(sampled_coord_df.index), f)

                for _, c_row in sampled_coord_df.iloc[::self.stride].iterrows(): # iloc[::3] takes every third row, inducing stride length of 3
                    x = c_row['x']
                    y = c_row['y']

                    # find rows corresponding to pixels around the sampled pixel coordinates
                    window_rows = filt_df[(filt_df['x'].between(x - self.w, x + self.w, inclusive = 'both'))
                                         & (filt_df['y'].between(y - self.w, y + self.w, inclusive = 'both'))]
                    # depending on how many items an ion/or formula (depending on self.ind_name parameter)
                    # occurs in the window, yield it
                    exp_inds = []
                    ind_counts = dict(window_rows.drop(columns=['y', 'x']).count())
                    for ind in ind_counts:
                        for i in range(0, ind_counts[ind]): exp_inds.append(ind)
                    random.shuffle(exp_inds) # shuffle ions in window
                    yield exp_inds
                    
    def get_ions2ids(self):
        ion2ids = {} # ion to int dictionary
        for f in os.listdir(self.ds_dir):
            try:
                ds_df = pd.read_pickle(os.path.join(self.ds_dir,f))
            except IsADirectoryError:
                continue
            all_ions = ds_df.drop(columns=['y','x']).columns.tolist()
            logger.info("ds_df pixel size %i ", len(ds_df[['x', 'y']].drop_duplicates().index))

            if self.ind_name != None:                                         
                ion_names = list(set(self.ind_name).intersection(all_ions)) # intersection between all ions in the ds and specified ions
                pop_ions = list(set(all_ions).difference(set(ion_names)))
                ds_df = ds_df.drop(columns=pop_ions) # drop ions not in ind_name

            else: ion_names = all_ions # either the specified ion names or all ions

            if not ion_names:
                pass # skip empty iterations

            for ion in ion_names:
                ion2ids[ion] = ion2ids.get(ion, len(list(ion2ids.values())) + 1 ) # reserve 0 for <pad>
        return ion2ids
    

# random walk implementation
# stride length implementation
# ion to int mapping
# drop all zero rows in the df
# drop all zero columns inside window

class PixelCorpusRW(object): #Replaced this one
    def __init__(self, fdr_thresh=0.1, pix_per=0.5, int_per=0.5, window=5, quan=99., ind_name=None, ds_dir=None, ds_ids=None, stride=1, no_samples = 5, walk_length = None):
        self.fdr = fdr_thresh
        self.ds_dir = ds_dir
        self.p = pix_per
        self.i = int_per
        self.w = window
        self.q = quan
        
        self.ind_name = ind_name # list of training ions
        self.ds_ids = ds_ids #new: list of metaspace dataset ids used as training data
        self.no_samples = no_samples # number of sampled random walks
        self.stride = stride # stride size of the image window
        self.walk_length = walk_length # length of random walk. Right now, it is not used but
                                       # fixed to the number of unique ions in the window
        
        
    def __iter__(self):
        ion2ids = {} #__iter__s own id dictionary, might be merged together later
        for f in os.listdir(self.ds_dir):
                try:
                    ds_df = pd.read_pickle(os.path.join(self.ds_dir,f))
                except IsADirectoryError:
                    continue
                all_ions = ds_df.drop(columns=['y','x']).columns.tolist()
                logger.info("ds_df pixel size %i ", len(ds_df[['x', 'y']].drop_duplicates().index))

                if self.ind_name != None:                                         
                    ion_names = list(set(self.ind_name).intersection(all_ions)) # intersection between all ions in the ds and specified ions
                    pop_ions = list(set(all_ions).difference(set(ion_names)))
                    ds_df = ds_df.drop(columns=pop_ions) # drop ions not in ind_name
                    
                else: ion_names = all_ions # either the specified ion names or all ions

                if not ion_names:
                    pass # skip empty iterations

                for ion in ion_names:
                    ion2ids[ion] = ion2ids.get(ion, len(list(ion2ids.values())) + 1 ) # reserve 0 for <pad>
                # filter out rows based on intensity and quantile param
                #int_thresh = percentile(ds_df, quan) * int_per  # this now uses a general intensity threshold,
                                                               # could also use an ion specific one
                filt_df = ds_df.rename(columns=ion2ids)
                filt_df = filt_df.loc[~(filt_df.drop(columns=['y', 'x'])==0).all(axis=1)] # drop all zero rows

                #filt_df[ion_names] = ds_df[ion_names][ds_df[ion_names] > int_thresh]
                #filt_df = filt_df.astype(pd.SparseDtype("int", np_nan)) # int should be replaced, when we care about the exact number of intensity

                # sample pixels
                if self.p != 1.0:
                    sampled_coord_df = filt_df.dropna(how='all').drop_duplicates().sample(frac=self.p)
                else: 
                    sampled_coord_df = filt_df.dropna(how='all').drop_duplicates()
                #logging.info("%i pixels selected for %s", len(sampled_coord_df.index), f)

                for _, c_row in sampled_coord_df.iloc[::self.stride].iterrows(): #iloc[::3] takes every third row, inducing stride length of 3
                    x = c_row['x']  # center x coordinate
                    y = c_row['y']  # center y coordinate

                    # find rows corresponding to pixels around the sampled pixel coordinates
                    window_rows = filt_df[(filt_df['x'].between(x - self.w, x + self.w, inclusive = 'both')) 
                                         & (filt_df['y'].between(y - self.w, y + self.w, inclusive = 'both'))]
                    ion_rows = window_rows.drop(columns=['y','x'])
                    ion_rows = ion_rows.loc[:, (ion_rows != 0).any(axis=0)] # drop columns with all zero entries
                    coloc_matrix = pairwise_kernels(ion_rows.T, metric='cosine') # Don't forget the Transpose
                    n = self.walk_length or int(len(ion_rows.columns.tolist()))
                    for i in range(self.no_samples):
                        ions_idx = random_coloc_walk(coloc_matrix, n=n )
                        yield ion_rows.columns[ions_idx].tolist()
                        
    def get_ions2ids(self):
        ion2ids = {} # ion to int dictionary
        for f in os.listdir(self.ds_dir):
            try:
                ds_df = pd.read_pickle(os.path.join(self.ds_dir,f))
            except IsADirectoryError:
                continue
            all_ions = ds_df.drop(columns=['y','x']).columns.tolist()
            logger.info("ds_df pixel size %i ", len(ds_df[['x', 'y']].drop_duplicates().index))

            if self.ind_name != None:                                         
                ion_names = list(set(self.ind_name).intersection(all_ions)) # intersection between all ions in the ds and specified ions
                pop_ions = list(set(all_ions).difference(set(ion_names)))
                ds_df = ds_df.drop(columns=pop_ions) # drop ions not in ind_name

            else: ion_names = all_ions # either the specified ion names or all ions

            if not ion_names:
                pass # skip empty iterations

            for ion in ion_names:
                ion2ids[ion] = ion2ids.get(ion, len(list(ion2ids.values())) + 1 ) # reserve 0 for <pad>
        return ion2ids
