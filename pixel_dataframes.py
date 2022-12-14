import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from metaspace import SMInstance

import gensim
from gensim import similarities
from gensim.similarities import Similarity
from gensim import models


METAHOST = 'https://metaspace2020.eu'
MOLDBNAME = 'HMDB-v4'
# DSID = "2016-09-22_11h16m17s" # example ID

# Temporary directory for output
OUTPUT_PATH = 'output'


def filter_ions(images, ann_ions):
    img_ions =[] 
    for image in images:
        s = str(image)
        img_ions.append(s[s.find("(")+1:s.find(")")])

    if len(ann_ions) != len(img_ions):
        drop_list = []
        for idx, ion in enumerate(img_ions):
            if ion + ann_ions[0][-1] not in ann_ions:
                drop_list.append(idx)

        img_ions = [img_ions[i] for i in range(len(img_ions)) if i not in drop_list]
        images = [images[i] for i in range(len(images)) if i not in drop_list]

    for i, ion in enumerate(ann_ions): # assert the same image order
        name_len = len(ion[:-1])
        assert ion[:name_len] == img_ions[i][:name_len], f"{ion[:-1]} != {img_ions_filt[i]} at  i = {i}" 
    return images

def create_pixel_df(dsid, dbase = MOLDBNAME, host = METAHOST, fdr = 0.1, only_first_isotope = True,
                    hotspot_clipping = False, norm = True):

    sm = SMInstance(host = host)
    molset = sm.dataset(dbase, dsid)
    
    ann_df = molset.results(fdr=fdr) #annotation data frame
    ann_ions = ann_df.ion.tolist() #list of ions and adduct
    columns = ['y', 'x'] + ann_ions
    
    images = molset.all_annotation_images(fdr= fdr, only_first_isotope=only_first_isotope, hotspot_clipping = hotspot_clipping)
    images = filter_ions(images, ann_ions) # here, we make sure, that the ions of ann_df and images match
    
    image_array = np.moveaxis((np.squeeze(np.array(images))), 0, -1) # move channel dimension to the last axis
    pixel_no = np.shape(image_array[:,:,0].flatten())[0]
    
    indices_array = np.moveaxis(np.indices((image_array.shape[0], image_array.shape[1])), 0, -1) # construct an index array
    all_array = np.dstack((indices_array, image_array)).reshape((pixel_no,-1)) # merge index and image array,
                                                                               # reshaping to (index, (y,x,intensity))
    pixel_df = pd.DataFrame(all_array, columns=columns).astype({'y': np.uint8, 'x': np.uint8})# create dataframe
    if norm:
        pixel_df[ann_ions] = pixel_df[ann_ions] / (pixel_df[ann_ions].sum()) #normalize
    del indices_array, all_array
    
    return pixel_df, ann_ions

#Adapted from Katya's
def create_sim_df(dsid = DSID, dbase = MOLDBNAME, host = METAHOST, fdr = 0.1, only_first_isotope = True,
                    hotspot_clipping = False):
    
    pixel_df, ion_list = create_pixel_df(dsid, dbase, host, fdr, only_first_isotope, hotspot_clipping)
    
    ion_corpus = []
    ions = []
    
    pixel_df = pixel_df.drop(columns=['y', 'x'])
    for ion, intensities in pixel_df.items():
        ions.append(ion)
        #build ion-pixel gensim corpus
        ion_doc = list(zip(intensities.index.tolist(), intensities.tolist())) # .tolist() is probably not needed
        ion_corpus.append(ion_doc)
    
    sim_index = gensim.similarities.docsim.Similarity(OUTPUT_PATH, ion_corpus, num_features = pixel_df.index.max()+1)
    sim_df = pd.DataFrame(np.array(sim_index), columns = ions, index = ions)
    
    pixel_df = None
    
    return sim_df

# Tim's
def create_coloc_df(ds_id, fdr = 0.1, database=('HMDB','v4'), only_first_isotope = True, scale_intensity = False, hotspot_clipping = False):
    sm = metaspace.SMInstance()
    ds = sm.dataset(id=ds_id)
    tmp = ds.all_annotation_images(fdr=fdr,
                                   database=database,
                                   only_first_isotope=only_first_isotope,
                                   scale_intensity=scale_intensity,
                                   hotspot_clipping=hotspot_clipping)
    ion_array = np.array(
        [scipy.signal.medfilt2d(x._images[0], kernel_size=3).flatten() for x in tmp])
    df = pd.DataFrame(pairwise_kernels(ion_array, metric='cosine'),
                      columns = [x.formula + x.adduct for x in tmp],
                      index=[x.formula + x.adduct for x in tmp])
    return df
