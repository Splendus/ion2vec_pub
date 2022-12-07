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
DSID = "2016-09-22_11h16m17s"

# Temporary directory for output
OUTPUT_PATH = 'output'


def filter_ions(images, ion_list):
    ions_large =[] 
    for image in images:
        s = str(image)
        ions_large.append(s[s.find("(")+1:s.find(")")])

    idx_list = []
    for idx, ion in enumerate(ions_large):
        if ion + ion_list[0][-1] not in ion_list:
            idx_list.append(idx)

    ions_large_filt = [ions_large[i] for i in range(len(ions_large)) if i not in idx_list]
    images_filt = [images[i] for i in range(len(images)) if i not in idx_list]
    
    for i, ion in enumerate(ion_list):
        assert ion[:-1] == ions_large_filt[i] # assert the same image order
    return images_filt

def create_pixel_df(dsid=DSID, dbase = MOLDBNAME, host = METAHOST, fdr = 0.1, only_first_isotope = True,
                    hotspot_clipping = False, norm = True):

    sm = SMInstance(host = host)
    molset = sm.dataset(dbase, dsid)
    
    ann_df = molset.results(fdr=fdr) #annotation data frame
    ion_list = ann_df.ion.tolist() #list of ions and adduct
    columns = ['y', 'x'] + ion_list
    
    images = molset.all_annotation_images(fdr= fdr, only_first_isotope=only_first_isotope, hotspot_clipping = hotspot_clipping)
    images = filter_ions(images, ion_list) # here, we make sure, that the ions of ann_df and images match
    
    image_array = np.moveaxis((np.squeeze(np.array(images))), 0, -1) # move channel dimension to the last axis
    pixel_no = np.shape(image_array[:,:,0].flatten())[0]
    
    indices_array = np.moveaxis(np.indices((image_array.shape[0], image_array.shape[1])), 0, -1) # construct an index array
    all_array = np.dstack((indices_array, image_array)).reshape((pixel_no,-1)) # merge index and image array,
                                                                               # reshaping to (index, (y,x,intensity))
    pixel_df = pd.DataFrame(all_array, columns=columns).astype({'y': np.uint8, 'x': np.uint8})# create dataframe
    if norm:
        pixel_df[ion_list] = pixel_df[ion_list] / (pixel_df[ion_list].sum()) #normalize
    del indices_array, all_array
    
    return pixel_df, ion_list


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
