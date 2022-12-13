import os
import ast

import numpy as np, pandas as pd
import scipy
import matplotlib.pyplot as plt, seaborn as sns
import umap 
import pickle

from metaspace import SMInstance

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_kernels
from sklearn.metrics.pairwise import cosine_similarity

def read_gensim_txt(f_path, dictionary = False):
    with open(f_path, 'r') as f:
        lines = [i.split() for i in f]
    vocab_size = int(lines[0][0])
    dimension = int(lines[0][1])
    ion_dict = {line[0]: list(map(float, line[1:])) for line in lines[1:]} # mapping ions to their vectors
    if dictionary:
        return ion_dict 
    else: 
        return pd.DataFrame(ion_dict)

# Constants
mol_df = pd.read_csv('Ions2Molecules.csv') # mapping ion to molecule names
hmdb_df  = pd.read_csv('datasets/HMDB4_database.csv') # containing metabolite info
id_df   = pd.read_csv('datasets/ions2ids.csv') # mapping ions to HMDB ids
ion2hmdb  = {ion: ast.literal_eval(id_df[id_df['ion']==ion]['hmdbID'].item())
                        for ion in id_df['ion']}
ds_df = pd.read_csv('datasets/theos_recom/ions_datasets.csv') # mapping ions to datasets

class post_processing:
    def __init__(self, vec_file, name_file = None, u_embed = False, 
                dsid_list = None, fdr = 0.1,
                metahost = 'https://metaspace2020.eu', molbase = 'HMDB-v4'):

        self.vec_file = vec_file    # embedded vectors, .tsv from TF, .txt from gensim
        self.name_file = name_file  # ion name file, only from TF
        self.u_embed = u_embed      # if a new UMAP embedding has to be created
        self.ds_ids = dsid_list     # list of datasets ids, the embeddings were trained on
        self.host = metahost        # host for the metaspace API
        self.fdr  = fdr             # False Detection Rate
        self.molbase = molbase      # annotation database
        self.iv_df, self.name_df = self.get_vecs() # ion_vector dataframe, ion_name dataframe
        self.ion_names = self.name_df[0].tolist() 

        self.ion2mol_dict = {}
        self.ion2ds_dict = {}
        for ion in self.ion_names: 
            self.ion2mol_dict[ion] = mol_df[mol_df['ions'] == ion]['moleculeNames'].item()
            self.ion2ds_dict[ion] = ast.literal_eval(ds_df[ds_df['ion'] == ion]['datasets'].item()) 

    def get_vecs(self):
        if self.name_file: # vecs from rw
            iv_df = pd.read_csv(self.vec_file, sep = '\t', header= None)       # read ion vectors and their labels
            name_df= pd.read_csv(self.name_file, sep = '\t', header = None )   # sep '\t' for tsv files
        else: # vecs from gensim vanilla
            with open(self.vec_file, 'r') as f:
                lines = [i.split() for i in f]
            # vocab_size = int(lines[0][0])
            # dimension = int(lines[0][1])
            ion2vec = {line[0]: list(map(float, line[1:])) for line in lines[1:]}

            iv_df = pd.DataFrame(ion2vec).T
            name_df= pd.DataFrame(iv_df.index.tolist()) 
        return iv_df, name_df
    
    def get_class(self, class_type): # super_class, class, sub_class etc. 
        ion2class_dict = {} #mapping ion to superclass
        for ion in self.ion_names: 
            for ID in ion2hmdb[ion]:
                try:
                    ion2class_dict[ion] = hmdb_df[hmdb_df['accession'] == ID][class_type].item()
                except ValueError: # some ions miss HMDB-ID
                    continue
        return ion2class_dict

    def get_true_ds(self, df):
        true_ids = [list(set(
            df_ids).intersection(set(self.ds_ids)))
            for df_ids in df['dataset_ids'].tolist()
            ]
        return true_ids
 
    def get_info_df(self):
        '''
        Parameters:
        -----------

        returns:
        ---------
        meta_df : Dataframe containing all meta information about the ions. 
        '''
        meta_df = self.name_df.rename(columns={0:'ion'})
        meta_df['mol_name'] = [mol_df[mol_df['ions'] == ion]['moleculeNames'].item() for ion in meta_df['ion']] # adding molecular names
        # adding ion classes
        for class_type in ['super_class', 'class', 'sub_class']:
            ion2class_dict = self.get_class(class_type)
            meta_df[class_type] = meta_df['ion'].map(ion2class_dict)

        # adding dataset_ids as list
        meta_df['dataset_ids'] = meta_df['ion'].map(self.ion2ds_dict) 
        # ion2ds_dict maps ion to all ds_ids in theos_recom folder
        # if not trained on all datasets, a ds_list has to be given 
        if self.ds_ids:
            true_ids = self.get_true_ds(meta_df)
            meta_df['dataset_ids'] = true_ids

        # Multihot encoding datasets
        multi_hot = MultiLabelBinarizer()
        encoded = multi_hot.fit_transform(meta_df['dataset_ids'])
        encoded = [tuple(l) for l in list(encoded)]
        meta_df['encoded'] = encoded # multi hot encoded datasets
        
        ds_ids2ds_names = {}
        ion2ionformula = {}
        ion2formula = {}
        ion2adduct = {}
        sm = SMInstance(host = self.host)
        for ds_id in list(multi_hot.classes_): # multi_hot.classes are the unique ds_ids
            molset = sm.dataset(self.molbase, ds_id)
            ds_ids2ds_names[ds_id] = molset.name #mapping ds_ids to ds_names
            results = molset.results(fdr=self.fdr)
            # temporary dicts for ionformula, formula and adduct
            temp_if = results.set_index('ion').to_dict()['ionFormula']
            temp_f = results.reset_index(level=['formula']).set_index('ion').to_dict()['formula']
            temp_a = results.reset_index(level=['adduct']).set_index('ion').to_dict()['adduct']

            #merging dicts, overwriting ion2formula        
            ion2ionformula = ion2ionformula | temp_if 
            ion2formula = ion2formula | temp_f
            ion2adduct = ion2adduct | temp_a
            
        name_list = [] # list of lists of dataset names
        for ds_ids in meta_df['dataset_ids'].tolist():
            name_list.append(tuple([ds_ids2ds_names[ID] for ID in ds_ids]))
        
        meta_df['dataset_names'] = name_list
        meta_df['ionFormula'] = meta_df['ion'].map(ion2ionformula)
        meta_df['adduct'] = meta_df['ion'].map(ion2adduct)
        meta_df['formula'] = meta_df['ion'].map(ion2formula)

        # adding a categorical column with ds_id in case ions are annotated in only one ds
        ion2single_dict = {ion :
                        meta_df[meta_df['ion'] == ion]['dataset_ids'].item()[0] 
                        if (sum(list(meta_df[meta_df['ion']==ion]['encoded'].item())) ==1)
                        else 'Multiple Datasets' for ion in meta_df['ion']
            }
        meta_df['single_dataset_id'] = meta_df['ion'].map(ion2single_dict)
        # mapping the same to ds_name
        ion2single_name_dict = {ion :
                                meta_df[meta_df['ion'] == ion]['dataset_names'].item()[0] 
                                if (sum(list(meta_df[meta_df['ion']==ion]['encoded'].item())) ==1) 
                                else 'Multiple Datasets' for ion in meta_df['ion']  }
        meta_df['single_dataset_name'] = meta_df['ion'].map(ion2single_name_dict)

        # in case u_map embeddings have to be generated
        if self.u_embed:
            reducer = umap.UMAP()

            # Do not scale for the time being, vectors should be scales already
            # scaled_vectors = StandardScalar().fit_transform(iv) 
            embedding = reducer.fit_transform(self.iv_df)

            name2x = {name: embedding[i][0] for i, name in enumerate(self.ion_names)}
            name2y = {name: embedding[i][1] for i, name in enumerate(self.ion_names)}

            meta_df['umap_x'] = meta_df['ion'].map(name2x)
            meta_df['umap_y'] = meta_df['ion'].map(name2y)

        meta_df.insert(1, 'formula', meta_df.pop('formula')) # reordering
        meta_df.insert(2, 'adduct', meta_df.pop('adduct')) # reordering
        meta_df.insert(3, 'ionFormula', meta_df.pop('ionFormula')) # reordering

        return meta_df

    def get_embed_sim(self, query_ions = None):
        '''
        returns cosine similarity matrix for embedding
        '''
        cos_df = pd.DataFrame(cosine_similarity(self.iv_df))

        idx2ion = self.name_df.to_dict()[0]
        cos_df.index = cos_df.index.map(idx2ion)
        cos_df.columns = cos_df.columns.map(idx2ion)
        if query_ions: # if only for query_ions
            cos_df = cos_df.loc[query_ions, query_ions]
        return cos_df 
    
    def get_coloc_df(self, ds_id, only_first_isotope = True, scale_intensity = False,
                    hotspot_clipping = False, offSample = False):
        '''
        returns a colocalization dataframe for the dataset given by ds_id.
        Note, that this is a work around for the validation datasets, where
        only positive adducts were considered.  
        '''
        sm = SMInstance()
        ds = sm.dataset(id=ds_id)
        tmp = ds.all_annotation_images(fdr=self.fdr,
                                    database=self.molbase,
                                    only_first_isotope=only_first_isotope,
                                    scale_intensity=scale_intensity,
                                    hotspot_clipping=hotspot_clipping,
                                    offSample = offSample)
        ion_array = np.array(
            [scipy.signal.medfilt2d(x._images[0], kernel_size=3).flatten() for x in tmp])
        df = pd.DataFrame(pairwise_kernels(ion_array, metric='cosine'),
                        columns = [x.formula + x.adduct + '+' for x in tmp], # + '+' is only a temporary fix
                        index=[x.formula + x.adduct + '+' for x in tmp])     # and of course only works for the positive mode 
        return df

    def get_mean_coloc(self, dsid_list, query_ions = None):
        '''
        returns a dataframe with colocalizations averaged over the datasets
        given by the IDs in dsid_list. If a list of query_ions is given
        the coloc dataframe is calculated for only those ions. 
        '''
        if query_ions:
            df_list = []
            for dsid in dsid_list:
                coloc_df = self.get_coloc_df(dsid)
                coloc_df = coloc_df.loc[coloc_df.index.isin(query_ions), 
                                        coloc_df.columns.isin(query_ions)]
                df_list.append(coloc_df)
        else:
            df_list = [self.get_coloc_df(dsid) for dsid in dsid_list]
 
        all_colocs_df = pd.concat(df_list)
        by_row_index = all_colocs_df.groupby(all_colocs_df.index)
        mean_coloc_df = by_row_index.mean() # average colocs
        # make sure, that column and row index match
        mean_coloc_df = mean_coloc_df[mean_coloc_df.index] 
        return mean_coloc_df



def get_meta_df(vec_file, meta_file=None, txt = False, embed=False,
                metahost = 'https://metaspace2020.eu', molbase = 'HMDB-v4'):
    
    map_df = pd.read_csv('Ions2Molecules.csv') # mapping ions to molecule names
    id_df = pd.read_csv('datasets/ions2ids.csv') # mapping ions to HMDB ids    
        
    hmdb = pd.read_csv("datasets/HMDB4_database.csv") # mapping ion_id to classes
    ion2hmdb = {ion: ast.literal_eval(id_df[id_df['ion']==ion]['hmdbID'].item())
                     for ion in id_df['ion']}
    
    if txt: # read in gensim wordvector.txt file
        with open(vec_file, 'r') as f:
            lines = [i.split() for i in f]
        #vocab_size = int(lines[0][0])
        #dimension = int(lines[0][1])
        ion2vec = {line[0]: list(map(float, line[1:])) for line in lines[1:]}

        iv = pd.DataFrame(ion2vec).T
        vec_names = pd.DataFrame(iv.index.tolist())
    
    else:
        iv = pd.read_csv(vec_file, sep = '\t', header= None)           # read ion vectors and their labels
        vec_names = pd.read_csv(meta_file, sep = '\t', header = None )   # sep '\t' for tsv files
        
    ion_names = vec_names[0].tolist()
    
    ds_df = pd.read_csv('datasets/theos_recom/ions_datasets.csv') # mapping ions to datasets
    ion2ds_dict = {ion: ast.literal_eval(ds_df[ds_df['ion'] == ion]['datasets'].item()) for ion in ion_names}
    
    i2m_dict = {} # mapping ions to molecule names
    for ion in ion_names: 
        i2m_dict[ion] = map_df[map_df['ions'] == ion]['moleculeNames'].item()
        
    ion2superclass_dict = {} #mapping ion to superclass
    for ion in ion_names: 
        for ID in ion2hmdb[ion]:
            try:
                ion2superclass_dict[ion] = hmdb[hmdb['accession'] == ID]['super_class'].item()
            except ValueError: # some ions miss HMDB-ID
                continue

    ion2class_dict = {} # mapping ion to class
    for ion in ion_names: 
        for ID in ion2hmdb[ion]:
            try:
                ion2class_dict[ion] = hmdb[hmdb['accession'] == ID]['class'].item()
            except ValueError: # some ions miss HMDB-ID
                continue

    ion2subclass_dict = {} # mapping ion to subclass
    for ion in ion_names:
        for ID in ion2hmdb[ion]:
            try:
                ion2subclass_dict[ion] = hmdb[hmdb['accession'] == ID]['sub_class'].item()
            except ValueError: # some ions miss HMDB-ID
                continue
                
    meta_df = vec_names.rename(columns={0:'ion'})
    meta_df['mol_name'] = [map_df[map_df['ions'] == ion]['moleculeNames'].item() for ion in meta_df['ion']] # adding molecular names
    meta_df['super_class'] = meta_df['ion'].map(ion2superclass_dict) # adding superclass
    meta_df['class'] = meta_df['ion'].map(ion2class_dict) # adding class
    meta_df['sub_class'] = meta_df['ion'].map(ion2subclass_dict) # adding subclass
    meta_df['dataset_ids'] = meta_df['ion'].map(ion2ds_dict) # adding dataset_ids
    
    multi_hot = MultiLabelBinarizer()
    encoded = multi_hot.fit_transform(meta_df['dataset_ids'])
    encoded = [tuple(l) for l in encoded]
    meta_df['encoded_ds'] = encoded # multi hot encoded datasets
    
    ds_ids2ds_names = {}
    ion2ionformula = {}
    ion2formula = {}
    ion2adduct = {}
    sm = SMInstance(host = metahost)
    for ds_id in list(multi_hot.classes_):
        molset = sm.dataset(molbase, ds_id)
        ds_ids2ds_names[ds_id] = molset.name
        results = molset.results(fdr=0.1)
        
        temp_if = results.set_index('ion').to_dict()['ionFormula']
        temp_f = results.reset_index(level=['formula']).set_index('ion').to_dict()['formula']
        temp_a = results.reset_index(level=['adduct']).set_index('ion').to_dict()['adduct']
                
        ion2ionformula = ion2ionformula | temp_if #merging dicts, overwriting ion2formula
        ion2formula = ion2formula | temp_f
        ion2adduct = ion2adduct | temp_a
        
    name_list = [] # list of lists of dataset names
    for lis in meta_df['dataset_ids'].tolist():
        name_list.append(tuple([ds_ids2ds_names[ID] for ID in lis]))
        
    meta_df['dataset_names'] = name_list
    meta_df['ionFormula'] = meta_df['ion'].map(ion2ionformula)
    meta_df['adduct'] = meta_df['ion'].map(ion2adduct)
    meta_df['formula'] = meta_df['ion'].map(ion2formula)

    
    ion2single_dict = {ion :
                       meta_df[meta_df['ion'] == ion]['dataset_ids'].item()[0] 
                       if (sum(list(meta_df[meta_df['ion']==ion]['encoded_ds'].item())) ==1)
                       else 'Multiple Datasets' for ion in meta_df['ion']
    }
    meta_df['single_dataset_id'] = meta_df['ion'].map(ion2single_dict)
    
    ion2single_name_dict = {ion :
                            meta_df[meta_df['ion'] == ion]['dataset_names'].item()[0] 
                            if (sum(list(meta_df[meta_df['ion']==ion]['encoded_ds'].item())) ==1) 
                            else 'Multiple Datasets' for ion in meta_df['ion']  }
    meta_df['single_dataset_name'] = meta_df['ion'].map(ion2single_name_dict)

    if embed:
        reducer = umap.UMAP()

        # Do not scale for the time being, vectors should be scales already
        # scaled_vectors = StandardScalar().fit_transform(iv) 
        embedding = reducer.fit_transform(iv)

        name2x = {name: embedding[i][0] for i, name in enumerate(ion_names)}
        name2y = {name: embedding[i][1] for i, name in enumerate(ion_names)}

        meta_df['umap_x'] = meta_df['ion'].map(name2x)
        meta_df['umap_y'] = meta_df['ion'].map(name2y)
    
    meta_df.insert(1, 'formula', meta_df.pop('formula')) # reordering
    meta_df.insert(2, 'adduct', meta_df.pop('adduct')) # reordering
    meta_df.insert(3, 'ionFormula', meta_df.pop('ionFormula')) # reordering

    
    return meta_df