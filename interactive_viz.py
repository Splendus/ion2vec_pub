#%% 
# imports
import os
import pickle

import numpy as np, pandas as pd
import scipy
from sklearn.metrics import pairwise_kernels
import matplotlib.pyplot as plt
import seaborn as sns 

import metaspace
from metaspace import SMInstance
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

from viz import get_ds_list, get_ion_imgs, ion_cluster, plot_ion_imgs, label_point, imshow_ions, cluster_viz
from metadata import get_meta_df, post_processing


#%%
# testing post_processing
vec_file = 'RW_output/vectors_TF_RW_validation_w1moreions_size100_ww5_rw.tsv'
name_file ='RW_output/metadata_TF_RW_validation_w1moreions_size100_ww5_rw.tsv' 
test = post_processing(vec_file, name_file, u_embed=True)
meta_df = test.get_info_df()
#%% 
# defaults, constants and helper functions
plt.rcParams['figure.figsize'] = (13,9)
plt.rcParams['figure.dpi'] = 300

ds_ids = ['2017-08-03_15h09m06s', '2017-08-11_07h59m58s', '2017-08-03_15h09m51s']
ds_names = [
    'Servier_Ctrl_mouse_wb_median_plane_chca',
    'Servier_Ctrl_mouse_wb_lateral_plane_DHB',
    'Servier_Ctrl_mouse_wb_lateral_plane_chca']
ds_id2name = dict(zip(ds_ids, ds_names))

color_dict = {'low' : 'red', 'high' : 'green', 'query' : 'black', 'none':'grey', 'offsample':'grey',
(1,0,0): 'red', (0,1,0): 'green', (0,0,1): 'blue',
(1,1,0): 'yellow', (1,0,1): 'purple', (1,1,1): 'black',
(0,1,1): 'cyan' }

# Label OffSample
off_sample = []
sm = SMInstance()
for ds_id in ds_ids:
    ds = sm.dataset(id=ds_id)
    results = ds.results(database=("HMDB", "v4"))
    tmp = results[results.offSample].ion.tolist()
    off_sample += tmp

# Loading coloc ions
with open("triple_ions.pickle", "rb") as trip: # ions annotated in all three datasets
    triple_ions = pickle.load(trip)
with open("top_ions.pickle", "rb") as top: # ions with high coloc w.r.t. triple_ions
    top_ions = pickle.load(top)
with open("bot_ions.pickle", "rb") as bot: # ions with low coloc w.r.t. triple_ions
    bot_ions = pickle.load(bot)
with open("train_ions.pickle", "rb") as query:
    query_ions = pickle.load(query)

# load the train_df (coloc of all ions in the 3 datasets with each other)
train_df = pd.read_csv('train_ions.csv', index_col=0)
# create an average coloc df
by_row_index = train_df.groupby(train_df.index)
mean_coloc_df = by_row_index.mean() # building the mean coloc spanning the 3 datasets
mean_coloc_df = mean_coloc_df[mean_coloc_df.index]

#%%
def get_coloc_dict(df):
    ions = df['ion']
    d = {}
    for ion in ions:
        if ion in off_sample:
            d[ion] = 'offsample'
        elif ion in triple_ions:
            d[ion] = 'query'
        elif ion in top_ions:
            d[ion] = 'high'
        elif ion in bot_ions:
            d[ion] = 'low'
        else:
            d[ion] = 'none'
    return d

def preprocess_meta_df(df, ds_ids = ds_ids):
    true_ids = [list(set(
        df_ids).intersection(set(ds_ids)))
        for df_ids in df['dataset_ids'].tolist()
        ]
    true_names = [list(set(
        df_names).intersection(set(ds_names)))
        for df_names in df['dataset_names'].tolist()
        ]
    df['dataset_ids'] = true_ids
    df['dataset_names'] = true_names

    # Multihot encoding datasets
    df.drop(columns=['encoded_ds'], inplace=True)
    multi_hot = MultiLabelBinarizer()
    encoded = multi_hot.fit_transform(df['dataset_names'])
    # turning lists into tuples
    encoded_tuples = [tuple(l) for l in list(encoded)]
    df['encoded'] = encoded_tuples
    ion2coloc=get_coloc_dict(df)
    df['coloc'] = df['ion'].map(ion2coloc)
    df.index = df['ion']
    return df


def noco_distances(mean_coloc_df, cos_df):
    '''
    params:
    --------
    mean_coloc_df : coloc_df averaged over train datasets
                    necessary for co-occurrence info
    cos_df        : cosine similarity df 

    returns:
    --------
    (no_dist, co_dist): a tuple of dictionaries, with entries for non and co-occurring ions
                        with individual entries (ion_1, ion_2) : cosine_sim(ion_1, ion_2)
    '''
    noco_ions = [(ion1, ion2)    
        for ion1 in mean_coloc_df.index for ion2 in mean_coloc_df.columns 
        if mean_coloc_df.isna().loc[ion1, ion2]]

    co_ions = [(ion1, ion2)    
        for ion1 in mean_coloc_df.index for ion2 in mean_coloc_df.columns 
        if not mean_coloc_df.isna().loc[ion1, ion2]] 

    # get cosine distance for not co-occurring ion pairs
    no_dist = {pair : cos_df.loc[pair[0],pair[1]]
        for pair in noco_ions}
    # get cosine distance for co-occuring ion pairs
    co_dist = {pair : cos_df.loc[pair[0],pair[1]]
        for pair in co_ions if pair[0] != pair[1]} # drop self similarity

    return no_dist, co_dist

def cos_coloc_dict(mean_coloc_df, cos_df):
    '''
    returns a dictionary with ion pair keys and (cos_sim, coloc) values
    '''
    cos_df = cos_df.loc[mean_coloc_df.index, mean_coloc_df.columns]
    cos_coloc_dict = {
        (ion1, ion2): (cos_df.loc[ion1,ion2], mean_coloc_df.loc[ion1, ion2]) 
        for ion1 in cos_df.index for ion2 in mean_coloc_df.index}
    return cos_coloc_dict

def pre_process(model, vec_file, meta_file = None, embed=True):
    if model == 'rw':
        df = get_meta_df(vec_file, meta_file, embed = embed)
        # process i2v embeddings
        iv = pd.read_csv(vec_file, sep = '\t', header= None)   # read ion vectors and their labels
        vec_names = pd.read_csv(meta_file, sep = '\t', header = None)
        idx2ion = vec_names.to_dict()[0]
        iv.index = iv.index.map(idx2ion)

        cos_df = pd.DataFrame(cosine_similarity(iv))
        cos_df.index = cos_df.index.map(idx2ion)
        cos_df.columns = cos_df.columns.map(idx2ion)

    elif model == 'vanilla':
        df = get_meta_df(vec_file, txt=True, embed = embed)
        # processing ionvecs
        with open(vec_file, 'r') as f:
            lines = [i.split() for i in f]
        ion2vec = {line[0]: list(map(float, line[1:])) for line in lines[1:]}

        iv = pd.DataFrame(ion2vec).T
        cos_df = pd.DataFrame(cosine_similarity(iv), columns=iv.index, index=iv.index)

    df = preprocess_meta_df(df)
    df = df.loc[mean_coloc_df.index, :]
    
    cos_df = cos_df.loc[mean_coloc_df.index, mean_coloc_df.columns]

    coord_dict = {
        (ion1, ion2): (cos_df.loc[ion1,ion2], mean_coloc_df.loc[ion1, ion2]) 
        for ion1 in cos_df.index for ion2 in mean_coloc_df.index}

    no_dist, co_dist = nocoloc_distances(mean_coloc_df, cos_df)

    return df, coord_dict, (no_dist, co_dist)

#%% 
# Preprocess gensim vanilla data trained on all ions, with embed size 50
vanilla_all50_dfs = {}
vanilla_all50_coords = {}
vanilla_all50_dists = {}

for i in range(1,6):
    try:
        gen_file = f"slurm_job/Vanilla_output/gensim_validation_w{i}_noshuff30size50moreions.model.txt"
        p = post_processing(gen_file, u_embed=True, dsid_list=ds_ids)
        df = p.get_info_df()
        col_df = p.get_mean_coloc(ds_ids, query_ions)
        cos_df = p.get_embed_sim(query_ions)
        
        ion2coloc=get_coloc_dict(df) # add coloc info
        df['coloc'] = df['ion'].map(ion2coloc) 

        vanilla_all50_dfs[f'w{i}'] = df
        vanilla_all50_coords[f'w{i}'] = cos_coloc_dict(col_df, cos_df)
        vanilla_all50_dists[f'w{i}'] = noco_distances(col_df, cos_df)
    except FileNotFoundError:
         continue


#%% 
# Preprocess gensim vanilla data trained on all ions, with embed size 100
vanilla_all100_dfs = {}
vanilla_all100_coords = {}
vanilla_all100_dists = {}

for i in range(1,6):
    try:
        gen_file = f"slurm_job/gensim_validation_w{i}_noshuff30size100moreions.model.txt"
        (vanilla_all100_dfs[f'w{i}'],
        vanilla_all100_coords[f'w{i}'], 
        vanilla_all100_dists[f'w{i}']) = pre_process('vanilla', gen_file) 
    except FileNotFoundError: continue

#%% 
# Preprocess gensim vanilla no_shuffling_20it only trained on query ions
vanilla_20_dfs = {}
vanilla_20_coords= {}
vanilla_20_dists = {}

for i in range(1,6):
    gen_file = f"slurm_job/gensim_validation_w{i}_noshuff.model.txt"

    (vanilla_20_dfs[f'w{i}'],
    vanilla_20_coords[f'w{i}'], 
    vanilla_20_dists[f'w{i}']) = pre_process('vanilla', gen_file) 

#%%
# Preprocessing RW data
rw_dfs = {}
rw_coords = {}
rw_dists = {}
for i in range(1,6):
    vec_file = f"RW_output/vectors_TF_RW_validation_w{i}_rw.tsv"
    meta_file = f"RW_output/metadata_TF_RW_validation_w{i}_rw.tsv"

    (rw_dfs[f'w{i}'],
     rw_coords[f'w{i}'],
     rw_dists[f'w{i}'] ) = pre_process('rw', vec_file, meta_file) 

#%%
# Preprocessing RW data where we trained on all ions, with ww10
rw_all10_dfs = {}
rw_all10_coords = {}
rw_all10_dists = {}
for i in range(1,6):
    vec_file = f"RW_output/vectors_TF_RW_validation_w{i}moreions_rw.tsv"
    meta_file = f"RW_output/metadata_TF_RW_validation_w{i}moreions_rw.tsv"
    
    (rw_all10_dfs[f'w{i}'],
     rw_all10_coords[f'w{i}'],
     rw_all10_dists[f'w{i}'] ) = pre_process('rw', vec_file, meta_file)
#%%
# Preprocessing RW data where we trained on all ions with ww5
rw_all5_dfs = {}
rw_all5_coords = {}
rw_all5_dists = {}
for i in range(1,6):
    vec_file = f"RW_output/vectors_TF_RW_validation_w{i}moreions_size50_ww5_rw.tsv"
    meta_file = f"RW_output/metadata_TF_RW_validation_w{i}moreions_size50_ww5_rw.tsv"
    
    (rw_all5_dfs[f'w{i}'],
     rw_all5_coords[f'w{i}'],
     rw_all5_dists[f'w{i}'] ) = pre_process('rw', vec_file, meta_file)
    
#%%
# Preprocessing RW data where we trained on all ions with ww5 and embed_size 100
rw_all5_100_dfs = {}
rw_all5_100_coords = {}
rw_all5_100_dists = {}
for i in range(1,6):
    vec_file = f"RW_output/vectors_TF_RW_validation_w{i}moreions_size100_ww5_rw.tsv"
    meta_file = f"RW_output/metadata_TF_RW_validation_w{i}moreions_size100_ww5_rw.tsv"
    
    (rw_all5_100_dfs[f'w{i}'],
     rw_all5_100_coords[f'w{i}'],
     rw_all5_100_dists[f'w{i}'] ) = pre_process('rw', vec_file, meta_file)
#%%
# Plotting vanilla vectors
for i, (df, coords, dists) in enumerate(
    zip(vanilla_all50_dfs.values(), vanilla_all50_coords.values(), vanilla_all50_dists.values()), 1):
    w = (2*i)+1 
    title = (f'Vanilla with 30 epochs and embedding dimension 50, window size ${w} \\times {w} $, trained all ions')
    save_str = f'plots/validation/vanilla/vanilla_30eps_50dims_w{i}_'

    ax = sns.scatterplot(df, x = 'umap_x', y='umap_y', hue='coloc', palette=color_dict, s=40)
    ax.set_title(title + 'colored by coloc')
    label_point(df['umap_x'], df['umap_y'], df['mol_name'], ax, size=4)
    plt.savefig(save_str + 'umap_coloc.png')
    plt.show()
    
    ax = sns.scatterplot(df, x = 'umap_x', y='umap_y', hue='encoded', palette=color_dict, s=40)
    ax.set_title(title + 'colored by co-occurrence')
    label_point(df['umap_x'], df['umap_y'], df['mol_name'], ax, size=4)
    plt.savefig(save_str + 'umap_encoded.png', format='png')
    plt.show()

    xy = np.array(list(coords.values()))
    ax = sns.scatterplot(x = xy[:,0], y= xy[:,1])
    ax.set_title(title + 'Mean coloc over cosine distance')
    ax.set_ylabel('Average Colocalization')
    ax.set_xlabel('Cosine distance in embedded space')
    plt.savefig(save_str + 'cos_over_coloc.png', format='png')
    plt.show()

    no_dist, co_dist = dists
    ax = sns.violinplot([list(no_dist.values()), list(co_dist.values())], 
    linewidth=2, fliersize=2)
    ax.set_ylabel('Cosine distance in embedded space')
    ax.set_title(title + 'Cosine distance w.r.t. co-occurrence')
    ax.set_xticklabels(labels=['no co-occurence', 'co-occurence'])
    plt.savefig(save_str + 'cos_violin.png', format='png')
    plt.show()
#%% 
# Plotting the random walk embeddings trained on all ions for ww5
for i, (df, coords, dists) in enumerate(
    zip(rw_dfs.values(), rw_coords.values(), rw_dists.values()), 1):
    w = (2*i)+1 
    title = f'RW, 30 epochs and 100 embedding dimensions, window size ${w} \\times {w} $, word window 10, query only'
    save_str = f'plots/validation/rw/rw_30eps_100dims_w{i}_ww10_queryonly' 

    ax = sns.scatterplot(df, x = 'umap_x', y='umap_y', hue='coloc', palette=color_dict, s=40)
    ax.set_title(title + 'colored by coloc')
    label_point(df['umap_x'], df['umap_y'], df['mol_name'], ax, size=4)
    plt.savefig(save_str + 'umap_coloc.png')
    plt.show()
    ax = sns.scatterplot(df, x = 'umap_x', y='umap_y', hue='encoded', palette=color_dict, s=40)
    ax.set_title(title + 'colored by co-occurrence')
    label_point(df['umap_x'], df['umap_y'], df['mol_name'], ax, size=4)
    plt.savefig(save_str + 'umap_encoded.png', format='png')
    plt.show()

    xy = np.array(list(coords.values()))
    ax = sns.scatterplot(x = xy[:,0], y= xy[:,1])
    ax.set_title(title + 'Mean coloc over cosine distance')
    ax.set_ylabel('Average Colocalization')
    ax.set_xlabel('Cosine distance in embedded space')
    plt.savefig(save_str + 'cos_over_coloc.png', format='png')
    plt.show()

    no_dist, co_dist = dists
    ax = sns.violinplot([list(no_dist.values()), list(co_dist.values())], 
    linewidth=2, fliersize=2)
    ax.set_ylabel('Cosine distance in embedded space')
    ax.set_title(title + 'Cosine distance w.r.t. co-occurrence')
    ax.set_xticklabels(labels=['no co-occurence', 'co-occurence'])
    plt.savefig(save_str + 'cos_violin.png', format='png')
    plt.show()
# %%
