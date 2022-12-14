import os
import ast

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.patches as mpatches

import pickle

import umap
import metaspace
from metaspace import SMInstance

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

metahost = 'https://metaspace2020.eu'
molbase = 'HMDB-v4'

map_df = pd.read_csv('datasets/Ions2Molecules.csv')
i2m_dict = map_df.set_index('ions').to_dict()['moleculeNames'] # mapping ions to molecule names


class cluster_viz:
    def __init__(self, df, xi,xf, yi,yf):
        
        self.all_df = df
        self.xs= (xi,xf)
        self.ys= (yi,yf)
        self.c_df = df[(df['umap_x'] >  self.xs[0]) & (df['umap_x'] <  self.xs[1])
                       & ( df['umap_y'] > self.ys[0]) & ( df['umap_y'] < self.ys[1]) ]
        self.c_df.dropna()
        
        self.all_single_ds_df = self.all_df[self.all_df['single_dataset_name'] != 'Multiple Datasets']
        self.all_multiple_ds_df = self.all_df[self.all_df['single_dataset_name'] == 'Multiple Datasets']

        self.single_ds_df = self.c_df[self.c_df['single_dataset_name'] != 'Multiple Datasets']
        self.multiple_ds_df = self.c_df[self.c_df['single_dataset_name'] == 'Multiple Datasets'] 
        
        self.ds_ids, self.ds_names = self.get_ds_list()
        self.ion_list = self.c_df['ion'].tolist()
        
        
        pal = sns.color_palette()
        self.single_color = {'whole body xenograft (1) [RMS norm]': pal[0], 'wb xenograft trp pathway dosed- rms_corrected': pal[1], 
                   'whole body xenograft (2) [RMS norm]': pal[2], 'Servier_Ctrl_mouse_wb_lateral_plane_9aa': pal[3], 
                   'Servier_Ctrl_mouse_wb_median_plane_9aa': pal[4],  'Servier_Ctrl_mouse_wb_median_plane_chca' : pal[5],
                   'Servier_Ctrl_mouse_wb_lateral_plane_chca': pal[6], 'Servier_Ctrl_mouse_wb_lateral_plane_DHB': pal[9]}

        self.multi_color = {'Multiple Datasets':'gray'}
        
    def overview_plot(self, hue='single_dataset_name', box=False, box_color='grey',
                      move_legend=False, title='test', out=None, single_trans = 1, multi_trans = 0.3 ):
        
        plt.title(f'{title}')

        if hue == 'single_dataset_name':
            ax = sns.scatterplot(data = self.all_single_ds_df, x = 'umap_x', y= 'umap_y',
                                 hue = hue, palette = self.single_color, alpha = single_trans)
            sns.scatterplot(data = self.all_multiple_ds_df, x = 'umap_x', y='umap_y',
                            hue = hue, palette = self.multi_color, alpha = multi_trans)
        else:     
            ax = sns.scatterplot(data = self.all_df, x = 'umap_x', y= 'umap_y', hue = hue)

        #label_point(all_df.umap_x, all_df.umap_y, all_df.mol_name, plt.gca())
        
        if box:
            left, bottom, width, height = (self.xs[0], self.ys[0],
                                           self.xs[1]-self.xs[0], self.ys[1]-self.ys[0])
            rect=mpatches.Rectangle((left,bottom),width,height, fill=False,
                                    color=box_color,linewidth=2)
            plt.gca().add_patch(rect)
            
        if move_legend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        if out:
            plt.savefig(f'plots/{out}_overview_{hue}.png', format='png')
        plt.show()
    
    def zoom_plot(self, hue='single_dataset_name', title='ion2vec', output=None,
                  marker_size=60, t_label='mol_name', t_size = 6, single_trans=1, mul_trans=0.3,
                 legend_loc = 'best'): 
        
        plt.title(title)    
        if hue == 'single_dataset_name':
            ax = sns.scatterplot(data = self.single_ds_df, x = 'umap_x', y= 'umap_y', hue = hue,
                                 palette = self.single_color, s=marker_size, alpha = single_trans)
            sns.scatterplot(data = self.multiple_ds_df, x = 'umap_x', y='umap_y', hue = hue,
                            palette = self.multi_color, s = marker_size, alpha = mul_trans)
        else:
            ax = sns.scatterplot(data = self.c_df, x = 'umap_x', y= 'umap_y', hue = hue,
                                 s=marker_size, alpha = single_trans)
            
        plt.legend(loc=legend_loc)
        label_point(self.c_df.umap_x, self.c_df.umap_y, self.c_df[t_label], plt.gca(), size = t_size)
        if output:
            plt.savefig(f'plots/{output}.png', format = 'png')
        plt.show()
    
        

    def get_ion_imgs(self, metahost= 'https://metaspace2020.eu', molbase = 'HMDB-v4'):
        '''
        out:
        ----
        imgs : array of images, where imgs[i,j] is the image for
               dataset[i] and ion[j]
        '''
        imgs = []
        sm = SMInstance(host=metahost)
        for j, dataset_id in enumerate(self.ds_ids):
            molset = sm.dataset(molbase, dataset_id)
            img_list = []
            for i, ion in enumerate(self.ion_list):
                query_row = self.c_df[self.c_df['ion'] == ion]
                sf = query_row['formula'].item()
                adduct = query_row['adduct'].item() 

                query_ids = query_row['dataset_ids'].tolist()
                try:
                    query_ids = [item for sublist in query_ids for item in ast.literal_eval(sublist)] # flatten list
                except ValueError:
                    query_ids = [item for sublist in query_ids for item in sublist] # flatten list

                querys_ids = list(set(query_ids)) 
                if dataset_id in query_ids: # check if ion is annotated in ds
                    img = molset.isotope_images(sf, adduct, only_first_isotope=True)
                else:
                    img = np.zeros((2,2))
                img_list.append(img)
            imgs.append(img_list)

        return imgs
    
    def get_ds_list(self):
        '''
        get dataset list of unique ids or names
        '''
        # get dataset names 
        ds_names = self.c_df['ds_names'].tolist()
        try:
            ds_names = [item for sublist in ds_names for item in ast.literal_eval(sublist)] # flatten list
        except ValueError:
            ds_names = [item for sublist in ds_names for item in sublist] # flatten list
            ds_names = list(set(ds_names)) # unique values
        #get dataset ids
        ds_ids = self.c_df['dataset_ids'].tolist()
        try:
            ds_ids = [item for sublist in ds_ids for item in ast.literal_eval(sublist)] # flatten list
        except ValueError:
            ds_ids = [item for sublist in ds_ids for item in sublist] # flatten list
            ds_ids = list(set(ds_ids)) # unique values
        return ds_ids, ds_names
    
    def plot_ion_imgs(self, mol_dict = i2m_dict, figsize=None, out=None):
        '''
        returns:
        --------------------------------------
        subplots with shape (#ions, #datasets)
        '''    
        n_ions   = len(self.ion_list)
        n_ds = len(self.ds_names)
        
        imgs = self.get_ion_imgs()
        
        if mol_dict:
            names = [mol_dict[ion] for ion in self.ion_list]   
        else: names = self.ion_list

        figsize = figsize or (15, 2*n_ions)
        fig, axs = plt.subplots(n_ions, n_ds, figsize=figsize) # rows: ions, columns: dataset
        for i, name in enumerate(names):
            if len(name) > 45: name = name[:45]
            if n_ds >1: # if ions come from more than one ds
                for j, ds_name in enumerate(self.ds_names):
                    axs[0,j].set_title(ds_name, size='x-small')
                    if isinstance(imgs[j][i],metaspace.sm_annotation_utils.IsotopeImages): # if ion is not detected in ds, hide it
                        axs[i,j].imshow(np.squeeze(imgs[j][i]))
                        axs[i,j].set_title(ds_name, size='x-small')
                        axs[i,j].set_xlabel(f'{name}', rotation=0, size='x-small')
                    else:
                        axs[i,j].set_visible(False)
            else:
                if isinstance(imgs[0][i],metaspace.sm_annotation_utils.IsotopeImages): # if ion is not detected in ds, hide it
                    axs[i].imshow(np.squeeze(imgs[0][i]))
                    axs[i].set_title(ds_names[0], size='x-small')
                    axs[i].set_xlabel(f'{name}', rotation=0, size='x-small')
                else:
                    axs[i].set_visible(False)


        plt.setp(axs, xticks=[], yticks=[]) # x/y-ticks offx
        plt.tight_layout()
        if out:
            plt.savefig(f'{out}.png', format='png')
        plt.show()

        
        
def ion_cluster(df, xi,xf, yi,yf, hue, t_label, output=None, title='ion2vec',
                text_size=6, marker_size=50, mul_trans=0.3):
    
    pal = sns.color_palette()
    color_dict_single = {'whole body xenograft (1) [RMS norm]': pal[0], 'wb xenograft trp pathway dosed- rms_corrected': pal[1], 
                   'whole body xenograft (2) [RMS norm]': pal[2], 'Servier_Ctrl_mouse_wb_lateral_plane_9aa': pal[3], 
                   'Servier_Ctrl_mouse_wb_median_plane_9aa': pal[4],  'Servier_Ctrl_mouse_wb_median_plane_chca' : pal[5],
                   'Servier_Ctrl_mouse_wb_lateral_plane_chca': pal[6], 'Servier_Ctrl_mouse_wb_lateral_plane_DHB': pal[9]}

    color_dict_multiple = {'Multiple Datasets':'gray'}
    
    c_df = df[(df['umap_x'] >  xi) & (df['umap_x'] <  xf) & ( df['umap_y'] > yi) & ( df['umap_y'] < yf) ]
    c_df.dropna()
    if hue == 'single_dataset_name':
        single_ds_df = c_df[c_df[hue] != 'Multiple Datasets']
        multiple_ds_df = c_df[c_df[hue] == 'Multiple Datasets'] 
        ax = sns.scatterplot(data = single_ds_df, x = 'umap_x', y= 'umap_y', hue = hue, palette = color_dict_single, s=marker_size)
        sns.scatterplot(data = multiple_ds_df, x = 'umap_x', y='umap_y', hue = hue, palette = color_dict_multiple, s = marker_size, alpha = mul_trans)
    else:
        ax = sns.scatterplot(data = c_df, x = 'umap_x', y= 'umap_y', hue = hue, s=marker_size)
    plt.legend(loc='best')
    plt.title(title)
    label_point(c_df.umap_x, c_df.umap_y, c_df[t_label], plt.gca(), size = text_size)
    if output:
        plt.savefig(f'plots/{output}.png', format = 'png')
    plt.show()
    return c_df, ax

def get_ds_list(df):
    '''
    get dataset list of unique ids or names
    '''
    # get dataset names 
    ds_names = df['ds_names'].tolist()
    try:
        ds_names = [item for sublist in ds_names for item in ast.literal_eval(sublist)] # flatten list
    except ValueError:
        ds_names = [item for sublist in ds_names for item in sublist] # flatten list
        ds_names = list(set(ds_names)) # unique values
    #get dataset ids
    ds_ids = df['dataset_ids'].tolist()
    try:
        ds_ids = [item for sublist in ds_ids for item in ast.literal_eval(sublist)] # flatten list
    except ValueError:
        ds_ids = [item for sublist in ds_ids for item in sublist] # flatten list
        ds_ids = list(set(ds_ids)) # unique values
    return ds_ids, ds_names

def get_ion_imgs(df, ion_list, ds_ids, fdr=0.1, metahost = 'https://metaspace2020.eu', molbase = 'HMDB-v4'):
    '''
    params:
    -------
    df       : input dataframe
    ion_list : list of query ions
    ds_ids   : list of query dataset ids
    
    out:
    ----
    imgs : array of images, where imgs[i,j] is the image for
           dataset[i] and ion[j]
    '''
    imgs = []
    sm = SMInstance(host=metahost)
    for j, dataset_id in enumerate(ds_ids):
        molset = sm.dataset(molbase, dataset_id)
        img_list = []
        for i, ion in enumerate(ion_list):
            query_row = df[df['ion'] == ion]
            sf = query_row['formula'].item()
            adduct = query_row['adduct'].item() 
            
            query_ids = query_row['dataset_ids'].tolist()
            try:
                query_ids = [item for sublist in query_ids for item in ast.literal_eval(sublist)] # flatten list
            except ValueError:
                query_ids = [item for sublist in query_ids for item in sublist] # flatten list

            querys_ids = list(set(ds_ids)) 
            if dataset_id in query_ids: # check if ion is annotated in ds
                img = molset.isotope_images(sf, adduct, only_first_isotope=True)
            else:
                img = np.zeros((2,2))
            img_list.append(img)
        imgs.append(img_list)
                
    return imgs

def plot_ion_imgs(imgs, ion_list, ds_names, mol_names = None,
                  figsize=(15,15), dpi=200, out=None):
    '''
    input:
    --------------------------------------------------------
    imgs    : image array with dimensions (#datasets, #ions)
    ion_list: list of ion names
    ds_names: list of dataset names
    
    returns:
    --------------------------------------
    subplots with shape (#ions, #datasets)
    '''    
    n_ions   = len(imgs[0])
    n_ds = len(imgs)
    
    if mol_names:
        names = [mol_names[ion] for ion in ion_list]   
    else: names = ion_list
        
    fig, axs = plt.subplots(n_ions, n_ds, figsize=figsize) # rows: ions, columns: dataset
    for i, name in enumerate(names):
        if len(name) > 45: name = name[:45]
        if n_ds >1: # if ions come from more than one ds
            for j, ds_name in enumerate(ds_names):
                axs[0,j].set_title(ds_name, size='x-small')
                if isinstance(imgs[j][i],metaspace.sm_annotation_utils.IsotopeImages): # if ion is not detected in ds, hide it
                    axs[i,j].imshow(np.squeeze(imgs[j][i]))
                    axs[i,j].set_title(ds_name, size='x-small')
                    axs[i,j].set_xlabel(f'{name}', rotation=0, size='x-small')
                else:
                    axs[i,j].set_visible(False)
        else:
            if isinstance(imgs[0][i],metaspace.sm_annotation_utils.IsotopeImages): # if ion is not detected in ds, hide it
                axs[i].imshow(np.squeeze(imgs[0][i]))
                axs[i].set_title(ds_names[0], size='x-small')
                axs[i].set_xlabel(f'{name}', rotation=0, size='x-small')
            else:
                axs[i].set_visible(False)
            
                
    plt.setp(axs, xticks=[], yticks=[]) # x/y-ticks offx
    plt.tight_layout()
    if out:
        plt.savefig(f'{out}.png', format='png')
    plt.show()

def label_point(x, y, val, ax, size = 2):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if len(str(point['val'])) > 30:
            t = str(point['val'])[:30]
            ax.text(point['x']+.0001, point['y'], t, size = size)
        else:       
            ax.text(point['x']+.0001, point['y'], str(point['val']), size = size)

def imshow_ions(df, out=None):
    ds_ids, ds_names = get_ds_list(df)
    ions = df['ion'].tolist()

    imgs = get_ion_imgs(df, ions, ds_ids)
    
    plot_ion_imgs(imgs, ions, ds_names, figsize=(15, len(ions)*2), mol_names=i2m_dict, out=out)
