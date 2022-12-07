import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pylint: disable=no-name-in-module
from Levenshtein import distance
from metaspace.sm_annotation_utils import SMInstance


#%% Functions for cleaning up metadata fields


def normalize_analyzer(analyzer):
    analyzer = (analyzer or '').lower()
    if any(phrase in analyzer for phrase in ['orbitrap', 'exactive', 'exploris', 'hf-x', 'uhmr']):
        return 'Orbitrap'
    if any(phrase in analyzer for phrase in ['fticr', 'ft-icr', 'ftms', 'ft-ms']):
        return 'FTICR'
    if any(phrase in analyzer for phrase in ['tof', 'mrt', 'exploris', 'synapt', 'xevo']):
        return 'TOF'
    return analyzer


def normalize_source(source):
    source = (source or '').lower()
    if any(phrase in source for phrase in ['maldi']):
        return 'MALDI'
    if any(phrase in source for phrase in ['sims', 'gcib']):
        return 'SIMS'
    if any(phrase in source for phrase in ['ir-maldesi', 'laesi', 'ldi', 'lesa']):
        # Some of these contain "ESI" but are pretty distinctive so shouldn't be lumped in with DESI
        return 'Other'
    if any(phrase in source for phrase in ['esi']):
        return 'ESI'
    return 'Other'


def normalize_resolving_power(ms_analysis):
    analyzer = normalize_analyzer(ms_analysis['Analyzer'])
    resolving_power = ms_analysis['Detector_Resolving_Power']['Resolving_Power']
    mz = ms_analysis['Detector_Resolving_Power']['mz']

    if analyzer == 'FTICR':
        return resolving_power * mz / 200.0
    elif analyzer == 'Orbitrap':
        return resolving_power * (mz / 200.0) ** 0.5
    else:
        return resolving_power


SOLVENT_MAPPING = {
    'tfa': 'TFA',
    'formic acid': 'FA',
    'acetone': 'Acetone',
    'dmf': 'DMF',
    'methanol': 'MeOH',
    'ethanol': 'EtOH',
    'toluene': 'Toluene',
    'acetonitrile': 'ACN',
}
SOLVENT_MAPPING.update({v.lower(): v for v in SOLVENT_MAPPING.values()})
SOLVENT_RE = re.compile('|'.join(SOLVENT_MAPPING.keys()))


def normalize_solvent(solvent):
    if not solvent:
        return 'Other'
    m = SOLVENT_RE.search(solvent.lower())
    if m:
        return SOLVENT_MAPPING[m[0]]
    return 'Other'


MATRIX_MAPPING = {
    # There are lots more matrixes, but these were the top 10 at time of writing
    '2,5-dihydroxybenzoic acid': 'DHB',
    '1,5-diaminonaphthalene': 'DAN',
    '9-aminoacridine': '9AA',
    'alpha-cyano-4-hydroxycinnamic acid': 'CHCA',
    'n-(1-naphthyl)ethylenediamine dihydrochloride': 'NEDC',
    'α-cyano-4-hydroxycinnamic acid': 'HCCA',
    '1,8-bis(dimethylamino)naphthalene': 'DMAN',
    '2,5-dihydroxyacetophenone': 'DHAP',
    'dha': 'DHAP',  # People use both DHA and DHAP for 2,5-dihydroxyacetophenone. Settle on DHAP
    'Norharmane': 'Norharmane',
    # 'None': 'None',
}
MATRIX_MAPPING.update({v.lower(): v for v in MATRIX_MAPPING.values()})
MATRIX_RE = re.compile('|'.join(re.sub('[()]', '\\$0', k) for k in MATRIX_MAPPING.keys()))


def normalize_matrix(matrix):
    if not matrix:
        return 'Other'
    m = MATRIX_RE.search(matrix.lower())
    if m:
        return MATRIX_MAPPING[m[0]]
    return 'Other'


#%%  Try to select a diverse but relatively representative sample of datasets
def get_stratified_sample(ds_df, count):
    weights = ds_df.batch_weight.copy()
    # For (col, count), take the top `count` values of `col` and compact the rest into an "Other"
    # category.  These are applied to the weights in order and the order has a big impact.
    # If a particular column isn't being proportionately represented in the results, try moving it
    # up or down in the list.
    ## The larger pop is, the more "diverse" the data becomes with respect to that column
    cols = [
        ('matrix_solvent', 10),
        ('source', 2),
        ('analyzer', 2),
        ('rp_range', 3),
        ('submitter', 50),
        ('group', 20),
        ('polarity', 20),
        #('organism', 50),
        #('organism_part', 20),
        #('condition', 20)
    ]
    for col, pop in cols:
        group_weights = weights.groupby(ds_df[col]).sum() ** 0.5
        group_weights = group_weights.sort_values(ascending=False) # largest weight on top
        group_weights /= group_weights.sum() # normalizing
        other = group_weights.index.isin([*group_weights.index[pop:], 'Other']) 
        adjustment = pd.Series(
            np.where(other, 1 / max(group_weights[other].sum(), 0.001), 1 / group_weights),
            index=group_weights.index,
        )

        weights *= adjustment[ds_df[col].values].values

    # Slightly prefer newer datasets, as they're more representative of future datasets
    newness = np.maximum(ds_df.ds_id.str.slice(0, 4).astype(int) - 2013, 0)
    weights *= newness

    ds_df = ds_df.assign(weight=weights)
    return ds_df.sample(count, weights=ds_df.weight).sort_values(['group', 'ds_id'])

def normalize_organism(organism):
    organism = (organism or '').lower()
    if any(phrase in organism for phrase in ['Homo sapiens (human)', 'Human', 'human', 'homospaiens', 'Homo sapiens (human) ']):
        return 'human'
    if any(phrase in organism for phrase in ['Mus musculus (mouse)', 'Mouse', 'Mouse ', 'mouse', 'mice']):
        return 'mouse'
    if any(phrase in organism for phrase in ['Rattus norvegicus (rat)', 'Rat', 'rat']):
        return 'rat'
    if any(phrase in organism for phrase in ['Poplar', 'poplar']):
        return 'poplar'
    return organism

def normalize_organism_part(organism_part):
    organism_part = (organism_part or '').lower()
    if any(phrase in organism_part for phrase in ['Kidney', 'kidney', 'kidney ', 'kideny']):
        return 'kidney'
    if any(phrase in organism_part for phrase in ['Brain', 'brain', 'Brain Hippocampus ', 'Brain (CSF)']):
        return 'brain'
    if any(phrase in organism_part for phrase in ['Liver', 'liver', 'Major part of liver tissue']):
        return 'liver'
    if any(phrase in organism_part for phrase in ['Whole organism', 'whole organism', 'whole body']):
        return 'whole organism'
    if any(phrase in organism_part for phrase in ['Leaf', 'leaf']):
        return 'leaf'
    if any(phrase in organism_part for phrase in ['Lung', 'lung']):
        return 'lung'    
    return organism_part

def normalize_condition(condition):
    condition = (condition or '').lower()
    if any(phrase in condition for phrase in ['Wildtype', 'wildtype', 'Wild type', 'Wildtipe', 'wildtype',
                                              'Wildtype and knock out', 'wild-type', 'Wildtype - Injured', 
                                             'wild type', 'Wtype']):
        return 'wildtype'
    if any(phrase in condition for phrase in ['Diseased', 'diseased']):
        return 'diseased'
    if any(phrase in condition for phrase in ['Frozen', 'frozen', 'Fresh frozen', 'fresh frozen']):
        return 'frozen'
    if any(phrase in condition for phrase in ['Normal', 'normal', 'normal ']):
        return 'normal'
    if any(phrase in condition for phrase in ['Leaf', 'leaf']):
        return 'leaf'
    if any(phrase in condition for phrase in ['Lung', 'lung']):
        return 'lung'    
    return condition