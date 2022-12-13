# ion2vec

## Data handling

### Preprocessing
[pixel_dataframes.py](pixel_dataframes.py) provides the basic functionalities to extract pixel dataframes from ion images given a metaspace dataset ID. 

### Functionalities
We have done basic explorative search of datasets in [DatasetExploration.ipynb](datasets/DatasetExploration.ipynb) which displayed that we loose much ions when we want ions to be annotated in multiple datasets. <br>
[CleaningMetadata.py](datasets/CleaningMetadata.py) provides the functionalities to normalize the metaspace dataset metadata. In [SelectingDatasets.ipynb](datasets/SelectingDatasets.ipynb), we  sample from all HMDB-v4 that were uploaded until 30.09.2022, clean and filter the data to come up with [datasets_filtered_weighted.csv](datasets/datasets_filtered_weighted.csv).
 These functionalities are used in [get_data.py](datasets/get_data.py), which we can run from the command line by 
```
python get_data.py -ds_df DATASETS_TO_SAMPLE_FROM.csv -org ORGANISMS -org_part ORGANISM_TYPE -cond CONDITION -n NUMBER_OF_DATASETS -output OUTPUT_FILE_NAME
```
For instance, sampling for `-n 100`, `-org mouse` and `-org_part brain`, leads to 100 sampled dfs, with their ids saved in a [csv](datasets/mouse_brain_datasets/mouse_brain.csv). We can then use [load_data.py](datasets/load_data.py) to create pixel dataframes from the ids, which can be either given in form of dataset IDs or a csv like the one previously created, 
```
python load_data.py -csv sampl`100.csv
```
The output files will be saved as pickle. Note, that the pickle protocol is set to `4` as the vanilla version runs on `python 3.6` and not all old packages (most importantly pandas) do support pickle protocol `5`. 

### Datasets
The [example dataframes](datasets/example_dfs/) are small datasets from the 100 sampled mouse brain data. They were mainly used to debug the training code, as training ran quickly on those small files. <br>
The 100 sampled mouse brain datasets can be loaded from [mouse_brain.csv](datasets/mouse_brain_datasets/mouse_brain.csv) via load_data.py. <br>
The datasets in [theos_recom](datasets/theos_recom/) were recommended by Theo as validation datasets. They're further split up into datasets with high ion coverage ([large files](datasets/theos_recom/No1/)) and whole body mouse datasets by the same lab, Servier, in [positve](datasets/theos_recom/mouse_wb_pos/) and [negative](datasets/theos_recom/mouse_wb_neg/) mode. The all display good spatial heterogeneity. 

## Training
Two conda environments are located in [conda-envs](conda-envs), where the main version is needed for the random walk application and is defaultly used for all other tasks - except when training the vanilla version when of course `VanillaEnv` has to be activated. We can create the environments from the yml files by
```
conda env create -f ENVIRONMENT.yml
```

The pixel_dataframes are the basic inputs to both <code>rw_train.py</code> and <code>Word2vec_pix.py</code>. 