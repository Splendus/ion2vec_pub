# ion2vec

## Data handling
We have done basic explorative search of datasets in [DatasetExploration.ipynb](datasets/DatasetExploration.ipynb) which displayed that we loose much ions when we want ions to be annotated in multiple datasets. <br>
[CleaningMetadata.py](datasets/CleaningMetadata.py) provides the functionalities to normalize the metaspace dataset metadata. In [SelectingDatasets.ipynb](datasets/SelectingDatasets.ipynb), we  sample from all HMDB-v4 that were uploaded until 30.09.2022, clean and filter the data to come up with [datasets_filtered_weighted.csv](datasets/datasets_filtered_weighted.csv).
 These functionalities are used in [get_data.py](datasets/get_data.py).  We can run get_data.py from the command line by 
```
python get_data.py -ds_df DATASETS_TO_SAMPLE_FROM.csv -org ORGANISMS -org_part ORGANISM_TYPE -cond CONDITION -n NUMBER_OF_DATASETS output OUTPUT_FILE_NAME
```
For instance, sampling for `-n` 100, `-org` mouse and `-org_part` brain, leads to 100 sampled dfs, with their ids saved in a [csv](datasets/sample100.csv). We can then use [load_data.py](datasets/load_data.py) to create pixel dataframes from the ids, which can be either given in form of dataset IDs or a csv like the one previously created, 
```python load_data.py -csv sample100.csv```

## Preprocessing
pixeldataframes.py provides the basic functionalities to extract pixel dataframes from ion images given a metaspace dataset ID. 

## Training
These are the basic inputs both <code>rw_train.py</code> and <code>vanilla_train.py</code> use. 