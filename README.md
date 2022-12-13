# ion2vec

## Data handling

### Preprocessing
[pixel_dataframes.py](pixel_dataframes.py) provides the basic functionalities to extract pixel dataframes from ion images given a metaspace dataset ID. 

### Functionalities
Basic explorative search of datasets can be seen in [DatasetExploration.ipynb](datasets/DatasetExploration.ipynb) which displayed how much query ions we loose when we require ions to be annotated in multiple datasets. <br>
[CleaningMetadata.py](datasets/CleaningMetadata.py) provides the functionalities to normalize the metaspace dataset metadata. In [SelectingDatasets.ipynb](datasets/SelectingDatasets.ipynb), we  sample from all HMDB-v4 that were uploaded until 30.09.2022, clean and filter the data to come up with [datasets_filtered_weighted.csv](datasets/datasets_filtered_weighted.csv).
 These functionalities are used in [get_data.py](datasets/get_data.py), which we can run from the command line by 
```
python get_data.py -ds_df DATASETS_TO_SAMPLE_FROM.csv -org ORGANISMS -org_part ORGANISM_TYPE -cond CONDITION -n NUMBER_OF_DATASETS -output OUTPUT_FILE_NAME
```
For instance, sampling for `-n 100 -org mouse -org_part brain`, leads to 100 sampled dfs, with their ids saved in a [csv](datasets/mouse_brain_datasets/mouse_brain.csv). We can then use [load_data.py](datasets/load_data.py) to create pixel dataframes from the ids, which can be either given in form of dataset IDs or a csv like the one previously created, 
```
python load_data.py -csv datasets/mouse_brain_datasets/mouse_brain.csv
```
The output files will be saved as pickle. Note, that the pickle protocol is set to `4` as the vanilla version runs on `python 3.6` and not all old packages (most importantly pandas) do support pickle protocol `5`. 

### Datasets
The [example dataframes](datasets/example_dfs/) are small datasets from the 100 sampled mouse brain data. They were mainly used to debug the training code, as training ran quickly on those small files. <br>
The 100 sampled mouse brain datasets can be loaded from [mouse_brain.csv](datasets/mouse_brain_datasets/mouse_brain.csv) via load_data.py. <br>
The datasets in [theos_recom](datasets/theos_recom/) were recommended by Theo as validation datasets. They're further split up into datasets with high ion coverage ([large files](datasets/theos_recom/No1/)) and whole body mouse datasets by the same lab, Servier, in [positve](datasets/theos_recom/mouse_wb_pos/) and [negative](datasets/theos_recom/mouse_wb_neg/) mode. The all display good spatial heterogeneity. 

## Training
Two conda environments are located in [conda-envs](conda-envs), where the main version is needed for the random walk application and is defaultly used for all other tasks - except when training the vanilla version when of course `VanillaEnv` has to be activated. We can create the environments from the yml files by
```
conda env create --name ENVIRONMENT_NAME --file ENVIRONMENT.yml
```

Importantly, the Vanilla version runs with *cbow* while the Random Walk uses *skipgram*. This is simply due to the fact, that the Vanilla version by Katja adapted only cbow for a start and Tensorflow has only skipgram implemented by default. One would have to write a [TensorFlow cbow version from scratch](https://gist.github.com/yxtay/a94d971955d901c4690129580a4eafb9)

### Vanilla
The vanilla training runs via [word2vec_pix.py](word2vec_pix.py), which bases on [gensim](https://radimrehurek.com/gensim/models/word2vec.html)[^1]. Note that it can run in either slow (python only) or fast mode (cython). For the fast version, the `word2vec_inner.*` files and the `build` directory are needed. `word2vec_pix.py` imports from the files by default but reverts to the slow python-only version in case an import fails. It should work the way the directory is set up. <br>

We can train the Vanilla model by running from the command line
```
python word2vec_pix.py -train DATA_DIR -ind_name TRAIN_IONS.pickle -iter EPOCHS -threads WORKERS -output OUTPUT_FILE_NAME -size VECTOR_DIMENSION -window IMAGE_WINDOW_SIZE`
```
Some more arguments - for instance to reduce the amount of data like window stride `-stride`, pixel sampling percentage `pix_per` - can be given and are explained in the python script itself. <br>
Note that a `window` of 5 relates to a $(2 \cdot 5 +1) \times (2 \cdot 5 +1)$ image window. <br>
The script yields two files, one .txt and one binary. Theoretically, the binary file can be loaded via the gensim functionalities and trained further or used for post-processing. We restricted ourselves to use the vectors which can be found in the text file. 

### Random Walk
The random walk training file [rw_train.py](rw_train.py) runs with Tensorflow[^2]. 

We can train the Random Walk model by running from the command line
```
python rw_train.py -train DATA_DIR -ind_name TRAIN_IONS.pickle -iter EPOCHS -threads WORKERS -output OUTPUT_FILE_NAME -size VECTOR_DIMENSION -window IMAGE_WINDOW_SIZE -word_window TEXT_WINDOW -rw 1 
```
Note the two additional parameters `word_window`and `rw`. In comparison to the Vanilla model, we do not naturally train on the whole image window, but rather pick an additional (one dimensional) `word_window` that sets the window size with which we go over the 'output text'. This can be advantageous as the random walk model encodes spatial information of the image window. <br>
We can also set `rw` to 0, to run with the regular corpus building instead of the random walk. We discourage from making use of the 'Vanilla TensorFlow' (i.e. `-rw 0`) for now as it was hardly tested and assumes some sense of locality which is not given in the vanilla image window output. <br>

The model yields two files, one vector and one meta file. The meta file holds the index-to-ion name mapping for the ion vectors in the vector file.

## Post Processing / Visualization

[metadata.py](metadata.py) covers the basic functionalities to connect the model outputs with meta data, e.g. molecule names, formulas, metabolite classes, dataset occurrence[^3] and more. We can also include UMAP embeddings in the dataframes. <br>

For visualization one can for instance simply read in a model output file (one text or two tsv files) in the [post_viz notebook](post_viz.ipynb) and run the code cells. An example run can be seen in [interactive_viz.ipynb](interactive_viz.ipynb) where we load, postprocess and visualiza data for different models and hyperparameters. <br>

Some model outputs can be found for the [Vanilla](slurm_job/Vanilla_output/) and [Random Walk](RW_output/). The file name generally follows the scheme 
```
vectors_TF_RW_validation_IMAGE_WINDOW_SIZE_*_EMBEDDING_DIMENSION_TEXT_WINDOW_SIZE_rw.tsv
```

[^1]: We have used an old version of gensim (version 3.4.0) as we built upon [Katja's old model](https://github.com/eovchinn/word2vec_pixel). 
[^2]: It can be run theoretically by the gensim vanilla version, too. One simply has to import `PixelCorpusRW` from [PixelCorpora.py](PixelCorpora.py) in the `word2vec_pix.py` script instead of using the default PixelCorpus class. 
[^3]: The functions were written with the [validation datasets](datasets/theos_recom/mouse_wb_pos/) in mind. In case, one uses ions trained on different datasets, one has to give a list of dataset ids as input to the `post_processing` class.