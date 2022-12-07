from os import listdir
from os.path import join
from pathlib import Path
import pandas as pd

DIR = "No1No2"
ion2ds = {}
for f in listdir(DIR):
    if f.endswith('.pickle'):
        file = join(DIR, f)
        ds_df = pd.read_pickle(file)
        
        ion_names = ds_df.drop(columns=['x', 'y']).columns.tolist()
        for ion in ion_names:
            if ion not in ion2ds.keys() or ion2ds[ion] == None:
                ion2ds[ion] = list((Path(file).stem.replace('pixel_df_', ''),))
            else:
                ion2ds[ion].append(Path(file).stem.replace('pixel_df_', ''))
    else: 
        continue
ion2ds_df = pd.DataFrame(ion2ds.items(), columns=['ion', 'datasets'])
ion2ds_df.to_csv("ions_datasets.csv", index=False)
print(ion2ds_df)
