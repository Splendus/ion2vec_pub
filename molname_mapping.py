import os, ast
import pandas as pd

ANN_DIR = os.path.abspath("/scratch/trose/hmdbv4_300922") # annotation csv files
ion_to_mols = {} # dictionary for ion to molecule name mapping

for file in os.listdir(ANN_DIR):
    if file.endswith(".csv"):
        ann_df = pd.read_csv(os.path.join(ANN_DIR, file))
        for ion in ann_df['ion']:
            if ion not in ion_to_mols:
                names = ast.literal_eval(ann_df[ann_df['ion'] == ion]['moleculeNames'].item())
                names.sort(key=len)
                short_name = names[0]
                ion_to_mols[ion] = ast.literal_eval(ann_df[ann_df['ion'] == ion]['moleculeNames'].item())[0]
# create df
ion_mol_df = pd.DataFrame(data = [ion_to_mols.keys(), ion_to_mols.values()]).T.rename(columns = {0:'ion', 1:'moleculeNames'})
ion_mol_df.to_csv('Ions2Molecules.csv', index=False) # save to csv

