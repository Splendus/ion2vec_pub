import pickle, os
import pandas as pd

ROOT = os.getcwd() 
for NODE in os.listdir(ROOT):
    try:
        for f in os.listdir(NODE):
            frame = os.path.join(NODE, f)
            df = pd.read_pickle(frame)
            save = df.to_pickle(frame, protocol = 4)
    except NotADirectoryError:
        continue
