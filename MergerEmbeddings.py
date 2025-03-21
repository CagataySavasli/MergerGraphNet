import pandas as pd
import numpy as np

frames = []

for idx in range(0, 14000, 100):
    tmp = pd.read_json(f'./data/sep/embeddings_{idx}_{idx+100}.json')
    tmp = tmp[['form', 'year', 'embeddings', 'label']].copy()
    tmp['embeddings'] = tmp['embeddings'].apply(lambda x: list(np.array(x).astype(np.float16)))
    frames.append(tmp)

data = pd.concat(frames, ignore_index=True)
data.reset_index(drop=True, inplace=True)

data.to_csv('data/processed/embedded_labeled.csv', index=False)