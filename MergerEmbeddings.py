import pandas as pd
import numpy as np
import os

frames = []
not_founds = []
print('Start merging embeddings')
for idx in range(0, 14000, 100):
    print(f"{round(((idx+100)/14000), 2)*100}%")
    if os.path.isfile(f'./data/sep/embeddings_labeled_{idx}_{idx+100}.json'):
        tmp = pd.read_json(f'./data/sep/embeddings_labeled_{idx}_{idx+100}.json')
        tmp = tmp[['form', 'year', 'embeddings', 'label']].copy()
        tmp['embeddings'] = tmp['embeddings'].apply(lambda x: list(np.array(x).astype(np.float16)))
        frames.append(tmp)
    else:
        not_founds.append(f"embeddings_labeled_{idx}_{idx+100}.json")


data = pd.concat(frames, ignore_index=True)
data.reset_index(drop=True, inplace=True)

data.to_csv('data/processed/embedded_labeled.csv', index=False)

print("## Not Found Files")
for not_found in not_founds:
    print(not_found)