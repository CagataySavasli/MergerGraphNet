from lib.config.config_loader import ConfigLoader

from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


import torch
import numpy as np
import pandas as pd
import sys

print("Loading model...")

config = ConfigLoader().load_config()
tqdm.pandas()

tokenizer = AutoTokenizer.from_pretrained(config['embedding']['model_name'])
model = AutoModel.from_pretrained(config['embedding']['model_name'])

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
#%%
print(f"Preprocessing {start_idx}-{end_idx}...")
df = pd.read_csv('./data/processed/reports_labeled.csv')

df = df.loc[start_idx:end_idx].copy()

df.reset_index(drop=True, inplace=True)
df['sentences'] = df['mda'].progress_apply(lambda x: sent_tokenize(x))
df.drop('mda', axis=1, inplace=True)

# %%
def get_finbert_embedding(texts, batch_size: int = 256):
    """
    Generate embeddings for a list of texts using batching.
    If a memory error occurs, recursively reduce the batch size.
    """
    embeddings = []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = tokenizer(batch, return_tensors='pt', padding=True,
                                        truncation=True, max_length=512)
                outputs = model(**tokens)
                batch_embeddings = outputs.pooler_output.half().cpu().numpy()
                embeddings.extend(batch_embeddings)
    except Exception as e:
        if batch_size > 2:
            return get_finbert_embedding(texts, batch_size=batch_size // 2)
        else:
            raise RuntimeError(f"Error during embedding generation: {e}")
    return embeddings


# %%
print("Start embedding...")
# Her rapor için tam metin üzerinden tek bir embedding hesaplanıyor.
df['embedding'] = df['sentences'].progress_apply(get_finbert_embedding)
# Belleği daha verimli kullanmak için, artık ihtiyaç duyulmayan orijinal metni kaldırıyoruz.
# df.drop('mda', axis=1, inplace=True)

# %%
print("Saving embeddings...")
# Parquet formatı, JSON'a göre daha verimli disk kullanımı sağlar.
df.to_parquet(f'./data/sep/x_embeddings_labeled_{start_idx}_{end_idx}.parquet', index=False)
print("Done!")
