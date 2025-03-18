#%%
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
#%%
def get_finbert_embeddings(texts) -> np.ndarray:
    """
    Verilen cümle listesindeki metinler için FinBERT embedding'lerini hesaplar.
    Her metin için, modelin son hidden state çıktısındaki token embedding'lerinin ortalaması alınır.

    Parametreler:
      texts: str tipinde öğeler içeren bir liste.

    Returns:
      texts listesindeki her bir metin için hesaplanmış embedding'lerin bulunduğu
      (n_metinsayısı, hidden_dim) boyutunda numpy dizisi.
    """
    if not type(texts) == list:
        texts = [texts]
    # Tokenize işlemi: tüm metinleri aynı anda tokenize edip, padding uyguluyoruz.
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Modeli değerlendirme modunda çalıştır
    with torch.no_grad():
        outputs = model(**inputs)

    # Her metin için token embedding'lerinin ortalamasını alıyoruz.
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings
#%%
print("Start embedding...")
df['embeddings'] = df['sentences'].progress_apply(get_finbert_embeddings)
#%%
print("Saving embeddings...")
df.to_json(f'./data/sep/embeddings_labeled_{start_idx}_{end_idx}.json', orient='records')
#%%
print("Done!")