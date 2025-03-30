# %%
from lib.config.config_loader import ConfigLoader
from transformers import AutoTokenizer, AutoModel
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

print(f"Preprocessing {start_idx}-{end_idx}...")
df = pd.read_csv('./data/processed/reports_labeled.csv')
df = df.loc[start_idx:end_idx].copy()
df.reset_index(drop=True, inplace=True)


# %%
def get_finbert_embedding(text: str) -> np.ndarray:
    """
    Verilen metin için FinBERT embedding'ini, akademik çalışmalarda sıklıkla tercih edilen
    pooler_output (CLS temsili) kullanarak hesaplar. Sonuç, hafıza kullanımını azaltmak için float16 tipinde döndürülür.

    Parametre:
      text: İşlenecek tam metin (str).

    Returns:
      text için hesaplanmış, (hidden_dim,) boyutunda float16 tipinde bir numpy array.
    """
    # Metni token'lara ayırırken truncation ve padding uygulanıyor.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Pooler_output, modelin cümlenin genel bilgisini içeren temsili olarak verilir.
    embedding = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float16)
    return embedding


# %%
print("Start embedding...")
# Her rapor için tam metin üzerinden tek bir embedding hesaplanıyor.
df['embedding'] = df['mda'].progress_apply(get_finbert_embedding)
# Belleği daha verimli kullanmak için, artık ihtiyaç duyulmayan orijinal metni kaldırıyoruz.
# df.drop('mda', axis=1, inplace=True)

# %%
print("Saving embeddings...")
# Parquet formatı, JSON'a göre daha verimli disk kullanımı sağlar.
df.to_parquet(f'./data/sep/embeddings_labeled_{start_idx}_{end_idx}.parquet', index=False)
print("Done!")
