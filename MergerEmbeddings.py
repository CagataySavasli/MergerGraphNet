import os
import pandas as pd
import numpy as np

# Çıktı dosya yolları
train_csv = 'data/processed/embedded_labeled.csv'

# Varolan çıktı dosyalarını temizle (varsa)
if os.path.exists(train_csv):
    os.remove(train_csv)

not_founds = []

print("Start merging embeddings in memory efficient way")
# Dosyalar 0'dan 14000'e kadar her 100 adımında işlenecek
for idx in range(0, 14000, 100):
    file_path = f'./data/sep/embeddings_labeled_{idx}_{idx + 100}.json'

    if os.path.isfile(file_path):
        try:
            # JSON dosyasını oku
            tmp = pd.read_json(file_path)

            # Dosyada yeterli veri varsa işle
            if len(tmp) >= 5:
                # İlgili sütunları seç ve embeddings'i float16'ya çevir
                tmp = tmp[['form', 'year', 'embeddings', 'label']].copy()
                tmp['embeddings'] = tmp['embeddings'].apply(lambda x: list(np.array(x).astype(np.float16)))


                # Verileri CSV'ye ekle (ilk seferde header yaz, sonraki seferde ekle)
                if not tmp.empty:
                    if os.path.exists(train_csv):
                        tmp.to_csv(train_csv, mode='a', header=False, index=False)
                    else:
                        tmp.to_csv(train_csv, mode='w', header=True, index=False)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    else:
        not_founds.append(file_path)
    print(f"Processed file: {file_path}")

print("## Not Found Files:")
for not_found in not_founds:
    print(not_found)
