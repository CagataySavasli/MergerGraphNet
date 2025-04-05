import os
import pandas as pd

# Çıktı dosya yolları
train_csv = 'data/processed/embedded_labeled.parquet'

# Varolan çıktı dosyalarını temizle (varsa)
if os.path.exists(train_csv):
    os.remove(train_csv)

not_founds = []
founds = []
print("Start merging embeddings in memory efficient way")
# Dosyalar 0'dan 14000'e kadar her 100 adımında işlenecek
for idx in range(0, 13100, 100):
    file_path = f'./data/sep/embeddings_labeled_{idx}_{idx + 100}.parquet'

    if os.path.isfile(file_path):
        try:
            tmp = pd.read_parquet(file_path)
            founds.append(tmp)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    else:
        not_founds.append(file_path)
    print(f"Processed file: {file_path}")

# Birleştirilmiş DataFrame'i oluştur
if founds:
    merged_df = pd.concat(founds, ignore_index=True)
    merged_df.to_parquet(train_csv, index=False)
    print(f"Merged DataFrame saved to {train_csv}")

print("## Not Found Files:")
for not_found in not_founds:
    print(not_found)
