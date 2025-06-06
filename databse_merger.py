import os
import json
from lib.database.database_connector import DatabaseConnector  # database_connector.py'den alınır

def ingest_embeddings_to_db(
    db_path: str = "./data/database.db",
    data_dir: str = "./data/sep",
    step: int = 100,
    max_idx: int = 14000
):
    # 1) Veritabanına bağlan / yoksa oluştur
    db = DatabaseConnector(db_path)

    table_name = "embeddings"
    columns = None

    # 2) JSON dosyalarını tek tek işle
    for start in range(0, max_idx, step):
        end = start + step
        file_path = os.path.join(data_dir, f"embeddings_labeled_{start}_{end}.json")

        if not os.path.isfile(file_path):
            print(f"[SKIP] Dosya bulunamadı: {file_path}")
            continue

        # JSON'u yükle (bellekte sadece bu dosya olacak)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON okunamadı ({file_path}): {e}")
                continue

        if not records:
            print(f"[INFO] Dosya boş: {file_path}")
            continue

        # 3) İlk dosyada tabloyu oluştur
        if columns is None:
            columns = list(records[0].keys())
            db.create_table(table_name, columns)

        # 4) Her kaydı uygun formata çevir ve ekle
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders});"

        to_insert = []
        for rec in records:
            row = []
            for col in columns:
                val = rec.get(col)
                # List veya dict türündeyse JSON stringine çevir
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                row.append(val)
            to_insert.append(tuple(row))

        # Toplu ekleme ve commit
        db.cursor.executemany(insert_sql, to_insert)
        db.connection.commit()

        print(f"[OK] {len(to_insert)} kayıt eklendi: {file_path}")

    # 5) Bağlantıyı kapat
    db.close_connection()
    print("Bütün işlemler tamamlandı, bağlantı kapatıldı.")

if __name__ == "__main__":
    ingest_embeddings_to_db()
