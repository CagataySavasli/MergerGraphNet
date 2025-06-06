from torch.utils.data import Dataset
from lib.database.database_connector import DatabaseConnector
from lib.data.graph_generator import GraphGenerator
import json

class SQLiteDataset(Dataset):
    """
    PyTorch Dataset: Veritabanından satır satır okuma.
    Tarihe göre split yapmak için start_date/end_date kullanılır.
    """
    def __init__(
        self,
        db_path,
        table_name,
        date_column,
        graph_version = False,
        conn_sentence = None,
        start_date=None,
        end_date=None,
        transform=None
    ):
        self.db = DatabaseConnector(db_path)
        self.cursor = self.db.cursor
        self.table_name = table_name
        self.date_column = date_column
        self.graph_version = graph_version
        self.transform = transform

        self.graph_generator = GraphGenerator(conn_sentence)

        # Sadece rowid'leri çek
        clauses, params = [], []
        if start_date is not None:
            clauses.append(f"{date_column} >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"{date_column} < ?")
            params.append(end_date)
        where = " AND ".join(clauses) if clauses else "1=1"

        self.cursor.execute(
            f"SELECT rowid FROM {table_name} WHERE {where};",
            params
        )
        self.rowids = [r[0] for r in self.cursor.fetchall()]

    def __len__(self):
        return len(self.rowids)

    def __getitem__(self, idx):
        rowid = self.rowids[idx]
        self.cursor.execute(
            f"SELECT * FROM {self.table_name} WHERE rowid = ?;",
            (rowid,)
        )
        row = self.cursor.fetchone()
        cols = [d[0] for d in self.cursor.description]
        record = dict(zip(cols, row))

        # JSON string => obje
        for k, v in record.items():
            if isinstance(v, str) and (v.startswith('[') or v.startswith('{')):
                try:
                    record[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass

        if self.transform:
            return self.transform(record)

        label = record['label']
        data = record['embeddings']

        if self.graph_version:
            data = self.graph_generator.generate_graph(data)

        return data, label

    def close(self):
        self.db.close_connection()