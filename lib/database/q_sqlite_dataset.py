import sqlite3
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm
import torch

def preprocess_dataframe(df):
    """
    Group-aware preprocessing for panel data with warning fixes:
    - Convert percentage-like strings to floats
    - Cast TICKER and cusip to categorical
    - Parse date columns
    - Drop numeric columns >30% missing overall
    - Median fill for mid-missing numeric cols within each TICKER
    - Linear interpolate for low-missing numeric cols within each TICKER
    - Forward/backward fill numeric cols per TICKER (explicit ffill/bfill)
    """
    df = df.copy()

    # 0) Convert percent strings
    for col in df.select_dtypes(include=['object']):
        if df[col].astype(str).str.endswith('%').any():
            df[col] = (df[col].str.rstrip('%')
                       .replace('', np.nan)
                       .astype(float)
                       .divide(100))

    # 1) Cast IDs to category early
    for cat in ['TICKER', 'cusip']:
        if cat in df.columns:
            df[cat] = df[cat].astype('category')

    # 2) Parse dates
    for col in ['adate', 'qdate', 'public_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3) Drop sparse numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    miss = df[num_cols].isna().mean() * 100
    drop = miss[miss > 30].index.tolist()
    df.drop(columns=drop, inplace=True)

    # 4) Identify mid/low missing
    num_cols = df.select_dtypes(include=[np.number]).columns
    miss = df[num_cols].isna().mean() * 100
    mid = miss[(miss > 5) & (miss <= 30)].index.tolist()
    low = miss[miss <= 5].index.tolist()

    # 5) Impute per TICKER
    out = []
    for _, grp in tqdm(df.groupby('TICKER', observed=False),
                        desc='Imputing per TICKER',
                        total=df['TICKER'].nunique()):
        g = grp.sort_values('qdate')
        for c in mid:
            if c in g and g[c].notna().any():
                g[c] = g[c].fillna(g[c].median())
        for c in low:
            if c in g:
                g[c] = g[c].interpolate(method='linear',
                                         limit_direction='both')
        nums = g.select_dtypes(include=[np.number]).columns
        g[nums] = g[nums].ffill().bfill()
        out.append(g)

    df = pd.concat(out, ignore_index=True)
    return df


def get_date_n_days_before(date_str, n_days):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return (dt - timedelta(days=n_days)).strftime('%Y-%m-%d')


class QSQLiteDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        table_name: str,
        date_column: str,
        q_data_path: str,
        n_days: int = 365,
        seq_len: int = 12,
        start_date: str = None,
        end_date: str = None,
        transform=None,
        graph_version: bool = False
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.date_column = date_column
        self.n_days = n_days
        self.seq_len = seq_len
        self.transform = transform
        self.graph_version = graph_version

        # Load and preprocess
        df = pd.read_csv(q_data_path)
        print('Preprocessing panel data...')
        self.q_df = preprocess_dataframe(df)
        print('Done preprocessing.')

        # Build rowid list
        tickers = self.q_df['TICKER'].astype(str).unique().tolist()
        clauses, params = [], []
        if start_date:
            clauses.append(f"{date_column} >= ?"); params.append(start_date)
        if end_date:
            clauses.append(f"{date_column} < ?");  params.append(end_date)
        placeholders = ','.join('?' for _ in tickers)
        clauses.append(f"ticker IN ({placeholders})"); params.extend(tickers)
        where = ' AND '.join(clauses)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(f"SELECT rowid FROM {self.table_name} WHERE {where};", params)
        self.rowids = [r[0] for r in cur.fetchall()]
        self.conn, self.cursor = conn, cur

    def __len__(self):
        return len(self.rowids)

    def _build_edge_index(self, x_arr: np.ndarray):
        N = x_arr.shape[0]
        edges = [[i, i+1] for i in range(N-1)] + [[i+1, i] for i in range(N-1)]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __getitem__(self, idx: int):
        rid = self.rowids[idx]
        self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE rowid = ?;", (rid,))
        cols = [d[0] for d in self.cursor.description]
        rec = dict(zip(cols, self.cursor.fetchone()))

        # decode JSON fields
        for k, v in rec.items():
            if isinstance(v, str) and v.strip().startswith(('[','{')):
                try: rec[k] = json.loads(v)
                except: pass

        # embeddings and panel data as numpy arrays
        seq = np.array(rec['embeddings'], dtype=np.float32)
        date_str = rec[self.date_column].split()[0]
        start_dt = get_date_n_days_before(date_str, self.n_days)

        panel_df = self.q_df.query(
            "TICKER == @rec['ticker'] and @start_dt <= public_date <= @date_str"
        )
        panel_vals = panel_df.drop(
            columns=['permno','gvkey','cusip','TICKER','adate','qdate','public_date'],
            errors='ignore'
        ).values.astype(np.float32)

        # pad/truncate
        if panel_vals.shape[0] >= self.seq_len:
            panel = panel_vals[-self.seq_len:]
        else:
            pad = np.zeros((self.seq_len - panel_vals.shape[0], panel_vals.shape[1]), np.float32)
            panel = np.vstack((pad, panel_vals))

        label = rec['label']

        if self.transform:
            return self.transform((seq, panel, label))

        if self.graph_version:
            data = Data(x=torch.tensor(seq), edge_index=self._build_edge_index(seq))
            return data, panel, label

        return seq, panel, label

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop('conn', None); st.pop('cursor', None)
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn, self.cursor = conn, conn.cursor()

    def close(self):
        self.conn.close()


# import sqlite3
# import json
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
#
#
# def preprocess_dataframe(df):
#     date_cols = ["adate", "qdate", "public_date"]
#     for col in date_cols:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
#
#     nan_ratios = df.isna().mean() * 100
#     high_nan = nan_ratios[nan_ratios > 30].index.tolist()
#     df.drop(columns=high_nan, inplace=True)
#
#     mid_nan = nan_ratios[(nan_ratios > 5) & (nan_ratios <= 30)].index.tolist()
#     for col in mid_nan:
#         if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
#             df[col].fillna(df[col].median(), inplace=True)
#
#     low_nan = nan_ratios[nan_ratios <= 5].index.tolist()
#     for col in low_nan:
#         if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].interpolate(method='linear', limit_direction='both')
#
#     df.fillna(method='ffill', inplace=True)
#     df.fillna(method='bfill', inplace=True)
#
#     for col in ['TICKER', 'cusip']:
#         if col in df.columns:
#             df[col] = df[col].astype('category')
#
#     return df
#
#
# def get_date_n_days_before(date_str, n_days):
#     date_obj = datetime.strptime(date_str, "%Y-%m-%d")
#     date_before = date_obj - timedelta(days=n_days)
#     return date_before.strftime("%Y-%m-%d")
#
#
# class QSQLiteDataset(Dataset):
#     def __init__(
#         self,
#         db_path,
#         table_name,
#         date_column,
#         q_data_path,
#         n_days=365,
#         seq_len=12,
#         start_date=None,
#         end_date=None,
#         transform=None,
#         graph_version=False
#     ):
#         self.db_path = db_path
#         self.table_name = table_name
#         self.date_column = date_column
#         self.n_days = n_days
#         self.seq_len = seq_len
#         self.transform = transform
#         self.graph_version = graph_version
#
#         self.q_df = pd.read_csv(q_data_path)
#         self.q_df = preprocess_dataframe(self.q_df)
#         self.ticker_list = self.q_df['TICKER'].astype(str).unique().tolist()
#
#         clauses = []
#         params = []
#         if start_date is not None:
#             clauses.append(f"{date_column} >= ?")
#             params.append(start_date)
#         if end_date is not None:
#             clauses.append(f"{date_column} < ?")
#             params.append(end_date)
#         placeholders = ','.join('?' for _ in self.ticker_list)
#         clauses.append(f"ticker IN ({placeholders})")
#         params.extend(self.ticker_list)
#         where_clause = " AND ".join(clauses)
#
#         conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         cursor = conn.cursor()
#         cursor.execute(f"SELECT rowid FROM {self.table_name} WHERE {where_clause};", params)
#         self.rowids = [r[0] for r in cursor.fetchall()]
#         self.conn = conn
#         self.cursor = cursor
#
#     def __len__(self):
#         return len(self.rowids)
#
#     def _build_edge_index(self, x_tensor):
#         N = x_tensor.shape[0]
#         edge_index = []
#         for i in range(N - 1):
#             edge_index.append([i, i + 1])
#             edge_index.append([i + 1, i])
#         return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#     def __getitem__(self, idx):
#         rowid = self.rowids[idx]
#         self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE rowid = ?;", (rowid,))
#         row = self.cursor.fetchone()
#         cols = [d[0] for d in self.cursor.description]
#         record = dict(zip(cols, row))
#
#         for k, v in record.items():
#             if isinstance(v, str) and (v.startswith('[') or v.startswith('{')):
#                 try:
#                     record[k] = json.loads(v)
#                 except json.JSONDecodeError:
#                     pass
#
#         sequences = record['embeddings']
#         label = record['label']
#         ticker = str(record['ticker'])
#         date_str = record[self.date_column].split(' ')[0] if ' ' in record[self.date_column] else record[self.date_column]
#
#         start_dt = get_date_n_days_before(date_str, self.n_days)
#         end_dt = date_str
#         q_sub = self.q_df.query("TICKER == @ticker and @start_dt <= public_date <= @end_dt").copy()
#
#         for drop_col in ['permno', 'gvkey', 'cusip', 'TICKER', 'adate', 'qdate', 'public_date']:
#             if drop_col in q_sub.columns:
#                 q_sub.drop(columns=[drop_col], inplace=True)
#         arr = q_sub.values.astype(np.float32)
#
#         if arr.shape[0] >= self.seq_len:
#             arr = arr[-self.seq_len:]
#         else:
#             pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
#             arr = np.vstack((pad, arr))
#
#         if self.transform:
#             return self.transform(record)
#
#         if self.graph_version:
#             x_tensor = torch.tensor(sequences, dtype=torch.float32)
#             edge_index = self._build_edge_index(x_tensor)
#             data = Data(x=x_tensor, edge_index=edge_index)
#             return data, arr, label
#         else:
#             return sequences, arr, label
#
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         state.pop('conn', None)
#         state.pop('cursor', None)
#         return state
#
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         self.conn = conn
#         self.cursor = conn.cursor()
#
#     def close(self):
#         self.conn.close()
