from threading import Lock, Thread
import sqlite3
import os

class DatabaseConnectorMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnector(metaclass=DatabaseConnectorMeta):
    def __init__(self, db_path):
        self.db_path = db_path

        parent_dir = os.path.dirname(self.db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def create_table(self, table_name, columns):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});")
        self.connection.commit()

    def insert_into_table(self, table_name, columns, values):
        self.cursor.execute(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(values))});", values)
        self.connection.commit()

    def create_table_with_dataframes(self, table_name, df):
        df['filing_date'] = df['filing_date'].dt.strftime('%Y-%m-%d')
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(df.columns)});")
        self.connection.commit()
        self.cursor.executemany(f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(df.columns))});", df.values)
        self.connection.commit()

    def insert_many(self, table_name, columns, rows):
        cols = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders});"
        self.cursor.executemany(sql, rows)
        self.connection.commit()

    def execute_query(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        self.connection.commit()

    def close_connection(self):
        self.connection.close()

    def __del__(self):
        self.close_connection()