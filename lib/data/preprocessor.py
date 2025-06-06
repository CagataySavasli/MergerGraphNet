import pandas as pd
from lib.database.database_connector import DatabaseConnector
import datetime
import os


class Preprocessor:
    def __init__(self, data):

        self.data = data

        self.chunk_size = 3 #Mounth
        self.added_time = datetime.timedelta(self.chunk_size * (365/12))

        self.minimum_token_limit = 100

        self.last_accapted_date = None

        self.base_dir = os.path.dirname(os.path.abspath(__file__).split("lib")[0])

        self.db_path = os.path.join(self.base_dir, "data", "data.db")
        self.db = DatabaseConnector(self.db_path)

        self.path_ThomsonRouter = os.path.join(base_dir, "data", "auxiliary", "thomson_routers_merger")
        self.data_ThomsonRouter = self.load_trm_data()


    def preprocess(self):

        self.data.reset_index(drop=True, inplace=True)
        self.data.drop(self.data[(self.data['filing_date'] > self.last_accapted_date) & (self.data['label'] == 0)].index, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data['word_count'] = self.data['mda'].apply(lambda x: len(x.split()))
        self.data = self.data[self.data['word_count'] >= self.minimum_token_limit]
        self.data.drop(columns=['word_count'], inplace=True)
        self.data.drop_duplicates(keep='first', inplace=True)
        self.data.reset_index(drop=True, inplace=True)


    def load_trm_data(self):

        frame = []
        for file_name in sorted(os.listdir(self.path_ThomsonRouter)):
            if not file_name.endswith('.xlsx'): continue
            print(file_name)
            frame.append(pd.read_excel(self.path_ThomsonRouter+"/"+file_name))

        df_ThomReuters = pd.concat(frame, ignore_index=True)
        last_merge_date = sorted(df_ThomReuters['Date Announced'])[-1]
        self.last_accapted_date = last_merge_date - self.added_time
        print("Last Merge Date Annonced : ", self.last_accapted_date)

        return df_ThomReuters

    def label(self):

        self.data['filing_date'] = pd.to_datetime(self.data['filing_date'])

        labels = []
        for idx, row in self.data.iterrows():
            str_date = row['filing_date']
            end_date = row['filing_date'] + self.added_time
            ticker = row['ticker']

            mask = (self.data_ThomsonRouter['Date Announced'] >= str_date) & (self.data_ThomsonRouter['Date Announced'] <= end_date)
            tmp_ThomsonRouter = self.data_ThomsonRouter[mask]

            if ticker in list(tmp_ThomsonRouter['Acquiror Primary Ticker Symbol']):
                labels.append(1)
            elif ticker in list(tmp_ThomsonRouter['Target Primary Ticker Symbol']):
                labels.append(1)
            else:
                labels.append(0)

        self.data['label'] = labels


    def main(self):
        self.label()
        self.preprocess()
        self.db.create_table_with_dataframes("reports", self.data)
        # self.data.to_csv('./data/processed/reports.csv', index=False)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__).split("lib")[0])
    data_path = os.path.join(base_dir, "data", "raw", "reports.csv")

    data = pd.read_csv(data_path)

    preprocessor = Preprocessor(data)
    preprocessor.main()