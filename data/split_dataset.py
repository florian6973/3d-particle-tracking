import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


# use bead radius to estimate height?

class Dataset:
    def __init__(self, path: Path=None):
        if path is None:
            self.path = Path(__file__).parent / 'all/data_all.csv'
        else:
            self.path = path
        self.data = self.read_data()

        self.data_grouped = None
        self.train = None
        self.test = None
        self.val = None
        self.split_data()

        # self.data = self.data.drop(labels=['folder', 'path', 'nb'], axis=1)
        # self.data.to_csv(self.path.parent / 'data_all.csv', index=False)
        # self.train, self.test = self.split_data()
        # self.save_data()

    @staticmethod
    def reformat(path):
        dataset = Dataset(path)
        dataset.data = dataset.data.drop(labels=['Unnamed: 0', 'folder', 'path', 'nb'], axis=1)
        dataset.data.to_csv(dataset.path.parent / 'data_all.csv', index=False)

    @property
    def dataset_folder(self):
        return Path(self.path).parent

    def read_data(self):
        data = pd.read_csv(self.path) # index_col=0
        return data
    
    def group_data(self):
        data = self.data.groupby(
            ['type', 'n', 'v', 'i', 'f']).agg(
                {'top': 'sum'}
            ).reset_index()
        return data

    def split_data(self):
        self.data_grouped = self.group_data()
        X = self.data_grouped.index
        y = self.data_grouped['top'] > 0

        # n split for cross validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_index, test_global_index = next(sss.split(X, y))
        # print(len(train_index), len(test_global_index))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        test_index, val_index = next(sss2.split(X[test_global_index], y[test_global_index]))
        # print(len(test_index), len(val_index))

        self.train = self.data_grouped.iloc[train_index]
        self.test = self.data_grouped.iloc[test_index]
        self.val = self.data_grouped.iloc[val_index]

        # train = self.data.sample(frac=0.8, random_state=200)
        # test = self.data.drop(train.index)
        # return train, test

    # def save_data(self):
    #     self.train.to_csv(os.path.join(os.path.dirname(self.path), 'train.csv'), index=False)
    #     self.test.to_csv(os.path.join(os.path.dirname(self.path), 'test.csv'), index=False)

if __name__ == '__main__':
    # Dataset.reformat(Path(__file__).parent / 'all/data_all_beads.csv')
    dataset = Dataset()
    # print(dataset.data.head())
    # print(len(dataset.data_grouped[dataset.data_grouped['type'] == 1]))
    # print(dataset.test.head())
    # data = dataset.group_data()
    # # data.sort_values(by='top', ascending=False, inplace=True)
    # # print(data.head(100))
    # print(len(data[data['top'] > 0]))
    # print(len(data[data['top'] == 0]))