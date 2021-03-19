import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class HighD_Dataset(Dataset):
    # 继承Dataset, 重载了__init__, __getitem__, __len__

    def __init__(self, file_path, time_len):
        self.df = pd.read_csv(file_path)
        self.time_len = time_len
        self.max_nodes = self.df.groupby("frame").count()['id'].max()

    def __getitem__(self, index):
        relative_data = self.df[(self.df['frame'] >= index) & (
            self.df['frame'] < index+self.time_len)]
        id_set = set(relative_data['id'])
        id_map = {}
        for idx, _id in enumerate(id_set):
            id_map[_id] = idx
        A = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        X = np.zeros((self.time_len, self.max_nodes, 8), dtype=np.float32)

        for i in range(len(relative_data)):

            x = relative_data.iloc[i]['id']
            y = set(relative_data.iloc[i][10:17])-{0}
            A[id_map[x]][id_map[x]] = 1
            for yy in y:
                A[id_map[x]][id_map[yy]] = 1
                A[id_map[yy]][id_map[x]] = 1

            X[relative_data.iloc[i]['frame']-index, id_map[x], 0:7] = relative_data.iloc[i][2:9]
            X[relative_data.iloc[i]['frame']-index, id_map[x], 7] = 1 if relative_data.iloc[i][9] == 'Car' else -1

        Y = np.copy(X[self.time_len-1,...])
        X[self.time_len-1, np.random.randint(len(id_map)), :] = 0
        return A, X, Y

    def __len__(self):
        return np.max(self.df['frame'])-self.time_len+1


if __name__ == "__main__":
    HighD_dataset = HighD_Dataset('data/highD/36.csv', time_len=6)
    HighD_dataloader = DataLoader(HighD_dataset, batch_size=1, shuffle=False, num_workers=8)

    for i, (A, X, Y) in enumerate(HighD_dataloader):
        if (i==2):
            print(X, Y)
