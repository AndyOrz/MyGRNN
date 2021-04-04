import numpy as np
import pandas as pd
# from torch.utils.data import DataLoader, Dataset
from dgl.data import DGLDataset
import dgl
import torch
import pickle


class HighD_Dataset(DGLDataset):

    def __init__(self, X_len, X_step, Y_len, Y_step, diff=0, mask_rate=0.2,
                 url=None,
                 name=None, #data_22
                 raw_dir=None, #'./data/HighD/'
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(HighD_Dataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

        '''
        输出数据集的初始化参数:
        X_len: X的采样点数目
        X_step: 采样点之间的时间片间隔 即实际经历帧数=X_len*X_step
        Y_len: X的采样点数目
        Y_step: 采样点之间的时间片间隔 即实际经历帧数=Y_len*Y_step
        diff: Y的起始时间相对于X晚的帧数

        mask_rate: 在Y中，屏蔽多少比例的数据点（至少屏蔽1个）
        '''
        self.X_len = X_len
        self.X_step = X_step
        self.diff = diff
        self.Y_len = Y_len
        self.Y_step = Y_step
        self.mask_rate = mask_rate

    def process(self):
        # 原始数据的性质
        f = open(self.raw_path+'.pkl','rb')
        self.graphs, self.id_maps, self.features = pickle.load(f)

    def __getitem__(self, index):
        '''
        输出：
        X:{"graph":[dgl.graph],"feature":torch.Tensor(in_seq_len,num_nodes,num_feats)}
        Y:{"graph":[dgl.graph],"feature":torch.Tensor(out_seq_len,num_nodes,num_feats)}
        mask:torch.Tensor(in_seq_len,num_nodes,num_feats)
        '''

        X_indexs = [i*self.X_step+index for i in range(self.X_len)]
        Y_indexs = [i*self.Y_step+index+self.diff for i in range(self.Y_len)]

        # 获取所有用到的帧的车辆ID
        node_set = set()
        for i in list(set.union(set(X_indexs),set(Y_indexs))):
            node_set = set.union(node_set,set(self.id_maps[i]))
        num_nodes = len(node_set)
        id_map = {}
        for idx, _id in enumerate(node_set):
            id_map[_id] = idx

        num_feats = 4 #暂时只用到x,y坐标作为features
        X = {"graph":[],"feature":torch.zeros((self.X_len,num_nodes,num_feats))}
        Y = {"graph":[],"feature":torch.zeros((self.Y_len,num_nodes,num_feats))}

        # 将graph复制到X,Y,GT中，补全graph节点，将features填入对应位置
        for seq_idx, frame_id in enumerate(X_indexs):
            new_ids = [id_map[id] for id in self.id_maps[frame_id]]
            edges = self.graphs[frame_id].adj().coalesce().indices()
            new_edges = edges.clone()
            # 对节点id进行更新
            for old_id, new_id in enumerate(new_ids):
                new_edges[edges==old_id] = new_id
                X["feature"][seq_idx,new_id,:]=torch.from_numpy(self.features[frame_id][old_id,0:4])
            X["graph"].append(dgl.graph((new_edges[0],new_edges[1]),num_nodes=num_nodes))

        for seq_idx, frame_id in enumerate(Y_indexs):
            new_ids = [id_map[id] for id in self.id_maps[frame_id]]
            edges = self.graphs[frame_id].adj().coalesce().indices()
            new_edges = edges.clone()
            # 对节点id进行更新
            for old_id, new_id in enumerate(new_ids):
                new_edges[edges==old_id] = new_id
                Y["feature"][seq_idx,new_id,:]=torch.from_numpy(self.features[frame_id][old_id,0:4])
            Y["graph"].append(dgl.graph((new_edges[0],new_edges[1]),num_nodes=num_nodes))


        # 对Y生成mask
        np.random.seed(2021)
        mask = np.random.random(size=X["feature"].shape)
        mask[mask>self.mask_rate] = 1
        mask[mask<self.mask_rate] = 0
        # TODO:判断是否至少有一个mask


        return X, Y, mask

    def __len__(self):
        return len(self.graphs)-(self.diff+(self.Y_len-1)*self.Y_step+1)+1


if __name__ == "__main__":
    from dgl.dataloading import GraphDataLoader
    import networkx as nx
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    HighD_dataset = HighD_Dataset(X_len=20,X_step=1,Y_len=20,Y_step=2,diff=5,name='data_22',raw_dir='./')
    HighD_dataloader = GraphDataLoader(HighD_dataset, batch_size=1, shuffle=False)
    print("Dataset Ready!")

    with tqdm(total=len(HighD_dataloader)) as pbar:
        for i, (X,Y,mask) in enumerate(HighD_dataloader):
            pbar.set_postfix({"mask_shape":mask.shape})
            pbar.update(1)
        # if (i==466):
        #     print(X["feature"][0,10])
        #     print(mask[0,10])
        #     nx.draw(X["graph"][10].to_networkx(), with_labels=True)
        #     plt.show()
 