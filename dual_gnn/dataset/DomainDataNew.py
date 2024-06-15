import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from scipy.io import loadmat
import scipy.sparse as sp

class DomainDataNew(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,pre_filter=None):
        self.name = name
        super(DomainDataNew, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dblpv7.mat", "acmv9.mat", "citationv1.mat"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.name == 'dblp':
            path = self.root + "/dblpv7.mat"
        if self.name == 'acm':
            path = self.root + "/acmv9.mat"
        if self.name == 'citation':
            path = self.root + "/citationv1.mat"

        data = loadmat(path)

        A = data['network']
        edge_temp = sp.coo_matrix(A)
        self.edge_temp = edge_temp

        indices = np.vstack((edge_temp.row, edge_temp.col))
        edge_index = torch.LongTensor(indices)

        Y = data['group']
        Y = [np.argmax(one_hot) for one_hot in Y]
        Y = np.array(Y)
        Y = torch.from_numpy(Y).to(torch.int64)
        X = data['attrb'].todense()
        X = torch.FloatTensor(X)

        data_list = []
        data = Data(x=X, edge_index=edge_index, y=Y)

        random_node_indices = np.random.permutation(Y.shape[0])
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([Y.shape[0]], dtype=torch.uint8)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([Y.shape[0]], dtype=torch.uint8)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([Y.shape[0]], dtype=torch.uint8)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
