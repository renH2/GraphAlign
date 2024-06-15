import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_dense_adj
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import block_diag

def numpy2csr(array, ):
    values = array[np.nonzero(array)]
    row_indices, col_indices = np.nonzero(array)
    result_matrix = csr_matrix((values, (row_indices, col_indices)), shape=array.shape)
    return result_matrix

def tensor2csr(edge_index, ):
    coo = coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())))
    result_matrix = csr_matrix(coo)
    return result_matrix


class MyDataGraph:
    def __init__(self, dataset, source_data, target_data, **kwargs):
        if dataset in ["acm", "citation", "dblp", "paper"]:
            source_adj = to_dense_adj(source_data.edge_index)[0].numpy()
            target_adj = to_dense_adj(target_data.edge_index)[0].numpy()
            source_feature = source_data.x.numpy()
            target_feature = target_data.x.numpy()

            adj_full = block_diag(source_adj, target_adj)
            feature_full = np.concatenate([source_feature, target_feature], axis=0)
            adj_full = numpy2csr(adj_full)
            idx_train = list(range(source_adj.shape[0]))
            idx_test = [i + source_adj.shape[0] for i in range(target_adj.shape[0])]
            labels = np.concatenate([source_data.y.numpy(), target_data.y.numpy()], axis=0)
        elif dataset in ["cora"]:
            adj_full = to_dense_adj(source_data.edge_index)[0].numpy()
            adj_full = numpy2csr(adj_full)
            feature_full = source_data.x.numpy()
            idx_train = np.where(source_data["train_mask"].numpy() > 0)[0]
            idx_test = np.where(source_data["test_mask"].numpy() > 0)[0]
            labels = source_data.y.numpy()
        elif dataset in ["arxiv"]:
            adj_full = tensor2csr(source_data.edge_index)
            feature_full = source_data.x.numpy()
            idx_train = np.where(source_data["source_mask"] > 0)[0]
            idx_test = np.where(source_data["target_mask"] > 0)[0]
            labels = source_data.y.numpy()

        self.nnodes = adj_full.shape[0]
        if dataset == 'arxiv':
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        feat_train = feature_full[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feature_full)

        self.feat_train = feat[idx_train]
        self.feat_test = feat[idx_test]
        self.labels_train = labels[idx_train]
        self.labels_test = labels[idx_test]

        self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_test)
        self.class_dict = None
        self.class_dict2 = None
        self.class_dict_test = None

        self.adj_full = adj_full
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_test = np.array(idx_test)
        self.samplers = None
        self.test_samplers = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_target_sampler(self, c, adj, transductive, num=256, args=None):
        sizes = [10, 5]

        idx_test = np.array(self.idx_test)
        idx = idx_test

        if self.test_samplers is None:
            self.test_samplers = []
            node_idx = torch.LongTensor(idx)
            self.test_samplers.append(NeighborSampler(adj,
                                                      node_idx=node_idx,
                                                      sizes=sizes, batch_size=num,
                                                      num_workers=8, return_e_id=False,
                                                      num_nodes=adj.size(0),
                                                      shuffle=True))
        batch = np.random.permutation(idx)[:num]
        out = self.test_samplers[0].sample(batch)
        return out

    def retrieve_target_class_sampler(self, c, adj, transductive, num=256, args=None):
        sizes = [10, 5]

        idx_test = np.array(self.idx_test)
        idx = idx_test

        if self.class_dict_test is None:
            print(sizes)
            self.class_dict_test = {}
            for i in range(self.nclass):
                idx = np.arange(len(self.labels_test))[self.labels_test == i]
                self.class_dict_test[i] = idx

        if self.test_samplers is None:
            self.test_samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict_test[i])
                if len(node_idx) == 0:
                    continue
                self.test_samplers.append(NeighborSampler(adj,
                                                          node_idx=node_idx,
                                                          sizes=sizes, batch_size=num,
                                                          num_workers=8, return_e_id=False,
                                                          num_nodes=adj.size(0),
                                                          shuffle=True))
        batch = np.random.permutation(self.class_dict_test[c])[:num]
        out = self.test_samplers[c].sample(batch)
        return out

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]
        elif args.nlayers == 3:
            sizes = [10, 5, 5]

        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out


class GraphData:
    def __init__(self, adj, features, labels, idx_train, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_test = idx_test
