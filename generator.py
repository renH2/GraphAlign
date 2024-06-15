import torch.nn.functional as F
from utils import match_loss, regularization, DataGraphSAINT
import deeprobust.graph.utils as utils
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from torch_sparse import SparseTensor
from tqdm import tqdm
import time
from dual_gnn.models.augmentation import *
import os
from torch_geometric.nn.inits import uniform
import scipy.sparse as sp
from scipy.special import expit
from scipy.sparse import csr_matrix
from collections import Counter
from torch_geometric.data import Data
import random
import scipy.linalg as spl
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances


def calculate_hop_laplacian_target(graph, graph1, train_candidate, source_num, all_num, class_sample_num):
    distance_list = []
    source_list = []
    target_list = []
    source_index_list = []
    for i in tqdm(range(1000)):
        # source_nd = random.randint(0, source_num)
        source_nd = int(random.choice(train_candidate))
        source_p = k_hop_subgraph(source_nd, 2, graph.edge_index, relabel_nodes=True)
        source_node_set, source_edge_index = source_p[0], source_p[1]
        source_index_list.append(source_nd)

        target_nd = random.randint(source_num, all_num - 1)
        target_p = k_hop_subgraph(target_nd, 2, graph1.edge_index, relabel_nodes=True)
        target_node_set, target_edge_index = target_p[0], target_p[1]

        source_sub_g = nx.Graph()
        source_sub_g.add_edges_from(source_edge_index.t().tolist())
        if check_empty(source_sub_g):
            continue
        source_L = nx.normalized_laplacian_matrix(source_sub_g)
        source_L = torch.from_numpy(source_L.todense()).float()
        # (eva_L_source, evt) = np.linalg.eig(source_L)
        # print(eva_L_source)
        eva_L_source, evt = spl.eigh(source_L)
        # print(eva_L_source)

        target_sub_g = nx.Graph()
        target_sub_g.add_edges_from(target_edge_index.t().tolist())
        if check_empty(target_sub_g):
            continue
        target_L = nx.normalized_laplacian_matrix(target_sub_g)
        target_L = torch.from_numpy(target_L.todense()).float()
        # (eva_L_target, evt) = np.linalg.eig(target_L)
        eva_L_target, evt = spl.eigh(target_L)

        distance = wasserstein_distance(eva_L_source, eva_L_target)
        distance_list.append(distance)
        source_list.append(eva_L_source.tolist())
        target_list.append(eva_L_target.tolist())

    distance_node = []
    source_flat_list = [item for sublist in source_list for item in sublist]
    target_flat_list = [item for sublist in target_list for item in sublist]
    for sublist in source_list:
        distance_node.append(wasserstein_distance(sublist, target_flat_list))
    train_idx = np.argsort(distance_node)[:class_sample_num]
    return train_idx


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight).reshape(-1, 1)
        return torch.sum(x * h, dim=1)


class MyGenerator:
    def __init__(self, data: DataGraphSAINT, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.dataset = args.dataset

        n = int(len(data.idx_train) * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)
        self.num_class_dict = dict()
        self.syn_class_indices = dict()
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)

        self.max_performance = None
        self.max_adj_syn = None
        self.max_feat_syn = None
        self.max_labels_syn = None
        print('adj_syn:', (n, n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data: DataGraphSAINT) -> list:
        """The distribution of the synthetic labels is (almost) the same as the original label distribution.
        """
        counter = Counter(data.labels_train)
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])  # descending order.
        sum_ = 0
        labels_syn = []
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                self.num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + self.num_class_dict[c]]
                labels_syn += [c] * self.num_class_dict[c]
            else:
                self.num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += self.num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + self.num_class_dict[c]]
                labels_syn += [c] * self.num_class_dict[c]
        return labels_syn


    def test_with_train(self, iteration, verbose=True):
        res = []
        args = self.args
        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
            self.pge, self.labels_syn
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        if args.source == "cora":
            model = GCN(nfeat=data.feat_train.shape[1], nhid=256, dropout=0.3, weight_decay=2e-2, nlayers=2,
                        nclass=data.nclass, device=device).to(device)
        elif args.version == "old":
            model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden, dropout=0.2, weight_decay=5e-3, nlayers=5,
                        nclass=data.nclass, device=device, lr=0.001, plot=args.plot).to(device)
        else:
            model = GCN(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=dropout,
                        weight_decay=5e-4, nlayers=2,nclass=data.nclass, device=device).to(device)
        adj_syn = pge.inference(feat_syn)
        args = self.args
        noval = True
        model.fit_with_train(feat_syn, adj_syn, labels_syn, data, train_iters=600, normalize=True, verbose=False,
                             noval=noval)
        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        output = model.predict(data.feat_test, data.adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)

        res.append(acc_test.item())
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))
        if self.max_performance is None or self.max_performance < acc_test.item():
            self.max_performance = acc_test.item()
            self.max_adj_syn = adj_syn
            self.max_feat_syn = feat_syn
            self.max_labels_syn = labels_syn
        return res, loss_test.item(), acc_test.item()

    def save(self):
        args = self.args
        if not os.path.exists(f'./{args.savefile}'):
            os.makedirs(f'./{args.savefile}')
            print(f"Directory '{args.savefile}' created.")
        print("Max performance:",self.max_performance)
        torch.save(self.max_adj_syn,
                   f'./{args.savefile}/adj_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')
        torch.save(self.max_feat_syn,
                   f'./{args.savefile}/feat_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')
        torch.save(self.max_labels_syn,
                   f'./{args.savefile}/labels_{args.source}_{args.target}_{args.reduction_rate}_{args.seed}_{self.max_performance:.4f}.pt')

    def train(self, method=None, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels, adj_full, feat_full, labels_full = data.feat_train, data.adj_train, data.labels_train, data.adj_full, data.feat_full, data.labels_full
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        features_full, adj_full, labels_full = utils.to_tensor(feat_full, adj_full, labels_full, device=self.device)

        feat_sub, _ = self.get_sub_adj_feat(features, args.init_method)
        self.feat_syn.data.copy_(feat_sub)
        if self.dataset == "cora":
            adj = adj.to_sparse()
        if utils.is_sparse_tensor(adj):
            # adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        else:
            # adj_norm = utils.normalize_adj_tensor(adj)
            adj_full_norm = utils.normalize_adj_tensor(adj_full)

        adj_full = adj_full_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1], value=adj._values(),
                           sparse_sizes=adj.size()).t()
        # adj_full = SparseTensor(row=adj_full._indices()[0], col=adj_full._indices()[1], value=adj_full._values(),
        #                         sparse_sizes=adj_full.size()).t()

        outer_loop, inner_loop = get_loops(args)
        loss_list, acc_list = [], []
        for it in range(args.epochs + 1):
            if args.sgc == 1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False, device=self.device).to(self.device)
            elif args.sgc == 2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=args.dropout,
                             nlayers=args.nlayers, with_bn=False, device=self.device).to(self.device)
            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, device=self.device).to(self.device)

            if args.surrogate is True:
                model1 = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden, nclass=data.nclass, dropout=args.dropout,
                             nlayers=args.nlayers, with_bn=False, device=self.device).to(self.device)
                discriminator = Discriminator(256).to(self.device)
                discriminator.to()

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            if args.surrogate is True:
                model1.initialize()
                model1_parameters = list(model1.parameters()) + list(discriminator.parameters())
                optimizer_model1 = torch.optim.Adam(model1_parameters, lr=args.lr_model)
                model1.train()

            t1 = time.time()
            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)  # use MLP to construct the structure of the synthetic graph.
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name():
                        BN_flag = True
                if BN_flag:
                    model.train()
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()

                if args.surrogate is True:
                    for module in model1.modules():
                        if 'BatchNorm' in module._get_name():
                            BN_flag = True
                    if BN_flag:
                        model1.train()
                        for module in model1.modules():
                            if 'BatchNorm' in module._get_name():
                                module.eval()

                loss_mimic = torch.tensor(0.0).to(self.device)
                mmd_record = 0
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue
                    batch_size, n_id, adjs = data.retrieve_class_sampler(c, adj, transductive=False, num=256, args=args)
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_source = F.nll_loss(output, labels[n_id[:batch_size]])
                    gradient_source = torch.autograd.grad(loss_source, model_parameters)
                    gradient_source = list((_.detach().clone() for _ in gradient_source))

                    ind = syn_class_indices[c]
                    adj_syn_norm_list = [adj_syn_norm] * (args.nlayers - 1) + [adj_syn_norm[ind[0]: ind[1]]]
                    output_generate = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_generate = F.nll_loss(output_generate, labels_syn[ind[0]: ind[1]])

                    gradient_generate = torch.autograd.grad(loss_generate, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss_mimic += coeff * match_loss(gradient_generate, gradient_source, args, device=self.device)

                if method == "mmd":
                    idx_target = torch.LongTensor(np.random.choice(data.idx_test, size=self.nnodes_syn)).to(self.device)
                    loss_alignment = MMD(self.feat_syn, features_full[idx_target])
                    loss_mimic += self.args.beta * loss_alignment
                elif method == "mmd-un":
                    output_1 = model1.forward_sampler(features[n_id], adjs)
                    output_syn_1 = model1.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_alignment = MMD(output_1, output_syn_1)
                    loss_mimic += self.args.beta * loss_alignment
                    mmd_record += (self.args.beta * loss_alignment).item()
                elif method == "mmd-sup":
                    output_1 = model.forward_sampler(features[n_id], adjs)
                    output_syn_1 = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_alignment = MMD(output_1, output_syn_1)
                    loss_mimic += self.args.beta * loss_alignment
                    mmd_record += (self.args.beta * loss_alignment).item()

                # TODO: regularize
                if args.alpha > 0:
                    loss_prop = args.alpha * regularization(adj_syn, self.tensor2onehot(labels_syn)).to(self.device)
                else:
                    loss_prop = torch.tensor(0).to(self.device)

                loss_all = loss_mimic + loss_prop

                # update synthetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss_all.backward()  # the gradients are based on the condensation loss.

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if ol == outer_loop - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()  # update gnn param

                    if args.surrogate is True:
                        optimizer_model1.zero_grad()
                        output_emb = model.forward_sampler(features[n_id], adjs)
                        shuffled_indices = torch.randperm(features[n_id].size(0))
                        output_emb_sf = model.forward_sampler(features[n_id][shuffled_indices], adjs)

                        positive_score = discriminator(output_emb, torch.mean(output_emb, dim=1))
                        negative_score = discriminator(output_emb_sf, torch.mean(output_emb_sf, dim=1))

                        func = nn.BCEWithLogitsLoss()
                        loss_infonce = func(positive_score, torch.ones_like(positive_score)) + func(
                            negative_score, torch.zeros_like(negative_score))
                        loss_infonce.backward()
                        optimizer_model1.step()

            if verbose and it % 5 == 0:
                _, loss, acc = self.test_with_train(it)
                loss_list.append(loss)
                acc_list.append(acc)

            if it % 10 == 0:
                print('Epoch {}, time gap: {:.4f}'.format(it, time.time() - t1))

    def ppr(self, adj, alpha=0.15, normalization="symmetric"):
        if sp.issparse(adj):
            adj = adj.toarray()
        elif isinstance(adj, np.ndarray):
            pass
        else:
            raise ValueError(f"adj tead)")
        eps = 1e-6
        deg = adj.sum(1) + eps
        deg_inv = np.power(deg, -1)

        num_nodes = adj.shape[0]
        if normalization == "right":
            M = np.eye(num_nodes) - (1 - alpha) * adj * deg_inv[:, None]
        elif normalization == "symmetric":
            deg_inv_root = np.power(deg_inv, 0.5)
            M = (np.eye(num_nodes) - (1 - alpha) * deg_inv_root[None, :] * adj * deg_inv_root[:, None])

        return alpha * np.linalg.inv(M)

    def get_sub_adj_feat(self, features, init_method=None):
        data = self.data
        idx_selected = []

        counter = Counter(self.labels_syn.cpu().numpy())
        if init_method == "spec":
            # g = nx.from_scipy_sparse_array(data.adj_full)
            feat = features.cpu().numpy()
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            adj_mat = expit(np.dot(feat, feat.T))
            threshold = 0.52
            adj_mat[adj_mat <= threshold] = 0
            adj_mat[adj_mat > threshold] = 1
            matrix = csr_matrix(adj_mat)
            matrix.eliminate_zeros()
            coo_mat = matrix.tocoo()
            edge_index = torch.tensor([coo_mat.row, coo_mat.col], dtype=torch.long)
            g = Data(edge_index=edge_index)
            coo_mat1 = data.adj_full.tocoo()
            edge_index1 = torch.tensor([coo_mat1.row, coo_mat1.col], dtype=torch.long)
            g1 = Data(edge_index=edge_index1)
            for c in range(data.nclass):
                train_idx = np.arange(len(data.labels_train))
                data.class_dict = {}
                for i in range(data.nclass):
                    data.class_dict['train_class_%s' % i] = (data.labels_train == i)
                train_idx = train_idx[data.class_dict['train_class_%s' % c]]

                tmp = calculate_hop_laplacian_target(g, g1, train_idx, features.shape[0], data.adj_full.shape[0],
                                                     counter[c])
                idx_selected = idx_selected + tmp.tolist()
            idx_selected = np.array(idx_selected).reshape(-1)
            features = features[idx_selected]
        elif init_method == "pprmax":
            ppr_vector = self.ppr(data.adj_train)
            ppr_dist = pairwise_distances(ppr_vector)
            ppr_final = np.sum(ppr_dist, axis=1)
            for c in range(data.nclass):
                train_idx = np.arange(len(data.labels_train))
                data.class_dict = {}
                for i in range(data.nclass):
                    data.class_dict['train_class_%s' % i] = (data.labels_train == i)
                train_idx = train_idx[data.class_dict['train_class_%s' % c]]
                sorted_indices = np.argsort(ppr_final[train_idx])[-counter[c]:]
                tmp = train_idx[sorted_indices]
                idx_selected = idx_selected + tmp.tolist()
            idx_selected = np.array(idx_selected).reshape(-1)
            features = features[idx_selected]
        else:
            for c in range(data.nclass):
                tmp = data.retrieve_class(c, num=counter[c])
                tmp = list(tmp)
                idx_selected = idx_selected + tmp
            idx_selected = np.array(idx_selected).reshape(-1)
            features = features[idx_selected]
        adj_knn = 0
        return features, adj_knn

    def tensor2onehot(self, labels):
        """Convert label tensor to label onehot tensor.
        """
        labels = labels.long()
        eye = torch.eye(labels.max() + 1).to(labels.device)
        onehot_mx = eye[labels]
        return onehot_mx.to(labels.device)


def get_loops(args):
    if args.one_step:
        return 10, 0
    if args.dataset in ['arxiv']:
        return 10, 0
    if args.dataset in ['cora']:
        return 20, 10
    return args.outer, args.inner
