# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
from dual_gnn.utils.switch_aug import switch_aug
from dual_gnn.utils.visual import visualization

import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
from scipy.stats import wasserstein_distance
import networkx as nx
from torch_geometric.utils.convert import to_networkx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--UDAGCN", type=bool,default=False)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--aug1",type=str,default='permE')
parser.add_argument("--aug_ratio1",type=float,default=0.5)

args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim



id = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed, use_UDAGCN,  encoder_dim)

print(id)



rate = 0.0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
print(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print(target_data)

# calculate sim between source graph and target graph 
# sim before aug
source_G_before = to_networkx(source_data)
# sim after aug 
source_data = switch_aug(source_data,args.aug1,args.aug_ratio1)
print(source_data)
source_G = to_networkx(source_data)
target_G = to_networkx(target_data)
source_data = source_data.to(device)
target_data = target_data.to(device)













class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, 128,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)

encoder = GNN(type="gcn").to(device)
if use_UDAGCN:
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)


cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


# att_model = Attention(encoder_dim).cuda()
att_model = Attention(encoder_dim).to(device)

models = [encoder, cls_model, domain_model]
if use_UDAGCN:
    models.extend([ppmi_encoder, att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=3e-3)


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_UDAGCN:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        outputs = att_model([gcn_output, ppmi_output])
        return outputs
    else:
        return gcn_output

def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy



epochs = 200
def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    if use_UDAGCN:
        # use domain classifier loss:
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + loss_grl

        # use target classifier loss:
        target_logits = cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy* (epoch / epochs * 0.01)


    else:
        loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
source_acc = []
target_acc = []

for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    source_acc.append(source_correct)
    target_correct = test(target_data, "target")
    target_acc.append(target_correct)
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch

# 度中心性
# dc1 = nx.degree_centrality(source_G)
# dc2 = nx.degree_centrality(target_G)
 
# # 介数中心性
# bc1 = nx.betweenness_centrality(source_G)
# bc2 = nx.betweenness_centrality(target_G)
 
# # 接近度中心性
# cc1 = nx.closeness_centrality(source_G)
# cc2 = nx.closeness_centrality(target_G)
 
# 特征向量中心性
# ec1 = nx.eigenvector_centrality(source_G)
# ec2 = nx.eigenvector_centrality(target_G)

print("=============================================================")
# dis_af = wasserstein_distance(nx.degree_histogram(source_G),nx.degree_histogram(target_G))
# dis_be = wasserstein_distance(nx.degree_histogram(source_G_before),nx.degree_histogram(target_G))
# dis_clu_af = wasserstein_distance(list(nx.clustering(source_G).values()),list(nx.clustering(target_G).values()))
# dis_deg_cen = wasserstein_distance(list(dc1.values()),list(dc2.values()))
# dis_bet_cen = wasserstein_distance(list(bc1.values()),list(bc2.values()))
# dis_clo_cen = wasserstein_distance(list(cc1.values()),list(cc2.values()))
# dis_ec_cen = wasserstein_distance(list(ec1.values()),list(ec2.values()))

# line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}, ratio: {}, dis_distan: {}, dis_af: {}, dis_clu_af: {}, dis_deg_cen:{}, dis_bet_cen:{}, dis_clo_cen:{}"\
#     .format(id, best_epoch, best_source_acc, best_target_acc,args.aug_ratio1,(dis_af-dis_be),dis_af,dis_clu_af,dis_deg_cen,dis_bet_cen,dis_clo_cen)
def laplacian_matrix(graph):
  # 求邻接矩阵
  A = np.array(nx.adjacency_matrix(graph).todense())
  A = -A
  for i in range(len(A)):
  	# 求顶点的度
    degree_i = graph.degree(i) # 节点编号从1开始，若从0开始，将i+1改为i
    A[i][i] = A[i][i] + degree_i  
  return A

# source_L_bf = laplacian_matrix(source_G_before)
source_L = laplacian_matrix(source_G)
target_L = laplacian_matrix(target_G)
## source_L 是 ndarray 类型

# (eva_L_sour_bf, evt) = np.linalg.eig(source_L_bf)
(eva_L_sour,evt) = np.linalg.eig(source_L)
(eva_L_tar, evt) = np.linalg.eig(target_L)

# dis_bf = wasserstein_distance(eva_L_sour_bf,eva_L_tar)
dis_af = wasserstein_distance(eva_L_sour,eva_L_tar)
print(dis_af)
# dis_dis = dis_bf - dis_af
line = "best_source_acc: {}, best_target_acc: {},ratio: {},dis_L_af: {}".format(best_source_acc,best_target_acc,args.aug_ratio1,dis_af)

print(line)
print("=============================================================")
# visualization(source_acc,target_acc,args.aug_ratio1,'line')


