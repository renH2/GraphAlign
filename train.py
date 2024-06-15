import random
import argparse
from generator import MyGenerator
from data import MyDataGraph
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.dataset.DomainDataNew import DomainDataNew
from dual_gnn.models.augmentation import *
from Utils.pre_data import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--savefile", type=str, default='tune')
parser.add_argument("--version", type=str, default="new")
parser.add_argument("--method", type=str, default="mmd-un")
parser.add_argument("--init_method", type=str, default="pprmax")
parser.add_argument('--surrogate', type=bool, default=True)

parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=1.0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--one_step', type=int, default=0)
parser.add_argument("--beta", type=float, default=1.0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)

root = "./"
dataset_path = "./data"

if args.source in ["acm", "dblp", "citation"] and args.target in ["acm", "dblp", "citation"]:
    if args.version == "new":
        source_dataset = DomainDataNew(root + "data/new/{}".format(args.source), name=args.source)
        target_dataset = DomainDataNew(root + "data/new/{}".format(args.target), name=args.target)
    elif args.version == "old":
        source_dataset = DomainData(root + "data/old/{}".format(args.source), name=args.source)
        target_dataset = DomainData(root + "data/old/{}".format(args.target), name=args.target)
    dataset = source_dataset
    source_data = source_dataset[0]
    target_data = target_dataset[0]
    src_msk = None
    tgt_msk = None
elif args.source == "cora" or args.target == "cora":
    dataset = datasets.prepare_cora(
        dataset_path,
        args.domain_split, "covariate")
    source_data = dataset
    target_data = dataset
    src_msk = source_data.train_mask
    tgt_msk = target_data.test_mask

data = MyDataGraph(args.dataset, source_data, target_data)
data.nclass = dataset.num_classes
agent = MyGenerator(data, args, device='cuda:' + str(args.gpu_id))
agent.train(method=args.method)
agent.save()
