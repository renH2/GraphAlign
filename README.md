# GraphAlign: Can Modifying Data Address Graph Domain Adaptation?

## About

This repo is the official code for KDD-24 "Can Modifying Data Address Graph Domain Adaptation?".
<span id='introduction'/>

## Brief Introduction 
Examination reveals the limitations inherent to these model-centric methods, while a data-centric method that is allowed to modify the source graph provably demonstrates considerable potential.

- By revisiting the theoretical generalization bound for UGDA, we identify two data-centric principles for UGDA: alignment principle and rescaling principle. 
- Guided by these principles, we propose novel approach GraphAlign, that generates a small yet transferable graph. By exclusively training a GNN on this new graph with classic Empirical Risk Minimization (ERM), GraphAlign attains exceptional performance on the target graph.

For more technical details, kindly refer to the following links (to be continued)

[//]: # (<a href='https://ojs.aaai.org/index.php/AAAI/article/view/29156'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> )
[//]: # (<a href='https://underline.io/lecture/93719-measuring-task-similarity-and-its-implication-in-fine-tuning-graph-neural-networks-video'><img src='https://img.shields.io/static/v1?label=Video/Poster&message=underline&color=blue'></a> )
[//]: # ()

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#File structure'>1. File structure</a>
* <a href='#Environment dependencies'>2. Environment dependencies </a>
* <a href='#Usage'>3. Usage: How to run the code </a>
  * <a href='#Training GraphAlign'>3.1. Generate new source graph under two principles</a>
  * <a href='#Evaluating model'>3.2. Evaluating performance with ERM</a>


<span id='File structure'/>

##  1. File Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
├── data
│   ├── new
│   │   ├── acm
│   │   │   ├── acmv9.mat
│   │   ├── citation
│   │   │   └── citationv1.mat
│   │   └── dblp
│   │       ├── dblpv7.mat
│   └── old
│       ├── acm
│       │   ├── processed
│       │   │   ├── data.pt
│       │   │   ├── pre_filter.pt
│       │   │   └── pre_transform.pt
│       │   └── raw
│       │       ├── acm_docs.txt
│       │       ├── acm_edgelist.txt
│       │       └── acm_labels.txt
│       └── dblp
│           ├── processed
│           │   ├── data.pt
│           │   ├── pre_filter.pt
│           │   └── pre_transform.pt
│           └── raw
│               ├── dblp_docs.txt
│               ├── dblp_edgelist.txt
│               └── dblp_labels.txt
├── dual_gnn
│   ├── cached_gcn_conv.py
│   ├── dataset
│   │   ├── DomainDataNew.py
│   │   ├── DomainData.py
│   │   ├── __init__.py
│   │   ├── pre_cora.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── augmentation.py
│   │   ├── basic_gnn.py
│   │   └── SAGEEncoder.py
│   ├── ppmi_conv.py
├── data.py
├── generator.py
├── models
│   ├── gat.py
│   ├── gcn.py
│   ├── mygatconv.py
│   ├── mygraphsage.py
│   ├── parametrized_adj.py
│   ├── sgc_multi.py
│   └── sgc.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py

```

*****

Below, we will specifically explain the meaning of important file folders to help the user better understand the file structure.

`data.zip`: **need to unzip**, contains the data of "ACMv9 (A), DBLPv7 (D), Citationv1 (C)".

`dual_gnn`: contains the code for loading data

`models`: contains classic GNN models used in GraphAlign.

`utils`: contains the class definition for GraphAlign.

<span id='Environment dependencies'/>


## 2. Environment dependencies <a href='#all_catelogue'>[Back to Top]</a>

The script has been tested running under Python 3.9, with the following packages installed (along with their dependencies):
```
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
deeprobust==0.2.4
scikit_learn==1.0.2
```
Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

`pip install -r requirements.txt`


<span id='Usage'/>

## 3. Usage: How to run the code  <a href='#all_catelogue'>[Back to Top]</a>
GraphAlign paradigm consists of two stages: (1) Generate new source graph under two principles (2) Evaluate performance with ERM.
<span id='Training Graph-Align'/>

### 3.1.  Generate new source graph under two principles

To conduct GraphAlign, you can execute `train.py` as follows:

```bash
python train.py \
  --source <source dataset> \
  --target <target dataset> \
  --epoch <epoch for GraphAlign> \
  --alpha <coefficient for GraphAlign> \
  --gpu <gpu id>
```

For more detail, the help information of the main script `train.py` can be obtained by executing the following command.

```bash
python train.py -h

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID       gpu id
  --source SOURCE
  --target TARGET
  --savefile SAVEFILE  where to save the newly generated source graph
  --method METHOD      computation method for alignment principle (default: mmd-un)
  --init_method INIT_METHOD  initialization method 
  --surrogate SURROGATE   whether to use surrogate model for alignment principle (default: True)
  --epochs EPOCHS   epochs number (default: 500)
  --nlayers NLAYERS number of layers for GNN (default:2)
  --hidden HIDDEN  (default: 256)
  --lr_adj LR_ADJ  (default: 0.01)
  --lr_feat LR_FEAT  (default: 0.01)
  --lr_model LR_MODEL  (default: 0.01)
  --normalize_features NORMALIZE_FEATURES
  --reduction_rate REDUCTION_RATE  (default: 0.01)
  --seed SEED           Random seed.
  --alpha ALPHA         coefficient for regularization term (default: 30).
  --beta BETA    coefficient for alignment term (default: 30).
```

It is worth noting that the newly generated source graph is saved at the location specified by '--savefile'.
**Demo:**	

Using DBLPv7 as source graph and  Citationv1 (C) as target graph.
```bash
python -u train.py --source dblp --target citation --epoch 500 --dis_metric ours --alpha 30 --method mmd-un --gpu_id 0
```

<span id='Evaluating model'/>

### 3.2. Evaluate performance with ERM

`generate.py` file helps generate embeddings on a specific dataset. The help information of the main script `generate.py` can be obtained by executing the following command.

```bash
python generate.py -h

optional arguments:
  --load-path LOAD_PATH
  --dataset Dataset
  --gpu GPU  GPU id to use.
```
The embedding will be used for evaluation in node classification. The script `evaluate.sh` is available to simplify the evaluation process as follows: 

```
bash evaluate.sh <model_path> <name> <dataset> <gpu id>
```
Here, `<saved_path>` refers to the main directory for finetuning, and `<name>` is the name of specific model directory.

**Demo:**
Here is the demo instruction, after the user has trained using the demo provided above.
```
bash scripts/evaluate.sh saved path_bridge10_usa_airport usa_airport 0
```

## Contact
If you have any questions about the code or the paper, feel free to contact me.
Email: renh2@zju.edu.cn

## Cite
If you find this work helpful, please cite (to be continued)

[//]: # (```)

[//]: # (@article{huang2024measuring,)

[//]: # (  title={Can Modifying Data Address Graph Domain Adaptation?},)

[//]: # (  author={Huang, Renhong and Xu, Jiarong and Jiang, Xin and Pan, Chenglu and Yang, Zhiming and Wang, Chunping and Yang, Yang},)

[//]: # (  booktitle={AAAI},)

[//]: # (  volume={38},)

[//]: # (  number={11}, )

[//]: # (  pages={12617-12625},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (```)

## Acknowledgements
Part of this code is inspired by Jin et al.'s [GCond](https://github.com/ChandlerBang/GCond).