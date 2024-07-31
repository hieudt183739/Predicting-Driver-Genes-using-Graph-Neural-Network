import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn import metrics
from model_all import Net
# from my_model_all import Net
from sklearn import linear_model
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# fixed seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def load_data(path, threshold_sim):
    # load network
    network1 = []
    adj1 = sp.load_npz(path + "PP.adj.npz")      # gene-gene network
    adj2 = sp.load_npz(path + f"GG_{threshold_sim}.adj.npz")   # similarity gene-gene network
    adj3 = sp.load_npz(path + "PO.adj.npz")      # gene-outlying gene network
    adj4 = sp.load_npz(path + "PR.adj.npz")      # gene-miRNA network

    network1.append(adj1.tocsc())
    network1.append(adj2.tocsc())
    network1.append(adj3.tocsc())
    network1.append(adj4.tocsc())

    # netwroks for bilinear aggregation layer
    network2 = []
    adj5 = sp.load_npz(path + "O.adj_loop.npz")
    adj6 = sp.load_npz(path + "O.N_all.npz")

    network2.append(adj5.tocsc())
    network2.append(adj6.tocsc())

    # load node features
    l_feature = []      # gene
    feat1 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat1 = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(path + "G.feat-final.csv", sep=",").values[:, 1:]
    feat2 = torch.Tensor(feat2).to(device)
    feat3 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat3 = torch.Tensor(feat3).to(device)
    feat4 = pd.read_csv(path + "P.feat-pre.csv", sep=",").values[:, 1:]
    feat4 = torch.Tensor(feat4).to(device)

    l_feature.append(feat1)
    l_feature.append(feat2)
    l_feature.append(feat3)
    l_feature.append(feat4)

    r_feature = []
    feat5 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]       # gene
    feat5 = torch.Tensor(feat5).to(device)
    feat6 = pd.read_csv(path + "G.feat-final.csv", sep=",").values[:, 1:]       # gene
    feat6 = torch.Tensor(feat6).to(device)
    feat7 = pd.read_csv(path + "O.feat-final.csv", sep=",").values[:, 1:]       # outlying gene
    feat7 = torch.Tensor(feat7).to(device)
    feat8 = pd.read_csv(path + "R.feat-pre.csv", sep=",").values[:, 1:]         # miRNA
    feat8 = torch.Tensor(feat8).to(device)


    r_feature.append(feat5)
    r_feature.append(feat6)
    r_feature.append(feat7)
    r_feature.append(feat8)

    # load edge
    pos_edge = np.array(np.loadtxt(path + "PP_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()

    pb, _ = remove_self_loops(pos_edge)
    pos_edge1, _ = add_self_loops(pb)



    # label
    label = np.loadtxt(path + "label_file.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)

    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1, Y




