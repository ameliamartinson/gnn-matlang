import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GINConv, global_add_pool, GATConv, ChebConv, GCNConv)
import numpy as np

from libs.spect_conv import SpectConv, ML3Layer
from libs.utils import SRDataset
from chebyshev_approx.cheb_utils import ChebyshevSpectralDesign


transform = ChebyshevSpectralDesign(nmax=25, recfield=1, dv=2, nfreq=5, adddegree=True, num_probes=10, cheb_degree=30)
dataset = SRDataset(root="dataset/sr25/", pre_transform=transform)
train_loader = DataLoader(dataset, batch_size=100, shuffle=False)

class PPGN(torch.nn.Module):
    def __init__(self, nmax=25, nneuron=32):
        super(PPGN, self).__init__()

        self.nmax = nmax        
        self.nneuron = nneuron
        ninp = dataset.data.X2.shape[1]
        
        bias = True
        self.mlp1_1 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias) 
        self.mlp1_2 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias) 
        self.mlp1_3 = torch.nn.Conv2d(nneuron+ninp, nneuron, 1, bias=bias) 

        self.mlp2_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias) 
        self.mlp2_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2*nneuron, nneuron, 1, bias=bias) 

        self.mlp3_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias) 
        self.mlp3_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias) 
        self.mlp3_3 = torch.nn.Conv2d(2*nneuron, nneuron, 1, bias=bias) 
        
        self.h1 = torch.nn.Linear(2*3*nneuron, 10)
        

    def forward(self, data):
        x = data.X2 
        M = torch.sum(data.M, (1), True) 

        x1 = F.relu(self.mlp1_1(x)*M) 
        x2 = F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x = F.relu(self.mlp1_3(torch.cat([x1x2, x], 1))*M) 
        xo1 = torch.cat([torch.sum(x*data.M[:,0:1,:,:], (2)), torch.sum(x*data.M[:,1:2,:,:], (2))], 1)

        x1 = F.relu(self.mlp2_1(x)*M) 
        x2 = F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x = F.relu(self.mlp2_3(torch.cat([x1x2, x], 1))*M)        
        xo2 = torch.cat([torch.sum(x*data.M[:,0:1,:,:], (2)), torch.sum(x*data.M[:,1:2,:,:], (2))], 1)

        x1 = F.relu(self.mlp3_1(x)*M) 
        x2 = F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x = F.relu(self.mlp3_3(torch.cat([x1x2, x], 1))*M) 
        xo3 = torch.cat([torch.sum(x*data.M[:,0:1,:,:], (2)), torch.sum(x*data.M[:,1:2,:,:], (2))], 1)
        
        x = torch.cat([xo1, xo2, xo3], 1)  
        x = torch.sum(x, 2)              
        x = torch.tanh(self.h1(x))
        return x

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for part1 and part2
        nout1 = 32
        nout2 = 16

        nin = nout1 + nout2
        ne = dataset.data.edge_attr2.shape[1]
        ninp = dataset.num_features

        self.conv1 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=ninp, nout1=nout1, nout2=nout2)
        self.conv2 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=nin, nout1=nout1, nout2=nout2)
        self.conv3 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=nin, nout1=nout1, nout2=nout2) 

        self.fc1 = torch.nn.Linear(nin, 10)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index2
        edge_attr = data.edge_attr2

        x = (self.conv1(x, edge_index, edge_attr))
        x = (self.conv2(x, edge_index, edge_attr))
        x = (self.conv3(x, edge_index, edge_attr))  

        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x

M = 0
for iter in range(0, 10):
    torch.manual_seed(iter)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate with PPGN first (baseline which fails)
    # To test if Cheb. preprocessing ruins expressive power for GNNML3, 
    # we should actually run GNNML3 instead of PPGN since GNNML3 is what uses edge_attr2.
    # Note: sr25 defaults to PPGN in original script. Let's run GNNML3 since we care about its preproc.
    model = GNNML3().to(device)   

    embeddings = []
    model.eval()
    for data in train_loader:
        data = data.to(device)
        pre = model(data)
        embeddings.append(pre)

    E = torch.cat(embeddings).cpu().detach().numpy()    
    M = M + 1*((np.abs(np.expand_dims(E, 1)-np.expand_dims(E, 0))).sum(2) > 0.001)
    sm = ((M == 0).sum() - M.shape[0]) / 2
    print(f'Iter {iter} - similar: {sm}')
