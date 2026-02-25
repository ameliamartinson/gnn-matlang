import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chebyshev_approx.cheb_utils import ChebyshevSpectralDesign
from libs.utils import GraphCountDataset
import torch
import numpy as np

transform = ChebyshevSpectralDesign(num_probes=10, cheb_degree=30, nmax=0, recfield=1, dv=2, nfreq=5, adddegree=True)
dataset = GraphCountDataset(root="dataset/subgraphcount/", pre_transform=transform)

for i in range(5):
    data = dataset[i]
    if torch.isnan(data.edge_attr2).any():
        print(f"Graph {i} has NaNs in edge_attr2!")
    if torch.isnan(data.x).any():
        print(f"Graph {i} has NaNs in x!")

print("Done checking.")
