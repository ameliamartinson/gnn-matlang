import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from numpy.polynomial.chebyshev import Chebyshev

class ChebyshevSpectralDesign(object):   
    def __init__(self, nmax=0, recfield=1, dv=5, nfreq=5, adddegree=False, laplacien=True, addadj=False, num_probes=10, cheb_degree=30, vmax=2.0):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield = recfield  
        # b parameter (scale)
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien = laplacien
        # add adjacecny as edge feature
        self.addadj = addadj
        
        # Chebyshev specific parameters
        self.num_probes = num_probes
        self.cheb_degree = cheb_degree
        self.vmax = vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax    

    def __call__(self, data):
        n = data.x.shape[0]     
        nf = data.x.shape[1]  
        
        data.x = data.x.type(torch.float32)  
               
        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1
            
        A = np.zeros((n,n), dtype=np.float32)
        SP = np.zeros((nsup, n, n), dtype=np.float32) 
        A[data.edge_index[0], data.edge_index[1]] = 1
        
        if self.adddegree:
            data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field mask M
        if self.recfield == 0:
            M = np.copy(A)
        else:
            M = (A + np.eye(n))
            for i in range(1, self.recfield):
                M = M.dot(M) 
        M = (M > 0)
        
        d = A.sum(axis=0) 
        
        # We need the operator B. GNNML3 uses either Normalized Laplacian or Adjacency.
        if self.laplacien:
            with np.errstate(divide='ignore'):
                dis = 1 / np.sqrt(d)
            dis[np.isinf(dis)] = 0
            dis[np.isnan(dis)] = 0
            D = np.diag(dis)
            B = np.eye(n) - (A.dot(D)).T.dot(D)
            # For normalized laplacian, eigs are strictly in [0, 2]
            vmax = 2.0
            vmin = 0.0
        else:
            B = A
            # If not using Laplacian, we have to guess or compute vmax/vmin
            vmax = self.vmax
            # Approximate max eigenvalue using power iteration if not provided
            if vmax is None:
                z = np.random.rand(n)
                for _ in range(10): z = B.dot(z); z /= np.linalg.norm(z)
                vmax = z.T.dot(B.dot(z))
            vmin = -vmax # Usually adjacency is bound between -vmax, vmax

        data.lmax = float(vmax)
        
        # Define the center frequencies for the filter bank
        freqcenter = np.linspace(vmin, vmax, self.nfreq)
        
        # Normalize operator B to \tilde{B} with spectrum in [-1, 1]
        # x = (lambda - (vmax+vmin)/2) / ((vmax-vmin)/2)  --> x in [-1, 1]
        # so \tilde{B} = 2 * (B - I*(vmax+vmin)/2) / (vmax-vmin)
        mid = (vmax + vmin) / 2.0
        half_width = (vmax - vmin) / 2.0
        B_tilde = (B - mid * np.eye(n)) / half_width
        
        # To avoid dense matrix powers, we will only apply B_tilde to vectors
        # Generate random probe vectors (Rademacher +/- 1)
        z = np.random.choice([-1.0, 1.0], size=(n, self.num_probes))
        
        # Precompute the Chebyshev recurrence vectors T_k(B_tilde) @ z
        # T_0(x) = 1, T_1(x) = x, T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
        T_vectors = np.zeros((self.cheb_degree + 1, n, self.num_probes))
        T_vectors[0] = z
        if self.cheb_degree >= 1:
            T_vectors[1] = B_tilde.dot(z)
        for k in range(2, self.cheb_degree + 1):
            T_vectors[k] = 2 * B_tilde.dot(T_vectors[k-1]) - T_vectors[k-2]
            
        # For each spectral filter \Phi_s, compute the outer product trace estimator
        for s in range(len(freqcenter)):
            fc = freqcenter[s]
            
            # The filter function in the original domain
            def phi_s(lam):
                return np.exp(-self.dv * (lam - fc)**2)
                
            # Mapped to [-1, 1] domain
            def phi_s_tilde(x):
                lam = x * half_width + mid
                return phi_s(lam)
                
            # Interpolate Chebyshev coefficients for phi_s_tilde
            # Using numpy's Chebyshev interpolation
            c = Chebyshev.interpolate(phi_s_tilde, deg=self.cheb_degree).coef
            
            # Approximate Phi_s(B) @ z
            filtered_z = np.zeros((n, self.num_probes))
            for k in range(len(c)):
                filtered_z += c[k] * T_vectors[k]
                
            # Estimate matrix entries: (Phi_s(B))_{ij} \approx 1/R \sum_r (filtered_z)_i (z)_j
            # We only evaluate this over the mask M=1 to match GNNML3 sparse support!
            # SP[s, i, j] = M[i, j] * 1/num_probes * (filtered_z_i @ z_j)
            
            # For efficiency over the mask M: 
            # We can compute it explicitly for all i,j since graphs are typically small in GNNML3 (like max n=64)
            # If n is large, we'd only compute this for indices where M is True.
            
            for r in range(self.num_probes):
                # We only add to M-masked regions. We can use broadcasting.
                # filtered_z[:, r] is (n,), z[:, r] is (n,)
                # outer product is (n,1) * (1,n) = (n,n)
                outer = np.outer(filtered_z[:, r], z[:, r])
                SP[s] += (M * outer)
            SP[s] /= self.num_probes

        # add identity as the last (or second to last) support
        SP[len(freqcenter)] = np.eye(n)
        
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter) + 1] = A
           
        # set convolution support weigths as an edge feature
        E = np.where(M > 0)
        data.edge_index2 = torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)  

        # set tensor for Maron's PPGN         
        if self.nmax > 0:       
            H = torch.zeros(1, nf + 2, self.nmax, self.nmax)
            H[0, 0, data.edge_index[0], data.edge_index[1]] = 1 
            H[0, 1, 0:n, 0:n] = torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0, nf):      
                H[0, j+2, 0:n, 0:n] = torch.diag(data.x[:,j])
            data.X2 = H 
            M_tensor = torch.zeros(1, 2, self.nmax, self.nmax)
            for i in range(0, n):
                M_tensor[0, 0, i, i] = 1
            M_tensor[0, 1, 0:n, 0:n] = 1 - M_tensor[0, 0, 0:n, 0:n]
            data.M = M_tensor        

        return data
