import numpy as np
from torch_geometric.datasets import MNISTSuperpixels
from libs.utils_tf import *
from libs.utils import DegreeMaxEigTransform
   
#select if node degree and location of superpixel region would be used by model or not.
#after any chnageing please remove MNIST/processed folder in order to preprocess changes again.
transform=DegreeMaxEigTransform(adddegree=True,addposition=False,addmaxeig=False)

train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True, pre_transform=transform)
test_dataset = MNISTSuperpixels(root='dataset/MNIST', train=False, pre_transform=transform)

# nkernel+1 supports of GNNML3 to be prepared
nkernel=5
# receptive field and bandwidth parameter
recfield=3
dv=10

################

n=70000
nmax=75
# number of node per graph
ND=75*np.ones((n,1)) 
# node feature matrix
FF=np.zeros((n,nmax,2))
# one-hot coding output matrix 
YY=np.zeros((n,10))
# Convolution kernels, supports
SP=np.zeros((n,nkernel+1,nmax,nmax),dtype=np.float32)


d=train_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 

    FF[i,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i,gtrt]=1

    if recfield==0:
        M=A
    else:
        M=(A+np.eye(nd))
        for j in range(1,recfield):
            M=M.dot(M) 
    M=(M>0)

    with np.errstate(divide='ignore'):
        dis=1/np.sqrt(deg)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)

    vmax = 2.0
    vmin = 0.0
    freqcenter = np.linspace(vmin, vmax, nkernel)
    mid = (vmax + vmin) / 2.0
    half_width = (vmax - vmin) / 2.0
    B_tilde = (nL - mid * np.eye(nd)) / half_width

    num_probes = 10
    cheb_degree = 30
    z = np.random.choice([-1.0, 1.0], size=(nd, num_probes))
    T_vectors = np.zeros((cheb_degree + 1, nd, num_probes))
    T_vectors[0] = z
    if cheb_degree >= 1:
        T_vectors[1] = B_tilde.dot(z)
    for k in range(2, cheb_degree + 1):
        T_vectors[k] = 2 * B_tilde.dot(T_vectors[k-1]) - T_vectors[k-2]

    # design convolution supports (aka edge features)         
    from numpy.polynomial.chebyshev import Chebyshev
    for j in range(0,nkernel): 
        fc = freqcenter[j]
        def phi_s(lam):
            return np.exp(-dv * (lam - fc)**2)
        def phi_s_tilde(x):
            lam = x * half_width + mid
            return phi_s(lam)
        
        c = Chebyshev.interpolate(phi_s_tilde, deg=cheb_degree).coef
        filtered_z = np.zeros((nd, num_probes))
        for k in range(len(c)):
            filtered_z += c[k] * T_vectors[k]
            
        sp_layer = np.zeros((nd, nd))
        for r in range(num_probes):
            outer = np.outer(filtered_z[:, r], z[:, r])
            sp_layer += (M * outer)
        sp_layer /= num_probes
        
        SP[i, j, 0:nd, 0:nd] = sp_layer
        
    SP[i, nkernel, 0:nd, 0:nd]=np.eye(nd)
d=test_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 

    FF[i+60000,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i+60000,gtrt]=1

    if recfield==0:
        M=A
    else:
        M=(A+np.eye(nd))
        for j in range(1,recfield):
            M=M.dot(M) 
    M=(M>0)

    with np.errstate(divide='ignore'):
        dis=1/np.sqrt(deg)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)

    vmax = 2.0
    vmin = 0.0
    freqcenter = np.linspace(vmin, vmax, nkernel)
    mid = (vmax + vmin) / 2.0
    half_width = (vmax - vmin) / 2.0
    B_tilde = (nL - mid * np.eye(nd)) / half_width

    num_probes = 10
    cheb_degree = 30
    z = np.random.choice([-1.0, 1.0], size=(nd, num_probes))
    T_vectors = np.zeros((cheb_degree + 1, nd, num_probes))
    T_vectors[0] = z
    if cheb_degree >= 1:
        T_vectors[1] = B_tilde.dot(z)
    for k in range(2, cheb_degree + 1):
        T_vectors[k] = 2 * B_tilde.dot(T_vectors[k-1]) - T_vectors[k-2]

    # design convolution supports (aka edge features)         
    from numpy.polynomial.chebyshev import Chebyshev
    for j in range(0,nkernel): 
        fc = freqcenter[j]
        def phi_s(lam):
            return np.exp(-dv * (lam - fc)**2)
        def phi_s_tilde(x):
            lam = x * half_width + mid
            return phi_s(lam)
        
        c = Chebyshev.interpolate(phi_s_tilde, deg=cheb_degree).coef
        filtered_z = np.zeros((nd, num_probes))
        for k in range(len(c)):
            filtered_z += c[k] * T_vectors[k]
             
        sp_layer = np.zeros((nd, nd))
        for r in range(num_probes):
            outer = np.outer(filtered_z[:, r], z[:, r])
            sp_layer += (M * outer)
        sp_layer /= num_probes
        
        SP[i+60000, j, 0:nd, 0:nd] = sp_layer
        
    SP[i+60000, nkernel, 0:nd, 0:nd]=np.eye(nd)

np.save('supports',SP)
np.save('feats',FF)
np.save('output',YY)
np.save('nnodes',ND)


