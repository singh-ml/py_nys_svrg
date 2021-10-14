import time
import torch
import numpy as np
import sys
import scipy.io as spio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


torch.manual_seed(0)
device='cuda:2'

data = spio.loadmat(sys.argv[1])
x = torch.from_numpy(np.float32(data['x_train'])).cuda(device)
t = torch.from_numpy(np.float32(data['y_train'])).cuda(device)

n, d = x.shape
print(n,d)
#x=torch.randn(n,d,device=device)
#t=torch.sign(torch.randn(n,1,device=device)+0.1)
# w0=torch.randn(d,1,device=device,requires_grad=True)
w0=torch.matmul(torch.matmul(torch.inverse(torch.matmul(x.t(),x)+0.000001*torch.eye(d,d,device=device)),x.t()),t)

for eta in [0.6,0.5,0.4,0.3]:
    w=w0
    w.requires_grad=True
    loss=torch.nn.SoftMarginLoss()
    start=time.process_time()
    for j in range(11):
        y=torch.sigmoid(torch.matmul(x,w))
        l=loss(y,t)+0.001*torch.sum(w.norm(2)**2)
        print(j,l,time.process_time()-start)
        g=torch.autograd.grad(l,w,create_graph=True)[0]
        C=torch.zeros(d,d,device=device)
        for i in range(d):
            C[i]=torch.autograd.grad(g[i],w,create_graph=True)[0].t()
        w = w-eta*torch.matmul(torch.inverse(C.clone().detach()+0.000001*torch.eye(d,d,device=device)),g.clone().detach())
