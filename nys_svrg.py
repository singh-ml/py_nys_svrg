import time
import torch
import numpy as np
import sys
import scipy.io as spio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(0)

data = spio.loadmat(sys.argv[1])
X = data['X']
Y = data['Y']

n, d = X.shape
k = 20
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

x = torch.from_numpy(X).float().cuda()
y = torch.from_numpy(Y).float().cuda()
w = torch.randn(d, 1, requires_grad = True).cuda()

def solve_func(w, lam):
    f = (0.5 * torch.norm(torch.matmul(x, w)-y).pow(2)+lam*torch.matmul(w.t(),w)[0,0])/n
    return f

def gd(function, w, iteration=100, convergence=0.0001, learning_rate=torch.tensor(0.001,device='cuda'), lam = torch.tensor(0.01, device = 'cuda')):
    initial = w.clone()
    hist = [None]*iteration
    for i in range(iteration):
        previous_data = initial
        value = function(initial, lam)
        hist[i] = value
        idx = torch.randint(0, d, (k,)).cuda()
        grad = torch.autograd.grad(value, initial, create_graph = True)[0]
        initial -= learning_rate * grad
        print("epoch {}, obtain {}".format(i, value))
        if value < torch.tensor(convergence):
            print("break")
    return hist


def nys_svrg(function, w, iteration=100, convergence=0.0001, learning_rate=torch.tensor(0.001,device='cuda'), lam = torch.tensor(0.01, device = 'cuda'), rho = torch.tensor(1, device='cuda')):
    initial = w.clone()
    hist = [None]*iteration
    C=torch.zeros(k,d).cuda()
    for i in range(iteration):
        previous_data = initial
        value = function(initial, lam)
        hist[i] = value
        idx = torch.randint(0, d, (k,)).cuda()
        grad = torch.autograd.grad(value, initial, create_graph = True)[0]
        for j in range(k):
            C[j] = torch.autograd.grad(grad[idx[j]], initial, create_graph = True)[0].t()
        W=C[:,idx]
        u, s, v = torch.svd(W, some=True)
        #u = u.cuda()
        s = torch.diag(s)
        #print(s)
        r = torch.matrix_rank(s)
        s = s[:r,:r]
        u = u[:,:r]
        s = torch.sqrt(torch.inverse(s))
        Z = torch.matmul(C.t(),torch.matmul(u,s))
        #print(Z.device)
        Q = 1 / (rho * rho) * torch.matmul(Z, torch.inverse(torch.eye(r, device = 'cuda') + torch.matmul(Z.t(), Z) / rho))
        #print(Q.device)
        initial -= learning_rate * (grad / rho - torch.matmul(Q, torch.matmul(Z.t(), grad)))
        print("epoch {}, obtain {}".format(i, value))
        if value < torch.tensor(convergence):
            print("break")
    return hist

def newton(function, w, iteration=3, convergence=0.0001, learning_rate=torch.tensor(0.001,device='cuda'), lam = torch.tensor(0.01, device = 'cuda')):
    initial = w.clone()
    hist = [None]*iteration
    C=torch.zeros(d,d).cuda()
    for i in range(iteration):
        previous_data = initial
        value = function(initial, lam)
        hist[i] = value
        grad = torch.autograd.grad(value, initial, create_graph = True)[0]
        for j in range(d):
            C[j] = torch.autograd.grad(grad[j], initial, create_graph = True)[0].t()
            if j%100==0:
                print("hessian rows: ",j)
        initial -= learning_rate * torch.matmul(torch.inverse(C), grad)
        print("epoch {}, obtain {}".format(i, value))
        if value < torch.tensor(convergence):
            print("break")
    return hist

s=time.process_time()
hist = nys_svrg(solve_func, w, learning_rate = torch.tensor(float(sys.argv[2]), device = 'cuda'), lam = torch.tensor(float(sys.argv[3]), device = 'cuda'), rho = torch.tensor(float(sys.argv[4]), device = 'cuda'))
print('Nys_SVRG Time: ',time.process_time()-s)
plt.plot(np.log(hist))

s=time.process_time()
hist_gd = gd(solve_func, w, learning_rate = torch.tensor(float(sys.argv[2]), device = 'cuda'), lam = torch.tensor(float(sys.argv[3]), device = 'cuda'))
print('GD Time: ',time.process_time()-s)
plt.plot(np.log(hist_gd))

'''s=time.process_time()
hist_nm = newton(solve_func, w, learning_rate = torch.tensor(float(sys.argv[2]), device = 'cuda'), lam = torch.tensor(float(sys.argv[3]), device = 'cuda'))
print('Newton Time: ',time.process_time()-s)
plt.plot(np.log(hist_nm))'''

plt.show()
