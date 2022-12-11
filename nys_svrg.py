import time
import torch
import numpy as np
import sys
import scipy.io as spio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(0)
device='cuda:3'

data = spio.loadmat(sys.argv[1])
X = data['X']
Y = data['Y']

n, d = X.shape
k = 5
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

x = torch.from_numpy(X).float().cuda(device)
y = torch.from_numpy(Y).float().cuda(device)
w = torch.randn(d, 1, requires_grad = True, device=device)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


def gd(w, iteration=10, convergence=0.0001, \
	learning_rate=torch.tensor(0.001,device=device), \
	lam = torch.tensor(0.001, device = device)):
    initial = w.clone()
    hist = [None]*(1+iteration)
    time1 = [None]*(1+iteration)
    p=torch.sigmoid(torch.matmul(x,initial))
    loss=torch.nn.SoftMarginLoss()
    value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
    hist[0] = value
    time1[0] = 0
    print("epoch {}, obtain {} time {}".format(0, hist[0], time1[0]))
    start = time.process_time()
    for i in range(iteration):
        g = torch.autograd.grad(value, initial, create_graph=True)[0]
        initial -= learning_rate * g
        p=torch.sigmoid(torch.matmul(x,initial))
        value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
        hist[i + 1] = value.clone().detach()
        time1[i + 1] = time.process_time()-start
        print("epoch {}, obtain {} time {}".format(i + 1, hist[i + 1], time1[i + 1]))
        if value < torch.tensor(convergence):
            print("break")
    return hist, time1

def nys_svrg(w, iteration=10, convergence=0.0001, \
	learning_rate=torch.tensor(0.01,device=device), \
	lam = torch.tensor(0.01, device = device), \
	rho = torch.tensor(1000, device=device)):
    initial = w.clone()
    hist = [None]*(1+iteration)
    time1 = [None]*(1+iteration)
    p=torch.sigmoid(torch.matmul(x,initial))
    loss=torch.nn.SoftMarginLoss()
    value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
    hist[0] = value
    time1[0] = 0
    print("epoch {}, obtain {} time {}".format(0, hist[0], time1[0]))
    start = time.process_time()
    C=torch.zeros(k,d).cuda(device)
    for i in range(iteration):
        idx = torch.randint(0, d, (k,)).cuda(device)
        grad = torch.autograd.grad(value, initial, create_graph = True)[0]
        for j in range(k):
            C[j]=torch.autograd.grad(grad[idx[j]], initial,  create_graph = True)[0].t()
        W=C[:,idx]+0.00001*torch.eye(k,k)
        u, s, v = torch.svd(W, some=True)
        s = torch.diag(s)
        r = torch.matrix_rank(s)
        s = s[:r,:r]
        u = u[:,:r]
        s = torch.sqrt(torch.inverse(s))
        Z = torch.matmul(C.t(),torch.matmul(u,s))
        Q = 1 / (rho * rho) * torch.matmul(Z, \
		torch.inverse(torch.eye(r, device = device) + \
		torch.matmul(Z.t(), Z) / rho))
        initial -= learning_rate * (grad / rho - torch.matmul(Q, torch.matmul(Z.t(), grad)))
        p=torch.sigmoid(torch.matmul(x,initial))
        value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
        hist[i + 1] = value
        time1[i + 1] = time.process_time()-start
        print("epoch {}, obtain {}, time {}".format(i + 1, hist[i + 1], time1[i + 1]))
        if value < torch.tensor(convergence):
            print("break")
    return hist, time1

def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
        if idx%100==0:
            print("hessian rows: ",idx)
    return hessian

def newton(w, iteration=1, convergence=0.0001, \
	learning_rate=torch.tensor(0.001,device=device), \
	lam = torch.tensor(0.01, device = device)):
    initial = w.clone()
    hist = [None]*(1+iteration)
    time1 = [None]*(1+iteration)
    p=torch.sigmoid(torch.matmul(x,initial))
    loss=torch.nn.SoftMarginLoss()
    value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
    hist[0] = value
    time1[0] = 0
    print("epoch {}, obtain {} time {}".format(0, hist[0], time1[0]))
    start = time.process_time()
    C=torch.zeros(d,d).cuda(device)
    for i in range(iteration):
        grad = torch.autograd.grad(value, initial, create_graph = True)[0]
        for j in range(d):
            C[j,:]=torch.autograd.grad(grad[j], initial, create_graph = True)[0].t()
            '''if j%100==0:
                print("hessian rows: {} time {}".format(j,time.process_time()-start))
        ''' #C=eval_hessian(grad,initial)
        initial -= learning_rate * torch.matmul(torch.inverse(C), grad)
        p=torch.sigmoid(torch.matmul(x,initial))
        value = loss(p,y)+lam*torch.sum(initial.norm(2)**2)
        hist[i + 1] = value
        time1[i + 1] = time.process_time()-start
        print("epoch {}, obtain {}, time {}".format(i + 1, hist[i + 1], time1[i + 1]))
        if value < torch.tensor(convergence):
            print("break")
    return hist, time1

for l in [0.001]:
        s=time.process_time()
        hist_gd, time_gd = gd(w, \
		iteration = 100, \
                learning_rate = torch.tensor(l, device = device), \
                lam = torch.tensor(0.001, device = device))
        print('GD Time: ',time.process_time()-s)
        plt.plot(time_gd, np.log(hist_gd), label = 'GD_'+str(l))

for l in [1]:
	s=time.process_time()
	hist, time1 = nys_svrg(w, \
                iteration = 100, \
		learning_rate = torch.tensor(l, device = device), \
		lam = torch.tensor(0.001, device = device), \
		rho = torch.tensor(1000, device = device))
	print('Nys_SVRG Time: ',time.process_time()-s)
	plt.plot(time1, np.log(hist), label = 'NGD_'+str(l))

for l in [1]:
	s=time.process_time()
	hist_nm, time_nm = newton(w, \
                iteration = 5, \
		learning_rate = torch.tensor(l, device = device), \
		lam = torch.tensor(0.001, device = device))
	print('Newton Time: ',time.process_time()-s)
	plt.plot(time_nm, np.log(hist_nm), label = 'Newton'+str(l))

plt.legend()
plt.show()


