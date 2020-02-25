import numpy as np
import pickle
import copy
import itertools
import random
from itertools import chain
import time
from meta_neural_net import MetaNeuralnet

from nasbench import api
from nas_bench.cell import Cell

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9

nasbench = api.NASBench('nasbench_only108.tfrecord')



def dist(u, v):
    return np.arccosh(1+2*np.linalg.norm(u-v)**2/((1-np.linalg.norm(u)**2)*(1-np.linalg.norm(v)**2)))

def proj(x, eps=1e-5):
    if np.linalg.norm(x) >= 1:
        x = x/np.linalg.norm(x)
        for i in range(len(x)):
            if x[i] >= eps:
                x[i] -= eps
            elif x[i] <= -eps:
                x[i] += eps
        return x
    else:
        return x

def dgrad(theta, x):
    alpha = 1 - np.linalg.norm(theta)**2
    beta = 1 - np.linalg.norm(x)**2
    gamma = 1 + 2/(alpha*beta)*np.linalg.norm(theta-x)**2
    return 4/(beta*(gamma**2-1)**(.5))*((np.linalg.norm(x)**2-2*np.dot(theta, x)+1)/alpha**2*theta-x/alpha)

def lgrad(init_grad, theta_u, ind_u, theta_v, ind_v, theta_N, ind_v_N):
    grad = init_grad
    N_sum = np.sum([np.exp(-dist(theta_u, theta_v_N)) for theta_v_N in theta_N])
    dist_u_v = dist(theta_u, theta_v)
    dgrad_u_v = dgrad(theta_u, theta_v)
    grad_u = (-1)*dgrad_u_v+\
             np.sum([np.exp(-dist(theta_u, theta_v_N))*dgrad(theta_u, theta_v_N) for theta_v_N in theta_N])/N_sum
    grad_u *= (1-np.linalg.norm(theta_u)**2)**2/4
    grad[ind_u][-1] = grad_u
    grad_v = (-1)*dgrad(theta_v, theta_u)
    grad_v *= (1-np.linalg.norm(theta_v)**2)**2/4
    grad[ind_u][ind_v] = grad_v
    for i, (k,j) in enumerate(ind_v_N):
        grad_v_N = np.exp(-dist(theta_u, theta_N[i]))*dgrad(theta_N[i], theta_u)/N_sum
        grad_v_N *= (1-np.linalg.norm(theta_u)**2)**2/4
        grad[k][j] = grad_v_N
    return grad

def poincare_embed(embed_data, epochs=1000000000, eps=1e-7, lr=.01, decay=.01):
    init_grad = []
    for x in embed_data:
        init_grad.append(np.zeros((len(x), 5)))
    init_grad = np.array(init_grad)
    Thetas = []
    for x in embed_data:
        Thetas.append(np.random.uniform(-1e-3, 1e-3, (len(x), 5)))
    Thetas = np.array(Thetas)
    for i in range(epochs):
        ind_u = random.choice(range(len(embed_data)))
        ind_v = random.choice(range(len(embed_data[ind_u])-1))
        ind_v_N = []
        while len(ind_v_N) < 10:
            ind_N = ind_u
            while ind_N == ind_u:
                ind_N = random.choice(range(len(embed_data)))
            ind_v_N.append((ind_N, random.choice(range(len(embed_data[ind_N])))))
            ind_v_N = list(set(ind_v_N))
        grad = lgrad(init_grad, Thetas[ind_u][-1], ind_u, Thetas[ind_u][ind_v], ind_v, [Thetas[i][j] for (i,j) in ind_v_N], ind_v_N)
        step = lr/(1+decay*i)*grad
        step_norm = np.linalg.norm(np.concatenate(step, axis=0))**2
        new_Thetas = Thetas - step
        if step_norm < eps:
            break
        for j in range(Thetas.shape[0]):
            Thetas[j] = np.array(list(map(proj, new_Thetas[j])))
    return Thetas


def unique(matrix, ops, archs):
    if len(archs) > 0:
        if (np.array([(matrix == x).all() and (ops == y) \
        for (x,y) in zip([d['matrix'] for d in archs],[d['ops'] for d in archs])])).any():
            return False
    return True

def get_v_D(u):
    D = []
    new_matrix = copy.deepcopy(u['matrix'])
    new_ops = copy.deepcopy(u['ops'])
    for i in range(42):
        new_matrix[i//7][i%7] = 1 - new_matrix[i//7][i%7]
        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            D.append({
                'matrix': new_matrix,
                'ops': new_ops
            })
        new_matrix[i//7][i%7] = 1 - new_matrix[i//7][i%7]
    for i in range(1,6):
        available = [o for o in OPS if o != new_ops[i]]
        for o in available:
            orig_ops = new_ops[i]
            new_ops[i] = o
            api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                D.append({
                    'matrix': new_matrix,
                    'ops': new_ops
                })
            new_ops[i] = orig_ops
    return D


def create_embed(num=1000):
    begin = time.time()
    embed_data = []
    new_embed_data = []
    for i in range(num):
        # create a random adjacency matrix
        u = Cell.random_cell(nasbench)
        # if matrix is unique create a set of matrices with edit distance of 'u' matrix equal to 1
        if unique(u['matrix'], u['ops'], list(chain.from_iterable(embed_data))):
            D = Cell(**u).get_v_D(nasbench)
            D.append(u)
            embed_data.append(D)
    Thetas = poincare_embed(embed_data)
    embed_data = list(chain.from_iterable(embed_data))
    Thetas = np.concatenate(Thetas, axis=0)
    ind = np.random.permutation(len(embed_data))
    for i in ind:
        val_loss = Cell(**embed_data[i]).get_val_loss(nasbench, deterministic=True)
        test_loss = Cell(**embed_data[i]).get_test_loss(nasbench)
        new_embed_data.append((embed_data[i], Thetas[i], val_loss, test_loss))
    meta_neuralnet = MetaNeuralnet()
    test_res = meta_neuralnet.cross_validate(np.array([d[1] for d in new_embed_data]), np.array([d[3] for d in new_embed_data]),5)
    end = time.time()
    print('time = {}'.format(end - begin))
    with open('embed.pkl', 'wb') as data:
        pickle.dump(new_embed_data, data)

create_embed(2)

# with open('embed.pkl', 'rb') as f:
#    embed_data = pickle.load(f)
