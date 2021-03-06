import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

from minConf_PQN import minConf_PQN
from utils import unpackParam, projectParam, genConstraints, initParam

from objective import NegELBO, ELBO_terms

from scipy.optimize import check_grad
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import pickle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi

import time
import os

############################ Load Data Matrix ##############################

data_name = 'ThreeClusters'
X = scale(np.loadtxt("data/ThreeClusters_X.csv", delimiter=','))
Y_1 = np.loadtxt("data/ThreeClusters_Y_1.csv", delimiter=',')
Y_2 = np.loadtxt("data/ThreeClusters_Y_2.csv", delimiter=',')
K = 3

############################ Generate Constraints ##############################
# compute sizes of the data matrix
[N, D] = X.shape

# the number of pairwise constraints provided by each expert
ncon = 600
[num_ML, num_CL] = [ncon/2, ncon/2]
# accuracy parameters of two groups of experts
acc_G1 = [0.90, 0.85, 0.80, 0.75, 0.70]
acc_G2 = [0.70, 0.75, 0.80, 0.85, 0.90]
acc = [acc_G1, acc_G2]
# compute the number of experts and set the maximal number of expert groups 
M = len(acc_G1) + len(acc_G2)
G = M/2

# generate constraints
[alpha_G1, beta_G1] = [acc_G1, acc_G1]
[alpha_G2, beta_G2] = [acc_G2, acc_G2]
prng = np.random.RandomState(100)
# set random number generator
S_G1 = genConstraints(prng, Y_1, alpha_G1, beta_G1, num_ML, num_CL, \
        start_expert=0, flag_same=True)
S_G2 = genConstraints(prng, Y_2, alpha_G2, beta_G2, num_ML, num_CL, \
        start_expert=len(alpha_G1), flag_same=True)
S = np.vstack((S_G1, S_G2))

# compute the number of constraints generated by each expert to be used later
# for efficiency in computing the objective function
Ncon = []
for m in range(M):
    Ncon.append(num_ML+num_CL)

######################## Variational Parameter Settings ######################
prior = {'mu_w':0, 'sigma_w': 1, 'mu_b':0, 'sigma_b':1,\
        'tau_a1':10, 'tau_a2':1, 'tau_b1':10, 'tau_b2':1, 'gamma':1}

# define the objective f_param and gradient g_param
f_param = lambda param: NegELBO(param, prior, X, S, Ncon, G, M, K)

# apply autodifferentiation using python-autograd
g_param = grad(f_param)

# one evaluation to confirm the code works
param_test = np.random.rand( M*4 + G*(M+2+K*2) + G*K*(D*2) )
t1 = time.time()
fval = f_param(param_test)
t2 = time.time()
print "One function computation takes " + str(t2 - t1)
gval = g_param(param_test)
t3 = time.time()
print "One gradient computation takes " + str(t3 - t2)
# gradient check by comparing the autograd results to numerical results
flag_gradcheck = False
if flag_gradcheck == True:
    gval_err = check_grad(f_param, g_param, param_test)
    abs_avgerr = np.sqrt(gval_err**2/param_test.shape[0])
    print "Absolute_AvgError = " + str(abs_avgerr)
    print "Gradient: (Mean, Min, Max) = " + str(gval.mean()) + "    " + \
            str(gval.min()) + "    " + str(gval.max())
    print "Relative_AvgError = " + str(abs_avgerr/np.abs(gval).mean())
else:
    pass


######## Apply optimization based on L-BFGS with simple constraints ######
# parameter settings for the optimization
options = {'optTol':1e-3, 'progTol':1e-6, 'maxIter':200}

# create a set of initializations
# number of initializations
num_init = 20
# parameter of Dirichlet distribution when randomly initializing phi, eta
dir_param = 1
param_init_list = []
for idx_init in range(num_init):
    prng = np.random.RandomState(1000+idx_init)
    param_init_list.append(initParam(prior, X, N,D,G,M,K,dir_param,prng))

# create function that returns both function value and gradient value
funObj = lambda param: (f_param(param), g_param(param))

# create projection function that projects a point into the constraint set
funProj = lambda param: projectParam(param, N, D, G, M, K, lb=1e-12)

# return the optimal solution given any initialization
f_initParam = lambda param_init: minConf_PQN(funObj, param_init, funProj,\
        options=options)

# start timing the program
tStart = time.time()

result_set = []
for idx_init in range(num_init):
    print "=========================================================="
    print "Initialization ID: " + str(idx_init)
    print "=========================================================="
    result_set.append(f_initParam(param_init_list[idx_init]))
    
# select the optimal solution: (param_opt, fval_opt, funEvals_opt)
fval_opt = np.infty
for idx in np.arange(len(result_set)):
    item = result_set[idx]
    if item[1] < fval_opt:
        param_opt = item[0]            
        fval_opt = item[1]
        funEvals_opt = item[2]
        param_init_sel = param_init_list[idx]

# write the optimal solution to a pickle file
rs_opt = {'param_opt':param_opt, 'fval_opt':fval_opt, 'funEvals_opt':funEvals_opt, \
        'param_init':param_init_sel, 'data_name':data_name, 'NDKMG': [N,D,K,M,G], \
        'num_init':num_init, 'num_ML':num_ML, 'num_CL':num_CL, 'acc':acc,\
        'prior':prior, 'options':options, 'X':X, 'Y_1':Y_1, 'Y_2':Y_2, 'S':S}
basedir = "./result/"
if not os.path.exists(basedir):
    os.makedirs(basedir)

filename = basedir + data_name + "_N" + str(N) + "_D" + str(D) + \
        "_K" + str(K) + "_M" + str(M) + "_G" + str(G) + \
        "_Init" + str(num_init) + "_ML" + str(num_ML) + "_CL" + str(num_CL) +\
        ".pkl"
file_pkl = open(filename, "wb")
pickle.dump(rs_opt, file_pkl)
file_pkl.close()

# Analyze the result
# unpack the optimal parameter vector
[tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, sigma_w,\
        mu_b, sigma_b] = unpackParam(param_opt, N, D, G, M, K)

print "Optimal: fval = " + str(fval_opt)
print "Optimal: ELBO terms: ", ELBO_terms(param_opt, prior, X, S, Ncon, G, M, K)

print "Optimal: phi = ..."
print phi

phi_label = np.zeros((M, 1))
for m in np.arange(M):
    phi_label[m,0] = phi[m].argmax()
print "Optimal: phi_label = ..."
print phi_label

 
# Cluster samples using the mode of W and b
prob_GNK = np.zeros((G,N,K))
label_GN = np.zeros((N, G))
for g in np.arange(G):
    t1 = np.exp(np.dot(X, mu_w[g]) + mu_b[g])
    t2 = np.sum(t1, axis=1)
    prob_GNK[g] = np.dot(np.diag(1./t2), t1)
    label_GN[:,g] = np.argmax(prob_GNK[g], axis=1)
sim_labelWb_Y12 = np.zeros((G, 2))
for g in np.arange(G):
    sim_labelWb_Y12[g, 0] = nmi(label_GN[:,g], Y_1)
    sim_labelWb_Y12[g, 1] = nmi(label_GN[:,g], Y_2)
print "Optimal: sim_labelWb_Y12 = ..."
print sim_labelWb_Y12

# Cluster samples by integrating out W and b
label_bayes = np.zeros((N,G))
# number of W, b sample to use
num_Wb = 100
for g in np.arange(G):
    prob_mat = np.zeros((N,K))
    # sample from Wg and bg a few times
    for idx_Wb in np.arange(num_Wb):
        sample_w = (np.random.randn(D, K) * sigma_w[g]) + mu_w[g]
        sample_b = (np.random.randn(K) * sigma_b[g]) + mu_b[g]
        t1 = np.exp(np.dot(X, sample_w) + sample_b)
        t2 = np.sum(t1, axis=1)
        prob_mat = prob_mat + 1./num_Wb * np.dot(np.diag(1./t2), t1)
    label_bayes[:,g] = np.argmax(prob_mat, axis=1)
sim_labelBayes_Y12 = np.zeros((G,2))
for g in np.arange(G):
    sim_labelBayes_Y12[g, 0] = nmi(label_bayes[:,g], Y_1)
    sim_labelBayes_Y12[g, 1] = nmi(label_bayes[:,g], Y_2)
print "Optimal: sim_labelBayes_Y12 = ..."
print sim_labelBayes_Y12

tFinish = time.time()
print "Total Running Time : " + str(tFinish - tStart) + '  seconds'

