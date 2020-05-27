""" This function implements the objective of variational inference for
<Multiple Clusterings from Uncertain Experts>
"""
import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad
from autograd.scipy.special import gamma, gammaln, digamma
from autograd.scipy.misc import logsumexp

from minConf_PQN import minConf_PQN
from utils import unpackParam, projectParam, genConstraints, initParam

def NegELBO(param, prior, X, S, Ncon, G, M, K):
    """
    Parameters
    ----------
    param: length (2M + 2M + MG + 2G + GNK + GDK + GDK + GK + GK) 
        variational parameters, including:
        1) tau_a1: len(M), first parameter of q(alpha_m)
        2) tau_a2: len(M), second parameter of q(alpha_m)
        3) tau_b1: len(M), first parameter of q(beta_m)
        4) tau_b2: len(M), second parameter of q(beta_m)
        5) phi: shape(M, G), phi[m,:] is the paramter vector of q(c_m)
        6) tau_v1: len(G), first parameter of q(nu_g)
        7) tau_v2: len(G), second parameter of q(nu_g)
        8) mu_w: shape(G, D, K), mu_w[g,d,k] is the mean parameter of 
            q(W^g_{dk})
        9) sigma_w: shape(G, D, K), sigma_w[g,d,k] is the std parameter of 
            q(W^g_{dk})
        10) mu_b: shape(G, K), mu_b[g,k] is the mean parameter of q(b^g_k)
        11) sigma_b: shape(G, K), sigma_b[g,k] is the std parameter of q(b^g_k)

    prior: dictionary
        the naming of keys follow those in param
        {'tau_a1':val1, ...}

    X: shape(N, D)
        each row represents a sample and each column represents a feature

    S: shape(n_con, 4)
        each row represents a observed constrain (expert_id, sample1_id,
        sample2_id, constraint_type), where
        1) expert_id: varies between [0, M-1]
        2) sample1 id: varies between [0, N-1]
        3) sample2 id: varies between [0, N-1]
        4) constraint_type: 1 means must-link and 0 means cannot-link

    Ncon: shape(M, 1)
        number of constraints provided by each expert

    G: int
        number of local consensus in the posterior truncated Dirichlet Process

    M: int
        number of experts

    K: int
        maximal number of clusters among different solutions, due to the use of
        discriminative clustering, some local solution might have empty
        clusters

    Returns
    -------
    """
    
    eps = 1e-12

    # get sample size and feature size
    [N, D] = np.shape(X)

    # unpack the input parameter vector
    [tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, sigma_w,\
            mu_b, sigma_b] = unpackParam(param, N, D, G, M, K)
    
    # compute eta given mu_w and mu_b
    eta = np.zeros((0, K))
    for g in np.arange(G):
        t1 = np.exp(np.dot(X, mu_w[g]) + mu_b[g])
        t2 = np.transpose(np.tile(np.sum(t1, axis=1), (K,1)))
        eta = np.vstack((eta, t1/t2))
    eta = np.reshape(eta, (G,N,K))

    # compute the expectation terms to be used later
    E_log_Alpha = digamma(tau_a1) - digamma(tau_a1 + tau_a2)  # len(M)
    E_log_OneMinusAlpha = digamma(tau_a2) - digamma(tau_a1 + tau_a2) # len(M)
    E_log_Beta = digamma(tau_b1) - digamma(tau_b1+tau_b2) # len(M)
    E_log_OneMinusBeta = digamma(tau_b2) - digamma(tau_b1+tau_b2) # len(M)

    E_log_Nu = digamma(tau_v1) - digamma(tau_v1 + tau_v2) # len(G)
    E_log_OneMinusNu = digamma(tau_v2) - digamma(tau_v1 + tau_v2) # len(G)
    E_C = phi # shape(M, G)
    E_W = mu_w # shape(G, D, K)
    E_WMinusMuSqd = sigma_w**2 + (mu_w - prior['mu_w'])**2 # shape(G, D, K)
    E_BMinusMuSqd = sigma_b**2 + (mu_b - prior['mu_b'])**2 # shape(G, K)
    E_ExpB = np.exp(mu_b + 0.5*sigma_b**2) # shape(G, K) 

    E_logP_Alpha = (prior['tau_a1']-1) * E_log_Alpha + \
            (prior['tau_a2']-1) * E_log_OneMinusAlpha -  \
            gammaln(prior['tau_a1']+eps) - \
            gammaln(prior['tau_a2']+eps) + \
            gammaln(prior['tau_a1']+prior['tau_a2']+eps)

    E_logP_Beta = (prior['tau_b1']-1) * E_log_Beta + \
            (prior['tau_b2']-1) * E_log_OneMinusBeta - \
            gammaln(prior['tau_b1']+eps) - \
            gammaln(prior['tau_b2']+eps) + \
            gammaln(prior['tau_b1']+prior['tau_b2']+eps)

    E_logQ_Alpha = (tau_a1-1)*E_log_Alpha + (tau_a2-1)*E_log_OneMinusAlpha - \
            gammaln(tau_a1 + eps) - gammaln(tau_a2 + eps) + \
            gammaln(tau_a1+tau_a2 + eps)

    E_logQ_Beta = (tau_b1-1)*E_log_Beta + (tau_b2-1)*E_log_OneMinusBeta - \
            gammaln(tau_b1 + eps) - gammaln(tau_b2 + eps) + \
            gammaln(tau_b1+tau_b2 + eps)

    E_logQ_C = np.sum(phi * np.log(phi+eps), axis=1)
    
    eta_N_GK = np.reshape(np.transpose(eta, (1,0,2)), (N,G*K))

    # compute three terms and then add them up
    L_1, L_2, L_3 = [0., 0., 0.]
    # the first term and part of the second term
    for m in np.arange(M):
        idx_S = range(sum(Ncon[:m]), sum(Ncon[:m])+Ncon[m])
        tp_con = S[idx_S,3]
        
        phi_rep = np.reshape(np.transpose(np.tile(phi[m],(K,1))), G*K)
        E_A = np.dot(eta_N_GK, np.transpose(eta_N_GK * phi_rep))
        E_A_use = E_A[S[idx_S,1], S[idx_S,2]]
        tp_Asum = np.sum(E_A_use)
        tp_AdotS = np.sum(E_A_use * tp_con)

        L_1 = L_1 + Ncon[m]*E_log_Beta[m] + np.sum(tp_con)*\
                (E_log_OneMinusBeta[m]-E_log_Beta[m]) + \
                tp_AdotS * (E_log_Alpha[m] + E_log_Beta[m] - \
                E_log_OneMinusAlpha[m] - E_log_OneMinusBeta[m]) + \
                tp_Asum * (E_log_OneMinusAlpha[m] - E_log_Beta[m]) 
        
        fg = lambda g: phi[m,g] * np.sum(E_log_OneMinusNu[0:g-1])

        L_2 = L_2 + E_logP_Alpha[m] + E_logP_Beta[m] + \
                np.dot(phi[m],E_log_Nu) + np.sum(map(fg, np.arange(G)))
        
    # the second term
    for g in np.arange(G):
        tp_Nug = (prior['gamma']-1)*E_log_OneMinusNu[g] + \
                np.log(prior['gamma']+eps)

        t1 = np.dot(X,mu_w[g])
        t2 = 0.5*np.dot(X**2, sigma_w[g]**2)
        t3 = np.sum(eta[g],axis=1)
        t_mat_i = logsumexp(np.add(mu_b[g]+0.5*sigma_b[g]**2, t1 + t2), axis=1)
        tp_Zg = np.sum(eta[g]*np.add(t1, mu_b[g])) - np.dot(t3,t_mat_i)

        t5 = -np.log(np.sqrt(2*np.pi)*prior['sigma_w']) - \
                0.5/(prior['sigma_w']**2) * (sigma_w[g]**2 + \
                (mu_w[g]-prior['mu_w'])**2)
        tp_Wg = np.sum(t5)
        t6 = -np.log(np.sqrt(2*np.pi)*prior['sigma_b']+eps) - \
                0.5/(prior['sigma_b']**2) * (sigma_b[g]**2 + \
                (mu_b[g]-prior['mu_b'])**2)
        tp_bg = np.sum(t6)
        L_2 = L_2 + tp_Nug + tp_Zg + tp_Wg + tp_bg

    # the third term
    L_3 = np.sum(E_logQ_Alpha + E_logQ_Beta + E_logQ_C)
    for g in np.arange(G):
        tp_Nug3 = (tau_v1[g]-1)*E_log_Nu[g]+(tau_v2[g]-1)*E_log_OneMinusNu[g] -\
                np.log(gamma(tau_v1[g])+eps) - np.log(gamma(tau_v2[g])+eps) + \
                np.log(gamma(tau_v1[g]+tau_v2[g])+eps)
        tp_Zg3 = np.sum(eta[g]*np.log(eta[g]+eps))
        tp_Wg3 = np.sum(-np.log(np.sqrt(2*np.pi)*sigma_w[g]+eps)-0.5)
        tp_bg3 = np.sum(-np.log(np.sqrt(2*np.pi)*sigma_b[g]+eps)-0.5)
        L_3 = L_3 + tp_Nug3 + tp_Zg3 + tp_Wg3 + tp_bg3

    # Note the third term should have a minus sign before it
    ELBO = L_1 + L_2 - L_3 
    #ELBO = L_1 + L_2
    
    return -ELBO

def ELBO_terms(param, prior, X, S, Ncon, G, M, K):
    eps = 1e-12

    # get sample size and feature size
    [N, D] = np.shape(X)

    # unpack the input parameter vector
    [tau_a1, tau_a2, tau_b1, tau_b2, phi, tau_v1, tau_v2, mu_w, sigma_w,\
            mu_b, sigma_b] = unpackParam(param, N, D, G, M, K)

    # compute eta given mu_w and mu_b
    eta = np.zeros((0, K))
    for g in np.arange(G):
        t1 = np.exp(np.dot(X, mu_w[g]) + mu_b[g])
        t2 = np.transpose(np.tile(np.sum(t1, axis=1), (K,1)))
        eta = np.vstack((eta, t1/t2))
    eta = np.reshape(eta, (G,N,K))

    # compute the expectation terms to be used later
    E_log_Alpha = digamma(tau_a1) - digamma(tau_a1 + tau_a2)  # len(M)
    E_log_OneMinusAlpha = digamma(tau_a2) - digamma(tau_a1 + tau_a2) # len(M)
    E_log_Beta = digamma(tau_b1) - digamma(tau_b1+tau_b2) # len(M)
    E_log_OneMinusBeta = digamma(tau_b2) - digamma(tau_b1+tau_b2) # len(M)

    E_log_Nu = digamma(tau_v1) - digamma(tau_v1 + tau_v2) # len(G)
    E_log_OneMinusNu = digamma(tau_v2) - digamma(tau_v1 + tau_v2) # len(G)
    E_C = phi # shape(M, G)
    E_W = mu_w # shape(G, D, K)
    E_WMinusMuSqd = sigma_w**2 + (mu_w - prior['mu_w'])**2 # shape(G, D, K)
    E_BMinusMuSqd = sigma_b**2 + (mu_b - prior['mu_b'])**2 # shape(G, K)
    E_ExpB = np.exp(mu_b + 0.5*sigma_b**2) # shape(G, K) 

    E_logP_Alpha = (prior['tau_a1']-1) * E_log_Alpha + \
            (prior['tau_a2']-1) * E_log_OneMinusAlpha -  \
            gammaln(prior['tau_a1']+eps) - \
            gammaln(prior['tau_a2']+eps) + \
            gammaln(prior['tau_a1']+prior['tau_a2']+eps)

    E_logP_Beta = (prior['tau_b1']-1) * E_log_Beta + \
            (prior['tau_b2']-1) * E_log_OneMinusBeta - \
            gammaln(prior['tau_b1']+eps) - \
            gammaln(prior['tau_b2']+eps) + \
            gammaln(prior['tau_b1']+prior['tau_b2']+eps)

    E_logQ_Alpha = (tau_a1-1)*E_log_Alpha + (tau_a2-1)*E_log_OneMinusAlpha - \
            gammaln(tau_a1 + eps) - gammaln(tau_a2 + eps) + \
            gammaln(tau_a1+tau_a2 + eps)

    E_logQ_Beta = (tau_b1-1)*E_log_Beta + (tau_b2-1)*E_log_OneMinusBeta - \
            gammaln(tau_b1 + eps) - gammaln(tau_b2 + eps) + \
            gammaln(tau_b1+tau_b2 + eps)

    E_logQ_C = np.sum(phi * np.log(phi+eps), axis=1)
    
    eta_N_GK = np.reshape(np.transpose(eta, (1,0,2)), (N,G*K))

    # compute three terms and then add them up
    L_1, L_2, L_3 = [0., 0., 0.]
    # the first term and part of the second term
    for m in np.arange(M):
        idx_S = range(sum(Ncon[:m]), sum(Ncon[:m])+Ncon[m])
        tp_con = S[idx_S,3]
        
        phi_rep = np.reshape(np.transpose(np.tile(phi[m],(K,1))), G*K)
        E_A = np.dot(eta_N_GK, np.transpose(eta_N_GK * phi_rep))
        E_A_use = E_A[S[idx_S,1], S[idx_S,2]]
        tp_Asum = np.sum(E_A_use)
        tp_AdotS = np.sum(E_A_use * tp_con)

        L_1 = L_1 + Ncon[m]*E_log_Beta[m] + np.sum(tp_con)*\
                (E_log_OneMinusBeta[m]-E_log_Beta[m]) + \
                tp_AdotS * (E_log_Alpha[m] + E_log_Beta[m] - \
                E_log_OneMinusAlpha[m] - E_log_OneMinusBeta[m]) + \
                tp_Asum * (E_log_OneMinusAlpha[m] - E_log_Beta[m]) 
        
        fg = lambda g: phi[m,g] * np.sum(E_log_OneMinusNu[0:g-1])

        L_2 = L_2 + E_logP_Alpha[m] + E_logP_Beta[m] + \
                np.dot(phi[m],E_log_Nu) + np.sum(map(fg, np.arange(G)))
        
    # the second term
    for g in np.arange(G):
        tp_Nug = (prior['gamma']-1)*E_log_OneMinusNu[g] + \
                np.log(prior['gamma']+eps)

        t1 = np.dot(X,mu_w[g])
        t2 = 0.5*np.dot(X**2, sigma_w[g]**2)
        t3 = np.sum(eta[g],axis=1)
        t_mat_i = logsumexp(np.add(mu_b[g]+0.5*sigma_b[g]**2, t1 + t2), axis=1)
        tp_Zg = np.sum(eta[g]*np.add(t1, mu_b[g])) - np.dot(t3,t_mat_i)

        t5 = -np.log(np.sqrt(2*np.pi)*prior['sigma_w']) - \
                0.5/(prior['sigma_w']**2) * (sigma_w[g]**2 + \
                (mu_w[g]-prior['mu_w'])**2)
        tp_Wg = np.sum(t5)
        t6 = -np.log(np.sqrt(2*np.pi)*prior['sigma_b']+eps) - \
                0.5/(prior['sigma_b']**2) * (sigma_b[g]**2 + \
                (mu_b[g]-prior['mu_b'])**2)
        tp_bg = np.sum(t6)
        L_2 = L_2 + tp_Nug + tp_Zg + tp_Wg + tp_bg

    # the third term
    L_3 = np.sum(E_logQ_Alpha + E_logQ_Beta + E_logQ_C)
    for g in np.arange(G):
        tp_Nug3 = (tau_v1[g]-1)*E_log_Nu[g]+(tau_v2[g]-1)*E_log_OneMinusNu[g] -\
                np.log(gamma(tau_v1[g])+eps) - np.log(gamma(tau_v2[g])+eps) + \
                np.log(gamma(tau_v1[g]+tau_v2[g])+eps)
        tp_Zg3 = np.sum(eta[g]*np.log(eta[g]+eps))
        tp_Wg3 = np.sum(-np.log(np.sqrt(2*np.pi)*sigma_w[g]+eps)-0.5)
        tp_bg3 = np.sum(-np.log(np.sqrt(2*np.pi)*sigma_b[g]+eps)-0.5)
        L_3 = L_3 + tp_Nug3 + tp_Zg3 + tp_Wg3 + tp_bg3

    return (L_1, L_2, L_3)


if __name__ == "__main__":
    #[N,D,G,M,K] = [1000, 50, 20, 100, 10]
    #[N,D,G,M,K] = [10,2,3,10,2]
    pass
