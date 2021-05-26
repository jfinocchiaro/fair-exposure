#!/usr/bin/env python
# coding: utf-8

"""
Plartform optimizations, fair, half and unconstrained

"""


#############################################Required libraries####################################################

import numpy as np
import cvxpy as cp
from scipy.special import betainc
import sims

#############################################End####################################################################


# utility the platform has for user g clicking on article s        
# indexed by (article, user)        
unit_util = {(1,1)   : 1.,
             (1,-1)  : 1.,
             (-1,1)  : 1.,
             (-1,-1) : 1.}

def psi(c, v, F):
    """
    Set up 
    ------
    input
        :param  c: dictionary indexed (g,s), cost for clicking by group and article
        :param  v: dictionary indexed (g,s), value for sharing by group and article
        :param  F: dictionary indexed (g,s), alpha and beta parameters for beta distribution
    output
        :return ps: dictionary indexed (g,s), 
    """
    ps = {}
    for g in [-1,1]:
        for s in [-1,1]:
            alpha, beta = F[(g,s)]
            c_ind = c[(g,s)]
            v_ind = v[(g,s)]
            ps[(g,s)] =  1 - betainc(alpha, beta, float(c_ind) / v_ind)

    return ps


def l(g,s,t, pi, theta, q, c,v,F):
    """
    Set up definition of l from the paper
    #currently done by recursion; seems to be more efficient in closed form, but want to confirm it is correct before I switch code.
    ------
    input
        :param  g: int, group affliation 1 or -1
        :param  s: int, affliation of the article source 1, -1
        :param  t: int, time step
        :param  pi: dictionary, proportion of users in each group
        :param  theta: dictionary, indexed by g,  probability of shown a user in group g article s
        :param  q: int, total number of iterations
        :param  c: dictionary indexed (g,s), cost for clicking by group and article
        :param  v: dictionary indexed (g,s), value for sharing by group and article
        :param  F: dictionary indexed (g,s), alpha and beta parameters for beta distribution
    """
    ps = psi(c,v,F)
    ti = (g+1) / 2
    if t >= 2:
        return ((q[g] * l(g,s,t-1, pi, theta, q,c,v,F)) + (1 - q[-g]) * l(-g,s,t-1, pi, theta, q,c,v,F)) * ps[(g,s)]
    else:
        if s == -1:
            return pi[g] * theta[ti] * ps[(g,-1)]
        if s == 1:
            return pi[g] * (1 - theta[ti]) * ps[(g,1)]

        

def opt(policy, pi, q, T, c,v,F, epsilon = 0.1, exposure_e = 0.0, delta_low = 0.5, delta_high=2, u=unit_util):
    """
    Set up the optimization function 
    ------
    input
        :param  policy: string, whether optimization is unconstrained or fair: additive, ratio
        :param  pi: dictionary, proportion of users in each group
        :param  q: dictionary, the homophily variable
        :param  T: int, max total number of timesteps
        :param  epsilon: double, level of fairness violation allowed
        :param  c: dictionary indexed (g,s), cost for clicking by group and article
        :param  v: dictionary indexed (g,s), value for sharing by group and article
        :param  F: dictionary indexed (g,s), alpha and beta parameters for beta distribution
        :param  u: dictionary, platform's utility for a click which can be thought of as a price charged.
    output
        :return ps: 1d-array of shape (n_features, 1), weight of the classifier
    """
    
    # varaible theta_A, theta_B
    theta = cp.Variable(2)
    # define the optimization objective
    objective = cp.Maximize(cp.sum([u[(1,1)] * l(1,1,t, pi, theta, q, c,v,F) + u[(-1,1)] * l(-1,1,t,pi, theta, q, c,v,F) \
        + u[(1,-1)] * l(1,-1,t, pi, theta, q, c,v,F) + u[(-1,-1)] *  l(-1,-1,t, pi, theta, q, c,v,F) for t in range(T)]))
    # define the unconstrained constraint
    constraints_theta = [exposure_e <= theta[0], theta[0] <= 1 - exposure_e, exposure_e <= theta[1], theta[1] <= 1 - exposure_e]
    constraints = []

    if policy == 'additive':
        #generate eta, used as constraints.
        eta = {}
        for s in [-1,1]:
            for g in [-1,1]:
                ti = (g+1) / 2 #theta index.... moving from -1 to 0 and  1 to 1
                if s == 1:
                    eta[(s,g)] = pi[g] * theta[ti] + sum([ l(g,s,t, pi, theta,q,c,v,F) * q[g] + l(-g,s,t, pi, theta,q,c,v,F) * (1 - q[-g]) for t in range(1,T)])
                else:
                    eta[(s,g)] = pi[g] * (1-theta[ti]) + sum([ l(g,s,t, pi, theta,q,c,v,F) * q[g] + l(-g,s,t, pi, theta,q,c,v,F) * (1 - q[-g]) for t in range(1,T)])
                    
        
        
        constraints_eta = [eta[(1,1)] - eta[(-1,-1)] <= epsilon, eta[(-1,-1)] - eta[(1,1)] <= epsilon]
        constraints = constraints_theta+constraints_eta
        
    elif policy == 'ratio': 
        constraints_ratio = []
        constraints_ratio.append(delta_low * sum([l(-1,-1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]) <= sum([l(1,1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]))
        constraints_ratio.append(sum([l(1,1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]) <= delta_high * sum([l(-1,-1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]))
        constraints_ratio.append(delta_low *  sum([l(-1,1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)])<= sum([l(1,-1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]))
        constraints_ratio.append(sum([l(1,-1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)])<= delta_high * sum([l(-1,1,t,  pi, theta, q, c,v,F) for t in range(1,T+1)]))
        constraints = constraints_theta + constraints_ratio
    else:
        constraints = constraints_theta

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.solve is not None and theta.value is not None:
        th = {}
        th[-1] = max(min(theta.value[1], 1.), 0.)
        th[1] = max(min(theta.value[0], 1.), 0.)
    else:
        th = {1:0, -1:0}
        print("Constraints not feasible")
        # quit()

    return th

#test code
if __name__ == '__main__':
  
    dataset_name = 'twitter_abortion'
    T = 5
    pi,beta_dist,P,v,c,q = sims.get_params(dataset_name)
    
    print(opt('additive',pi, q, T, c,v,beta_dist))