#!/usr/bin/env python
# coding: utf-8

"""
Plartform optimizations, fair, half and unconstrained

"""


#############################################Required libraries####################################################

import numpy as np
import scipy.stats as stats

#############################################End####################################################################


class Player:
    """
    Set up what a player would look like  
    Each player has group membership (A= 1, B = -1), the article they are shown (a = 1, b =-1), 
    and booleans for whether or not they clicked and shared the article        
    ------
    input
        :param  group: int, group affliation 1 or -1
        :param  shared: bool, shared or not
        :param  article: int, affliation of the article source 1, -1
        :param  clicked: bool, clicked or not
    """
    def __init__(self, group=1, shared=False, article=0, clicked = 0): 
        self.group = group
        self.shared = shared
        self.article = article
        self.clicked = clicked
        #self.clickprob = calcclickprob(*clickprobparams)


def calcclick(player, t, P, q, theta, c = 1, v=1):
    """
    compute click        
    ------
    input
        :param  player: python object, as described in class Player
        :param  t: int, current timestep.  we have to make sure we actually got to this person
        :param  P: dictionary indexed by (g, s) of the probability distribution of utility.
        :param  q: dictionary, indexed by g, homophily parameter
        :param  theta: dictionary of the possible optimization trials we want to run on this same population.
        :param  c: int, cost for clicking
        :param  v: int, value for liking
    output
        :return 0 or 1
    """
    g = player.group
    s = player.article
    #print(theta)
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(g,s)]) + ((1 - q[ -1 * g]) * P[(g,s)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(g,1)] * thetag + P[(g,-1)] * (1-thetag)
        if p >= float(c/v):
            return 1
        else:
            return 0
 

def calcclickdict(player, t, P, q, theta, c, v):
    """
    compute click 
    ------
    input
        :param  player: python object, as described in class Player
        :param  t: int, current timestep.  we have to make sure we actually got to this person
        :param  P: dictionary indexed by (g, s) of the probability distribution of utility.
        :param  q: dictionary, indexed by g, homophily parameter
        :param  theta: dictionary, indexed by g,  probability of shown a user in group g article s
        :param  c: int, cost for clicking
        :param  v: int, value for liking
    output
        :return 0 or 1
    """
    g = player.group
    s = player.article
    thetag = theta[g]
    if t > 1:
        return (q[g] * calcclick(player, t-1, P, q, theta, c, v) * P[(g,s)]) + ((1 - q[ -1 * g]) * P[(g,s)] * calcclick(player, t-1, P, q, theta, c, v))
    if t == 1:
        p =  P[(g,1)] * thetag * (v[(g,1)] - c[(g,1)]) + P[(g,-1)] * (1-thetag) * (v[(g,-1)] - c[(g,1)])
        if p >= 0.:
            return 1
        else:
            return 0

   
def coin_toss(p):
    """
    Mechanism to decide between A (1) or B (-1).        
    ------
    input
        :param  p: dictionary indexed by (g, s) of the probability distribution of utility.
    output
        :return 
    """
    return (2 * np.random.binomial(1, p) - 1)


