#!/usr/bin/env python
# coding: utf-8

"""
File specifies how to run and store simulation results. 

"""


#############################################Required libraries####################################################

import numpy as np 
import random
import pickle as pkl
import pandas as pd
import platform_opt
from players import *
from statistics import variance

#############################################End####################################################################


def runModel(theta, T, pi, M, P, beta_dist, v,c,q): 
    """
    Set up runs of the model given a strategy theta for showing articles at time step 1.      
    ------
    input
        :param  theta: dictionary indexed by user group g, the proprtion of users in group g shown article A (1)
        :param  T: int, max total number of timesteps
        :param  pi: dictionary indexed by user group, proportion of users in each group.  pi[1] = 1 - pi[-1]
        :param  M: int, number of players in the mass
        :param  P: dictionary indexed by (article group, user group), expected value for probably liking given a click
        :param  beta_dist: dictionary indexed by (alpha, beta), parameters for the beta distribution for above distribution.
        :param  v: dictionary indexed (article group, user group),  value for liking an article. 
        :param  c: dictionary indexed (article group, user group), cost for clicking on an article. 
        :param  q: dictionary indexed by group homophily variable, probability of intra-group propogation at next timestep 
    output
        :returns num_players_in_model: list of dictionary entries, each for a giver user from player object
        :returns showns_dict: list of dictionary entries, each depicting user group, article group and int value for whether or not article was shown to that user
        :returns clicks_dict: list of dictionary entries, each depicting user group, article group and int value for whether or not user clicked on the article 
        :returns shares_dict: list dictionary entries, each depicting user group, article group and int value for whether or not user shared the article
    """
    
    #tracking data; this could be way more efficient :face-palm:
    old_u = []
    time_data_diff = []
    num_players_in_model = [M]
    tot_shown_A = 0
    tot_in_model = 0
    t = 1
    pi_a = pi[1]
    
    #dictionaries to keep track of who is shown which articles and who clicks on which articles
    #indexed (user group, article shown)
    shown_dict = {(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0}
    
    click_dict = {(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0}
    
    share_dict = {(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0}
    


    while (t <= T) and (t == 1 or len(old_u) > 0):
        num_shown_A = 0 #number of players at this timestep that are shown article A
        new_u = []  # list of new players that arrive at the timestep

        if t == 1:  # initial mass of users arrives
            for i in range(M): # iterating over the size of the unit mass
                tot_in_model = tot_in_model + 1
                g = coin_toss(pi[1]) # determine players group according to the true group distribution
                s = coin_toss(theta[g]) # show article A according to the platform's policy.  (right now, this is just a placeholder)
                player = Player(group=g, article=s)
                shown_dict[(g,s)] = shown_dict[(g,s)] + 1
                if s == 1:
                    num_shown_A = num_shown_A + 1
                    player.article = 1
                else:
                    player.article = -1

                P_personal = P
                P_personal[(g,s)] = np.random.beta(*beta_dist[(g,s)])
                P_personal[(g,-s)] = np.random.beta(*beta_dist[(g,-s)])

                player.clicked = calcclickdict(player, 1, 
                                                  P_personal, 
                                                  q, 
                                                  theta,
                                                  c,
                                                  v)
                if player.clicked:  
                    click_dict[(g,s)] = click_dict[(g,s)] + 1 
                    if random.uniform(0, 1) <= P[(player.group, player.article)]:
                        player.shared = True
                        share_dict[(g,s)] = share_dict[(g,s)] + 1
                old_u.append(player)
            

        else:
            for user in old_u:

                if user.shared == 1: # new user only added to the system if the previous user shared the article
                    tot_in_model = tot_in_model + 1
                    if random.uniform(0, 1) <= q[user.group]:  # if next person is drawn by homophily
                        new_user = Player(group=user.group)
                    else:
                        new_user = Player(group=-user.group)
                        
                    # show the previous person's article, regardless of the new user's group    
                    new_user.article = user.article
                    shown_dict[(new_user.group, new_user.article)] = shown_dict[(new_user.group, new_user.article)] + 1
                    if new_user.article == 1:
                        num_shown_A = num_shown_A + 1


                    g = new_user.group
                    s = new_user.article
                    P_personal = P
                    P_personal[(g,s)] = np.random.beta(*beta_dist[(g,s)])
                    P_personal[(g,-s)] = np.random.beta(*beta_dist[(g,-s)])
                    new_user.clicked = calcclickdict(new_user, 1, 
                                                  P_personal, 
                                                  q, 
                                                  theta,
                                                  c,
                                                  v)
                    # decide if user shares article, according to P.
                    if new_user.clicked == 1:  
                        click_dict[(new_user.group, new_user.article)] = click_dict[(new_user.group, new_user.article)] + 1
                        if random.uniform(0, 1) <= P[(new_user.group, new_user.article)]:
                            new_user.shared = True
                            share_dict[(new_user.group, new_user.article)] = share_dict[(new_user.group, new_user.article)] + 1
                    else:
                        new_user.shared = False

                    #add user to list
                    new_u.append(new_user)
                else: #only add a user to the next round if the previous user shared the article 
                    pass

            num_players_in_model.append(len(new_u)) #tracks how many players are being shown articles at all timesteps
            old_u = new_u



        t = t + 1
        tot_shown_A = tot_shown_A + num_shown_A

    
    return num_players_in_model, shown_dict, click_dict, share_dict

def saveRuns(lst,filename):    
    """
    Save simulation runs
    ------
    input
        :param  lst: list, list to be pickled
        :param  filename: string, filename of picked string
    """ 
    tommy_pickles = open(filename, "wb") # remember to open the file in binary mode 
    pkl.dump(lst, tommy_pickles)
    tommy_pickles.close()
    
    
def get_params(dataset_name):
    """
    Get parameters from a given real life dataset
    ------
    input
        :param  dataset_name: string, name of the dataset from which variables are extracted
    output
        :returns pi: dictionary, proportion of users in each group
        :returns beta_dist: dictionary indexed by (alpha, beta), parameters for the beta distribution for above distribution.
        :returns P: dictionary indexed by (article group, user group), expected value for probably liking given a click
        :returns v: dictionary indexed (g,s), value for sharing by group and article
        :returns c: dictionary indexed (g,s), cost for clicking by group and article
        :returns q: dictionary, the homophily variable
    """
    if dataset_name == 'twitter_uselections':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {1: 0.43294, 
             -1: 0.56706}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (41.45784070052453, 556.8653671739492),
                    (-1,1) : (0.7519296311195025, 413.4664888783973),
                    (1,-1) : (6.096475779403813, 1519.8459882514462),
                    (-1,-1): (2152.960173995409, 23647.671142918956)}

        # probability of like | click, user group, article group
        # P indexed ((user group, article shown)). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {( 1,  1):  0.0692, 
             ( -1, 1):  0.001815,
             (1,  -1):  0.003995,
             (-1, -1):  0.08344} 

        # player utility for liking, known to both user and platform,
        # v indexed by (user group, article shown) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        
        # cost of clicking, known to both user and platform,
        # c indexed by (user group, article shown)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.9877, 
             -1: 1.}
        
    if dataset_name == 'twitter_brexit':
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {-1: 0.47532, 
             1: 0.52468}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (1.6421893317945877, 62.9176081976947),
                    (-1,1) : (1.4779704026249152, 27.402822213177515),
                    (1,-1) : (1.7187375537951832, 380.1479381108044),
                    (-1,-1): (39.62421666552372, 506.9074863422272)}

        # probability of like | click, user group, article group
        # P indexed (user group, article shown). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {k: beta_dist[k][0] / sum(beta_dist[k])
                 for k in beta_dist}

        # player utility for liking, known to both user and platform,
        # v indexed by (user group, article shown) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        # cost of clicking, known to both user and platform,
        # c indexed by (user group, article shown)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.68052, 
             -1: 0.38406}
        
    if dataset_name == 'twitter_abortion':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {-1: 0.627787, 
             1: 0.372213}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (2.1955994970845176, 53.70406773206235),
                    (-1,1) : (0.2547399546219493, 7.443199476254677),
                    (1,-1) : (0.15985328765176798, 50.82834553323337),
                    (-1,-1): (2.2966486101771286, 27.58967380064497)}

        # probability of like | click, user group, article group
        # P indexed (article group, user group). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {k: beta_dist[k][0] / sum(beta_dist[k])
                 for k in beta_dist}

        # player utility for liking, known to both user and platform,
        # v indexed by (article group, user group) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        
        # cost of clicking, known to both user and platform,
        # c indexed by (article shown, user group)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.5529954, 
             -1: 0.8169399}

        
    elif dataset_name=='facebook':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # A = 1
        # parameters here come from Replication Exposure.ipynb
        pi = {1: 0.5, 
             -1: 0.5}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (article group, user group). 
        beta_dist = {(1,1) : (0.9541492709534125, 1.345006644515015),
                    (-1,1) : (0.1822515775580026, 2.7574965182522644),
                    (1,-1) : (0.09576097403924465, 3.09136619146736),
                    (-1,-1): (0.8828729918440646, 1.6247070146941363)}

        # probability of like | click, user group, article group
        # P indexed (article group, user group). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {}
        for a in [-1,1]:
            for g in [-1,1]:
                P[(g,a)] = beta_dist[(g,a)][0] / (beta_dist[(g,a)][0] +  beta_dist[(g,a)][1])


        # player utility for liking, known to both user and platform,
        # v indexed by (article group, user group) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        # cost of clicking, known to both user and platform,
        # c indexed by (article shown, user group)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        q = {1: 0.7192525700416023, 
             -1: 0.6797565445480234}

        # approximation parameter for approximately equal probability
        epsilon = 0.05  
        
    if dataset_name == 'twitter_uselections':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {1: 0.43294, 
             -1: 0.56706}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (41.45784070052453, 556.8653671739492),
                    (-1,1) : (0.7519296311195025, 413.4664888783973),
                    (1,-1) : (6.096475779403813, 1519.8459882514462),
                    (-1,-1): (2152.960173995409, 23647.671142918956)}

        # probability of like | click, user group, article group
        # P indexed ((user group, article shown)). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {( 1,  1):  0.0692, 
             ( -1, 1):  0.001815,
             (1,  -1):  0.003995,
             (-1, -1):  0.08344} 

        # player utility for liking, known to both user and platform,
        # v indexed by (user group, article shown) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        
        # cost of clicking, known to both user and platform,
        # c indexed by (user group, article shown)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.9877, 
             -1: 1.}
        
    if dataset_name == 'twitter_brexit':
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {-1: 0.47532, 
             1: 0.52468}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (1.6421893317945877, 62.9176081976947),
                    (-1,1) : (1.4779704026249152, 27.402822213177515),
                    (1,-1) : (1.7187375537951832, 380.1479381108044),
                    (-1,-1): (39.62421666552372, 506.9074863422272)}

        # probability of like | click, user group, article group
        # P indexed (user group, article shown). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {k: beta_dist[k][0] / sum(beta_dist[k])
                 for k in beta_dist}

        # player utility for liking, known to both user and platform,
        # v indexed by (user group, article shown) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        # cost of clicking, known to both user and platform,
        # c indexed by (user group, article shown)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.68052, 
             -1: 0.38406}
        
    if dataset_name == 'symmetric':
        # SIMULATION PARAMS DEPENDING ON DATASET
        # parameters here come from probability_sharing_distributions.ipynb
        pi = {-1: 0.5, 
             1: 0.5}          # number of members in groups a and b #estimated from probability_sharing_distributions.ipynb
        pi_a = pi[1]

        # (alpha, beta) values for the beta distribution as a function of article and user groups.
        # beta_dist indexed (user group, article shown). 
        beta_dist = {(1,1) : (1., 9.),(-1,1) : (1., 29.),(1,-1) : (1., 29.),(-1,-1): (1., 9.)}

        # probability of like | click, user group, article group
        # P indexed (article group, user group). expected values of above beta distribution
        #estimated from probability_sharing_distributions.ipynb
        P = {k: beta_dist[k][0] / sum(beta_dist[k])
                 for k in beta_dist}

        # player utility for liking, known to both user and platform,
        # v indexed by (article group, user group) pair
        #unclear what these values _should_ be!
        v = {( 1,  1):   2000.,
             (-1,  1):   200.,
             ( 1, -1):   200.,
             (-1, -1):   2000. }

        
        # cost of clicking, known to both user and platform,
        # c indexed by (article shown, user group)
        c = {( 1,  1):   1.,
             (-1,  1):   1.,
             ( 1, -1):   1.,
             (-1, -1):   1. }

        # transition probability across groups at time t + 1 
        # indexed by the first user's group membership
        # seems too high to be practical
        q = {1:  0.75, 
             -1: 0.75}

        
        
    return pi,beta_dist,P,v,c,q


def loadRuns(filename):
    """
    Load saved runs from pickel files
    ------
    input
        :param  filename: string, filename of the pickel file being loaded
    output
        :returns new_dict, data from the pickel file
    """
    infile = open(filename,'rb')
    new_dict = pkl.load(infile)
    infile.close()
    return new_dict


def runModel_samepop(T, pi, M, P, beta_dist, v,c,q, thetas={'opt': {-1:1, 1:0}, 'half': {-1:0.5, 1:0.5}}): 

    """
    Set up runs of the model given a strategy theta for showing articles at time step 1.      
    ------
    input
        :param  theta: dictionary indexed by user group g, the proprtion of users in group g shown article A (1)
        :param  T: int, max total number of timesteps
        :param  pi: dictionary indexed by user group, proportion of users in each group.  pi[1] = 1 - pi[-1]
        :param  M: int, number of players in the mass
        :param  P: dictionary indexed by (article group, user group), expected value for probably liking given a click
        :param  beta_dist: dictionary indexed by (alpha, beta), parameters for the beta distribution for above distribution.
        :param  v: dictionary indexed (article group, user group),  value for liking an article. 
        :param  c: dictionary indexed (article group, user group), cost for clicking on an article. 
        :param  q: dictionary indexed by group homophily variable, probability of intra-group propogation at next timestep 
        :param  theta: dictionary of the possible optimization trials we want to run on this same population.
    output
        :returns ret: dictionary, the number of clicked articles in the simulation.
    """    
    ret = {}
    time_data_diff = {}
    num_players_in_model = {}
    tot_shown_A = {}
    tot_in_model = {}
    shown_dict = {}
    click_dict = {}
    share_dict = {}
    for theta_type in thetas.keys():
        ret[theta_type] = pd.DataFrame(dtype=object,columns=['players_list', 'shown', 'click', 'share'])
        time_data_diff[theta_type] = []
        num_players_in_model[theta_type] = [M]
        tot_shown_A[theta_type] = 0
        tot_in_model[theta_type] = 0
        #dictionaries to keep track of who is shown which articles and who clicks on which articles
    #indexed (article, user)
        shown_dict[theta_type] = []
    
        click_dict[theta_type] = []
    
        share_dict[theta_type] = []
    
    #tracking data; this could be way more efficient :face-palm:
    
    
    t = 1
    pi_a = pi[1]
    
    num_shown_A = {} #number of players at this timestep that are shown article A
    new_u = {}  # list of new players that arrive at the timestep
    old_u = {} #maintain list of old players from previous timestep

    for theta_type in thetas.keys():
        old_u[theta_type] = []

    while (t <= T):# and (t == 1 or len(old_u) > 0):
        for theta_type in thetas.keys():
            num_shown_A[theta_type] = 0 #number of players at this timestep that are shown article A
            new_u[theta_type] = []  # list of new players that arrive at the timestep
            shown_dict[theta_type].append({(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0})
            click_dict[theta_type].append({(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0})
            share_dict[theta_type].append({(1,1):   0,
                  (-1,1):  0, 
                  (1,-1):  0,
                  (-1,-1): 0})

        if t == 1:  # initial mass of users arrives
            for i in range(M): # iterating over the size of the unit mass
                g = coin_toss(pi[1]) # determine players group according to the true group distribution
                for theta_type, theta in thetas.items():
                    a = coin_toss(theta[g]) # show article A according to the platform's policy.  (right now, this is just a placeholder)
                    player = Player(group=g, article=a)
                    shown_dict[theta_type][t-1][(g,a)] = shown_dict[theta_type][t-1][(g,a)] + 1
                    if a == -1:
                        num_shown_A[theta_type] = num_shown_A[theta_type] + 1
                        player.article = -1
                    else:
                        player.article = 1

                    P_personal = P
                    P_personal[(g,a)] = np.random.beta(*beta_dist[(g,a)])
                    P_personal[(g,-a)] = np.random.beta(*beta_dist[(g,-a)])

                    tot_in_model[theta_type] = tot_in_model[theta_type] + 1
                    player.clicked = calcclickdict(player, 1, 
                                                      P_personal, 
                                                      q, 
                                                      theta,
                                                      c,
                                                      v)
                    if player.clicked:  
                        click_dict[theta_type][t-1][(g,a)] = click_dict[theta_type][t-1][(g,a)] + 1 
                        if random.uniform(0, 1) <= P[(player.article, player.group)]:
                            player.shared = True
                            share_dict[theta_type][t-1][(g,a)] = share_dict[theta_type][t-1][(g,a)] + 1
                    old_u[theta_type].append(player)
            
                
        else:
            for theta_type in thetas.keys():
                for user in old_u[theta_type]:
                    if user.shared == 1: # new user only added to the system if the previous user shared the article
                        tot_in_model[theta_type] = tot_in_model[theta_type] + 1
                        if random.uniform(0, 1) <= q[user.group]:  # if next person is drawn by homophily
                            new_user = Player(group=user.group)
                        else:
                            new_user = Player(group=-user.group)

                        # show the previous person's article, regardless of the new user's group    
                        new_user.article = user.article
                        shown_dict[theta_type][t-1][(new_user.article, new_user.group)] = shown_dict[theta_type][t-1][(new_user.article, new_user.group)] + 1
                        if new_user.article == -1:
                            num_shown_A[theta_type] = num_shown_A[theta_type] + 1


                        P_personal = P
                        P_personal[(g,a)] = np.random.beta(*beta_dist[(new_user.group,new_user.article)])
                        P_personal[(g,-a)] = np.random.beta(*beta_dist[(new_user.group,-new_user.article)])
                        new_user.clicked = calcclickdict(new_user, 1, 
                                                      P_personal, 
                                                      q, 
                                                      theta,
                                                      c,
                                                      v)
                        # decide if user shares article, according to P.
                        if new_user.clicked == 1:  
                            click_dict[theta_type][t-1][(new_user.group, new_user.article)] = click_dict[theta_type][t-1][(new_user.group, new_user.article)] + 1
                            if random.uniform(0, 1) <= P[(new_user.group, new_user.article)]:
                                new_user.shared = True
                                share_dict[theta_type][t-1][(new_user.group, new_user.article)] = share_dict[theta_type][t-1][(new_user.group, new_user.article)] + 1
                        else:
                            new_user.shared = False

                        #add user to list
                        new_u[theta_type].append(new_user)
                    else: #only add a user to the next round if the previous user shared the article 
                        pass

                num_players_in_model[theta_type].append(len(new_u[theta_type])) #tracks how many players are being shown articles at all timesteps
                old_u[theta_type] = new_u[theta_type]
                tot_shown_A[theta_type] = tot_shown_A[theta_type] + num_shown_A[theta_type]


        t = t + 1
    
    for theta_type in thetas.keys():
        # columns: 'players_list',  'shown', 'click'
        ret[theta_type]['players_list'] = num_players_in_model[theta_type]
        ret[theta_type]['shown'] = shown_dict[theta_type]
        ret[theta_type]['click'] = click_dict[theta_type]
        ret[theta_type]['share'] = share_dict[theta_type]
    
    return ret


def average_dicts(dicts, col_name):
    """
    Compute average of dictionaries with matching keys    
    ------
    input
        :param  dicts: list, list of dictionaries with all matching keys
        :param  col_name: string, 
    output
        :returns d, dataframe, 
    """
    ret = {}
    vals = []
    num_rows = len(dicts[0].index)
    for i in range(num_rows):
        val = {}
        for (g,s) in dicts[0][col_name][0].keys():
            val[(g,s)] = np.mean([dic[col_name][i][(g,s)] for dic in dicts])
        vals.append(val)   
        #ret[i][(g,s)] = val

    d = pd.DataFrame(dtype = object)
    d[col_name] = vals
    
    return d

        
def average_dfs(dataframes):
    """
    Compute element-wise average of dataframes   
    dataframes[i][theta_type] is a dataframe with columns 'players_list', 'shown', 'click', and 'share' 
    ------
    input
        :param  dataframes: list of dicts, indexed by theta type
    output
        :returns ret: dictionary with same columns as above, and each element is the element-wise average
    """

    ret = {}
    for theta_type in dataframes[0].keys():
        #get the average number of players in each timestep
        df = pd.DataFrame(dtype=object,columns=['players_list', 'shown', 'click', 'share'])
        df['players_list'] = np.mean([dataf[theta_type]['players_list'] for dataf in dataframes], axis=0)
        
        #get the average of who is shown, clicks, and shares which article
        df['shown'] = average_dicts([dataf[theta_type] for dataf in dataframes], 'shown')
        df['click'] = average_dicts([dataf[theta_type] for dataf in dataframes], 'click')
        df['share'] = average_dicts([dataf[theta_type] for dataf in dataframes], 'share')
        
        ret[theta_type] = df
    return ret



if __name__ == '__main__':
    """
    Main function to test some functions
    ------
    
    """
    #load parameters from dataset
    dataset_name = 'twitter_abortion'
    pi,beta_dist,P,v,c,q = get_params(dataset_name)
    
    theta_additive = {1: 0.7383090139889638, -1: 0.9000000001963758}
    theta_ratio = {1: 0.6075487021940786, -1: 0.900000000045376}
    theta_opt = {1:0., -1:1.}
    theta_half = {1: 0.5, -1: 0.5}
    theta_proprtional = {}
    thetas = {'additive' : theta_additive, 'ratio': theta_ratio, 'opt': theta_opt, 'half': theta_half, 'proportional': pi}
    
    # SIMULATION PARAMETERS AGNOSTIC TO DATA
    T = 6                 # max number of timesteps
    M = 10000            # size of unit mass


    
    ret1 = runModel_samepop(T, pi, M, P, beta_dist, v,c,q, thetas)
    ret2 = runModel_samepop(T, pi, M, P, beta_dist, v,c,q, thetas)
    
    num_trials = 2
    
    avg = average_dfs([runModel_samepop(T, pi, M, P, beta_dist, v,c,q, thetas) for ix in range(num_trials)])
    print(avg['ratio']['share'])
    print(avg['proportional']['share'])

