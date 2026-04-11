# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 22:23:02 2025

@author: halajgr
"""

## load data
import pandas as pd
import numpy as np
import random
import itertools
import networkx as nx
#from cvxopt import matrix, solvers
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.colors
import chardet
import datetime
from scipy.stats import ortho_group
import pickle
from pathlib import Path

from FeinsteinHalaj_aux_functions import *
import config


COLORSkyBlue = (133/255, 195/255, 236/255)
COLORPastelOrange = (255/255, 179/255, 71/255)
COLORCrayolasOrangeRed = (255/255, 83/255, 72/255)
COLORTropicalRainForest = (23/255, 128/255, 109/255)
IFSAVEEXAMPLE = 0

IFLOADEXAMPLE = config.IF_LOAD_EXAMPLE

RUNOFFRATE = 1.0  

IFSIMULATED = 1 # data will be generated in a cell below, ignoring input from files (taken care of in this cell);

IFPROPORTIONALSELLING = 0 # 1: proportional selling, 0: strategic

Ntatonn = config.NTATONNEMENT

IF_DRAW_BIPARTITE = config.IF_DRAW_BIPARTITE

IFCENTRALBANK = config.IFCENTRALBANK

SHOCKEDBANKS = config.SHOCKED_BANKS

PRIMPACTPRCT = config.PRICE_IMPACT_PER_PRTC_SOLD

PRICEIMPACTFUNCTION = config.PRICE_IMPACT_FUNCTION

SHOCK_SIZE = config.SHOCK_SIZE/10**2

THRESHOLD_BIPARTITE = config.THRESHOLD_BIPARTITE

RESPONSEMODE = config.RESPONSEMODE

CREATESTABLESYSTEM = config.CREATESTABLESYSTEM
IFSTARTFROMSTEADY = config.IFSTARTFROMSTEADY
NRUNSTOSTABILIZE = config.NRUNSTOSTABILIZE
IF_LOAD_FROM_PICKLE = config.IF_LOAD_FROM_PICKLE

IFCBONLYPURCHASE = 0 # 2: small selling is allowed
CBLOWERBOUND = 0.01
CBUNLIMITEDPOOL = 0.0*10.0**9 # a large number that corresponds to unlimited liquidity resources of a central bank
IFFASTCB = 1 #1: central bank transact step by step to bring prices back (fraction of a distance to optimium); 2: full absorption by CB
IFQUANT = 0 #1:change to the optimum by fraction of x; otherwise by (y_response-y)
EXCLUDEASSETFROMCB = [] #trouble-makers derailing optimisation of the central bank

TURNOFFBENCHMARK = 1 #1: [baseline settings] in case banks internalize their actions, period == 0 is run as if banks internalize actions; 0: run assuming banks do not internalize

IFPARETOTOTALASSETS = 0 #sample from a pareto distribution
PARETOALPHA = 3.0
PARETOMODE = 2.0

IFDECAY = 0
EXPO = 0.4 #decay parameter in the step of the totannement

IF_TURNOFF_ITER_STATS = config.IF_TURNOFF_ITER_STATS

# Folder where this .py file lives
FPATH = Path(__file__).resolve().parent
FPATHOUTPUT = FPATH / 'output'

# Check if subfolder exists; create it if it doesn't
if not FPATHOUTPUT.exists():
    FPATHOUTPUT.mkdir(parents=True)
    print("Created subfolder:", FPATHOUTPUT)

RUNOFFRATE = 1.0
    
BANKNUMBER = config.NUMBER_OF_BANKS #number of banks in the system
ASSETNUMBER = config.NUMBER_OF_ASSETS #number of tradable assets

iba_frac = np.random.uniform(0,0.1,size=(BANKNUMBER,))
ibl_frac = np.random.uniform(0,0.1,size=(BANKNUMBER,))

print('settings loaded!')


#######################
#STYLIZED BALANCE SHEET
#######################

DISTRIBUTION_BANKS = 'u' # 'u': uniform, 'p': Pareto

FRACTIONCORE = 0.1 #percentage of banks in core

if IFSIMULATED==1:
    if IFPARETOTOTALASSETS==1:
        total_assets_bench = (np.random.pareto(PARETOALPHA, BANKNUMBER) + 1) * PARETOMODE# will come from FINREP
    else:
        total_assets_bench = np.random.uniform(50,1000,size = (BANKNUMBER,))

    total_securities_bench = np.random.uniform(0,0.5,size = (BANKNUMBER,))
    
    IBPROB = 0.05

#generate data!
if IFSIMULATED == 1 and IF_LOAD_FROM_PICKLE == 0:
    print('simulated')
    import string
    
    np.random.seed(11) #101
    
    LETTERS = list(string.ascii_uppercase)
    
    nSimulatedSys = 1
    
    xdata = []
    for ii in range(0,nSimulatedSys):
        if ii<10:
            xdata.append('ibsys00'+str(ii))
        else:
            if ii<100:
                xdata.append('ibsys0'+str(ii))
            else: 
                xdata.append('ibsys'+str(ii))
    
    
    optim_net_ts_dict = dict()
    for ixd in xdata:
        FRACT_LOANS = 0.5
        print('system '+str(ixd))

        ii = 0
        z_step = 0.0
        
        Nb = np.random.randint(BANKNUMBER,BANKNUMBER+1)
        M = np.random.randint(ASSETNUMBER,ASSETNUMBER+1) #number of assets x
        
        Nb_core = int(FRACTIONCORE*Nb)
        Nb_codes = []
        icode = 0
        jcode = 0
        xiter = 0
        while xiter<Nb:
            Nb_codes.append(LETTERS[icode]+LETTERS[jcode])
            if LETTERS[icode]==LETTERS[-1]:
                icode = 0
                jcode+=1
            else:
                icode+=1
            xiter+=1
        
        bank_names = [Nb_codes[ib]+'c' if ib<Nb_core else Nb_codes[ib]+'p' for ib in range(0,Nb)]

        HAIRCUTS = M*[0.0]

        e0_frac = 0.1

        ta = total_assets_bench
        
        tsec = total_securities_bench*ta
        loans = FRACT_LOANS*ta
        
        #x,temp = funOverlapPortfoliosV1(tsec,M,bank_names,0.3,0.1,compress=1,xconfig=1)
        # (sec,ns=ASSETNUMBER,cosine=0.3,alg='b',direction=1,groups=[]):
        x = funOverlapPortfoliosV3(tsec,M,0.1)
        asset_names = list(range(0,M))

        mu = np.random.uniform(0.005,0.01,size=(M,))
        
        iba = iba_frac*ta
        ibl = ibl_frac*ta
        # make total interbank liabilities and assets equal
        ibal_frac = np.sum(iba)/np.sum(ibl)
        if ibal_frac>1:
            iba = iba/ibal_frac
        else:
            ibl = ibl*ibal_frac

        e = e0_frac*ta   
        
        geomap = IBPROB* np.ones((Nb,Nb))
        for ii in range(0,Nb):
            geomap[ii,ii] = 0.0
        net = mLendIBank(iba, ibl, bank_names, e, geomap, bank_names, bank_names, 0, 0)
        L = nx.to_numpy_array(net)
        if np.sum(np.sum(L,1)==0)>0:
            print('interbank matrix empty in at least '+str(np.sum(np.sum(L,1)==0))+' columns')
        #L = np.random.uniform(2,5,size = (Nb,Nb))
        #for ii in range(0,Nb): L[ii,ii] = 0.0
        
        c = np.maximum(np.random.uniform(0,1,size=(Nb,))*(ta-tsec-loans-np.sum(L,1)),0.0)
        
        Q = 0.0001*np.eye(M)+0.00001*np.ones((M,M))

        alpha = 0.01*PRIMPACTPRCT/np.sum(x)*np.ones((M,)) #sensitivity of prices to changes in y, used in funcion f (in my other paper it is alpha)
        list_assets_sdjust_priceimp = list(PRICEIMPACTFUNCTION.keys())
        for el in list_assets_sdjust_priceimp:
            idx = -1
            try:
                idx = asset_names.index(el)
            except ValueError:
                print('no bank '+el+' on the price impact list for adjustment')
            if idx>-1:
                alpha[idx] = 0.01*PRICEIMPACTFUNCTION[el]/np.sum(x)
        b = alpha

        beta = np.array(M*[0.10]) # recovery rate of the revalued assets

        xoptim_params = dict()
        xoptim_params['xx_vec'] = x
        xoptim_params['xr_vec'] = x #fixed reference to the starting point securities composition
        xoptim_params['cc_vec'] = c

        N = x.shape[0]
        M = x.shape[1]

        xoptim_params['L_mat'] = L

        xoptim_params['zz_vec'] = np.maximum(0.0,c + np.sum(x,axis=1) + loans + np.sum(L,axis=1) - np.sum(L,axis=0) - e)

        xoptim_params['mu_vec'] = mu#returns

        xoptim_params['b_vec'] = b

        xoptim_params['q0_vec'] = np.array(M*[1.0]) #initial prices of assets

        xoptim_params['Q_mat'] = Q

        xoptim_params['ee_vec'] = e
        
        xoptim_params['tsec_vec'] = tsec
        
        xoptim_params['ta_beta'] = beta

        xoptim_params['gamma'] = 1.0 
        
        xoptim_params['ta_vec'] = ta
        
        xoptim_params['nodes'] = bank_names

        optim_net_ts_dict[ixd] = xoptim_params
    
if IFSIMULATED == 1 and IF_LOAD_FROM_PICKLE == 1:
    nSimulatedSys = 1
    xdata = []
    for ii in range(0,nSimulatedSys):
        if ii<10:
            xdata.append('ibsys00'+str(ii))
        else:
            if ii<100:
                xdata.append('ibsys0'+str(ii))
            else: 
                xdata.append('ibsys'+str(ii))
    optim_net_ts_dict = dict()
    test_list_N = list()
    test_list_M = list()
    for ixd in xdata:
        optim_net_ts_dict[ixd] = load_system_unpickled(FPATH,ixd)
        x=optim_net_ts_dict[ixd]['xx_vec']
        Naux = x.shape[0]
        Maux = x.shape[1]
        test_list_M.append(Maux)
        test_list_N.append(Naux)
    if np.max(test_list_N)!=np.max(test_list_N) or np.max(test_list_M)!=np.max(test_list_M):
        print('error: loaded system has different number of banks or assets than the one specified in settings')
    else:
        M = Maux
        N = Naux
        HAIRCUTS = M*[0.0]  
    bank_names = optim_net_ts_dict[xdata[0]]['nodes']


if IFSAVEEXAMPLE==1:
    balance_sheet_file = open(FPATH / 'balance_sheet_example','wb')
    pickle.dump(optim_net_ts_dict,balance_sheet_file)
    balance_sheet_file.close()
    
if IFLOADEXAMPLE==1:
    with open(FPATH / "balance_sheet_example", "rb") as f:
        optim_net_ts_dict = pickle.load(f)    
    
# table with statistics
import networkx as nx

FRAC = 0.01

def funStats(x):
    xs = np.zeros((6,))
    xs[0] = np.min(x)
    xs[1] = np.percentile(x,10)
    xs[2] = np.mean(x)
    xs[3] = np.std(x)
    xs[4] = np.percentile(x,90)
    xs[5] = np.max(x)
    return xs, ['min','prc10','mean','std','prc90','max']

nb_stat = list()
na_stat = list()
ta_mean_stat = list()
degree_mean_stat = list()
bness_mean_stat = list()
eigen_mean_stat = list()

for ixd in xdata:
    xx_v = optim_net_ts_dict[ixd]['xx_vec']
    cc_v = optim_net_ts_dict[ixd]['cc_vec']
    nb_stat.append(xx_v.shape[0])
    na_stat.append(xx_v.shape[1])
    xta = np.sum(xx_v,1)+cc_v
    ta_mean_stat.append(np.mean(xta))
    net_mat = optim_net_ts_dict[ixd]['L_mat']
    net_edges_l_of_tuple = list()
    for ii in range(0,xx_v.shape[0]):
        for jj in range(0,xx_v.shape[0]):
            w = net_mat[ii][jj]
            if jj!=ii and w>FRAC*xta[ii]:
                net_edges_l_of_tuple.append((ii,jj,w))
    net_stat = nx.Graph()
    net_stat.add_weighted_edges_from(net_edges_l_of_tuple)
    degree_mean_stat.append(np.sum([x[1] for x in list(net_stat.degree)])/xx_v.shape[0])
    bness = nx.betweenness_centrality(net_stat)
    bness_mean_stat.append(np.sum([bness[x] for x in list(bness.keys())])/xx_v.shape[0])
    eigen = nx.eigenvector_centrality(net_stat)
    eigen_mean_stat.append(np.sum([eigen[x] for x in list(eigen.keys())])/xx_v.shape[0])    
    
stat_dict = dict()
stat_dict['total assets'] = funStats(ta_mean_stat)[0]
stat_dict['n banks'] = funStats(nb_stat)[0]
stat_dict['n assets'] = funStats(na_stat)[0]
stat_dict['ibank dgree'] = funStats(degree_mean_stat)[0]
stat_dict['bness centrality'] = funStats(bness_mean_stat)[0]
stat_dict['eigen centrality'] = funStats(eigen_mean_stat)[0]

statstable = pd.DataFrame.from_dict(stat_dict)
statstable.index = funStats(degree_mean_stat)[1]


## draw bipartite
if IF_DRAW_BIPARTITE == 1:
    xn = fun_bipirtite(bank_names, list(range(0,M)), x, THRESHOLD_BIPARTITE)
    pos = nx.bipartite_layout(xn,nodes=bank_names)
    fig, ax = plt.subplots()
    nx.draw(xn, pos=pos, ax=ax, with_labels=True)

##################
##### SIMMULATIONS
##################

#import copy
import time

IFONLYNONZEROOPTIMIZED = 1 #1 means that bank optimizes only those exposures that are non-zero at start

SLACKOFCB = 0.0 # baseline is 0.0, >0.0 only if you want the central bank to allow for some drop in prices
def randslack(maxs):
    np.random.seed(datetime.datetime.now().microsecond)
    return np.random.uniform(0,maxs)

IFPARALLEL = 0

IFTESTCALIB = 0
IFCALIBBETAGAMMA = 1 #0: only gamma is calibrated
IFOVERWRITEX = 0
IFINTERNALISE = 1
IFTMIN2TOL = 0 #stop convergence when \|y_{t-2}-y_t\|<tol
STOPMIN2TOL = 0.001

SHOCKSIMNAME = 'BA'
IFCOUNTRYORBANK = 0 #1: country, 0 bank

if IFINTERNALISE==0:
    nointeraction = 'no_interact'
else:
    nointeraction = 'strategic'

if IFPROPORTIONALSELLING==1:
    xliqstrat_str = 'PROP'
else:
    xliqstrat_str = 'OPTM'

if IFCENTRALBANK==0:
    if IFCOUNTRYORBANK==1:
        FILEALIAS = 'cntr_'+SHOCKSIMNAME+'_'+xliqstrat_str+'_'+nointeraction
    else:
        FILEALIAS = 'bank_'+SHOCKSIMNAME+'_'+xliqstrat_str+'_'+nointeraction
else:
    if IFCOUNTRYORBANK==1:
        FILEALIAS = 'cntr_cb_hq_'+SHOCKSIMNAME+'_'+xliqstrat_str+'_'+nointeraction
    else:
        FILEALIAS = 'bank_cb_hq_'+SHOCKSIMNAME+'_'+xliqstrat_str+'_'+nointeraction

        
COLLECTRANGE = 20

IFSAVECOLLECT = 0 #saving output

q_ts = dict()
p_ts = dict()
x_collect = dict()
y_collect = dict()
y_sum_collect = dict()
ycb_sum_collect = dict()

collect_q_min = dict()
collect_q_vol = dict()
collect_q_all_range = dict()

y_calib_collect = dict()
beta_calib_collect = dict()
gamma_calib_collect = dict()

q_dict_ts = dict()
p_dict_ts = dict()
c_dict_ts = dict()

XGRID = 1 #would be 10
YGRID = 1 #would be 10

#xxx = np.linspace(0.0005, 0.005, XGRID)
xxx = np.linspace(0.25, 0.25, XGRID)
#yyy = np.linspace(0.0005, 0.005, YGRID)
yyy = np.linspace(0.001, 0.001, YGRID)
tolandstep_m = np.meshgrid(xxx, yyy)
tolandstep = list(zip(*(x.flat for x in tolandstep_m)))

for xtenum,xtols in enumerate(tolandstep):
    print('outer loop ',xtenum)
    print(xtols)

    q_ts_eq = dict()
    p_ts_eq = dict()
    pBar_ts_eq = dict()
    idef_ts_first = dict()
    ratio = list()
    q_b_list = list()
    q_avg_ts_nsim = dict()
    
    heldtoinit = list() #list presenting ratios of held assets (y) to initial assets (x) aggregated across banks
    ycb_list = list()
    y_list = list()
    ycb_tatonn = list()
    ycb_tatonn_min = list()
    zl_list = list() #slack in the linear constraint of the central bank
    q_list = list()

    Ntatonn = 50 # 400 #how many steps in tattonement?
    NSIM_Z = 2 # how many versions of how to set z (external liabilities to be paid back)
    
    gamma_start = 2.0 # starting multiplier of gamma
    gamma_step = 0.0 #in baseline case set it to 0
    b_start = 1.0
    b_step = 0.0
    beta_start = 1.0
    beta_step = 0.0
    cb_start = 1.0
    cb_step = 0.0#-0.95
    cb_capacity_start = 1.0
    cb_capacity_step = -0.95
    
    period_count = 0
    cb_list = list()

    def opt_for_parallel(en):
        global period_count
        print('period_count: '+str(period_count))
        
        period = xdata[en]
        start_time = time.time()

        gamma_fact = gamma_start + gamma_step*period_count
        b_fact = b_start + b_step*period_count
        beta_fact = beta_start + beta_step*period_count
        
        
        xoptim_params = optim_net_ts_dict[period]
        x = xoptim_params['xx_vec']
        xr = xoptim_params['xr_vec']
        q0 = xoptim_params['q0_vec']

        #if period_count==0: print('all exp =', x.sum())

        #z_start = 0.0 #what fraction of total external liabilities to start with?
        z_start = np.asarray(x.shape[0]*[0.0])
        
        
        #zstep = 0.00 #how to increase the fraction in each step of NSIM_Z
        xcntr_list = [y for y,x in enumerate(xoptim_params['nodes']) if x in SHOCKEDBANKS]

        zstep = np.asarray(x.shape[0]*[0.0])
        for iz in xcntr_list: #UNICRED
            zstep[iz] = SHOCK_SIZE

        
        c = xoptim_params['cc_vec']
        #print(c)
        N = x.shape[0]
        M = x.shape[1]
        print('x shape')
        print((N,M))

        L = np.maximum(0.0,xoptim_params['L_mat'])

        z = RUNOFFRATE*xoptim_params['zz_vec']

        print('shocks applied: ', z)

        mu = xoptim_params['mu_vec']#returns

        b = xoptim_params['b_vec']*b_fact
        alpha = xoptim_params['b_vec']*b_fact
        Q = xoptim_params['Q_mat']

        e = xoptim_params['ee_vec']

        beta = xoptim_params['ta_beta'] # recovery rate of the revalued assets

        pbar = np.sum(L,axis=1)
        pBar = np.zeros((N,1))

        gamma = gamma_fact

        ## CALIBRATE gamma and beta
        print('here!')
        if IFTESTCALIB==1:
            gamma_v = np.zeros((1,N))
            beta_v = np.zeros((M,N))
            y_calib = np.zeros((M,N))
            for ixb in range(0,N):
                if IFONLYNONZEROOPTIMIZED==1:
                    nz_items = list(np.where(x[ixb]>0.0)[0])
                    M_nz = len(nz_items)
                    x_ixb_diag = np.diag(x[ixb][nz_items])
                    x_ixb = x[ixb][nz_items]
                    b_diag = np.diag(b[nz_items])
                    mu_nz = mu[nz_items]
                    Q_nz = Q[np.ix_(nz_items,nz_items)]
                    b_nz = b[nz_items]
                else:
                    nz_items = list(range(0,M))
                    x_ixb_diag = np.diag(x[ixb])
                    x_ixb = x[ixb]
                    b_diag = np.diag(b)
                    mu_nz = mu
                    b_nz = b
                    M_nz = M
                    Q_nz = Q
                    
                print('bank '+str(ixb))

                # CVXPY implementation of calib and solver
                xeye = np.eye(M_nz+1)
                # Define cvxpy variable
                yy = cp.Variable(M_nz+1)
                # xbeta and xg as in original code
                xbeta = xeye[0:M_nz] @ yy
                xg = xeye[M_nz] @ yy
                g = b_diag @ x_ixb_diag @ xbeta + xg * (Q_nz @ x_ixb) - mu_nz - b_diag @ x_ixb
                # Objective: squared norm
                objective = cp.Minimize(cp.sum_squares(g))
                # Constraints: -I <= yy <= [0,...,0,1,...,1]
                constraints = []
                constraints.append(yy >= 0)
                constraints.append(yy[0:M_nz] <= 1)
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.CLARABEL, verbose=False)
                y_beta_gamma = yy.value if yy.value is not None else np.zeros(M_nz+1)

                beta_v[nz_items,ixb] = y_beta_gamma[0:M_nz]
                gamma_v[0,ixb] = y_beta_gamma[M_nz]

                y_calib[nz_items,ixb] = np.linalg.inv(2.0*b_diag@np.diag(beta_v[nz_items,ixb]) + gamma_v[0,ixb]*Q_nz) @ (mu_nz + b_diag@(np.eye(M_nz)+np.diag(beta_v[nz_items,ixb]))@x_ixb)

        #BANKS
        def response_II(i,y,c,q,x,Pi,p,pBar,ycb): #for bank i

            global period_count

            Nb = x.shape[0] #number of banks
            Na = x.shape[1]
            if IFONLYNONZEROOPTIMIZED==1:
                nz_items = list(np.where(x[i]>0.0)[0])
                x_ixb = x[i][nz_items]
                Maux = len(nz_items)
                q_nz = q[nz_items]
            else:
                Maux = y.shape[1]
                nz_items = list(range(0,Maux))
                x_ixb = x[i][nz_items]
                q_nz = q
            Nmini = [x for x in range(0,Nb) if x!=i]

            rhs_ib = np.dot(Pi.T[i],p) #0 position is cash!
            rhs = rhs_ib+c[i]+np.dot(q,x[i])-pBar[i]

            if np.sum(x[i])>0:
                cash_to_be_raised = np.maximum(-(rhs_ib+c[i] - pBar[i]),0.0)/np.sum(x[i])
            else:
                cash_to_be_raised = 0.0

            #print(rhs)
            if rhs<=0.0:
                print('Shock for bank '+bank_names[i]+' surpassing its capacity to react')
                return np.array(Na*[0])
            else:
                
                x0 = x.copy()
                II = i
                yminII = y.copy()
 
                if IFTESTCALIB==1:
                    if (TURNOFFBENCHMARK+period_count)==0:
                        beta = np.minimum(1.0,beta_v[:,i])
                        gamma = gamma_v[0,i]
                    else:
                        beta = np.minimum(1.0,beta_v[:,i]*beta_fact)
                        gamma = gamma_v[0,i]*gamma_fact
                else:
                    beta = np.array(M*[beta_fact])
                    gamma = gamma_fact
 
                beta_nz = beta[nz_items]
                b_nz = b[nz_items]
                Q_nz = Q[np.ix_(nz_items,nz_items)]

                # CVXPY implementation for strategic optimization
                if IFPROPORTIONALSELLING == 1:
                    xII = x0[i]
                    ysol = xII * np.minimum(1.0, rhs / np.sum(xII))
                    ysol_array = np.asarray([0.0]*Na)
                    ysol_array = ysol
                else:
                    xII = x0[i][nz_items]
                    xepsilon = 0.02
                    NminII = [x for x in range(0,Nb) if x!=II]
                    Qaug = np.dot(np.diag(beta_nz), np.diag(b_nz)) + gamma/2.0 * Q_nz
                    if IFINTERNALISE==1 or (TURNOFFBENCHMARK+period_count)==0:
                        xsum_minII = np.sum(x0,axis=0)[nz_items] - np.sum(yminII[NminII],axis=0)[nz_items]
                        xsum_minII = xsum_minII - ycb[nz_items]
                    else:
                        xsum_minII = np.sum(x0,axis=0)[nz_items] - np.sum(x0[NminII],axis=0)[nz_items]
                    Muaug = mu[nz_items] + np.dot(np.diag(b_nz), xII + np.dot(np.diag(beta_nz), xsum_minII))
                    # Define cvxpy variable
                    yy = cp.Variable(Maux)
                    # Quadratic form: (1/2)yy^T Q yy - Muaug^T yy
                    # But our original objective is yy^T Q yy - Muaug^T yy, so we double Q for cvxpy
                    objective = cp.Minimize(cp.quad_form(yy, Qaug) - Muaug @ yy)
                    # Constraints: q_nz @ yy <= rhs, yy >= 0
                    constraints = [q_nz @ yy <= rhs, yy >= 0]
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                    ysol_array_aux = yy.value if yy.value is not None else np.zeros(Maux)
                    ysol_array = np.asarray([0.0]*Na)
                    ysol_array[nz_items] = ysol_array_aux
                return ysol_array 
        #CENTRAL BANKS
        def response_CB(y,c,q,x,Pi,p,pBar,cb_acted,ycb,cb_capacity): #for bank i

            Nb = y.shape[0] #number of banks
            Maux = y.shape[1]
            Nmini = [x for x in range(0,Nb)] #with  'if x!=i]' for BANKS

            rhs = cb_capacity+ CBUNLIMITEDPOOL # with '+c[i]+np.dot(q,x[i])-pBar[i]' for BANKS

            cash_to_be_raised = 0.0

            if rhs<=0.0:
                zl = 0.0
                return np.asarray(len(cb_acted)*[0.0]), zl
            else:
                #G = matrix(len(cb_acted)*[1.0]).T 
                #print(G)

                #modify globals
                x0 = x.copy()
                #II = i
                yminII = y.copy()
                
                slackunif = np.random.uniform(size=1,low=0.0,high=SLACKOFCB)[0]#randslack(SLACKOFCB)

                if IFTESTCALIB==1:
                    #print('update')
                    1==1

                # CVXPY implementation of utilCBqp and solver
                if len(cb_acted) > 0:
                    pr_impact_external = 1.0 - flin(x, y, b)
                    b_acted_vec = b[cb_acted]
                    slack = slackunif
                    # Define cvxpy variable
                    yy = cp.Variable(len(cb_acted))
                    # Objective: squared norm
                    objective = cp.Minimize(cp.sum_squares(pr_impact_external[cb_acted] - cp.multiply(b_acted_vec, yy) - slack))
                    # Constraints from G and h (assume G, h are compatible with cvxpy)
                    constraints = []
                    if IFCBONLYPURCHASE == 0:
                        constraints.append(q[cb_acted] @ yy <= rhs)
                    elif IFCBONLYPURCHASE == 1:
                        constraints.append(q[cb_acted] @ yy <= rhs)
                        constraints.append(yy >= 0)
                    elif IFCBONLYPURCHASE == 2:
                        constraints.append(q[cb_acted] @ yy <= rhs)
                        for nz in range(len(cb_acted)):
                            constraints.append(yy[nz] >= CBLOWERBOUND * np.sum(x[:, nz]))
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                    ysol_array_aux = yy.value if yy.value is not None else np.zeros(len(cb_acted))
                    ysol_array = np.asarray(Maux * [0.0])
                    ysol_array[cb_acted] = ysol_array_aux
                    zl = 0.0  # cvxpy does not provide duals in the same way
                else:
                    ysol_array = np.asarray(Maux * [0.0])
                    zl = 0.0

                return ysol_array, zl


        def iteration_utilII(y,c,q,Pi,x,pBar,alpha,ycb,cb_acted,cb_capacity,xt):

            global period_count
            
            Nb = y.shape[0]

            if IFDECAY == 1:
                ddt = 1.0/EXPO*xt**(-EXPO)*xtols[0] # version 2: step defined as fraction of x
                tol = xtols[1]
            else:
                ddt = xtols[0]
                tol = xtols[1] # tolerance for adjustment of y

            y_response = np.copy(y)#np.zeros(y.shape)
            y_response_aux = np.copy(y_response)

            p, idef = FDAq(c,x,q,Pi,pBar) #idef: indicator of default

            #print('ycb')
            #print(ycb)
            if RESPONSEMODE in [0,1]:
                xrange = range(0,Nb)
            if RESPONSEMODE == 2:
                xrange = perms = list(itertools.permutations(range(0, Nb)))
            for i in xrange:
                #print(' processed '+str(i))
                if RESPONSEMODE in [1,2]:
                    yinext = response_II(i,y_response,c,q,x,Pi,p,pBar,ycb)
                if RESPONSEMODE==0:
                    yinext = response_II(i,y,c,q,x,Pi,p,pBar,ycb)

                signif_i = (np.abs(yinext-y[i])>tol*x[i]) + 0
                sgn_i = np.sign(yinext-y[i])

                if IFQUANT == 0:
                    yinext = y[i] + ddt*signif_i*(yinext-y[i])
                else:
                    yinext = y[i] + ddt*signif_i*sgn_i*x[i]

                y_response[i] = yinext
                #y_response_unconstr[i] = yinext_unconstr
            if IFCENTRALBANK==1:
                ynext_cb, zl = response_CB(y_response,c,q,x,Pi,p,pBar,cb_acted,ycb,cb_capacity)
                #print(ynext_cb)
            else:
                zl = 0.0

            ynext = y_response

            if IFCENTRALBANK==1:
                print('CB mode')

                signif_cb = (np.abs(ynext_cb-ycb)>tol*ycb) + 0
                #sgn = np.sign(y_response-y)
                sgn_cb = np.sign(ynext_cb-ycb)
                if IFFASTCB == 0:
                    ynext_cb = ycb + ddt*signif_cb*sgn_cb*np.mean(x,0)
                else:
                    if IFFASTCB == 1:
                        ynext_cb = ycb + ddt*signif_cb*(ynext_cb-ycb)
                    else:
                        ynext_cb = ycb + (ynext_cb-ycb)
            if IFCENTRALBANK==0: #!TO CHECK - STH WRONG
                ynext_cb = 0.0*ynext[0] #only size we need from y_next?

            y_concat_bank_cb = np.concatenate((ynext,ynext_cb.reshape((1,M))),axis=0)
            #print(y_concat_bank_cb)
            qnext = flin(x,y_concat_bank_cb,alpha)
            #print(qnext)

            cnext = np.array([c[i] + np.dot(Pi.T[i],p) + np.dot(q,x[i]) - np.dot(q,ynext[i]) for i in range(0,Nb)])

            return ynext, cnext, qnext, p, idef, ynext_cb, zl

        q_mat_eq = np.zeros((NSIM_Z,M))
        p_mat_eq = np.zeros((NSIM_Z,N))
        c_mat_eq = np.zeros((NSIM_Z,N))
        q_mat_dict = dict()
        p_mat_dict = dict()
        c_mat_dict = dict()
        y_sum_collect_dict = dict()
        ycb_sum_collect_dict = dict()
        y_collect_dict = dict()
        q_matFAM_eq = np.zeros((NSIM_Z,M))
        y_mat_eq_aggr = np.zeros((NSIM_Z,M))
        pBar_mat_eq = np.zeros((NSIM_Z,N))
        idef_mat_first = np.zeros((NSIM_Z,N))
        q_avg_mat_nsim = np.zeros((Ntatonn+1,NSIM_Z))
        start_time = time.time()
        for kkk in range(0,NSIM_Z):
            #np.random.seed(108)
            print('period_count: ',str(period_count))
            print('banks and assets '+str(N)+' and '+str(M))
            
            Nb = N

            if IFOVERWRITEX == 1:
                print(x)
                x = y_calib.T.copy()
                print(x)

            z_frac = z_start+zstep*kkk
            
            cb_fact = cb_start + cb_step*kkk
            print((HAIRCUTS,M))
            cb_action_fact = [xz for xz in range(0,M) if HAIRCUTS[xz]<=cb_fact and xz not in EXCLUDEASSETFROMCB]

            cb_capacity_fact = cb_capacity_start + cb_capacity_step*kkk
            #cb_capacity_fact = cb_capacity_start + cb_capacity_step*period_count
            cb_list.append(cb_capacity_fact)
    
            print('z_frac: ',z_frac)
            print('gamma_fact: ',gamma_fact)
            print('beta_fact: ',beta_fact)
            print('cb_haircut', cb_fact)
            print('cb_capacity_fact',cb_capacity_fact)
            if b_step>0.0: print('b_fact: ',b_fact)

            #x0 = x.copy()
            yminII = xr.copy()
            II = 0
            c0 = c.copy()
            
            Pi=L.copy()
            #print(L)
            e0 = e.copy()
            pbar = np.sum(Pi,1)

            pBar = z_frac*z+pbar

            print(pBar)
            for i in range(0,Nb):
                Pi[i,i] = 0.0
                if np.sum(pBar[i])>0.0:
                    Pi[i] = Pi[i]/np.sum(pBar[i])

            #x_k = x0.copy()
            y_k = x.copy()
            c_k = c0.copy()
            p_k = pbar.copy()
            ycb_k = np.asarray(M*[0.0])
            k = 0 #steps

            q_k = q0.copy()

            D_k = set()

            cb_capacity_fix = cb_capacity_fact*np.sum(x)

            y_seq = list()
            y_stab_seq = list()
            y_stab_seq.append(y_k)
            ycb_seq = list()
            zl_seq = list()
            ycb_seq.append(ycb_k)
            p_seq = list()
            p_seq.append(pBar)
            c_mat = np.zeros((Ntatonn+1,Nb))
            c_mat[0,:] = c0
            q_mat = np.ones((Ntatonn+1,M))
            q_mat[0,:] = q0
            y_sum_mat = np.zeros((Ntatonn+1,M))
            ycb_sum_mat = np.zeros((Ntatonn+1,M))
            y_sum_mat[0,:] = np.sum(x,0)
            idef_mat = np.zeros((Ntatonn+1,Nb))
            x0_1 = np.sum(xr,0)

            #print(c.shape)
            #print(x.shape)
            pFAM,idefFAM,qFAM = FDAsale(c,x,Pi,pBar,alpha)

            convergencestop = 0
            if IFSTARTFROMSTEADY == 1 or CREATESTABLESYSTEM == 1:
                #temp_cb = IFCENTRALBANK
                #IFCENTRALBANK = 0 # turn off the CB temporarily
                q_mat = np.ones((Ntatonn+1,M))
                q_avg_mat = np.ones((Ntatonn+1,1))
                
                for t in range(0,NRUNSTOSTABILIZE):
                    #print('iteration pre-run '+str(t))
                    #cb_capacity_prerun = cb_capacity_start*np.sum(x)
                    y_k, c_k, q_k, p_k, idef, ycb_k, zl_k = iteration_utilII(y_k,c0,q_k,Pi,xr,pbar,alpha,ycb_k,cb_action_fact,cb_capacity_fix,t+1.0)
                    #print(ycb_k)
                    y_stab_seq.append(y_k)
                    y_min2 = y_stab_seq[-2]
                    if IF_TURNOFF_ITER_STATS==0:
                        xnorm_x_rel = np.linalg.norm(y_min2-y_k)/np.linalg.norm(y_min2)
                        print('pre-run xnorm_x_rel: ',xnorm_x_rel)

                q_mat[0,:] = q_k
                c_mat[0,:] = c_k
                #IFCENTRALBANK = 1
            
            if CREATESTABLESYSTEM==1:
                optim_dict_to_save_dict = optim_net_ts_dict[period].copy()
                optim_dict_to_save_dict['xx_vec'] = y_k.copy()
                optim_dict_to_save_dict['q0_vec'] = q_k.copy()
                #optim_dict_to_save_dict['cc_vec'] = c_k.copy()
                file_stable = 'balance_sheet_stable_system_'+period
                balance_sheet_file = open(FPATH / file_stable, 'wb')
                pickle.dump(optim_dict_to_save_dict, balance_sheet_file)
                balance_sheet_file.close()
                print('stable system created for period '+period+'!')

            if CREATESTABLESYSTEM==0:
                convergencestop = 0

                q_avg_mat = np.ones((Ntatonn+1,1))
                y_seq.append(y_k)

                for t in range(0,Ntatonn):
                    print('iteration '+str(t))
                    if convergencestop==0:
                        y_k, c_k, q_k, p_k, idef, ycb_k, zl_k = iteration_utilII(y_k,c0,q_k,Pi,xr,pBar,alpha,ycb_k,cb_action_fact,cb_capacity_fix,t+1.0)
                        #print(ycb_k)
                    else:
                        y_k = y_seq[-2]
                        p_k = p_seq[-2]
                        c_k = c_mat[t-2,:]
                        q_k = q_mat[t-2,:]
                        ycb_k = ycb_seq[-2]

                    y_seq.append(y_k)
                    ycb_seq.append(ycb_k)
                    p_seq.append(p_k)
                    zl_seq.append(zl_k)

                    c_mat[t+1,:] = c_k
                    q_mat[t+1,:] = q_k
                    #print(q_k)
                    
                    y_sum_mat[t+1,:] = np.sum(y_k,0)
                    ycb_sum_mat[t+1,:] = ycb_k

                    idef_mat[t+1,:] = idef

                    q_avg_mat[t+1,0] = np.dot(q_k,x0_1)/np.sum(xr)

                    y_min2 = y_seq[-2]
                    if IF_TURNOFF_ITER_STATS==0:
                        xnorm_x_rel = np.linalg.norm(y_min2-y_k)/np.linalg.norm(y_min2)
                        print('xnorm_x_rel, mean(q): ',xnorm_x_rel,np.mean(q_k))

                    if IFTMIN2TOL==1:
                        if t>2 and convergencestop==0:
                            y_min2 = y_seq[-2]
                            xnorm_rel = np.linalg.norm(y_min2-y_k)/np.linalg.norm(y_min2)
                            if xnorm_rel<STOPMIN2TOL:
                                convergencestop = 1
                                print('stopped at t = '+str(t))
                

            q_mat_eq[kkk,:] = q_k
            q_matFAM_eq[kkk,:] = qFAM
            p_mat_eq[kkk,:] = p_k
            c_mat_eq[kkk,:] = c_k
            pBar_mat_eq[kkk,:] = pBar
            
            y_mat_eq_aggr[kkk,:] = np.sum(y_k,0)
            q_avg_mat_nsim[:,kkk] = q_avg_mat[:,0]
            
            q_mat_dict['sim'+str(kkk)] = q_mat
            p_mat_dict['sim'+str(kkk)] = p_mat_eq
            c_mat_dict['sim'+str(kkk)] = c_mat
            
            y_sum_collect_dict['sim'+str(kkk)] = y_sum_mat
            ycb_sum_collect_dict['sim'+str(kkk)] = ycb_sum_mat
            y_collect_dict['sim'+str(kkk)] = y_k
            for kj in range(0,Nb):
                if sum(idef_mat[:,kj])>0.0:
                    idef_mat_first[kkk,kj] = np.where(idef_mat[:,kj]==1)[0][0]+1
                        
            period_count+=1
            print('idef: ',idef_mat_first[kkk,:])

        end_time = time.time()
        print(f"Execution time: {(end_time - start_time)/60.0:.4f} minutes")
        q_ts[period+'_'+str(xtenum)] = q_mat
        q_dict_ts[period+'_'+str(xtenum)] = q_mat_dict
        p_ts[period+'_'+str(xtenum)] = p_seq
        p_dict_ts[period+'_'+str(xtenum)] = p_mat_dict
        c_dict_ts[period+'_'+str(xtenum)] = c_mat_dict

        x_collect[period+'_'+str(xtenum)] = y_k.copy()
        y_collect[period+'_'+str(xtenum)] = y_collect_dict
        y_sum_collect[period+'_'+str(xtenum)] = y_sum_collect_dict
        ycb_sum_collect[period+'_'+str(xtenum)] = ycb_sum_collect_dict
        if IFTESTCALIB == 1:
            y_calib_collect[period+'_'+str(xtenum)] = y_calib.copy()
            beta_calib_collect[period+'_'+str(xtenum)] = beta_v.copy()
            gamma_calib_collect[period+'_'+str(xtenum)] = gamma_v.copy()

        collect_q_min[period+'_'+str(xtenum)] = np.min(np.dot(q_mat[(Ntatonn-5):Ntatonn,:],np.sum(x,0))/np.sum(x))
        collect_q_vol[period+'_'+str(xtenum)] = np.std(np.dot(q_mat[(Ntatonn-20):Ntatonn,:],np.sum(x,0))/np.sum(x))
        collect_q_all_range[period+'_'+str(xtenum)] = q_mat[(Ntatonn-COLLECTRANGE):Ntatonn,:]

    
    for idx in range(0,len(xdata)):
        opt_for_parallel(idx)

#Save output
today_d = datetime.datetime.now().day
if today_d<10:
    today_d = '0'+str(today_d)
today_m = datetime.datetime.now().month
if today_m<10:
    today_m = '0'+str(today_m)
today_y = datetime.datetime.now().year

xfile = 'q_ts_'+FILEALIAS+'_'+str(today_y)+str(today_m)+str(today_d)+'.xlsx'
convertDictNumpyToPd(q_ts).to_excel(FPATHOUTPUT / xfile)
xfile = 'p_ts_'+FILEALIAS+'_'+str(today_y)+str(today_m)+str(today_d)+'.xlsx'
convertDictNumpyToPd(p_ts).to_excel(FPATHOUTPUT / xfile)
xfile = 'x_collect_'+FILEALIAS+'_'+str(today_y)+str(today_m)+str(today_d)+'.xlsx'
convertDictNumpyToPdWithKeysCols(x_collect,bank_names).to_excel(FPATHOUTPUT / xfile)
xfile = 'y_collect_sim0_'+FILEALIAS+'_'+str(today_y)+str(today_m)+str(today_d)+'.xlsx'
convertDictNumpyToPdWithKeysCols_sim(y_collect,'sim0',bank_names).to_excel(FPATHOUTPUT / xfile)
xfile = 'y_collect_sim1_'+FILEALIAS+'_'+str(today_y)+str(today_m)+str(today_d)+'.xlsx'
convertDictNumpyToPdWithKeysCols_sim(y_collect,'sim1',bank_names).to_excel(FPATHOUTPUT / xfile)
