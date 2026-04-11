# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:50:39 2025

@author: halajgr
"""

#Auxiliary functions
def convertDictNumpyToPd(dn): #converts a dictionary of numpy arrays to pandas DataFrame
    import pandas as pd    
    #dn: dictionary of numpy arrays (must have the same number of columns)
    
    xkeys = list(dn.keys())
    
    df = pd.DataFrame()
    
    for xk in xkeys:
        xn = dn[xk]
        xn2df = pd.DataFrame(xn)
        xn2df['row'] = xn2df.index
        xn2df['key'] = xk
        xn2df.set_index(['key','row'],inplace=True)
        df = pd.concat([df, xn2df], axis = 0)
    return df

def convertDictNumpyToPdWithKeysCols(dn,rows): #converts a dictionary of numpy arrays to pandas DataFrame
    import pandas as pd    
    #dn: dictionary of numpy arrays (must have the same number of columns)
    
    xkeys = list(dn.keys())
    
    df = pd.DataFrame()
    
    for xk in xkeys:
        xn = dn[xk]
        xn2df = pd.DataFrame(xn)
        xn2df['row'] = rows
        xn2df['key'] = xk
        xn2df.set_index(['key','row'],inplace=True)
        df = pd.concat([df, xn2df], axis = 0)
    return df

def convertDictNumpyToPd_sim(dn,sim): #converts a dictionary of numpy arrays to pandas DataFrame, using a sim sub-dictionary
    #dn: dictionary of numpy arrays (must have the same number of columns)
    import pandas as pd
    xkeys = list(dn.keys())
    
    df = pd.DataFrame()
    
    for xk in xkeys:
        xn = dn[xk][sim]
        xn2df = pd.DataFrame(xn)
        xn2df['row'] = xn2df.index
        xn2df['key'] = xk
        xn2df.set_index(['key','row'],inplace=True)
        df = pd.concat([df, xn2df], axis = 0)
    return df

def convertDictNumpyToPdWithKeysCols_sim(dn,sim,rows): #converts a dictionary of numpy arrays to pandas DataFrame, using a sim sub-dictionary
    #dn: dictionary of numpy arrays (must have the same number of columns)
    import pandas as pd
    xkeys = list(dn.keys())
    
    df = pd.DataFrame()
    
    for xk in xkeys:
        xn = dn[xk][sim]
        xn2df = pd.DataFrame(xn)
        xn2df['row'] = rows
        xn2df['key'] = xk
        xn2df.set_index(['key','row'],inplace=True)
        df = pd.concat([df, xn2df], axis = 0)
    return df

# generate portfolios trying to best match cosine similarity measures

def funCosine(a,b):
    import numpy as np
    return np.sum(a*b)/(np.sqrt(a@a)*np.sqrt(b@b))

def funPorfCosineSimilarity(x):
    import numpy as np
    #x: portfolios (in columns) of banks (in rows)
    sx = x.shape
    mat = np.zeros((sx[0],sx[0]))
    
    for ii in range(0,sx[0]):
        for jj in range(0,sx[0]):
            if np.sum(x[ii])!=0.0 and np.sum(x[jj])!=0.0:
                mat[ii,jj] = funCosine(x[ii],x[jj])
            
    return mat

def funVecCosineSim(v,xc,alg='e'):
    import numpy as np
    
    SMALLNUMBER = 1e-9
    SMALLNUMBER2 = 1e-3
    #alg='e': expensive algorithm sampling until vector r\in R^na_+
    #
    bottom_n_iter = 100
    
    sv = v.shape[0]
    
    w = np.asarray(sv*[-1])
    xiter = 0
    while np.sum(w<SMALLNUMBER)>0.2*sv and xiter<bottom_n_iter:
        u = v/np.linalg.norm(v)
    
        r = np.random.uniform(0,1,size=(v.shape[0],))
    
        while np.linalg.norm(r)<=SMALLNUMBER:
            r = np.random.uniform(0,1,size=(v.shape[0],))
    
        uperp = r - r.dot(u)*u
    
        uperp = uperp/ np.linalg.norm(uperp)
    
        w = xc*u+np.sqrt(1-xc**2)*uperp
        #print(w)
        xiter+=1
    
    #print(np.sum(w<0))
    return np.maximum(w,0.0)


def funOverlapPortfoliosV3(sec,ns,thresh=0.1):
    import numpy as np
    
    nb = sec.shape[0]
    xdistr_rand = np.random.uniform(0.0,1.0,size=(nb,ns))
    xdistr_rand[np.where(xdistr_rand<thresh)]=0.0
    xdistr_rand = xdistr_rand/np.tile(np.sum(xdistr_rand,1),(ns,1)).T
    
    mat = xdistr_rand*np.tile(sec,(ns,1)).T
    
    return mat
    

def mLendIBank(ia, il, nodes, cap, pm, pm_r, pm_c, r_nstr, c_nstr):
    #Halaj & Kok (2013), Computational Management Science
    #https://doi.org/10.1007/s10287-013-0168-4
    
    #uses geographical maps, 2-digit country code and a third char for c: core, p: periphery
    
    import networkx as nx
    import numpy as np

    #ia = bs_structures[15,:]
    #il = bs_structures[23,:]
    #nodes = list_names_str

    FRACCAP = 0.00001

    LBUNIF = 0.01
    UBUNIF = 0.20

    REPETS = 100000
    # print(ia)
    aia = ia.copy()
    ail = il.copy()

    SYSSIZE = ia.shape[0]

    Net = nx.DiGraph()
    Net.add_nodes_from(nodes)

    tpl_wght = list()
    tpl_edges = list()

    ridx = np.random.randint(0, SYSSIZE, size=(REPETS, 2))
    ridx = np.asarray([x for x in ridx if x[0]!=x[1]])
    REPETS = ridx.shape[0]

    dict_cap = dict((x, y) for x, y in zip(nodes, cap))
    if r_nstr==0:
        dict_pm_r = dict((y, x) for x, y in enumerate(pm_r))
    else:
        dict_pm_r = dict((y[0:r_nstr], x) for x, y in enumerate(pm_r))
    if c_nstr==0:
        dict_pm_c = dict((y, x) for x, y in enumerate(pm_c))
    else:
        dict_pm_c = dict((y[0:c_nstr], x) for x, y in enumerate(pm_c))
    #print(dict_pm_r)
    #print(dict_pm_c)

    ii = 0
    while ii < REPETS and sum(aia) > 0.0 and sum(ail) > 0.0:
        # print(ii)
        x = ridx[ii, 0]
        y = ridx[ii, 1]

        nin = nodes[x]
        nout = nodes[y]
        #if x==6:
        #    print(nin,nout)
        xr = np.random.uniform(LBUNIF, UBUNIF)

        try:
            if r_nstr == 0:
                pm_pos_r = dict_pm_r[nin]
            else:
                pm_pos_r = dict_pm_r[nin[0:r_nstr]]
            # print('lala')
        except KeyError:
            pm_pos_r = -1
        try:
            if c_nstr == 0:
                pm_pos_c = dict_pm_c[nout]  # country code
            else:
                pm_pos_c = dict_pm_c[nout[0:c_nstr]]  # country code
        except:
            pm_pos_c = -1
        #print(pm_pos_r)
        #print(pm_pos_c)
        if pm_pos_r > -1 and pm_pos_c > -1:
            #if x==6:
            #    print(pm[pm_pos_r][pm_pos_c])
            if np.random.uniform(0, 1) < pm[pm_pos_r][pm_pos_c]:

                xexp = xr * min(aia[x], ail[y])
                
                aia[x] += -xexp
                ail[y] += -xexp

                if not ([nin, nout] in tpl_edges):  # reverse order
                    xedge = tuple([nin, nout, {'weight': xexp}])
                    tpl_wght.append(xedge)
                    tpl_edges.append([nin, nout])
                    #print(('edge ',nin,nout,' added with',float(row1[8]),' weight!'))
                else:
                    #print(('edge ',nin,nout,' exists!'))
                    vale = [xx[2] for xx in tpl_wght if xx[0] == nin and xx[1] == nout][0]['weight']
                    tpl_wght.remove(tuple([nin, nout, {'weight': vale}]))
                    xedge = tuple([nin, nout, {'weight': vale + xexp}])
                    tpl_wght.append(xedge)

        ii += 1

    Net.add_edges_from(tpl_wght)
    Net_aux = Net.copy()

    for xe in Net_aux.edges():
        #print((dict_cap[xe[0]],Net[xe[0]][xe[1]]['weight']))
        if Net[xe[0]][xe[1]]['weight']<FRACCAP*dict_cap[xe[0]]:
            Net.remove_edge(xe[0],xe[1])
    
    return Net


# DEFINITIONS OF FUNCTIONS/ METHODS

#y - array, each row i corresponds to a bank i, and its lenght is the size of non-interbank assets

def f(x,y,alpha):
    import numpy as np
    return np.exp(-np.multiply(alpha,np.sum(x-y,0))) #alphas are asset specific

def flin(x,y,b):
    import numpy as np
    return 1.0-np.multiply(b,np.sum(x,0)-np.sum(y,0)) #alphas are asset specific

def FDAq(c,x,q,Pi,pBar):
    import numpy as np
    M = x.shape[1]
    Nb = x.shape[0]
    #print(N)
    I = np.identity(Nb)
    PiT = Pi.transpose()
    
    iter = 1
    p = pBar
    D = set()
    
    capacity = c + np.dot(x,q)
    E = capacity + PiT @ p - pBar
    D2 = set(np.where(E < 0)[0])
    #print(D)
    #print(D2)
    
    idef = Nb*[0]
    while (not D2.issubset(D)) or (not D.issubset(D2)):
        #print('truncation of payments')
        idef = [1 if x in D2 else 0 for x in range(0,Nb)]
        
        Lam = np.diag(idef)
        #print(Lam)
        #print(Lam @ (c+np.dot(x,q)))
        
        p = np.linalg.inv(I - Lam @ PiT) @ ((I - Lam) @ pBar + Lam @ (c+np.dot(x,q)))
        E = capacity + PiT @ p - pBar
        D = D2.copy()
        D2 = set(np.where(E < 0)[0])
        
        
    return p, idef

# clearing with proportional sales

def FDAsale(c,x,Pi,pBar,b):
    import numpy as np
    # System with n banks and m illiquid assets
    # x = n by 1 vector of liquid assets (cash)
    # S = n by m matrix of illiquid holdings
    # Pi = n by n relative liabilities matrix
    # pbar = n by 1 vector of total liabilities
    # f = function handle for inverse demand function (ex: f = @(s)ones(m,1)-C*s for linear impacts C)

    Nb,M = x.shape
    tol = 1e-10
    
    I = np.identity(Nb)

    xiter = 1
    p = pBar.copy()
    q = flin(x,x,b)
    D = set()
    D2 = set(np.where((c + Pi.T@p + x@q) - pBar < tol)[0])
    while (not D2.issubset(D)) or (not D.issubset(D2)) or xiter < 2:
        xiter+=1;
        idef = [1 if x in D2 else 0 for x in range(0,Nb)]
        Lam = np.diag(idef)
        q1 = q.copy()
        q2 = np.zeros(q.shape)
        while np.linalg.norm(q1 - q2) > 1e-8:
            q2 = q1.copy()
            
            p = np.linalg.inv(I - Lam@Pi.T)@((I-Lam)@pBar + Lam@(c+np.dot(x,q1)))
            
            div = np.divide(np.maximum(pBar - c - Pi.T@p , np.asarray(Nb*[0.0])),np.dot(x,q1))
            #print(div.shape)
            divtile = np.tile(div,(M, 1)).T
            #print(divtile.shape)
            g = np.multiply(divtile , x)
            g[np.isnan(g)] = 0.0
            q1 = flin(x,x-np.minimum(x,g),b)
        
        q = q1.copy();
        p = np.linalg.inv(I - Lam@Pi.T)@((I-Lam)@pBar + Lam@(c+np.dot(x,q)))
        D = D2.copy()
        D2 = set(np.where((c + Pi.T@p + x@q) - pBar < tol)[0])
    return p, idef, q

def fun_bipirtite(banks, assets, mat, thresh):
    import networkx as nx
    b = nx.Graph()
    b.add_nodes_from(banks,bipartite=0)
    b.add_nodes_from(assets,bipartite=1)
    list_e = list()
    for y1,y2 in enumerate(banks):
        for z1,z2 in enumerate(assets):
            if mat[y1][z1]>thresh:
                list_e.append((y2,z2))
    b.add_edges_from(list_e)
    return b

def best_response(xo_params,y): #for bank i

    import numpy as np
    import cvxpy as cp

    x = xo_params['xx_vec']

    bank_names = xo_params['nodes']
    L = xo_params['L_mat']
    c = xo_params['cc_vec']

    beta = xo_params['ta_beta']
    gamma = xo_params['gamma']
    b = xo_params['b_vec']
    mu = xo_params['mu_vec']
    Q = xo_params['Q_mat']

    Nb = x.shape[0] #number of banks
    Na = x.shape[1]

    q = np.asarray(Na*[1.0])

    Pi = L.copy()
    pBar = np.sum(Pi,1)

    list_ysol = list()
    for i in range(0,Nb):
        nz_items = list(np.where(x[i]>0.0)[0])
        x_ixb = x[i][nz_items]
        Maux = len(nz_items)
        q_nz = q[nz_items]

        Nmini = [x for x in range(0,Nb) if x!=i]

        p, idef = FDAq(c,x,q,Pi,pBar)

        rhs_ib = np.dot(Pi.T[i],p) #0 position is cash!
        rhs = rhs_ib+c[i]+np.dot(q,x[i])-pBar[i]

        if np.sum(x[i])>0:
            cash_to_be_raised = np.maximum(-(rhs_ib+c[i] - pBar[i]),0.0)/np.sum(x[i])
        else:
            cash_to_be_raised = 0.0

        #print(rhs)
        if rhs<=0.0:
            print('Shock for bank '+bank_names[i]+' surpassing its capacity to react')
            list_ysol.append(np.array(Na*[0]))
        else:
            
            x0 = x.copy()
            II = i
            yminII = y.copy()

            beta_nz = beta[nz_items]
            b_nz = b[nz_items]
            Q_nz = Q[np.ix_(nz_items,nz_items)]

            # CVXPY implementation for strategic optimization
            xII = x0[i][nz_items]
            xepsilon = 0.02
            NminII = [x for x in range(0,Nb) if x!=II]
            Qaug = np.dot(np.diag(beta_nz), np.diag(b_nz)) + gamma/2.0 * Q_nz

            xsum_minII = np.sum(x0,axis=0)[nz_items] - np.sum(yminII[NminII],axis=0)[nz_items]

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

            list_ysol.append(ysol_array)
        ysol_ndarray = np.stack(list_ysol)
    return ysol_ndarray 


def load_system_unpickled(FPATH,period):
    import pickle
    file_stable = 'balance_sheet_stable_system_' + period
    balance_sheet_file = open(FPATH / file_stable, 'rb')
    optim_dict_to_load = pickle.load(balance_sheet_file)
    balance_sheet_file.close()
    return optim_dict_to_load