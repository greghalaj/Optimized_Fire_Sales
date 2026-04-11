# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:41:18 2025

@author: Grzegorz Halaj (https://www.ecb.europa.eu/pub/research/authors/profiles/grzegorz-halaj.en.html)
"""

#config

IF_DRAW_BIPARTITE = 0 #draw bipartite (1) or not (0), just for visualization of the interbank network
THRESHOLD_BIPARTITE = 10.0 #threshold set for drawing edges of the bank-asset network

### the next toggle allows a user to create a system that is stable, ie absent a shock the optimum is near the observed starting point structure.
#It is a useful feature to control the experiments,otherwise, it might be challenging to disentangle the effects of the shock and the rebalancing related to unstable (not in equilibrium) system. Toggle set to 1 might be necessary for comparative statics
CREATESTABLESYSTEM = 0 #create a stable system and save it in the selected location (1) or not, i.e., use the simulated one (0) 

IFSTARTFROMSTEADY = 0 # start at steady state of no shock case

NRUNSTOSTABILIZE = 50 #number of runs to stabilize the system before the shock is applied, e.g., 20

EXAMPLE_FILE = 'blance_sheet_example'
IF_LOAD_EXAMPLE = 0 #load the example file from EXAMPLE_FILE (1) or not (0); if 1, the file should be in the selected location; if 0, the system will be simulated with the parameters set below
IF_LOAD_FROM_PICKLE = 1 #load the system from a pickle file (1) or not (0); if 1, the file should have been created with CREATESTABLESYSTEM=1 and saved in the selected location; if 0, the system will be simulated with the parameters set below

#set the number of banks in the system, e.g., 100
NUMBER_OF_BANKS = 20
#print(NUMBER_OF_BANKS)
#set the number of assets types each banks hold in their portfolios that can be used to cover liquidity needs, e.g., 40
NUMBER_OF_ASSETS = 40

#dynamics of responses
RESPONSEMODE = 1 #0: independently, i.e., loop for all banks and only after update y, 1: synchronously, i.e., update y after each bank's response; 2: randomly, i.e., update y after each bank's response but in random order

IFPROPORTIONALSELLING = 0 # 1: proportional selling, 0: optimized
IFINTERNALISE = 1 # 1: strategic, 0: banks decide about how and what to transact irrespective of the peers

NTATONNEMENT = 100 #number of iterations to run the system to reach equilibrium in tatonnement steps, e.g., 100

#set a linear price impact function by choosing the impact (in bps) of how much the price would change in case of selling of 1% of the total volume of securities 
PRICE_IMPACT_PER_PRTC_SOLD = 20# in basis points

NSIM_Z = 2 #number of parameterisation of the shocks to funding sources (in this way, on can run on a grid of shocks to see the sensitivity of the results to the shock structure)

#in basis points, list impact for assets 0 to NUMBER_OF_ASSETS
PRICE_IMPACT_FUNCTION = {1:50,5:40,3:60} 

#activate the central bank to transact in order to stabilise prices
IFCENTRALBANK = 0 #0: the central bank is passive; 1: the CB can take the slack of the demand/ supply of the securities

# list banks that would be shocked with funding outflow
SHOCKED_BANKS = ['PBp','AAc','BAc','CAc','DAc','EAc']

#shock size (i.e., percentage of uncecured funding runnig off, in %)
SHOCK_SIZE = 20.00

IF_TURNOFF_ITER_STATS = 1 # 0: basic statistics of the iterations are displayed

