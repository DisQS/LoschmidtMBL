import numpy as np
import H_real as hr
import os
import time
import params

LOCAL = os.path.abspath('.')

### PARAMETERS ###
Jzz = params.Jz
L = params.L
D = params.D
D_q = params.D_quench
NN_RR = params.N_real

### PATH DIRECTORY FOR OUTPUTS ###
PATH_now = LOCAL
directory = 'DATA/L_'+str(L)+'/D_'+str(D)
PATH_now = LOCAL+os.sep+directory+os.sep
if not os.path.exists(PATH_now):
    os.makedirs(PATH_now)

### REALIZATION LOOP ###
for n in range(NN_RR):
    data = [L,D,n+1]
    AA=time.clock()
    hr.ExDiag(PATH_now,data[0],data[1], D_q, Jzz)
    print(L, D, time.clock()-AA)
