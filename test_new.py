import numpy as np
import H_real as hr
import os
from time import gmtime, strftime
import time
import params

LOCAL = os.path.abspath('.')

### PARAMETERS ###
Jzz = params.Jz
L = params.L
D = params.D
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
    hr.ExDiag(PATH_now,data[0],data[1], Jzz)
    print(L, D, time.clock()-AA)
