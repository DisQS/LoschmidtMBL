import numpy as np
import H_real as hr
import os
from time import gmtime, strftime
import time

LOCAL = os.path.abspath('.')

### INTERACTIONS ###
Jzz = 1.0

### SIZE ###
L_i = 8
L_f = 8
L_D = 2

L_n = int(1+(L_f-L_i)/L_D)
L_tab = [int(L_i+j*L_D) for j in range(L_n)]

for L in L_tab:
	nomefile = 'LevStat_'+str(L)+'.npy'

### DISORDER ###
D_i = 1.0
D_f = 1.0
D_D = 0.5

D_n = int(1+(D_f-D_i)/D_D)
D_tab = [D_i+j*D_D for j in range(D_n)]

PATH_now = LOCAL

### NUMBER OF REAL ###
NN_RR = 500

for i in L_tab:
	for j in D_tab:
		directory = 'DATA/L_'+str(i)+'/D_'+str(j)
		PATH_now = LOCAL+os.sep+directory+os.sep
		if not os.path.exists(PATH_now):
			os.makedirs(PATH_now)
		for n in range(NN_RR):
			data = [i,j,n+1]
			AA=time.clock()
			hr.ExDiag(PATH_now,data[0],data[1], Jzz)
			print(i, j, time.clock()-AA)

	#n0 += 1
