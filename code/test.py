import numpy as np
import diaglosch as diag
import functions as ff
import os
from time import gmtime, strftime
import time

LOCAL = os.path.abspath('.')

#....................................SIZE
L_i = 6
L_f = 6
L_D = 2

L_n = int(1+(L_f-L_i)/L_D)

#print L_i, L_f, L_n
L_tab = [int(L_i+j*L_D) for j in range(L_n)]

for L in L_tab:
	nomefile = 'LevStat_'+str(L)+'.npy'

#....................................DISORDER
D_i = 2.0
D_f = 2.0
D_D = 0.25

D_n = int(1+(D_f-D_i)/D_D)

#print D_i, D_f, D_n
D_tab = [D_i+j*D_D for j in range(D_n)]

D1 = 2
#....................................REAL


PATH_now = LOCAL

NN_RR = 1

for i in L_tab:

	for j in D_tab:
		#directory = 'DATA/L_'+str(i)+'/D_'+str(j)
		directory = 'DATA/'
		PATH_now = LOCAL+os.sep+directory+os.sep
		if not os.path.exists(PATH_now):
			os.makedirs(PATH_now)

		for n in range(NN_RR):
			data = [i,j,n+1]
			nomefile = 'LevStat_'+str(i)+str(j)+'.npy'
			AA=time.clock()
			diag.ExDiag(PATH_now,data[0],data[1], D1)
			print(time.clock()-AA)

	#n0 += 1
