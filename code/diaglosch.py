import numpy as np
import time
import scipy.special as special
import functions as ff

def ExDiag(PATH_now,L,D,D1):
	t0=time.time()

	#### BOUNDARY CONDITIONS (0 ---> open, 1 ---> periodic) ###
	BC = 0

	### DISORDER TYPE (0 ---> random, 1 ---> quasiperiodic) ###
	Dis_gen = 0

	### HILBERT SPACE DIMENSION ###
	LL = int(L)
	DD = float(D)
	DD1 = float(D1)
	NN = int(LL/2)
	Dim = int(ff.comb(LL, NN))

	### BASE CREATION ###
	Base_num = ff.Base_prep(LL,NN)
	Base_bin = [int(Base_num[i], 2) for i in range(Dim)]

	### HOPPING CREATION ###
	if BC == 1:
		Hop_dim=LL-1
	else:
		Hop_dim=LL

	Hop_num = ff.Hop_prep(LL,BC)
	Hop_bin = [int(Hop_num[i],2) for i in range(Hop_dim)]

	### LOOKUP TABLE CREATION ###
	LinTab = ff.LinTab_Creation(int(LL),Base_num,Dim)

	### DISORDER CREATION ###
	Dis_real = ff.Dis_Creation(LL,Dis_gen)

	### CREATION OF HAMILTONIAN MATRICES ###
	HAM = ff.Ham_Dense_Creation(LL,NN,Dim,DD,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)
	#HAM1 = ff.Ham_Dense_Creation(LL,NN,Dim,DD1,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)

	#### DIAGONALIZATION ###
	Eval, Evec = ff.eigval(HAM)
	#Eval1, Evec1 = ff.eigval(HAM1)

	### NORMALIZED SPECTRUM ###
	E_norm = (Eval[1:]-min(Eval[1:]))/(max(Eval[1:])- min(Eval[1:]))

	### LEVEL STATISTICS (MIDDLE OF THE SPECTRUM) ###
	Eval_mid = E_norm[int(Dim/3):int(2*Dim/3)]
	#r, ravg = ff.levstat(Eval_mid)
	delta = Eval_mid[1:]-Eval_mid[:-1]
	r = list(map(lambda x,y: min(x,y)*1./max(x,y), delta[1:], delta[:-1]))
	ravg = np.mean(r)

	nomefile_lev = str(PATH_now+'levst_L'+str(LL)+'_D'+str(D)+'.dat')
	with open(nomefile_lev, 'a') as ee:
		ee.write('%f' % ravg +"\n")

	## INVERSE PARTICIPATION RATIO ###
	#ipr = ff.InvPartRatio(Evec)
	#ipr_norm = (1/Dim)*ipr
	#avg_ipr = np.mean(ipr)
	#nomefile_ipr_avg = str(PATH_now+'IPRavg_L'+str(LL)+'_D'+str(D)+'.dat')
	#with open(nomefile_ipr, 'a') as ee:
	#	ee.write('%f' % avg_ipr % "\n")

	#.............................Time Evolution starting from a random state
	t_i   = float(1.0)
	t_f   = float(1.0)
	Nstep = int(1)
	t_tab = np.linspace(t_i, t_f, Nstep)
	print(t_tab)
	
	Psi0 = ff.Psi_0(Dim)
	ProjPsi0 = ff.Proj_Psi0(Psi0, Evec)
	Psit = ff.TimEvolve(ProjPsi0, Eval, t_i)
	G = np.dot(ProjPsi0, Psit)
	Losch = (abs(G))**2
	print(Losch)
	#Losch = ff.Loschmidt(Psit, ProjPsi0))

	for t in t_tab:
		Psit = ff.TimEvolve(ProjPsi0, Eval, t)
		Losch = ff.Loschmidt(Psit, ProjPsi0)
		#nomefile_losch = str(PATH_now+'losch_L'+str(LL)+'.dat')
		#with open(nomefile_losch, 'a') as file:
		#	file.write('%f' % L +"\n")
