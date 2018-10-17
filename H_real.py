import numpy as np
import time
import scipy.special as special
import H_functions as hf
import H_ent as ent

def ExDiag(PATH_now,L,D):
	t0=time.time()

	#### BOUNDARY CONDITIONS (0 ---> open, 1 ---> periodic) ###
	BC = 0

	### DISORDER FLAG (0 ---> random, 1 ---> quasiperiodic) ###
	Dis_gen = 0

	### HILBERT SPACE DIMENSION ###
	LL = int(L)
	DD = float(D)
	#DD1 = float(D1)
	NN = int(LL/2)
	Dim = int(hf.comb(LL, NN))

	### BASE CREATION ###
	Base_num = hf.Base_prep(LL,NN)
	Base_bin = [int(Base_num[i], 2) for i in range(Dim)]

	### HOPPING CREATION ###
	if BC == 1:
		Hop_dim=LL-1
	else:
		Hop_dim=LL

	Hop_num = hf.Hop_prep(LL,BC)
	Hop_bin = [int(Hop_num[i],2) for i in range(Hop_dim)]

	### LOOKUP TABLE CREATION ###
	LinTab = hf.LinTab_Creation(int(LL),Base_num,Dim)

	### DISORDER CREATION ###
	Dis_real = hf.Dis_Creation(LL,Dis_gen)

	### CREATION OF HAMILTONIAN MATRICES ###
	HAM = hf.Ham_Dense_Creation(LL,NN,Dim,DD,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)
	#HAM1 = hf.Ham_Dense_Creation(LL,NN,Dim,DD1,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)

	#### DIAGONALIZATION ###
	Eval, Evec = hf.eigval(HAM)
	#Eval1, Evec1 = hf.eigval(HAM1)

	### NORMALIZE SPECTRUM ###
	E_norm = (Eval[1:]-min(Eval[1:]))/(max(Eval[1:])- min(Eval[1:]))

	### LEVEL STATISTICS (MIDDLE OF THE SPECTRUM) ###
	#Eval_mid = E_norm[int(Dim/3):int(2*Dim/3)]
	#ravg = hf.levstat(Eval_mid)
	#nomefile_lev = str(PATH_now+'Levst_L'+str(LL)+'_D'+str(D)+'.dat')
	#with open(nomefile_lev, 'a') as ee:
	#	ee.write('%f' % ravg +"\n")

	# INVERSE PARTICIPATION RATIO ###
	#ipr = hf.InvPartRatio(Evec)
	#ipr_norm = (1/Dim)*ipr
	#avg_ipr = np.mean(ipr)
	#nomefile_ipr = str(PATH_now+'IPRavg_L'+str(LL)+'_D'+str(D)+'.dat')
	#with open(nomefile_ipr, 'a') as ee:
	#	ee.write('%f' % avg_ipr % "\n")

	#.............................Time Evolution starting from a random state
	t_i   = float(0.0)
	t_f   = float(10.0)
	Nstep = int(50)
	tmp_tab = np.logspace(t_i, t_f, Nstep)
	t_tab = np.sort(np.append(tmp_tab, [0, 0.25, 0.5, 0.75]))

	in_flag = 1

	Psi0 = hf.Psi_0(Dim, LL, Base_num, in_flag)
	ProjPsi0 = hf.Proj_Psi0(Psi0, Evec)

	for t in t_tab:
		Psit = hf.TimEvolve(ProjPsi0, Eval, t)
		#Losch = hf.Loschmidt(Psit, ProjPsi0)
		#nomefile_losch = str(PATH_now+'Losch_t' + str(t) + '.dat')
		#with open(nomefile_losch, 'a') as file:
		#	file.write('%f' % Losch +"\n")
		VnEnt = ent.compute_entanglement_entropy(Psit, LL, NN, NN)
		nomefile_ent = str(PATH_now+'Ent_t' + str(t) + '.dat')
		with open(nomefile_ent, 'a') as file:
			file.write('%f' % VnEnt +"\n")
