import numpy as np
import time
import scipy.special as special
import H_functions as hf
import H_ent as ent
import scipy.sparse as _sp
from scipy.sparse import linalg
import params

def ExDiag(PATH_now,L,D, D_quench, Jzz):
	t0=time.time()

	BC = params.BC
	Dis_gen = params.Dis_gen
	form_flag = params.Format_flag
	in_flag = params.In_flag
	LL = int(L)
	DD = float(D)
	DD_quench = float(D_quench)
	NN = int(LL/2)
	Dim = int(hf.comb(LL, NN))

	### BASE CREATION ###
	Base_num = hf.Base_prep(LL,NN)
	Base_bin = [int(Base_num[i], 2) for i in range(Dim)]
	Base_NumRes = hf.BaseNumRes_creation(Dim,LL,Base_num)

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
	if form_flag == 0:
		HAM = hf.Ham_Dense_Creation(LL,NN,Dim,DD,Jzz,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)
	else:
		HAM, dis0 = hf.Ham_Sparse_Creation(LL,NN,Dim,DD,Jzz,Dis_real,BC,Base_bin,Base_num,Hop_bin,LinTab)


	#### DIAGONALIZATION ###
	Eval, Evec = hf.eigval(HAM)

	### IF HAM DENSE, CALCULATE STATIC OBSERVABLES, IF SPARSE CALCULATE DYNAMICS ###
	if form_flag == 0:
		### NORMALIZE SPECTRUM ###
		E_norm = (Eval[1:]-min(Eval[1:]))/(max(Eval[1:])- min(Eval[1:]))

		### LEVEL STATISTICS (MIDDLE OF THE SPECTRUM) ###
		Eval_mid = E_norm[int(Dim/3):int(2*Dim/3)]
		ravg = hf.levstat(Eval_mid)
		nomefile_lev = str(PATH_now+'Levst_L'+str(LL)+'_D'+str(D)+'.dat')
		with open(nomefile_lev, 'a') as ee:
			ee.write('%f' % ravg +"\n")

		### INVERSE PARTICIPATION RATIO ###
		ipr = (1/Evec.shape[1])*(hf.InvPartRatio(Evec, HAM))
		nomefile_ipr = str(PATH_now+'IPR_L'+str(LL)+'_D'+str(D)+'.dat')
		with open(nomefile_ipr, 'a') as ee:
			ee.write('%f' % ipr + "\n")

	#Psi0 = hf.Psi_0(Dim, LL, Base_num, in_flag)
	#ProjPsi0 = hf.Proj_Psi0(Psi0, Evec)

	else:
		Psi0 = Evec[:,0]
		Diag_quench = HAM.diagonal()-(DD_quench*(np.asarray(dis0)))
		HAM.setdiag(Diag_quench)
		HAM_quench = HAM

		A = -1.0j*HAM_quench

		t_i = params.t_i
		steps = params.t_steps
		dt = 0.1
		t_f = dt*steps
		t_tab = np.linspace(t_i, t_f, steps)

		Psit = linalg.expm_multiply(A, Psi0, start=t_i, stop=t_f, num=steps, endpoint=True)
		SurvP = [(-1/LL)*np.log(hf.Loschmidt(Psit[i], Psi0)) for i in range(len(Psit))]

		nomefile_losch = str(PATH_now+'Losch.dat')
		np.savetxt(nomefile_losch, np.c_[SurvP, t_tab], fmt='%.9f')

	# ### TIME EVOLUTION AND OBSERVABLES ###
	# t_i   = params.t_i
	# t_f   = float(10.0)
	# Nstep = params.t_steps
	# tmp_tab = np.logspace(t_i, t_f, Nstep)
	# t_tab = np.sort(np.append(tmp_tab, [0, 0.25, 0.5, 0.75]))
	#
	# for t in tmp_tab:
	# 	Psit = hf.TimEvolve(ProjPsi0, Eval, t)
	#
	# 	Losch = hf.Loschmidt(Psit, ProjPsi0)
	# 	nomefile_losch = str(PATH_now+'Losch_t' + str(t) + '.dat')
	# 	with open(nomefile_losch, 'a') as file:
	# 		file.write('%f' % Losch +"\n")
	#
	# 	VnEnt = ent.compute_entanglement_entropy(Psit, LL, NN, NN)
	# 	nomefile_ent = str(PATH_now+'Ent_t' + str(t) + '.dat')
	# 	with open(nomefile_ent, 'a') as file:
	# 		file.write('%f' % VnEnt +"\n")
	#
	# 	Exp_Sz, Tot_Sz = hf.magnetization(Psit, Base_NumRes)
	# 	nomefile_mag_profile = str(PATH_now+'Mag_t' + str(t) + '.dat')
	# 	nomefile_totmag = str(PATH_now+'SzHalf_t' + str(t) + '.dat')
	# 	with open(nomefile_mag_profile, 'w') as file:
	# 		for i in range(len(Exp_Sz)):
	# 			file.write('%f' % np.real(Exp_Sz[i]) +'   %i' % int(i+1) + '   %f'% t +'   %f' %np.sum(np.real(Exp_Sz))+"\n")
	# 	with open(nomefile_totmag, 'w') as file:
	# 		file.write('%f' % Tot_Sz + '\n')

	return 1
