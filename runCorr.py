from quspin.operators import hamiltonian, exp_op, quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time # entropies
import numpy as np # generic math functions
from quspin.tools.evolution import expm_multiply_parallel
from numpy.random import ranf, seed, uniform, choice # pseudo random numbers
import scipy
import scipy.linalg as _la
from scipy.sparse import linalg
import math
import sys,os
from time import time
import h_functions as hf
import params

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

### BOUNDARY CONDITIONS
BC = 1 # 1 --> OPEN, 0 ---> PERIODIC
### DISORDER TYPE ###
dis_flag = 1 # 1 --> QUASIPERIODIC, 0 ---> RANDOM
### INITIAL STATE ###
in_flag  = params.In_flag

### SYSTEM SIZE ###
L = params.L
N = L//2
### DISORDER STRENGTH ###
W_i = 0.5
W = params.D

### COUPLING AND INTERACTIONS ###
Jxy = 1.0 # hopping
Jz = 1.0/2 # zz interaction

### RUN ###
ti = time() # start timing function for each realization

sp_basis = spin_basis_1d(L,L//2)
int_list, hop_list = hf.coupling_list(L, Jxy, Jz, BC)
oplist_static = [["+-",hop_list],["-+",hop_list],["zz",int_list]]
Dim = sp_basis.Ns
phi = np.random.uniform(0,1)
t_i, t_f, t_steps = 0.0, 2.0, 200
t_tab = np.linspace(t_i,t_f,num=t_steps,endpoint=True)

operator_dict = dict(H0 = oplist_static)
for i in range(L):
    operator_dict["z"+str(i)] = [["z", [[1.0, i]]]]

no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
H_dict = quantum_operator(operator_dict, basis = sp_basis, **no_checks)
params_dict = dict(H0=1.0)
if dis_flag == 1:
    for j in range(L):
        params_dict["z"+str(j)] = W_i*np.cos(2*math.pi*0.721*j+2*math.pi*phi) # create quasiperiodic fields list
else:
    for j in range(L):
        params_dict["z"+str(j)] = W_i*np.random.uniform(0,1) # create random fields list

HAM = H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict
HAM = np.real(HAM.tocsc())
psi_0 = linalg.eigsh(HAM, k=1)[1].flatten() # initialize system in GS of pre-quench Hamiltonian

params_dict_quench = dict(H0=1.0)
if dis_flag ==1:
    for j in range(L):
        params_dict_quench["z"+str(j)] = W*np.cos(2*math.pi*0.721*j+2*math.pi*phi) # create quench quasiperiodic fields list
else:
    for j in range(L):
        params_dict_quench["z"+str(j)] = W*np.random.uniform(0,1) # create quench random fields list

HAM_quench = H_dict.tohamiltonian(params_dict_quench) # build post-quench Hamiltonian
psi_t = HAM_quench.evolve(psi_0, 0.0, t_tab) # evolve with post-quench Hamiltonian
SxSx_t = hf.SxSxCorr(L, psi_t, sp_basis)

if dis_flag == 1:
    directory = '../DATA/GSQPCorr/L'+str(L)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)
else:
    directory = '../DATA/GSrandomCorr/L'+str(L)+'/D'+str(W)+'/'
    PATH_now = LOCAL+os.sep+directory+os.sep
    if not os.path.exists(PATH_now):
        os.makedirs(PATH_now)

nomefile = str(PATH_now+'LoschL_'+str(L)+'D_'+str(W)+'phi'+str(phi)+'.dat')
np.savetxt(nomefile, np.c_[t_tab, SxSx_t], fmt = '%.9f')

print("Realization completed in {:2f} s".format(time()-ti))
