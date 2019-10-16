from quspin.operators import hamiltonian, exp_op, quantum_operator, quantum_LinearOperator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, ent_entropy, diag_ensemble, ED_state_vs_time # entropies
import scipy
import scipy.linalg as _la
import sys, os
from time import time
import h_functions as hf
import math
import params
import numpy as np # generic math functions
from numpy.random import uniform, choice # pseudo random numbers
import matplotlib.pyplot as plt

LOCAL = os.path.abspath('.')
PATH_now = LOCAL

### BOUNDARY CONDITIONS
BC = 1 # 1 --> OPEN, 0 ---> PERIODIC
### DISORDER TYPE ###
dis_flag = 1 # 1 --> QUASIPERIODIC, 0 ---> RANDOM
in_flag = 1
epsilon = 0.01

### SYSTEM SIZE ###
L = 12
N = L//2
### DISORDER STRENGTH ###
W_i = 0.5
w_tab = np.linspace(0.5, 6.0, 50)

### COUPLING AND INTERACTIONS ###
Jxy = 1.0 # hopping
Jz = 1.0/2 # zz interaction

N_real = 1

### RUN ###
ti = time() # start timing function for each realization

sp_basis = spin_basis_1d(L, N)
int_list, hop_list = hf.coupling_list(L, Jxy, Jz, BC)
oplist_static = [["+-",hop_list],["-+",hop_list],["zz",int_list]]
Dim = sp_basis.Ns
phi = np.random.uniform(0,1)
t_i, t_f, t_steps = 0.0, 1000000.0, 200
t_tab = np.linspace(t_i,t_f,num=t_steps,endpoint=True)
phi=np.random.uniform(0,1)

operator_dict = dict(H0 = oplist_static)
for i in range(L):
    operator_dict["z"+str(i)] = [["z", [[1.0, i]]]]

no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
H_dict = quantum_operator(operator_dict, basis = sp_basis, **no_checks)
params_dict = dict(H0=1.0)
if dis_flag == 1:
    for j in range(L):
        params_dict["z"+str(j)] = (W_i/2)*np.cos(2*math.pi*0.721*j+2*math.pi) # create quasiperiodic fields list
else:
    for j in range(L):
        params_dict["z"+str(j)] = (W_i/2)*np.random.uniform(0,1) # create random fields list

HAM = H_dict.tohamiltonian(params_dict) # build initial Hamiltonian through H_dict
e_n, v_n = scipy.linalg.eigh(HAM.todense())

def real(H_dict, W, v_n, t_tab, epsilon, l=None):
    phi=np.random.uniform(0,1)
    t0 = time()
    params_dict_quench = dict(H0=1.0)
    if dis_flag ==1:
        for j in range(L):
            params_dict_quench["z"+str(j)] = (W/2)*np.cos(2*math.pi*0.721*j+2*math.pi*phi) # create quench quasiperiodic fields list
    else:
        for j in range(L):
            params_dict_quench["z"+str(j)] = (W/2)*np.random.uniform(0,1) # create quench random fields list

    HAM_quench = H_dict.tohamiltonian(params_dict_quench) # build post-quench Hamiltonian
    E_nquench, v_nquench = scipy.linalg.eigh(HAM_quench.todense())

    m_tab = np.zeros(Dim)
    for i, n in enumerate(v_n.T):
        psi_tdense = ED_state_vs_time(n, E_nquench, v_nquench, t_tab, iterate=False)
        Losch = np.square(np.abs(np.dot(n, psi_tdense)))
        m = len(Losch[Losch<epsilon])
        m_tab[i]=m
    print("realization completed in {:.2f} s".format(time()-t0))
    return m_tab, Losch

# for w in w_tab:
#     Losch = np.vstack([real(H_dict, w, v_n, t_tab, epsilon, i)[1] for i in range(N_real)])
#     Losch_avg = np.mean(Losch, axis=0)
#     m_tot = np.vstack([real(H_dict, w, v_n, t_tab, epsilon, i)[0] for i in range(N_real)])
#     m_avg = np.mean(m_tot, axis=0)
#     m_avg = (m_avg-m_avg.min())/(m_avg.max()-m_avg.min())
#     if dis_flag == 1:
#         directory = '../DATA/SingleTest/L'+str(L)+'Wi'+str(W_i)+'e/'
#         PATH_now = LOCAL+os.sep+directory+os.sep
#         if not os.path.exists(PATH_now):
#             os.makedirs(PATH_now)
#     else:
#         directory = '../DATA/SingleShotrandomlongtime/L'+str(L)+'Wi'+str(W_i)+'/'
#         PATH_now = LOCAL+os.sep+directory+os.sep
#         if not os.path.exists(PATH_now):
#             os.makedirs(PATH_now)
#     nomefile_measure = str(PATH_now+'measureL_'+str(L)+'D_'+str(w)+'.dat')
#     nomefile_losch = str(PATH_now+'LoschL_'+str(L)+'D_'+str(w)+'.dat')
#     np.savetxt(nomefile_measure, m_avg, fmt = '%.9f')
#     np.savetxt(nomefile_losch, m_avg, fmt = '%.9f')

for w in w_tab:
    m_avg, Losch = real(H_dict, w, v_n, t_tab, epsilon)
    m_avg = (m_avg-m_avg.min())/(m_avg.max()-m_avg.min())
    if dis_flag == 1:
        directory = '../DATA/SingleTest/L'+str(L)+'Wi'+str(W_i)+'e'+str(epsilon)+'/'
        PATH_now = LOCAL+os.sep+directory+os.sep
        if not os.path.exists(PATH_now):
            os.makedirs(PATH_now)
    else:
        directory = '../DATA/SingleShotrandomlongtime/L'+str(L)+'Wi'+str(W_i)+'/'
        PATH_now = LOCAL+os.sep+directory+os.sep
        if not os.path.exists(PATH_now):
            os.makedirs(PATH_now)
    nomefile_measure = str(PATH_now+'measureL_'+str(L)+'D_'+str(w)+'.dat')
    nomefile_losch = str(PATH_now+'LoschL_'+str(L)+'D_'+str(w)+'.dat')
    np.savetxt(nomefile_measure, m_avg, fmt = '%.9f')
    np.savetxt(nomefile_losch, Losch, fmt = '%.9f')
