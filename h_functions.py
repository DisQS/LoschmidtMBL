from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy, diag_ensemble # entropies
import numpy as np # generic math functions
from numpy.random import ranf,seed # pseudo random numbers
import scipy.sparse as _sp
import scipy.linalg as _la
from scipy.sparse import eye
from scipy.sparse.linalg import LinearOperator, eigsh, minres
from itertools import combinations_with_replacement
from math import factorial

def mat_format(A):
    """Find the type of a matrix(dense or sparse)
    Args:
        A(2d array of floats)           = Dense or sparse Matrix
    Returns:
        mat_type(string)                = Type of A, "Sparse" or "Dense"
    """
    form = str(type(A))
    spar_str = "<class 's"
    if form.startswith(spar_str):
        mat_type = "Sparse"
    else:
        mat_type = "Dense"
    return mat_type

### FROM CONFIGURATION TO BIN NUMBER ###
def TO_bin(xx):
    return int(xx,2)

### FROM BIN NUMBER TO CONFIGURATION ###
def TO_con(x,L):
    x1=int(x)
    L1=int(L)
    return np.binary_repr(x1, width=L1)

### BINOMIAL ###
def comb(n, k):
	kk = factorial(n) / factorial(k) / factorial(n - k)
	uga= int(kk)
	return uga

### BASIS CREATION ###
def basis_prep(L):
    basis = spin_basis_1d(L,pauli=False, Nup=L//2)
    return basis

### COUPLING (HOPPING AND INTERACTION) LISTS ###
def coupling_list(L, Jxy, Jzz, BC):
    if BC == 1:
        J_zz = [[Jzz,i,i+1] for i in range(L-1)] # OBC
        J_xy = [[Jxy/2.0,i,i+1] for i in range(L-1)] # OBC
    else:
        J_zz = [[Jzz,i,i+1] for i in range(L)] # PBC
        J_xy = [[Jxy/2.0,i,i+1] for i in range(L)] # PBC
    return J_zz, J_xy

### RANDOM FIELD LIST ###
def field_list(L, W, Dis_type, seed = None):
    if seed is None:
        seed = np.random.randint(1,10000)
    if seed is not None:
        np.random.seed(seed)
    if Dis_type == 0:
        dis = 2*np.random.rand(L)-1.0
        mag_field = [[dis[i], i] for i in range(L)]
        dis_hz = [["z", mag_field]]
    else:
        for i in range(L):
            mag_field = [np.cos(2*math.pi*0.721*i/L), i]
        dis_hz = [["z", mag_field]]

    return mag_field, dis_hz

### OPERATORS ASSOCIATED TO COUPLINGS ###
def op_list(Jxy, Jzz):
    oplist_static = [["+-",Jxy],["-+",Jxy],["zz",Jzz]]
    oplist_dynamic = [] # empty if Hamiltonian is time-independent
    return oplist_static, oplist_dynamic

### RANDOM HEISENBERG HAMILTONIAN ###
def ham(L, basis, hop, int, W, Dis_type, seed = None):
    op_static, op_dynamic = op_list(hop, int)
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
    clean_ham = hamiltonian(op_static, op_dynamic, basis=basis, dtype=np.float64,**no_checks) #clean XXZ Hamiltonian with no disorder
    h_z, dis_static = field_list(L, Dis_type, seed)
    dis_ham = hamiltonian(dis_static, op_dynamic, basis=basis,dtype=np.float64,**no_checks)
    tot_ham = clean_ham + W*dis_ham #disordered XXZ Hamiltonian
    return tot_ham

### LEVEL STATISTICS (HUSE OBSERVABLE) ###
def levstat(E):
    gap = E[1:]-E[:-1]
    r = list(map(lambda x,y:min(x,y)*1./max(x,y), gap[1:], gap[0:-1]))
    return np.mean(r)

def eigPsi_0(H, Dim, L, psi0_flag, sigma=None):
    """Compute the initial state as the groundstate of the Hamiltonian or a middle eigenstate
    Args:
        H(2d sparse matrix)                 = Hamiltonian matrix in sparse form
        Dim(int)                            = Hilbert space dimension
        L(int)                              = Size of the chain
        psi0_flag(int)                      = Flag for initial state (0 ---> GS, 1 ---> Middle state)
        sigma(float)                        = Target energy for shift-invert method
    """
    if psi0_flag == 1:
        if sigma is None:
            print("No target energy given. Check psi0_flag and/or sigma")
        else:
            if L<14:
                e, v = np.linalg.eig(H.todense())
                idx = e.argsort()[::1]
                e_0 = e[idx]
                v_0 = v[:,idx][Dim //2]
            elif L == 14 or L == 16:
                e_0, v_0 = eigsh(H, 1, sigma=sigma, maxiter=1E4)
                v_0 = np.asarray(v_0).ravel()

            else:
                OP = H - sigma*eye(Dim)
                OPinv = LinearOperator(matvec=lambda v: minres(OP, v, tol=1e-5)[0],shape=H.shape, dtype=H.dtype)
                e_0, v_0 = eigsh(H, sigma=sigma, k=1, tol=1e-9, OPinv=OPinv)
                v_0 = np.asarray(v_0).ravel()

    if psi0_flag == 0:
        e_0, v_0 = eigsh(H, k=1)

    return e_0, v_0

def Psi_0(Dim, L, in_flag):
    """Index of the initial state for time evolution
    Args:
        Dim(int)                        = Dimension of Hilbert space
        L(int)                          = Size fo the chain
        Base_num(1d array of str)       = Basis states in binary representation
        in_flag(int)                    = Flag for initial state (0 ---> random,
                                                                  1 ---> 1010101010,
                                                                  2 ---> 1111100000)
    Returns:
        n(int)                          = Index of the chosen initial state in Base_num
    """
    if in_flag == 0:
        n = np.random.randint(0,Dim-1)
    elif in_flag == 1:
        n = TO_con(sum([2**i for i in range(1, L, 2)]), L)
    else:
        n = TO_con(sum([2**i for i in range(0,L//2, 1)]),L)[::-1]
    return n

def Loschmidt(Psi_t, Proj_Psi0):
    """Calculate survival probability at time t
    (see Markus Heyl review, Rep. Prog. Phys. 81, 054001 (2018))
    Args:
        Psit(1d array of complex)       = State evolved at time t
        Proj_Psi0(1d array of floats)   = Initial state
    Returns:
        L(float)                        = Loschmidt echo at time t
    """
    L = np.square(np.absolute(np.dot(Proj_Psi0, Psi_t))).flatten()
    return L

def SxSxCorr(L, psi_t, basis, SS='xx'):
    """Calculate spin correlation matrix for all times
    Args:
        L(int)                          = Size of the chain
        psi_t(2d array of complex)      = Matrix of evolved state psi(t) at all unix_times
        S(str)                          = Operator label (xx,yy or zz). Default is S_xx
    Returns:
        corr_mean(2d array of float)    = L x t matrix of correlations (rows --> distance, columns ---> time)
    """
    sx_list = [i for i in combinations_with_replacement(np.arange(L), 2)]
    Sxx = [[1.0, i, j] for (i, j) in sx_list]

    Cxx = np.zeros([len(psi_t.T), len(Sxx)])
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
    for i in range(len(Sxx)):
        xx_op = hamiltonian([[SS, [Sxx[i]]]], [], basis=basis, **no_checks)
        Exp_xx = [np.real(xx_op.expt_value(psi_t[:,i])) for i in range(len(psi_t.T))]
        Cxx[:,i] = Exp_xx

    corr_mat = np.zeros((L,L))
    corr_mean = np.zeros((len(psi_t.T), L))
    for r in range(len(Cxx)):
        corr_mat[np.triu_indices(L,0)] = Cxx[r]
        corr_mean[r] = [np.mean(corr_mat.diagonal(i)) for i in range(L)]

    return corr_mean

def Sent(psit, basis, subsys=None, return_rdm=None):
    """Return the entanglement entropy of psi, living in basis <basis>, computed in the reduced subsystem specified by subsys
    subsys = list of site labels [0, 1, ..., k] specifying the subsystem. If subsys=None,  defaults to 0....N/2 -1

    return_rdm can be specified as 'A' (the subsystem of interest), 'B', or both; if so a dictionary is returned
    """
    if subsys is None:
        #default partition ---> half-chain
        subsys=tuple(range(basis.N//2))
    S_tab = np.zeros(len(psit.T))
    for i, psi in enumerate(psit.T):
        sdict= basis.ent_entropy(psi, sub_sys_A=subsys,return_rdm=return_rdm)
        # the quspin value is normalized by the subsystem size
        SA= sdict['Sent_A'] * len(subsys)
        if return_rdm is not None:
            sdict['Sent_A']=SA
            return sdict
        S_tab[i] = SA
    return S_tab

### DATA DIRECTORIES AND DATA FILES CREATION ###

def generate_directory(basename, para, namesub):
    #namesub is a string example namesub = 'L'
    path = os.getcwd()
    path_now = os.path.join(basename,namesub+str(para))
    if not os.path.exists(path_now):
        os.makedirs(path_now)
    return path_now

def generate_filename(basename):

    unix_timestamp = int(time.time())
    local_time = str(int(round(time.time() * 1000)))
    xx = basename + local_time + ".dat"
    if os.path.isfile(xx):
        time.sleep(1)
        return generate_filename(basename)
    return xx

def creation_all_subdirectory(L, W):
    PATH = os.getcwd()
    PATH_L=generate_directory(PATH, L, 'L_')
    PATH_W= generate_directory(PATH_alpha, W, 'W_')
    return 0
