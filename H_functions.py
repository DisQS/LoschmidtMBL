import numpy as np
import math
import scipy.linalg as _la
from math import factorial
import itertools
import time
import scipy.sparse as _sp
from scipy.sparse import linalg
import os
import os.path
from datetime import datetime
import time


POPCOUNT_TABLE16 = [0] * 2**16
for index in range(len(POPCOUNT_TABLE16)):
    POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]

def one_count(v):
    """Count the number of bits different from 0 for a certain value
    Args:
        v(int)                          = value
    Returns:
        (int)                           = # of bits different from 0 in binary representation of v
    """
    return (POPCOUNT_TABLE16[ v & 0xffff] + POPCOUNT_TABLE16[(v >> 16) & 0xffff])

### FROM CONFIGURATION TO BIN NUMBER ###
def TO_bin(xx):
    return int(xx,2)

### FROM BIN NUMBER TO CONFIGURATION ###
def TO_con(x,L):
    x1=int(x)
    L1=int(L)
    return np.binary_repr(x1, width=L1)

def comb(n, k):
    """Binomial coefficient function
    Args:
        n,k(int)
    Returns:
        bin_coef(int)
    """
    kk = factorial(n) / factorial(k) / factorial(n - k)
    bin_coef = int(kk)
    return bin_coef

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

def Base_prep(n,k):
    """Prepare basis in configurational representation (01010101010...etc)
    Args:
        n(int)                          = Size of the chain
        k(int)                          = Number of spin-up
    Returns:
        conf(1d array of str)           = Configuration basis
    """
    conf = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        conf.append(''.join(s))
    return conf

def BaseNumRes_creation(Dim,LL,B):
    A=np.zeros((Dim,LL), dtype=np.float)

    for i in range(Dim):
        k=0
        for j in list(B[i]):
            A[i,k] = float(j)-0.5
            k+=1
    return A

def Hop_prep(L,BC):
    """Hopping list
    Args:
        L(int)                          = Size of the chain
        BC(int)                         = Boundary conditions flag (0 ---> periodic
                                                                    1 ---> open)
    Returns:
        Hopping(list of floats)
    """
    if BC == 1:
        Hop_dim=L-1
    else:
        Hop_dim=L
    return [TO_con(2**i+2**((i+1)%L),L) for i in range(Hop_dim)]

def Dis_Creation(LL,Dis_gen):
    """Random fields creation
    Args:
        LL(int)                         = Size of the chain
        Dis_gen(int)                    = Disorder flag (0 ---> Random
                                                         1 ---> Quasiperiodic)
    Returns:
        dis(1d array of floats)         = Random field per site
    """
    dis = np.zeros(LL, dtype=np.float)
    for i in range(LL):
        if Dis_gen==0:
            dis[i] = 2*np.random.random()-1
        else:
            dis[i] = np.cos(2*math.pi*0.721*i/LL)
    return dis

def LinTab_Creation(LL,Base,di):
    """Create lookup table
    Args:
        LL(int)                         = Size of the chain
        Base(1d array of str)           = Basis states in configuration representation
        di(int)                         = Dimension of Hilbert space
    Returns:
        LinTab(2d array of int)         = Lookup table
    """

    L = int(LL)
    Dim=int(di)

    MaxSizeLINVEC = sum([2**(i-1) for i in range(1,int(L/2+1))])

    LinTab   = np.zeros((MaxSizeLINVEC+1,4),dtype=int)
    Jold     = JJ=j1=j2=0
    Conf_old = TO_con(0,int(L/2))

    for i in range(Dim):
        Conf_lx = Base[i][0:int(L/2)]
        Bin_lx  = TO_bin(Conf_lx)
        Conf_rx = Base[i][int(L/2):L]
        Bin_rx  = TO_bin(Conf_rx)

        if Conf_lx==Conf_old:
            j1 = Jold
        else:
            j1 += j2

        Conf_old = Conf_lx

        if Jold != j1:
            JJ = Jold = 0

        j2   = JJ+1
        Jold = j1
        JJ  += 1

        #print(Conf_lx, int(Bin_lx), int(j1), Conf_rx, int(Bin_rx), int(j2))

        LinTab[Bin_lx,0]= int(Bin_lx)
        LinTab[Bin_lx,1]= int(j1)
        LinTab[Bin_rx,2]= int(Bin_rx)
        LinTab[Bin_rx,3]= int(j2)

    #print(LinTab)
    return LinTab

def LinLook(vec,LL,arr):
    """Look for complete table
    Args:
        vec(1d array of int)            = Vector
        LL(int)                         = Size of the chain
        arr(2d array of int)            = Lookup table
    """
    Vec  = TO_con(vec,LL)
    v1   = Vec[0:int(LL/2)]
    v2   = Vec[int(LL/2):LL]
    ind1 = TO_bin(v1)
    ind2 = TO_bin(v2)
    return arr[ind1,1]+arr[ind2,3]-1

def Ham_Dense_Creation(LL,NN,Dim,D,Jzz,Dis_real,BC,Base_Bin,Base_Num,Hop_Bin,LinTab):
    """Compute the Hamiltonian matrix elements H_ij = <i|H|j> and create a dense H
    Args:
        LL(int)                         = Size of the chain
        NN(int)                         = Number of spin-up
        Dim(int)                        = Dimension of the Hilbert SPACE
        D(float)                        = Disorder strength
        Jzz(float)                      = Interaction strength
        Dis_real(1d array of floats)    = Random fields
        BC(int)                         = Boundary conditions flag (0 ---> periodic
                                                                    1 ---> open)
        Base_bin(1d array of int)       = Configuration basis
        Base_num(1d array of str)       = Basis states in binary representation
        Hop_bin(1d array of floats)     = Hopping list
        LinTab(2d array of int)         = Lookup table
    Returns:
        ham(2d array of floats)         = Hamiltonian matrix
    """

    J=1.

    ham = np.zeros((Dim,Dim), dtype=np.float)

    if BC == 1:
        Hop_dim=LL-1
    else:
        Hop_dim=LL

    for i in range(Dim):
        n_int = 0.0
        n_dis = 0.0
        bra = LinLook(Base_Bin[i],LL,LinTab)

        for j in range(Hop_dim):
            xx  = Base_Bin[i]^Hop_Bin[j]
            ket = LinLook(xx,LL,LinTab)

            if one_count(xx) == NN:
                ham[bra,ket] = J/2
            uu = Base_Bin[i] & Hop_Bin[j]

            if one_count(uu) == 1:
                n_int -= 0.25
            else:
                n_int += 0.25

            n_ones = Base_Bin[i] & int(2**(LL-j-1))
            if n_ones != 0:
                n_dis += 0.5*Dis_real[j]
            else:
                n_dis -= 0.5*Dis_real[j]

        ham[bra,bra] = J*(Jzz*n_int + D*n_dis)

    return ham

def Ham_Sparse_Creation(LL,NN,Dim,D,Jzz,Dis_real,BC,Base_Bin,Base_Num,Hop_Bin,LinTab):
    """Save i, j, <i|H|j> and create a sparse H
    Args:
        LL(int)                         = Size of the chain
        NN(int)                         = Number of spin-up
        Dim(int)                        = Dimension of the Hilbert SPACE
        D(float)                        = Disorder strength
        Jzz(float)                      = Interaction strength
        Dis_real(1d array of floats)    = Random fields
        BC(int)                         = Boundary conditions flag (0 ---> periodic
                                                                    1 ---> open)
        Base_bin(1d array of int)       = Configuration basis
        Base_num(1d array of str)       = Basis states in binary representation
        Hop_bin(1d array of floats)     = Hopping list
        LinTab(2d array of int)         = Lookup table
    Returns:
        ham(CSC sparse matrix)          = Hamiltonian matrix
    """

    J=1.

    if BC == 1:
        Hop_dim=LL-1
    else:
        Hop_dim=LL
    i_ind = []
    j_ind = []
    vals  = []
    dis = []
    for i in range(Dim):
        n_int = 0.0
        n_dis = 0.0
        bra = LinLook(Base_Bin[i],LL,LinTab)

        for j in range(Hop_dim):
            xx  = Base_Bin[i]^Hop_Bin[j]
            ket = LinLook(xx,LL,LinTab)

            if one_count(xx) == NN:
                i_ind.append(bra)
                j_ind.append(ket)
                vals.append(J/2)

            uu = Base_Bin[i] & Hop_Bin[j]

            if one_count(uu) == 1:
                n_int -= 0.25
                #0.5 perche spin 1/2*1/2
            else:
                n_int += 0.25

            n_ones = Base_Bin[i] & int(2**(LL-j-1))
            #diventa diverso da zero solamente se ce un 1 in quel sito
            if n_ones != 0:
                n_dis += 0.5*Dis_real[j]
                #0.5 perche spin 1/2
            else:
                n_dis -= 0.5*Dis_real[j]

        dis.append(n_dis)
        i_ind.append(bra)
        j_ind.append(bra)
        vals.append(J*(Jzz*n_int + D*n_dis))

    ham = _sp.coo_matrix((vals,(i_ind, j_ind))).tocsc()
    return ham, dis

def eigval(A):
    """Diagonalize the Hamiltonian
    Args:
        A(2d array of floats)           = Matrix to be diagonalized (Hamiltonian)
    Returns:
        E(1d array of floats)           = Eigenvalues
        V(2d array of floats)           = Eigenvectors (in columns!)
    """
    type = mat_format(A)
    if type == 'Sparse':
        E,V = _sp.linalg.eigsh(A, k = 1, which='SA', return_eigenvectors=True)
    else:
        E, V = _la.eigh(A)
    return E, V

def levstat(E):
    """Calculate the statistics of the ratio r between adjacent energy gaps
    (see A.Pal and D.Huse, Phys. Rev. B 82, 174411)
    Args:
        E(1d array of floats)           = Eigenvalues
    Returns:
        avg(float)                      = Mean of r
    """
    di = len(E)
    delta = E[1:]-E[:-1]
    r = list(map(lambda x,y:min(x,y)*1./max(x,y), delta[1:], delta[:-1]))
    avg = np.mean(r)
    return avg

def InvPartRatio(Evec,A):
    """Calculate participation ratios
    Args:
        Evec(2d array of floats)        = Eigenvectors (in columns)
    Returns:
        IPR(1d array of floats)         = Inverse participation ratio for each eigenstate
    """
    type = mat_format(A)
    if type == 'Sparse':
        IPR = np.zeros(50)
        for i in range(len(IPR)):
            IPR[i] = np.sum(Evec[:,i]**4)
        IPR_sum = np.sum(IPR)
    else:
        IPR = np.zeros(len(Evec))
        for i in range(len(IPR)):
            IPR[i] = np.sum(Evec[:,i]**4)
        IPR_sum = np.sum(IPR)
    return IPR_sum

def Psi_0(Dim, L, Base_num, in_flag):
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
        ind = TO_con(sum([2**i for i in range(1, L, 2)]), L)
        n = Base_num.index(ind)
    else:
        ind = TO_con(sum([2**i for i in range(0,L//2, 1)]),L)[::-1]
        n = Base_num.index(ind)
    return n

def Proj_Psi0(a,V):
    """Initial state projected on the configuration basis
    Args:
        a(int)                          = Index of the chosen initial state
        V(2d array of floats)           = Eigenvectors (in columns)
    Returns:
        V[a](1d array of floats)        = Initial state in the configuration basis
    """
    return V[a]

def TimEvolve(Proj_Psi0, E, t):
    """Time evolution of initial state
    Args:
        Proj_Psi0(1d array of floats)   = Initial state
        E(1d array of floats)           = Eigenspectrum
        t(float)                        = Time
    Returns:
        psit(1d array of complex)       = State evolved at time t
    """
    psit0 = Proj_Psi0
    psit = np.exp(-1j*E*t)*psit0
    return psit

def Loschmidt(Psi_t, Proj_Psi0):
    """Calculate survival probability at time t
    (see Markus Heyl review, Rep. Prog. Phys. 81, 054001 (2018))
    Args:
        Psit(1d array of complex)       = State evolved at time t
        Proj_Psi0(1d array of floats)   = Initial state
    Returns:
        L(float)                        = Loschmidt echo at time t
    """
    L = np.square(np.absolute(np.dot(Proj_Psi0, Psi_t)))
    return L

def magnetization(V,Base_NumRes):
    """Calculate magnetization profile and total magnetization for half-chain (not conserved)
    (see R. Singh et al., NJP 18, 023046 (2016))
    Args:
        V(1d array of floats)           = State vector
        Base_NumRes(1d array of int)    = Spin operator basis
    Returns:
        Sz(1d array of floats)          = Magnetization value at each site
        Tot_Sz(float)                   = Total magnetization for half chain
    """
    Sz   = np.dot(np.transpose(V**2),Base_NumRes)
    Ch_L = len(Sz)
    Tot_Sz = np.sum(np.real(Sz[0:int((Ch_L/2-1))]))
    return Sz, Tot_Sz

def generate_filename(basename):
    unix_timestamp = int(time.time())
    local_time = str(int(round(time.time() * 1000)))
    xx = basename + local_time + ".dat"
    if os.path.isfile(xx):
        time.sleep(1)
        return generate_filename(basename)
    return xx
