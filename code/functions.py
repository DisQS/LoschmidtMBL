import numpy as np
import os.path
import math
import scipy.linalg as _la
from math import factorial
import itertools
import time
import scipy.special as special
import os
from datetime import datetime
import time


#..................................counting number of zero
POPCOUNT_TABLE16 = [0] * 2**16
for index in range(len(POPCOUNT_TABLE16)):
    POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]

def one_count(v):
    return (POPCOUNT_TABLE16[ v & 0xffff] + POPCOUNT_TABLE16[(v >> 16) & 0xffff])

#..................................from configuration to bin number
def TO_bin(xx):
    return int(xx,2)

#..................................from bin number to configuration
def TO_con(x,L):
    x1=int(x)
    L1=int(L)
    return np.binary_repr(x1, width=L1)

#..................................Binomial
def comb(n, k):
    kk = factorial(n) / factorial(k) / factorial(n - k)
    uga= int(kk)
    return uga

#..................................Base preparation
def Base_prep(n,k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result

def BaseNumRes_creation(Dim,LL,B):
    A=np.zeros((Dim,LL), dtype=np.float)

    for i in range(Dim):
        k=0
        for j in list(B[i]):
            A[i,k] = float(j)-0.5
            k+=1
    return A

#..................................Hopping preparation
def Hop_prep(L,BC):
    if BC == 1:
        Hop_dim=L-1
    else:
        Hop_dim=L
    return [TO_con(2**i+2**((i+1)%L),L) for i in range(Hop_dim)]

#..................................Disorder creation
def Dis_Creation(LL,Dis_gen):

    dis = np.zeros(LL, dtype=np.float)
    for i in range(LL):
        if Dis_gen==0:
            dis[i] = 2*np.random.random()-1
        else:
            dis[i] = np.cos(2*math.pi*0.721*i/LL)
    return dis

#..................................creation Lin Tables
def LinTab_Creation(LL,Base,di):

    L = int(LL)
    Dim=int(di)

	#..........................Table Creation
    MaxSizeLINVEC = sum([2**(i-1) for i in range(1,int(L/2+1))])

    #....creates a table LinTab_L+LinTab_R
    #.....................[  ,  ]+[  ,  ]
    LinTab   = np.zeros((MaxSizeLINVEC+1,4),dtype=int)
    Jold     = JJ=j1=j2=0
    Conf_old = TO_con(0,int(L/2))

	#...........................Table Filling
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

#..................................Lin Look for complete state
def LinLook(vec,LL,arr):

    Vec  = TO_con(vec,LL)
    v1   = Vec[0:int(LL/2)]
    v2   = Vec[int(LL/2):LL]
    ind1 = TO_bin(v1)
    ind2 = TO_bin(v2)
    return arr[ind1,1]+arr[ind2,3]-1

#..................................Lin Look for RIGHT state
def LinLook_LL(vec,arr):
    ind=TO_bin(vec)
    return arr[ind+1,1]


#..................................Lin Look for RIGHT state
def LinLook_RR(vec,arr):
    ind=TO_bin(vec)
    return arr[ind+1,3]

#..................................Hamiltonian creation
def Ham_Dense_Creation(LL,NN,Dim,D,Dis_real,BC,Base_Bin,Base_Num,Hop_Bin,LinTab):

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
                #ham[bra,ket] = J
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

        ham[bra,bra] = J*(n_int + D*n_dis)

    return ham

#..................................Eigenspectrum of Ham
def eigval(A):
    E, V = _la.eigh(A)
    return E, V

#..................................Level statistics (Huse-ratio)
def levstat(E):
    delta = E[1:]-E[:-1]
    r = list(map(lambda x,y:min(x,y)*1./max(x,y), delta[1:], delta[:-1]))
    avg = np.mean(r)
    return r, avg

def InvPartRatio(Evec):
    IPR = np.zeros(len(Evec))
    for i in range(len(Evec)):
        IPR[i] = np.sum(Evec[i]**4)
    return IPR
    
#..................................................Initial state
def Psi_0(Dim):
    n = np.random.randint(0,Dim-1)
    return n

#.................................................Initial state projected on eigenket
def Proj_Psi0(a,V):
    return V[a]

#.................................Time evolution
def TimEvolve(Proj_Psi0, E, t):
    psit0 = Proj_Psi0
    psit = np.exp(-1j*E*t)*psit0
    return psit

#.................................Loschmidt Amplitude and Loschmidt Echo
def Loschmidt(Psi_t, Proj_Psi0):
    G = np.dot(Proj_Psi0, Psi_t)
    L = (abs(G))**2
    return L

def generate_filename(basename):
    unix_timestamp = int(time.time())
    local_time = str(int(round(time.time() * 1000)))
    xx = basename + local_time + ".dat"
    if os.path.isfile(xx):
        time.sleep(1)
        return generate_filename(basename)
    return xx

def time_tab(t_i,t_f,Nstep,Lo_li):
    if Lo_li == 0:
        t_tab = np.insert(10**-5,1,np.linspace(t_i,t_f, num=Nstep, dtype=float))
    else:
        t_tab = np.insert(10**-5,1,np.logspace(np.log10(t_i),np.log10(t_f), num=Nstep, dtype=float))
    return t_tab
