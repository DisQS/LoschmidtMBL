import numpy as np
import scipy.linalg as _la
import H_functions as hf

def compute_entropy_singular_vals(psi, L, N, i):
    """Compute the singular values of a state decomposition.
    We divide the state in two parts given by a position and then
    do return its singular values. The space partition is done between
    i-1 and i.
    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.
    Returns:
        svals (1darray of floats): singular values.
    """
    svals = None

    # States in the whole lattice with N particles.
    states = hf.Base_prep(L, N)
    Dim = len(states)
    statesBin = sorted([int(states[i], 2) for i in range(Dim)])

    # Get the maximum and minimum number of particles that fit in the
    # subspace 0 to i-1 (both inclusive).
    num_min = N - min(L-i, N)
    num_max = min(i, N)

    for n in range(num_min, num_max+1):
        # Generate all states in the interval (0, i-1) with n
        # particles.
        states_a = hf.Base_prep(i, n)
        Dim_A = len(states_a)
        statesBin_a = sorted([int(states_a[i], 2) for i in range(Dim_A)])

        # Generate all states in the interval (i, L-1) with N-n
        # particles.
        states_b = hf.Base_prep(i, N-n)
        Dim_B = len(states_b)
        statesBin_b = sorted([int(states_b[i], 2) for i in range(Dim_B)])

        A = np.zeros((Dim_A, Dim_B), dtype=np.complex_)

        for ia, a in enumerate(statesBin_a):
            for ib, b in enumerate(statesBin_b):
                # Tensor multiply a and b to produce a state in (0, L).
                ab = np.left_shift(a, L-i) + b
                A[ia, ib] = psi[np.nonzero(statesBin == ab)]

        if n == num_min:
            svals = _la.svdvals(A)
        else:
            svals = np.concatenate((svals, _la.svdvals(A)))

    return svals


def compute_entanglement_entropy(psi, L, N, i):
    """Compute the entanglement entropy of a state.
    We divide the state in two parts between i-1 and i.
    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.
    Returns:
        S (float): entanglement entropy.
    """
    svals = compute_entropy_singular_vals(psi, L, N, i)
    # Remove 0 singular values.
    svals = svals[np.nonzero(svals)]
    S = -np.dot(svals, np.log2(svals))
    return S


def compute_entanglement_spectrum(psi, L, N, i):
    """Compute the entanglement spectrum of a state.
    We divide the state in two parts between i-1 and i.
    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.
    Returns:
        es (1darray of floats): entanglement spectrum without infinite
            values.
    """
    svals = compute_entropy_singular_vals(psi, L, N, i)
    # Remove 0 singular values.
    svals = svals[np.nonzero(svals)]
    es = -np.log(svals)
    return es
