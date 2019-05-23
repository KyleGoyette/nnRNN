import numpy as np
import scipy.linalg


def henaff_init(n):
    # Initialization of skew-symmetric matrix
    s = np.random.uniform(-np.pi, 0., size=int(np.floor(n / 2.)))
    return create_diag(s, n)

def cayley_init(n):
    s = np.random.uniform(0, np.pi/2., size=int(np.floor(n / 2.)))
    s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))
    return create_diag(s, n)

def create_diag(s, n):
    diag = np.zeros(n-1)
    diag[::2] = s
    A_init = np.diag(diag, k=1)
    A_init = A_init - A_init.T
    return A_init.astype(np.float32)

def random_orthogonal_init(n):
    # NOTE: For a matrix logarithm to be real valued, need orthogonal
    # matrices to have an even number of negative eigenvalues. That is,
    # they should be in SO(n) with det 1. Then the orthogonal_kernel init 
    # can be mapped to log_orthogonal_kernel.
    def get_orthogonal():
        # As in https://arxiv.org/pdf/math-ph/0609050.pdf
        q, _ = np.linalg.qr(np.random.normal(size=(n,n)))
        return q * np.sign(np.diag(q))
    orthogonal = get_orthogonal()
    while np.linalg.det(orthogonal) < 0:
        orthogonal = get_orthogonal()
    A_init = scipy.linalg.logm(orthogonal).real
    return A_init.astype(np.float32)