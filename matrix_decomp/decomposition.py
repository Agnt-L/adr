import numpy as np
import scipy as scipy

# model parameters
epsilon = 10 ** (-16)

# Decomposes matrix R into W and H using the parameters k and alpha.
def DecompositionAlgorithm(R, k, alpha):
    # termination tolerance
    tolx = 10 ** (-3)
    maxiter = 2000
    # initial variance
    variance = 0.01

    # Returns True if x is the frequency of an observed side-effect.
    def observed(x):
        return int(x > 0)

    # Returns True if x is the frequency of an unobserved side-effect, ie. 0.
    def unobserved(x):
        return int(x == 0)

    # Make matrices of truth values weather the side-effects are observed.
    observed_vec = np.vectorize(observed)
    OBS = observed_vec(R)
    unobserved_vec = np.vectorize(unobserved)
    MIS = unobserved_vec(R)

    # Calculates the hadamard product.
    def had(a, b):
        return np.multiply(a, b)

    # Calculates the loss function.
    def loss(W, H):
        WH = np.matmul(W, H)
        return (1 / 2) * scipy.linalg.norm(had(OBS, (R - WH))) + (alpha / 2) * scipy.linalg.norm(had(MIS, WH))

    # Updates matrix W in matrix decomposition process.
    def updateW(W, H):
        WH = np.matmul(W, H)
        Ht = np.transpose(H)
        num = np.matmul(R, Ht)
        sum1 = np.matmul(had(OBS, WH), Ht)
        sum2 = np.matmul(alpha * had(MIS, WH), Ht) + epsilon
        den = sum1 + sum2
        return np.maximum(had(W, num / den), 0)

    # Updates matrix H in matrix decomposition process.
    def updateH(W, H):
        WH = np.matmul(W, H)
        Wt = np.transpose(W)
        num = np.matmul(Wt, R)
        sum1 = np.matmul(Wt, had(OBS, WH))
        sum2 = np.matmul(alpha * Wt, had(MIS, WH)) + epsilon
        den = sum1 + sum2
        return np.maximum(had(H, num / den), 0)

    # Normalizes H by dividing each entry by the frobenius norm of the corresponding row.
    def normalizeH(H):
        row_sums = np.linalg.norm(H, axis=1, keepdims=True)
        return H / row_sums[:]

    # get dimensions of the drug-side effect matrix
    (nchems, nses) = np.shape(R)

    # Initialization
    W0 = np.random.rand(nchems, k) * np.sqrt(variance)
    H0 = np.random.rand(k, nses) * np.sqrt(variance)

    # Normalization
    H0 = normalizeH(H0)
    sqrteps = np.sqrt(epsilon)

    # values of cost function
    J = []
    deltas = []
    for iter in range(maxiter):
        W = updateW(W0, H0)
        H = updateH(W0, H0)

        # compute cost function
        J.append(loss(W, H))

        # Get norm of difference and max change in factors
        dw = np.max(np.absolute(W - W0) / (sqrteps + np.max(np.absolute(W0))))
        dh = np.max(np.absolute(H - H0) / (sqrteps + np.max(np.absolute(H0))))
        delta = max(dw, dh)
        deltas.append(delta)

        # Check for convergence
        if iter > 0:
            if delta <= tolx:
                print(f'iter = {iter}, delta = {delta}')
                break

        # Remember previous iteration results
        W0 = W
        H0 = updateH(W, H)  # update H
        print(f'iter = {iter}, J(iter) = {J[iter]}')

    return W, H, J, deltas