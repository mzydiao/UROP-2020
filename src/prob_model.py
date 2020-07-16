def get_der_fast(S, B, gene_list, prediction):
    d1_vec, d2_vec = get_d1_d2(B, prediction)
    grad = -d1_vec @ S
    hess_c = -d2_vec @ S_mat#a bit concerned about whether or not S_mat will work
    hess = np.zeros((S_cols, S_cols))#does this work?
    counter = 0
    for i in range(S_cols):
        l = S_cols - i - 1
        hess[i, i:S_cols] = hess_c[counter: counter+l]
        hess[i, i] /= 2
        counter += l + 1
    hess += hess.T
    return dict("grad"=grad, "hess"=hess)

def get_d1_d2(B, prediction):
    #returns tuple of d1_vec, d2_vec
    bead, epsilon, X_max, delta = B, 1e-4, max(X_vals), 1e-5#are we sure we want to modify B? maybe B.copy
    x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
    Q_cur = calc_Q_all(x, bead)
    bead = np.maximum(K_val, bead)
    Q_k, Q_k1, Q_k2 = Q_cur
    Q_d1 = 1/x * \
            (bead * (Q_k - Q_k1) - Q_k1)
    Q_d2 = 1/(x**2) * \
            ((bead+1) * (bead+2) * Q_k2 - bead * (2 * (bead+1) * Q_k1 - (bead-1) * Q_k))
    d1_vec, d2_vec = Qd_1 / Q_k, Q_d2 / Q_k - Q_d1**2 / Q_k**2
    return d1_vec, d2_vec

def calc_Q_all(x, bead):
    #return tuple of r1, r2, r3
    epsilon, X_max, delta = 1e-4, max(X_vals), 1e-5
    X = np.minimum(np.maximum(epsilon, x), X_max - epsilon)#this is repeated from before
    l = np.floor((x / delta) ** (2/3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900, 0) / 30)
    prop = (X_vals[l+1] - x) / (X_vals[l+1] - X_vals[l])
    output = []
    for i in range(1,4):
        v = Q_mat[np.append(bead + i, l + 1, axis=1)]#axis?
        k = Q_mat[cbind(bead+i, l)] - v
        r = k * prop + v
        output.append(r)
    return output
        
def psd(H):
    eig_values, eig_vectors = np.linalg.eig(H)
    if len(H) == 1:
        return eig_vectors @ np.maximum(eig_values, epsilon) @ eig_vectors.T
    return eig_vectors @ np.diag(np.maximum(eig_values, epsilon)) @ eig_vectors.T
