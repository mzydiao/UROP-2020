import quadprog

def get_der_fast(S, B, prediction, X_vals, Q_mat):
    d1_vec, d2_vec = get_d1_d2(B, prediction, X_vals, Q_mat)
    S_rows, S_cols = S.shape
    grad = -d1_vec @ S
    hess_c = -d2_vec @ S_mat#a bit concerned about whether or not S_mat will work
    hess = np.zeros((S_cols, S_cols))#does this work?
    counter = 0
    for i in range(S_cols):
        l = S_cols - i
        hess[i, i:S_cols] = hess_c[counter: counter+l]
        hess[i, i] /= 2
        counter += l
    hess += hess.T
    return {"grad":grad, "hess":hess}

def get_d1_d2(B, prediction, X_vals, Q_mat):
    #returns tuple of d1_vec, d2_vec
    bead, epsilon, X_max, delta = B.copy(), 1e-4, max(X_vals), 1e-5#are we sure we want to modify B? maybe B.copy
    K_val = 100 #?
    x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
    Q_cur = calc_Q_all(x, bead, X_vals, Q_mat)
    bead = np.maximum(K_val, bead)
    Q_k, Q_k1, Q_k2 = Q_cur
    Q_d1 = 1/x * \
            (bead * (Q_k - Q_k1) - Q_k1)
    Q_d2 = 1/(x**2) * \
            ((bead+1) * (bead+2) * Q_k2 - bead * (2 * (bead+1) * Q_k1 - (bead-1) * Q_k))
    d1_vec, d2_vec = Q_d1 / Q_k, Q_d2 / Q_k - Q_d1**2 / Q_k**2
    return d1_vec, d2_vec

def calc_Q_all(x, bead, X_vals, Q_mat):
    #return tuple of r1, r2, r3
    epsilon, X_max, delta = 1e-4, max(X_vals), 1e-5
    X = np.minimum(np.maximum(epsilon, x), X_max - epsilon)#this is repeated from before
    l = np.floor((x / delta) ** (2/3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900, 0) / 30)
    l = l.astype(int)
    prop = (X_vals[l+1] - x) / (X_vals[l+1] - X_vals[l])
    output = []
    bead = bead.astype(int)#do this earlier?
    for i in range(1,4):
        v = Q_mat[bead + i, l + 1]#axis?
        k = Q_mat[bead+i, l] - v
        r = k * prop + v
        output.append(r)
    return output
        
def psd(H):
    eig_values, eig_vectors = np.linalg.eig(H)
    epsilon = 1e-7
    if len(H) == 1:
        return eig_vectors @ np.maximum(eig_values, epsilon) @ eig_vectors.T
    return eig_vectors @ np.diag(np.maximum(eig_values, epsilon)) @ eig_vectors.T


def solve_WLS(S, B, initial_sol, nUMI, X_vals, Q_mat,
        constrain = False):
    S_rows, S_cols = S.shape
    epsilon = 1e-7
    solution = np.maximum(initial_sol, 0)
    prediction = np.abs(S @ solution)
    threshold = max(1e-4, max(nUMI) * epsilon)
    prediction = np.maximum(prediction, threshold)
    derivatives = get_der_fast(S, B, prediction, X_vals, Q_mat)
    d_vec = -derivatives["grad"]
    D_mat = psd(derivatives["hess"]) #positive semidefinite part
    norm_factor = np.linalg.norm(D_mat)
    D_mat /= norm_factor
    d_vec /= norm_factor
    D_mat += epsilon * np.identity(len(d_vec))
    A = np.identity(S_cols)
    bzero = -solution
    alpha = 0.3
    if constrain:
        A_const = np.append(np.ones((S_cols, 1)), A, axis=1)#are you sure axis is right?
        b_const = np.append(1 - sum(solution), bzero)
        solution += alpha * quadprog.solve_qp(D_mat, d_vec, C = A_const, b =b_const, meq=1)[0]#not sure if this is how it works
    else:
        solution += alpha * quadprog.solve_qp(D_mat, d_vec, C = A, b = bzero, meq=0)[0]

    #names(solution)<-colnames(S)?
    return solution


def solve_IRWLS(S, B, nUMI, X_vals, Q_mat,
        constrain = True, verbose = False, n_iter = 50, 
        MIN_CHANGE = 0.001):
    S_rows, S_cols = S.shape
    solution = np.ones(S_cols)/S_cols
    #names(solution) <- colnames(S)#?
    S_mat = np.zeros((S_rows, int(S_cols*(S_cols+1)/2)))
    counter = 0
    for i in range(S_cols):
        for j in range(i, S_cols):
            S_mat[:,counter] = S[:,i] * S[:,j]
            counter += 1

    iterations = 0 #now use dampened WLS, iterate weights until convergence
    change = 1
    while change > MIN_CHANGE and iterations < n_iter:
        new_solution = solve_WLS(S, B, solution, nUMI, X_vals, Q_mat,
                constrain = constrain)
        change = np.linalg.norm(new_solution - solution)
        if verbose:
            print("Change:",change)
            print(solution)
        solution = new_solution
        iterations += 1
    converged = (change <= MIN_CHANGE)
    return solution, converged
