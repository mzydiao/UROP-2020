import quadprog

class IRWLS:
    def __init__(self, X_vals = None, Q_mat = None):
        self.X_vals = X_vals
        self.Q_mat = Q_mat

    def get_der_fast(S, B, prediction, S_mat):
        d1_vec, d2_vec = self.get_d1_d2(B, prediction)
        S_rows, S_cols = S.shape
        grad = -d1_vec @ S
        hess_c = -d2_vec @ S_mat#a bit concerned about whether or not S_mat will work
        hess = np.zeros((S_cols, S_cols))#does this work?
        counter = 0
        for i in range(S_cols):
            l = S_cols - i
            hess[i, i:] = hess_c[counter: counter+l]
            hess[i, i] /= 2
            counter += l
        hess += hess.T
        return {"grad":grad, "hess":hess}

    def get_d1_d2(B, prediction):
        #returns tuple of d1_vec, d2_vec
        X_vals = self.X_vals
        bead, epsilon, X_max, delta = B.copy(), 1e-4, max(X_vals), 1e-5#are we sure we want to modify B? maybe B.copy
        K_val = 100 #?
        x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
        Q_cur = self.calc_Q_all(x, bead)
        bead = np.minimum(K_val, bead)
        Q_k, Q_k1, Q_k2 = Q_cur
        Q_d1 = 1/x * \
                (bead * (Q_k - Q_k1) - Q_k1)
        Q_d2 = 1/(x**2) * \
                ((bead+1) * (bead+2) * Q_k2 - bead * (2 * (bead+1) * Q_k1 - (bead-1) * Q_k))
        d1_vec, d2_vec = Q_d1 / Q_k, Q_d2 / Q_k - Q_d1**2 / Q_k**2
        return d1_vec, d2_vec

    def calc_Q_all(x, bead):
        #return tuple of r1, r2, r3
        Q_mat = self.Q_mat
        X_vals = self.X_vals
        epsilon, X_max, delta = 1e-4, max(X_vals), 1e-5
        X = np.minimum(np.maximum(epsilon, x), X_max - epsilon)#this is repeated from before
        l = np.floor((x / delta) ** (2/3))
        l = np.minimum(l, 900) + np.floor(np.maximum(l - 900, 0) / 30)
        l = l.astype(int)
        prop = (X_vals[l] - x) / (X_vals[l] - X_vals[l-1])
        output = []
        bead = bead.astype(int)#do this earlier?
        for i in range(3):
            v = Q_mat[bead+i, l]#axis?
            k = Q_mat[bead+i, l-1] - v
            r = k * prop + v
            output.append(r)
        return output

    def psd(H):
        eig_values, eig_vectors = np.linalg.eig(H)
        epsilon = 1e-7
        if len(H) == 1:
            return eig_vectors @ np.maximum(eig_values, epsilon) @ eig_vectors.T
        return eig_vectors @ np.diag(np.maximum(eig_values, epsilon)) @ eig_vectors.T


    def solve_WLS(S, B, initial_sol, nUMI, S_mat,
            constrain = False):
        S_rows, S_cols = S.shape
        epsilon = 1e-7
        solution = np.maximum(initial_sol, 0)
        prediction = np.abs(S @ solution)
        threshold = max(1e-4, max(nUMI) * epsilon)
        prediction = np.maximum(prediction, threshold)
        derivatives = self.get_der_fast(S, B, prediction, S_mat)
        d_vec = -derivatives["grad"]
        D_mat = self.psd(derivatives["hess"]) #positive semidefinite part
        norm_factor = np.linalg.norm(D_mat, ord=2)
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
        return solution


    def solve_IRWLS(S, B, nUMI,
            constrain = True, verbose = False, n_iter = 50, 
            MIN_CHANGE = 0.001):
        #preprocess row, col nums
        S_rows, S_cols = S.shape
        
        #generate initial naive solution
        solution = np.ones(S_cols)/S_cols
        S_mat = np.zeros((S_rows, int(S_cols*(S_cols+1)/2)))
        ##fill in S_mat
        counter = 0
        for i in range(S_cols):
            for j in range(i, S_cols):
                S_mat[:,counter] = S[:,i] * S[:,j]
                counter += 1
        
        #now use dampened WLS, iterate weights until convergence
        iterations, change = 0, 1
        while change > MIN_CHANGE and iterations < n_iter:
            new_solution = self.solve_WLS(S, B, solution, nUMI, S_mat,
                    constrain = constrain)
            change = np.linalg.norm(new_solution - solution,ord=1)
            if verbose:
                print("Change:",change)
                print(solution)
            solution = new_solution
            iterations += 1
        converged = (change <= MIN_CHANGE)
        return solution, converged
