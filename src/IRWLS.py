import quadprog
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import itertools

class IRWLS:
    def __init__(self, X_vals = None, Q_mat = None, nUMI = None, B = None):
        self.X_vals = X_vals
        self.Q_mat = Q_mat
        self.nUMI = nUMI
        self.B = B

    def get_der_fast(self, S, bead, prediction, S_mat):
        d1_vec, d2_vec = self.get_d1_d2(bead, prediction)
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
        self.der = {"grad":grad, "hess":hess}
        return grad, hess

    def get_d1_d2(self, bead, prediction):
        #returns tuple of d1_vec, d2_vec
        X_vals = self.X_vals
        epsilon, X_max, delta = 1e-4, max(X_vals), 1e-5
        x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
        Q_cur = self.calc_Q_all(x, bead)
        Q_k, Q_k1, Q_k2 = Q_cur
        Q_d1 = 1/x * \
                (bead * (Q_k - Q_k1) - Q_k1)
        Q_d2 = 1/(x**2) * \
                ((bead+1) * (bead+2) * Q_k2 - bead * (2 * (bead+1) * Q_k1 - (bead-1) * Q_k))
        d1_vec, d2_vec = Q_d1 / Q_k, Q_d2 / Q_k - Q_d1**2 / Q_k**2
        return d1_vec, d2_vec

    def calc_Q_all(self, x, bead, option='all'):
        #return tuple of r1, r2, r3
        Q_mat = self.Q_mat
        X_vals = self.X_vals
        epsilon, X_max, delta = 1e-4, max(X_vals), 1e-5
        x = np.minimum(np.maximum(epsilon, x), X_max - epsilon)#this is repeated from before
        l = np.floor((x / delta) ** (2/3))
        l = np.minimum(l, 900) + np.floor(np.maximum(l - 900, 0) / 30)
        l = l.astype(int)
        prop = (X_vals[l] - x) / (X_vals[l] - X_vals[l-1])
        output = []
        bead = bead.astype(int)#do this earlier?
        up_to = 3 if option == 'all' else 1
        for i in range(up_to):
            v = Q_mat[bead+i, l]#axis?
            k = Q_mat[bead+i, l-1] - v
            r = k * prop + v
            output.append(r)
        return output

    #negative log likelihood
    def calc_log_l_vec(self, pred_lambda, bead,
                       prior = None, 
                       weights = False
                      ):
        if prior is not None:
            return -np.log(
                self.calc_Q_all(pred_lambda, bead, option='single')[0]
            ).sum() - ((prior-1)*(np.log(weights) - np.log(weights.sum()))).sum()
        return -np.log(self.calc_Q_all(pred_lambda, bead, option='single')[0]).sum()
    
    def psd(self, H):
        eig_values, eig_vectors = np.linalg.eig(H)
        self.eig = eig_values, eig_vectors
        epsilon = 1e-3
        if len(H) == 1:
            return eig_vectors @ np.maximum(eig_values, epsilon) @ eig_vectors.T
        self.VLVT = VLVT = eig_vectors @ np.diag(
            np.maximum(eig_values, epsilon)
        ) @ eig_vectors.T
        return VLVT


    def solve_WLS(self, S, bead, initial_sol, S_mat,
                  prior = None,
                  solmin = 0, 
                  constrain = False
                 ):
        K_val = 100
        bead = np.minimum(bead, K_val)
        S_rows, S_cols = S.shape
        epsilon = 1e-7
        solution = np.maximum(initial_sol, solmin)
        prediction = np.abs(S @ solution)
        threshold = 1e-4 #max(1e-4, max(self.nUMI) * epsilon)
        prediction = np.maximum(prediction, threshold)
        grad, hess = self.get_der_fast(S, bead, prediction, S_mat)
        #if dirichlet prior
        if prior is not None:
            s = solution.sum()
            alph_minus = prior - 1
            grad -= alph_minus * (1/solution) - alph_minus.sum() * (1/s)
            delt_hess = np.ones((S_cols, S_cols)) * (1/(s**2)) * alph_minus.sum()
            delt_hess -= np.diag((1/(solution ** 2))*alph_minus)
            hess -= delt_hess    
        d_vec = -grad
        D_mat = self.psd(hess) #positive semidefinite part
        self.D_mat = D_mat.copy()
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
            solution += alpha * quadprog.solve_qp(D_mat, d_vec, C = A_const, b = b_const, meq = 1)[0]
            #not sure if this is how it works
        else:
            solution += alpha * quadprog.solve_qp(D_mat, d_vec, C = A, b = bzero, meq = 0)[0]
        solution = np.maximum(solmin,solution)
        return solution

    def solve_IRWLS(self, S, bead, 
                    solution = None,
                    constrain = False, verbose = False, n_iter = 1000,
                    MIN_CHANGE = 0.000001, prior = None, solmin = 0,
                    progress = False
                   ):
        #preprocess row, col nums
        S_rows, S_cols = S.shape
        
        #generate initial naive solution
        if solution is None: 
            solution = np.ones(S_cols)/S_cols
        
        #fill in S_mat
        S_mat = np.zeros((S_rows, int(S_cols*(S_cols+1)/2)))
        counter = 0
        for i in range(S_cols):
            for j in range(i, S_cols):
                S_mat[:,counter] = S[:,i] * S[:,j]
                counter += 1
                
        #now use dampened WLS, iterate weights until convergence
        iterations, change, small = 0, 1, True
        while change > MIN_CHANGE and iterations < n_iter and small:
            new_solution = self.solve_WLS(S, bead, solution, S_mat,
                                          solmin = solmin,
                                          prior = prior,
                                          constrain = constrain)
            change = np.linalg.norm(new_solution - solution,ord=1)
            if change > 50:
                print("past sol", solution)
                small = False
                if verbose:
                    print("Change:",change)
                    print(solution)
            solution = new_solution
            #solution = new_solution
            iterations += 1
        
        if not small:
            print('not small', change)
            print('solution', solution)
            iterations, change = 0, 1
            solution = np.ones(S_cols)/S_cols
            while change > MIN_CHANGE and iterations < n_iter:
                new_solution = self.solve_WLS(S, bead, solution, S_mat,
                                              solmin = solmin,
                                              prior = prior,
                                              constrain = constrain)
                change = np.linalg.norm(new_solution - solution,ord=1)
                solution = new_solution
                if change > 1:
                    if verbose:
                        print("Change:",change)
                        print(solution)
                #solution = new_solution
                iterations += 1
        converged = (change <= MIN_CHANGE)
        if verbose: print(iterations)
        #print(iterations)
        if progress: print('■ ', end='', flush=True)
        return solution, converged
    
    #3: -20; 5: -13; 7: ?
    def alt_max(self, init_S, reset = False, init_W = None, prior = None, solmin_W = 1e-4, solmin_S=np.exp(-16),
                constrain = False, constrain_W = False, verbose = False, n_iter = 1000,
                MIN_CHANGE_S = 0.00000001, MIN_CHANGE_W = 0.00001, iters = 3, parallel = False, start_W = False,
                return_lls = False,
               ):
        '''
        plan of attack:
        (1) iterate through rows of B
        --1 for each row, do solve_IRWLS with curr_S, B_row, ...
        --2 combine resultant rows into W_all
        (2) iterate through columns of B
        --1 do the same as above but with W_all
        @return: curr_S, curr_W
        '''
        B = self.B
        curr_S = init_S
        if reset:
            curr_S = np.ones(init_S.shape)/init_S.shape[0]
        if init_W is None:
            curr_W = np.ones((B.shape[1], init_S.shape[1])) / init_S.shape[1]
        else:
            curr_W = np.maximum(solmin_W, init_W)
        print('before')
        lls = []
        ll = self.calc_log_l_vec(self.nUMI.T * (curr_S @ curr_W.T), np.minimum(B, 100), 
                                  weights = curr_W,
                                  prior = prior)
        lls.append(ll)
        print(ll)
        prog_W, prog_S = max(1, B.shape[1]//20), max(1, B.shape[0]//20)
        def W_dict(i):
            return dict(
                constrain = constrain_W,#constrain,
                verbose = verbose, 
                n_iter = n_iter,
                MIN_CHANGE = MIN_CHANGE_W,
                solmin = solmin_W,
                solution = np.maximum(curr_W[i], solmin_W),
                prior = prior,
                progress = (i+1) % prog_W == 0
            )
        def S_dict(i):
            return dict(
                constrain = constrain,
                verbose = verbose, 
                n_iter = n_iter,
                MIN_CHANGE = MIN_CHANGE_S,
                solmin = solmin_S,
                solution = np.maximum(curr_S[i], solmin_S),
                progress = (i+1) % prog_S == 0
            )
        cores = max(4, min(20,mp.cpu_count()-5))
        if start_W:
            print('starting S')
            if parallel:
                with mp.Pool(processes=cores) as pool:
                    curr_S = np.array(pool.starmap(
                        self.solve_S,
                        zip(itertools.count(),
                            B,
                            itertools.repeat(curr_W),
                            map(S_dict, itertools.count())
                           )))
            else:
                curr_S = np.array([self.solve_S(*args) for args in 
                                   zip(itertools.count(),
                                       B,
                                       itertools.repeat(curr_W),
                                       map(S_dict, itertools.count())
                                      )])
            print('after')
            new_ll = self.calc_log_l_vec(self.nUMI.T * (curr_S @ curr_W.T), np.minimum(B, 100), weights = curr_W,
                                  prior = prior)
            ll = new_ll
            lls.append(ll)
                
        for i in range(iters):
            if parallel:
                #(1)
                print('starting W')
                with mp.Pool(processes=cores) as pool:
                    curr_W = np.array(pool.starmap(
                        self.solve_W,
                        zip(itertools.count(),
                            B.T,
                            itertools.repeat(curr_S),
                            map(W_dict, itertools.count())
                           )))
                #(2)
                print('starting S')
                with mp.Pool(processes=cores) as pool:
                    curr_S = np.array(pool.starmap(
                        self.solve_S,
                        zip(itertools.count(),
                            B,
                            itertools.repeat(curr_W),
                            map(S_dict, itertools.count())
                           )))
            else:
                #(1)
                print('starting W')
                curr_W = np.array([self.solve_W(*args) for args in
                                   zip(itertools.count(),
                                   B.T,
                                   itertools.repeat(curr_S),
                                   map(W_dict, itertools.count())
                                      )])
                #(2)
                print('starting S')
                curr_S = np.array([self.solve_S(*args) for args in 
                                   zip(itertools.count(),
                                       B,
                                       itertools.repeat(curr_W),
                                       map(S_dict, itertools.count())
                                      )])
            print('after')
            new_ll = self.calc_log_l_vec(self.nUMI.T * (curr_S @ curr_W.T), np.minimum(B, 100), weights = curr_W,
                                  prior = prior)
            d_ll = ll - new_ll
            print(new_ll)
            if d_ll < 30:
                break
            ll = new_ll
            lls.append(ll)
        if return_lls:
            return (curr_S, curr_W), lls
        return curr_S, curr_W
    
    def solve_W(self, i, B_col, curr_S, args_dict):
        return self.solve_IRWLS(
                curr_S * self.nUMI[i], B_col.copy(),
                **args_dict
            )[0]

    def solve_S(self, i, B_row, curr_W, args_dict):
        return self.solve_IRWLS(
                curr_W * self.nUMI[:,np.newaxis], B_row.copy(),
                **args_dict
            )[0]
