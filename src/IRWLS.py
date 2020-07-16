import numpy as np
import pandas as pd
import quadprog
import prob_model

def solve_WLS(S, B, initial_sol, nUMI,
        constrain = False):
    S_rows, S_cols = S.shape
    solution = np.maximum(initial_sol, 0)
    prediction = np.abs(S @ solution)
    threshold = max(1e-4, nUMI * 1e-7)
    prediction = np.maximum(prediction, threshold)
    gene_list = list(S.index)
    derivatives = prob_model.get_der_fast(S, B, gene_list, prediction)
    d_vec = -derivatives["grad"]
    D_mat = prob_model.psd(derivatives["hess"]) #positive semidefinite part
    norm_factor = np.linalg.norm(D_mat)
    D_mat /= norm_factor
    d_vec /= norm_factor
    epsilon = 1e-7
    D_mat += epsilon * np.diag(len(d_vec))
    A = np.diag(S_cols)
    bzero = -solution
    alpha = 0.3
    if constrain:
        A_const = np.append(np.ones((S_cols, 1)), A, axis=1)
        b_const = np.append(1 - sum(solution), bzero)
        solution += alpha * quadprog.solve_QP(D_mat, d_vec, C = A_const, b =b_const, meq=1)["solution"]#not sure if this is how it works
    else:
        solution += alpha * quadprog.solve_QP(D_mat, d_vec, C = A, b = bzero, meq=0)["solution"]

    #names(solution)<-colnames(S)?
    return solution
}

def solve_IRWLS(S, B, nUMI, 
        constrain = True, verbose = False, n_iter = 50, 
        MIN_CHANGE = 0.001, bulk_mode = False):
    S_rows, S_cols = S.shape
    solution = np.ones(S_cols)/S_cols
    #names(solution) <- colnames(S)#?

    S_mat = np.zeros((S_rows, S_cols * (S_cols + 1)/2))
    counter = 0
    for i in range(S_cols):
        for j in range(i, S_cols):
            S_mat[,counter] = S[,i] @ S[,j]
            counter += 1

    iterations = 0 #now use dampened WLS, iterate weights until convergence
    change = 1
    while change > MIN_CHANGE and iterations < n_iter:
        new_solution = solve_WLS(S, B, solution, nUMI,
                constrain = constrain, bulk_mode = bulk_mode)
        change = np.linalg.norm(new_solution - solution)
        if verbose:
            print("Change:",change)
            print(solution)
        solution = new_solution
        iterations += 1
    converged = (change <= MIN_CHANGE)
    return solution, converged
