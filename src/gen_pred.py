if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from IRWLS import *
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='find prior and nonprior predictions, save to csv')
    
    parser.add_argument('--types', '-t', nargs='?', type=int, 
                        default=3, help='number of types')
    parser.add_argument('--pixels', '-p', nargs='?', type=int, 
                        default=2000, help='number of pixels')
    parser.add_argument('--index', '-i', nargs='?', type=int, 
                        default=0, help='index of cluster file')
    parser.add_argument('--iters', '-it', nargs='?', type=int,
                         default=15, help='number of iterations for alt max')
    args = parser.parse_args()
    
    data_path  = os.path.join(os.getcwd(),"data")
    X_vals = np.genfromtxt(os.path.join(data_path,"X_vals.csv"),delimiter=",",skip_header=1)
    Q_mat = np.genfromtxt(os.path.join(data_path,"Q_mat.csv"),delimiter=",",skip_header=1)

    clusters = pd.read_csv(os.path.join(data_path,
                                        "clusters_B-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"))
    
    B = pd.read_csv(os.path.join(data_path, "B-"+
                                str(args.types)+"-"+
                                str(args.pixels)+"-"+
                                str(args.index)+".csv"), delimiter=',', index_col=0).to_numpy()
    
    nUMI = np.loadtxt(os.path.join(data_path, "nUMI-"+
                                str(args.types)+"-"+
                                str(args.pixels)+"-"+
                                str(args.index)+".csv"), delimiter=',',)
    
    test = IRWLS(nUMI = nUMI
             , B = B, X_vals = X_vals, Q_mat = Q_mat)
    
    cluster_init = np.array([B.T[clusters['x'].eq(i)].mean(axis=0) for i in range(args.types)]).T
    
    start_time = time.time()
    prior = np.ones(args.types) * 0.3 / args.types
    prior_output, plls = test.alt_max(cluster_init,
                                reset=False, prior = prior, iters=args.iters, parallel = True, return_lls=True)
    print("time elapsed",time.time()-start_time)
    
    start_time = time.time()
    noprior_output, nlls = test.alt_max(cluster_init,
                                  reset=False, prior = None, iters=args.iters, parallel = True, return_lls=True)
    print("time elapsed",time.time()-start_time)
    
    
    
    np.savetxt(os.path.join(data_path,"prior-curr_S"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), prior_output[0])
    np.savetxt(os.path.join(data_path,"prior-curr_W"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), prior_output[1])
    
    np.savetxt(os.path.join(data_path,"noprior-curr_S"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), noprior_output[0])
    np.savetxt(os.path.join(data_path,"noprior-curr_W"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), noprior_output[1])
    
    np.savetxt(os.path.join(data_path,"prior-lls"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), plls)
    np.savetxt(os.path.join(data_path,"noprior-lls"+"-"+
                                        str(args.types)+"-"+
                                        str(args.pixels)+"-"+
                                        str(args.index)+".csv"), nlls)
