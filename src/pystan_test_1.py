import pystan
import pandas as pd
import os
import pickle
import numpy as np

###### same preprocessing as before ######

data_path = os.path.join(os.getcwd(),"cerebellum_data")
def process_data():
    cached = {}
    output = []
    for s in ["counts", "genes", "metadata"]:
        #check if we have already processed the dataset
        filename = os.path.join(data_path,"cerebellum_%s.pickle" % s)
        cached = os.path.isfile(filename)
        if cached:
            output.append(pd.read_pickle(filename))
        else:
            #if not processed, read from csv and save pickled file
            csv_file = os.path.join(data_path, "cerebellum_%s.csv" % s)
            df = pd.read_csv(csv_file,index_col=0)
            if s in ['counts','genes']:
                df = df.transpose()
            df.to_pickle(filename)
            output.append(df)
    return output

counts, genes, metadata = process_data()

considered_types = ['Granule', 'Purkinje', 'Astrocytes']
#filter by cell type
c_subset = counts.loc[metadata['liger_ident_coarse'].isin(considered_types)]
#filter genes
c_subset = c_subset.filter(genes.loc['gene'].tolist())
#normalize matrix of counts
#(note: these proportions are conditioned on genes being in frequent set)
c_subset = c_subset.div(c_subset.sum(axis=1), axis=0)

###### pystan testing: poisson model with constant number of reads (5000) ######

def gen_data(I = 5000, P = 1000, J = 899, C = 3000, mag = 0.00007, beta = c_subset.to_numpy()):
    alpha = mag * np.ones(C)
    rng = np.random.default_rng()
    theta = rng.dirichlet(alpha)
    lam = I * np.reshape(theta,(-1,C)).dot(beta)
    N = rng.poisson(lam, size=(P,J))
    return N

fake_data = gen_data(P=1000)

constant_I_model = """
data {    
    int<lower=0> P; //number of pixels    
    int<lower=0> J; //number of genes    
    int<lower=0> N[P,J]; //count number of gene j at pixel p    
    int<lower=0> types; //number of cell types   
}    
parameters {
    simplex[types] theta; //theta is cell type proportion    
    simplex[J] beta[types]; //beta is matrix of gene frequencies    
}    
model {    
    matrix[types, J] matrixBeta;
    for (type in 1:types)
        matrixBeta[type] = beta[type]'; 
    for (p in 1:P)    
        N[p] ~ poisson(5000 * theta' * matrixBeta); //consider truncating pdf because don't want N to be negative
}
"""

constant_sm = pystan.StanModel(model_code=constant_I_model)

parameter_samples = constant_sm.sampling(data = {
    "P": 1000,
    "J": 899,
    "N": fake_data,
    "types": 3
}, verbose=True, iter=1000)

with open('constant_sm.pickle', 'wb') as handle:
    pickle.dump(constant_sm, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('parameter_samples.pickle', 'wb') as handle:
    pickle.dump(parameter_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
