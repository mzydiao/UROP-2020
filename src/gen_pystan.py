import pystan
import pandas as pd
import os
import pickle
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

gen_model = """
data {    
  int<lower=0> P; //number of pixels    
  int<lower=0> J; //number of genes    
  int<lower=0> C; //number of cells    
  real<lower=0> a; //we let alpha=<a, a, a, ..., a>
  matrix[J,C] beta; //beta is matrix of gene frequencies   
}    
generated quantities{
  vector[C] theta = dirichlet_rng(rep_vector(a, C)); 
  int<lower=0> N[P,J];
  for (p in 1:P)    
    N[p] = poisson_rng(5000 * theta' * beta'); //count number of gene j at pixel p
}
"""

metadata = metadata.loc[metadata['liger_ident_coarse'].isin(considered_types)]

data = {
    "P": 1000,
    "J": 899,
    "C": 3000,
    "a": 0.00005,
    "beta": c_subset.transpose().to_numpy()
}

gsm = pystan.StanModel(model_code=gen_model)
gen = gsm.sampling(data=data,algorithm="Fixed_param")

with open('gen_out.pickle', 'wb') as handle:
    pickle.dump(gen, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('gsm_out.pickle', 'wb') as handle:
    pickle.dump(gsm, handle, protocol=pickle.HIGHEST_PROTOCOL)
