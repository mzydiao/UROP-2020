if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IRWLS import *
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='generate B matrix, save to csv')
#     parser.add_argument('output', help='output file/folder')
#     parser.add_argument('--folder', '-f', action='store_true',
#                         help='output as folder')
    parser.add_argument('--types', '-t', nargs='?', type=int, 
                        default=3, help='number of types')
    parser.add_argument('--pixels', '-p', nargs='?', type=int, 
                        default=2000, help='number of pixels')
    args = parser.parse_args()

    data_path  = os.path.join(os.getcwd(),"data")
    X_vals = np.genfromtxt(os.path.join(data_path,"X_vals.csv"),delimiter=",",skip_header=1)
    Q_mat = np.genfromtxt(os.path.join(data_path,"Q_mat.csv"),delimiter=",",skip_header=1)

    # sc_counts = pd.read_csv(os.path.join(data_path,"sc_counts.csv"), index_col=0)
    # sn_counts = pd.read_csv(os.path.join(data_path,'cerebellum_counts.csv'), index_col=0)
    # sc_counts.to_pickle(os.path.join(data_path,"sc_counts.pickle"))
    # sn_counts.to_pickle(os.path.join(data_path,"cerebellum_counts.pickle"))
    sc_celltype = pd.read_csv(os.path.join(data_path,"sc_celltype.csv"))
    sc_nUMI = pd.read_csv(os.path.join(data_path,"sc_nUMI.csv"))
    sn_metadata = pd.read_csv(os.path.join(data_path,"cerebellum_metadata.csv"), index_col=0)
    print("reading sc counts")
    with open(os.path.join(data_path,"sc_counts.pickle"), 'rb') as f:
        sc_counts = pickle.load(f)
    print("reading sn counts")
    with open(os.path.join(data_path,"cerebellum_counts.pickle"), 'rb') as f:
        sn_counts = pickle.load(f)
    print("finished reading counts")
    
    index = 0
    while os.path.exists(os.path.join(data_path, 'B-'+str(args.types)+'-'+str(args.pixels)+'-'+str(index)+'.csv')):
        index += 1

    ###cell type filtration

    #split metadata
    sn_nUMI = sn_metadata[['nUMI']].to_numpy().T
    sn_celltype = sn_metadata[['liger_ident_coarse']]

    #fix labels
    sc_celltype = sc_celltype.rename({
        i: val for i, val in enumerate(sc_counts.columns.values)
    })

    #get the intersecting cell types
    #int_celltypes = np.unique(
    #    sn_celltype.to_numpy()
    #)[
    #    np.in1d(np.unique(sn_celltype.to_numpy()), np.unique(sc_celltype.to_numpy()))
    #]
    int_celltypes = ['Astrocytes', 'Purkinje', 'Granule', 'Bergmann', 'Oligodendrocytes', 'MLI1', 'MLI2']
    int_celltypes = int_celltypes[:args.types]

    #get cell type filters
    sn_ctfilter = sn_celltype['liger_ident_coarse'].isin(int_celltypes)
    sc_ctfilter = sc_celltype['x'].isin(int_celltypes)

    #do the filtration
    sc_ctf_counts=sc_counts.T[sc_ctfilter]
    sn_ctf_counts=sn_counts.T[sn_ctfilter]
    sn_celltype=sn_celltype[sn_ctfilter]
    sc_celltype=sc_celltype[sc_ctfilter]

    sn_filtered_counts = sn_ctf_counts.copy()#.sample(3000, random_state=0)
    sc_filtered_counts = sc_ctf_counts.copy()#.sample(3000, random_state=0)

    cell_dict = {celltype : index for index, celltype in enumerate(int_celltypes)}

    def select_genes(info, counts, fc_thresh = 3, expr_thresh = .001, MIN_OBS = 3):
        total_gene_list = None
        epsilon = 1e-9
        bulk_vec = counts.sum(axis=1)
        gene_list = counts.T.filter(regex='^(?!mt-)').columns.values
        gene_list = gene_list[bulk_vec[gene_list] >= MIN_OBS]
        for celltype in int_celltypes:
            this_mean = counts.T[info.eq(celltype)].mean()[gene_list]
            other_mean = counts.T[-info.eq(celltype)].mean()[gene_list]
            logFC = np.log(this_mean+epsilon) - np.log(other_mean + epsilon)
            type_gene_list = (logFC > fc_thresh) & (this_mean > expr_thresh)
            if total_gene_list is None:
                total_gene_list = type_gene_list
            else:
                total_gene_list |= type_gene_list

        total_gene_list = gene_list[total_gene_list]
        return total_gene_list

    ###gene filtration and subsampling

    #get pre-filtration umis
    sn_filtered_nUMI = sn_filtered_counts.sum(axis=1)
    sc_filtered_nUMI = sc_filtered_counts.sum(axis=1)
    sn_ctf_nUMI = sn_ctf_counts.sum(axis=1)
    sc_ctf_nUMI = sc_ctf_counts.sum(axis=1)

    #construct filter
    # dylan_filter = list(set(
    #     pd.read_csv(os.path.join(data_path, "cerebellum_genes.csv"),index_col=0)['gene']) &
    #                     set(sc_counts.index))
    dylan_filter = select_genes(sc_celltype['x'],sc_filtered_counts.T, fc_thresh=3, expr_thresh=0.00015, MIN_OBS = 3)
    dylan_filter = dylan_filter[np.in1d(dylan_filter, sn_filtered_counts.columns.values, assume_unique=True)]

    #filter
    sn_filtered_counts = sn_filtered_counts.filter(dylan_filter)
    sc_filtered_counts = sc_filtered_counts.filter(dylan_filter)
    sn_ctf_counts = sn_ctf_counts.filter(dylan_filter)
    sc_ctf_counts = sc_ctf_counts.filter(dylan_filter)

    #create 'other' gene
    sn_filtered_counts[
        'Other'
    ] = sn_filtered_nUMI - sn_filtered_counts.sum(axis=1)
    sc_filtered_counts[
        'Other'
    ] = sc_filtered_nUMI - sc_filtered_counts.sum(axis=1)

    #create np arrays
    sn_filtered_counts_np = sn_filtered_counts.to_numpy()
    sc_filtered_counts_np = sc_filtered_counts.to_numpy()
    sn_B_ds = sn_filtered_counts_np.copy()
    sc_B_ds = sc_filtered_counts_np.copy()


    #subsample
    rng = np.random.default_rng(seed=0)
    for i, row in enumerate(sn_filtered_counts_np):
        if row.sum() > 1000:
            sn_B_ds[i,:] = rng.multinomial(1000,row/row.sum())
    for i, row in enumerate(sc_filtered_counts_np):
        if row.sum() > 1000:
            sc_B_ds[i,:] = rng.multinomial(1000,row/row.sum())

    sc_UMI = sn_B_ds.sum(axis=1)
    sn_UMI = sn_B_ds.sum(axis=1)

    # #delete other
    # sc_B_ds = sc_B_ds[:,:-1]
    # sn_B_ds = sn_B_ds[:,:-1]

    sc_celltype_int = sc_celltype[sc_celltype['x'].isin(int_celltypes)]
    sn_celltype_int = sn_celltype[sn_celltype['liger_ident_coarse'].isin(int_celltypes)]

    # generate fake data

    def gen_data(orig_data, P = 2000, mag = 0.00015/args.types):
        alpha = mag * np.ones(orig_data.shape[0])
        theta = rng.dirichlet(alpha,P)
        #create fake data
        return (np.floor(theta @ orig_data + 0.5)).astype(int), theta

    B, true_theta = gen_data(sc_B_ds, P = args.pixels)
    nUMI = B.sum(axis=1)
    B = B[:,:-1]
    B = np.maximum(B, 0)
    true_W = np.array([
        true_theta.T[sc_celltype_int['x'].eq(celltype).to_numpy()].sum(axis=0)
        for celltype in int_celltypes])
    
    np.savetxt(os.path.join(data_path, 'nUMI-'+str(args.types)+'-'+str(args.pixels)+'-'+str(index)+'.csv'), nUMI, delimiter=',')
    
    np.savetxt(os.path.join(data_path, 'true_W-'+str(args.types)+'-'+str(args.pixels)+'-'+str(index)+'.csv'), true_W, delimiter=',')

    b_df = pd.DataFrame(B.T,index=dylan_filter,columns=["cell_"+str(i) for i in range(args.pixels)])
    b_df.to_csv(os.path.join(data_path, 'B-'+str(args.types)+'-'+str(args.pixels)+'-'+str(index)+'.csv'))
    
    print("saved B matrix")
    
    ###retrieve initial S matrices

    sc_S_init = np.array([
        (sc_ctf_counts[sc_celltype_int['x'].eq(celltype)] /
             sc_ctf_nUMI[sc_celltype_int['x'].eq(celltype)][:,np.newaxis]
        ).mean(axis=0)
        for celltype in int_celltypes])

    sn_S_init = np.array([
        (sn_ctf_counts[
            sn_celltype_int['liger_ident_coarse'].eq(celltype)
        ] 
         /
         sn_ctf_nUMI[
             sn_celltype_int['liger_ident_coarse'].eq(celltype)
         ][:,np.newaxis]
        ).mean(axis=0)
        for celltype in int_celltypes])
    
    np.savetxt(os.path.join(data_path, 'sc_S-'+str(args.types)+'.csv'), sc_S_init, delimiter=',')
    np.savetxt(os.path.join(data_path, 'sn_S-'+str(args.types)+'.csv'), sn_S_init, delimiter=',')
    
    print("saved S matrices")
