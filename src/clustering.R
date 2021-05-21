library(Seurat)
gen_clusters <- function(filename, resolution = 0.45) {
  B <- read.csv(paste("~/Documents/data/",filename, sep=""), row.names=1)
  B <- as.matrix(B)
  test <- CreateSeuratObject(B)
  test <- FindVariableFeatures(test)
  test <- ScaleData(test)
  test <- RunPCA(test, assay = "RNA", verbose = FALSE)
  test <- FindNeighbors(test, reduction = "pca", dims = 1:30)
  print(resolution)
  test <- FindClusters(test, resolution = resolution)
  write.csv(test@meta.data[["seurat_clusters"]],
            paste("~/Documents/data/clusters_",filename, sep=""),
            row.names = FALSE)
}
