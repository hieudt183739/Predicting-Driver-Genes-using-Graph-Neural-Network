# Tiến trình cần 70GB Ram để có thể chạy

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GOSemSim")
BiocManager::install("org.Hs.eg.db")

library(GOSemSim)
library(org.Hs.eg.db)

gene_ids <- keys(org.Hs.eg.db, keytype = "ENTREZID")

gene_symbols <- readLines("data/gene_names.txt")
# gene_symbols <- c('STIM1', 'TRPC1', 'NOS1', 'ATP2B4', 'ABCC9', 'KCNJ11', 'HADHA', 'HADHB', 'GTF2E2', 'GTF2E1')
gene_info <- select(org.Hs.eg.db, keys = gene_symbols, columns = "ENTREZID", keytype = "SYMBOL")
go_db <- godata(annoDb = 'org.Hs.eg.db', ont="BP", computeIC=TRUE)
genes <- gene_info$ENTREZID
sim_matrix <- mgeneSim(genes, semData=go_db, measure="Wang", combine="BMA")

# lưu ma trận 
colnames(sim_matrix) <- gene_info$SYMBOL
rownames(sim_matrix) <- gene_info$SYMBOL
write.table(sim_matrix, file = "Similarity_matrix.csv", row.names = FALSE, col.names = T)
