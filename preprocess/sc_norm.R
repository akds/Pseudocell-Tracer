library(SingleCellExperiment)
library(Mus.musculus)
library(scater)
library(scran)
library(batchelor)
library(BiocSingular)
library(Rtsne)
library(sva)

gene_df <- read.table("all.cells.tsv", sep="\t", row.names=1, stringsAsFactors=F)
names(gene_df) <- gene_df[1,]
gene_df <- gene_df[-c(1),]
gene_df[,] <- sapply(gene_df[,], as.numeric)
num_samples <- 

gene.ids <- read.table("B6ensemblIDtoGeneName.tsv", sep="\t", row.names=1, stringsAsFactors=F)
gene.symb = gene.ids[-c(1),]
ensemble.ids <- rownames(gene.ids)[-c(1)]
gene_df = gene_df[rowSums(gene_df!=0) > 0,]


map_vec = vector()
for (i in 1:length(rownames(gene_df))){
  idx = match(rownames(gene_df)[i], ensemble.ids)
  map_vec <- append(map_vec, gene.symb[idx])
}
"Aicda" %in% map_vec
"Ighm" %in% map_vec
"Ighg3" %in% map_vec

num_samples = length(names(gene_df))

keep <- !is.na(map_vec) & !duplicated(map_vec)
gene_df <- gene_df[keep,]
rownames(gene_df) <- map_vec[keep]
summary(keep)

samples <- names(gene_df)
group = vector()

for(s in samples){
  print(strsplit(s,"cell")[[1]][1])
  group <- append(group, strsplit(s,"cell")[[1]][1])
}
gene_df = gene_df[rowSums(gene_df)!=0,]

unique_groups <- unique(group)


sce1 <- SingleCellExperiment(list(counts=as.matrix(gene_df[,group==unique_groups[1]])),
                            colData=DataFrame(Run=rep(unique_groups[1], 
                            length(colnames(gene_df[,group==unique_groups[1]])))), 
                            rowData=DataFrame(Symbol=map_vec[keep]))

sce1 <- calculateQCMetrics(sce1, compact=TRUE)
QC <- sce1$scater_qc
low.lib <- isOutlier(QC$all$log10_total_counts, type="lower", nmad=3)
low.genes <- isOutlier(QC$all$log10_total_features_by_counts, type="lower", nmad=3)
discard <- low.lib | low.genes
sce1 <- sce1[,!discard]
sce1 <- computeSumFactors(sce1)
sce1 <- normalize(sce1)

fit <- trendVar(sce1, parametric=TRUE, use.spikes=F) 
dec1 <- decomposeVar(sce1, fit)
dec1$Symbol <- rowData(sce1)$Symbol

sce2 <- SingleCellExperiment(list(counts=as.matrix(gene_df[,group==unique_groups[2]])),
                             colData=DataFrame(Run=rep(unique_groups[2], 
                             length(colnames(gene_df[,group==unique_groups[2]])))), 
                             rowData=DataFrame(Symbol=map_vec[keep]))

sce2 <- calculateQCMetrics(sce2, compact=TRUE)
QC <- sce2$scater_qc
low.lib <- isOutlier(QC$all$log10_total_counts, type="lower", nmad=3)
low.genes <- isOutlier(QC$all$log10_total_features_by_counts, type="lower", nmad=3)
discard <- low.lib | low.genes
sce2 <- sce2[,!discard]
sce2 <- computeSumFactors(sce2)
sce2 <- normalize(sce2)

sce3 <- SingleCellExperiment(list(counts=as.matrix(gene_df[,group==unique_groups[3]])),
                             colData=DataFrame(Run=rep(unique_groups[3], 
                            length(colnames(gene_df[,group==unique_groups[3]])))), 
                             rowData=DataFrame(Symbol=map_vec[keep]))

sce3 <- calculateQCMetrics(sce3, compact=TRUE)
QC <- sce3$scater_qc
low.lib <- isOutlier(QC$all$log10_total_counts, type="lower", nmad=3)
low.genes <- isOutlier(QC$all$log10_total_features_by_counts, type="lower", nmad=3)
discard <- low.lib | low.genes
sce3 <- sce3[,!discard]
sce3 <- computeSumFactors(sce3)
sce3 <- normalize(sce3)

sce4 <- SingleCellExperiment(list(counts=as.matrix(gene_df[,group==unique_groups[4]])),
                             colData=DataFrame(Run=rep(unique_groups[4], 
                             length(colnames(gene_df[,group==unique_groups[4]])))), 
                             rowData=DataFrame(Symbol=map_vec[keep]))


sce4 <- calculateQCMetrics(sce4, compact=TRUE)
QC <- sce4$scater_qc
low.lib <- isOutlier(QC$all$log10_total_counts, type="lower", nmad=3)
low.genes <- isOutlier(QC$all$log10_total_features_by_counts, type="lower", nmad=3)
discard <- low.lib | low.genes
sce4 <- sce4[,!discard]
sce4 <- computeSumFactors(sce4)
sce4 <- normalize(sce4)

rescaled <- multiBatchNorm(sce1, sce2, sce3, sce4)

res1 <- rescaled[[1]]
res2 <- rescaled[[2]]
res3 <- rescaled[[3]]
res4 <- rescaled[[4]]

unc1 <- logcounts(res1)
unc2 <- logcounts(res2)
unc3 <- logcounts(res3)
unc4 <- logcounts(res4)


tmp1 <- merge(unc1, unc2, by="row.names")
rownames(tmp1) <- tmp1$Row.names; tmp1$Row.names <- NULL
tmp2 <- merge(unc3, unc4, by="row.names")
rownames(tmp2) <- tmp2$Row.names; tmp2$Row.names <- NULL

final <- merge(tmp1, tmp2, by="row.names")
rownames(final) <- final$Row.names; final$Row.names <- NULL

new_samples <- names(final)
new_group = vector()

for(s in new_samples){
  print(strsplit(s,"cell")[[1]][1])
  new_group <- append(new_group, strsplit(s,"cell")[[1]][1])
}

mnn.out <- batchelor::fastMNN(unc1, unc2, unc3, unc4, cos.norm = F)

write.table(assay(mnn.out,"reconstructed"), "mnn.nocos.full.genes.tsv", sep="\t")


