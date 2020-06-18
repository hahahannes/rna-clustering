from pybedtools import BedTool

snps = BedTool('snps.bed.gz')  # [1]
genes = BedTool('hg19.gff')    # [1]

intergenic_snps = snps.subtract(genes)                       # [2]
nearby = genes.closest(intergenic_snps, d=True, stream=True) # [2, 3]

for gene in nearby:             # [4]
    if int(gene[-1]) < 5000:    # [4]
        print gene.name         # [4]
