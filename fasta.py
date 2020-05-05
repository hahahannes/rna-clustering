from Bio import SeqIO

# https://lncipedia.org/download
for seq_record in SeqIO.parse("data/lncipedia_5_2.fasta", "fasta"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
