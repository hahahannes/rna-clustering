import urllib.request 
import os

DATA_DIR = "data"

# Download lncipedia fasta files:
fasta_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2.fasta"
urllib.request.urlretrieve(fasta_url, os.path.join(DATA_DIR, fasta_url.split("/")[-1]))
print("Done with fasta files")

# Download lncipedia bed files:
bed19_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2_hg19.bed"
urllib.request.urlretrieve(fasta_url, os.path.join(DATA_DIR, bed19_url.split("/")[-1]))

bed38_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2_hg38.bed"
urllib.request.urlretrieve(fasta_url, os.path.join(DATA_DIR, bed38_url.split("/")[-1]))
print("Done with bed files")
