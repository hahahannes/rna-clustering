import urllib.request 
import os

DATA_DIR = "data"

# Download lncipedia fasta files:
fasta_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2.fasta"
file_path = os.path.join(DATA_DIR, fasta_url.split("/")[-1])
if not os.path.exists(file_path):
    urllib.request.urlretrieve(fasta_url, file_path)
    print("Done with fasta files")
else:
    print("fasta file already downloaded")

# Download lncipedia bed files:
bed38_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2_hg38.bed"
file_path = os.path.join(DATA_DIR, bed38_url.split("/")[-1])
if not os.path.exists(file_path):
    urllib.request.urlretrieve(bed38_url, file_path)
    print("Done with bed files")
else:
    print("bed file already downloaded")

# Download lncipedia gff files:
gff38_url = "https://lncipedia.org/downloads/lncipedia_5_2/full-database/lncipedia_5_2_hg38.gff"
file_path = os.path.join(DATA_DIR, gff38_url.split("/")[-1])
if not os.path.exists(file_path):
    urllib.request.urlretrieve(gff38_url, file_path)
    print("Done with bed files")
else:
    print("gff file already downloaded")
