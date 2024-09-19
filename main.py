from dotenv import load_dotenv
import os
from Bio import Entrez, SeqIO

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL")


def fetch_and_save_sequence(accession_number, filename):
    with Entrez.efetch(db="nucleotide", id=accession_number, rettype="gbwithparts", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")
    
    sequence = str(record.seq)
    
    
    with open(filename, "w") as file:
        file.write(sequence)
    
    print(f"Sequence Length for {accession_number}: {len(sequence)}")
    print(f"Sequence (first 100 bp) for {accession_number}: {sequence[:100]}") 


accessions = {
    "NZ_CP046872.1": "NZ_CP046872.1.gb",
    "AY353394.1": "AY353394.1.gb"
}


for accession, filename in accessions.items():
    fetch_and_save_sequence(accession, filename)
