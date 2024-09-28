import os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt

def read_sequences(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    sequences = []
    if filename.endswith(".fasta"):
        for record in SeqIO.parse(filename, "fasta"):
            sequences.append((record.id, str(record.seq)))  # Return sequence ID and the sequence
    elif filename.endswith(".gb"):
        for record in SeqIO.parse(filename, "genbank"):
            sequences.append((record.id, str(record.seq)))
    else:
        raise ValueError("Unsupported file format")
    
    return sequences

def calculate_kmer_frequencies(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    return kmer_counts

def generate_fcgr_image(kmer_counts, image_size):
    kmer_list = list(kmer_counts.keys())
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    
    image = np.zeros((image_size, image_size))
    
    for kmer, count in kmer_counts.items():
        x = kmer_index[kmer] % image_size
        y = kmer_index[kmer] // image_size
        if y < image_size:
            image[x, y] = count
    
    return image

def plot_fcgr_image(image, filename):
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('FCGR Image')
    plt.savefig(filename)
    plt.close()


fasta_files = [
    r"C:\Users\LENOVO\Desktop\ResistGen\ResistGen\acito.fasta",
    r"C:\Users\LENOVO\Desktop\ResistGen\ResistGen\clost.fasta",
    r"C:\Users\LENOVO\Desktop\ResistGen\ResistGen\Chlamydomonas.fasta"
]

k = 4  
image_size = 16  


for filepath in fasta_files:
    print(f"Reading file: {filepath}")
    
    try:
        sequences = read_sequences(filepath)
        
        #  FCGR image for each strain
        for strain_id, sequence in sequences:
            kmer_counts = calculate_kmer_frequencies(sequence, k)
            fcgr_image = generate_fcgr_image(kmer_counts, image_size)
            
            # unique image filename for each strain
            if filepath.endswith(".fasta"):
                image_filename = filepath.replace(".fasta", f"_{strain_id}_fcgr.png")
            else:
                image_filename = filepath.replace(".gb", f"_{strain_id}_fcgr.png")
            
            plot_fcgr_image(fcgr_image, image_filename)
            print(f"FCGR image for {strain_id} saved as {image_filename}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error processing file {filepath}: {e}")
