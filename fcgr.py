import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt


def read_sequence(filename):
    with open(filename, "r") as file:
        sequence = file.read().strip()
    return sequence


def calculate_kmer_frequencies(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    return kmer_counts


def generate_fcgr_image(kmer_counts, k, image_size):
    kmer_list = list(kmer_counts.keys())
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    
   
    image = np.zeros((image_size, image_size))
    
    
    for kmer, count in kmer_counts.items():
        x = kmer_index[kmer] % image_size
        y = kmer_index[kmer] // image_size
        image[x, y] = count
    
    return image


def plot_fcgr_image(image, filename):
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('FCGR Image')
    plt.savefig(filename)
    plt.close()


accessions = {
    "NZ_CP046872.1": "NZ_CP046872.1.gb",
    "AY353394.1": "AY353394.1.gb"
}


for accession, filename in accessions.items():
    sequence = read_sequence(filename)
    k = 4  
    image_size = 16  
    
    kmer_counts = calculate_kmer_frequencies(sequence, k)
    fcgr_image = generate_fcgr_image(kmer_counts, k, image_size)
    
    
    image_filename = filename.replace(".gb", "_fcgr.png")
    plot_fcgr_image(fcgr_image, image_filename)
    
    print(f"FCGR image saved as {image_filename}")
