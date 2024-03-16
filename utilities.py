import numpy as np
#from sklearn.metrics import pairwise_distances;
from sklearn.manifold import MDS
import pandas as pd
import matplotlib.pyplot as plt
import os


def read_fasta(file_path) -> list:
    sequences = []
    cur = ''
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    sequences.append(cur)
                    cur = ''
            else:
                cur += line.strip()
        sequences.append(cur)
    return sequences

def filter_on_length(sequences : list, lo, hi):
    res = [sequence for sequence in sequences if lo <= len(sequence) <= hi]
    return res

def clean_fasta(sequences) -> list:
    for i in range(len(sequences)):
        sequences[i] = sequences[i].upper()
        sequences[i] = \
            ''.join(c for c in sequences[i] if c in {'A','C','G','T'})
    return sequences

def write_fastas(sequences, subdir, dir, header=''):
    path_to_dir = f'./{dir}/{subdir}'
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    for i in range(len(sequences)):
        with open(f'{path_to_dir}/{subdir}_{i+1}.fasta', 'w') as f:
            f.write(f'>{header}\n')
            for j in range(0, len(sequences[i]), 60):
                f.write(f'{sequences[i][j:j+60]}\n')
              

def read_sequence(file_path) -> str:
    return read_fasta(file_path)[0]

def cgr(seq, order, k) -> np.ndarray:
    ln = len(seq)
    pw = 2**k
    HALF_LENGTH = pw//2
    out = [[0 for i in range(pw)] for j in range(pw)]
    x = HALF_LENGTH
    y = HALF_LENGTH
    for i in range(ln):
        x //= 2
        y //= 2
        if(seq[i] == order[2] or seq[i] == order[3]):
            x += HALF_LENGTH

        if(seq[i] == order[0] or seq[i] == order[3]):
            y += HALF_LENGTH
        if i >= k-1:
            out[y][x] += 1
    return np.array(out)

def perform_mds(diatances_matrix, n=3, random_state=99) -> np.ndarray:
    mds = MDS(n_components=n, dissimilarity='precomputed', random_state=random_state, normalized_stress='auto')
    embeddings = mds.fit_transform(diatances_matrix)
    return embeddings

# takes list of ids, places them in a subdir
def download_fasta(ids, subdir):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    print('Writing script ...')
    with open('script.sh', 'w') as f:
        f.write(f'cd {subdir}\n')
        f.write('rm -r \n')
        if isinstance(ids, str):  # If ids is a single string, convert it to a list
            ids = [ids]
        for id in ids:
            f.write(f'curl -OJX GET "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{id}/download?include_annotation_type=GENOME_FASTA&filename={id}.zip" -H "Accept: application/zip"\n')
        f.write('echo "Extracting and cleaning ... "\n')
        f.write("unzip -o '*.zip'\n")
        f.write("rm *.zip\n")
        f.write("find . -type f -exec mv {} . \\;\n")
        f.write("rm -r *.json* *.md ncbi_dataset*\n")
    print('Running script ...')
    os.system('chmod 777 script.sh')
    os.system('./script.sh')
    print('Cleaning ...')
    print('Success!')

def generate_accession_ids(subdir):
    ids = []
    labels = []
    for file in os.listdir(subdir):
        with open(os.path.join(subdir, file), 'r') as f:
            for line in f:
                if line.startswith('Accession'):
                    continue
                ids.append(line.split(',')[0])
                labels.append(line.split(';s__')[1].split(' ')[0])

    return ids, labels

def plot_3d(reduced_data, clusters):
    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_label in np.unique(clusters):
        idxs = np.where(np.array(clusters) == cluster_label)[0]

        ax.scatter(reduced_data[idxs, 0], reduced_data[idxs, 1], reduced_data[idxs, 2],\
                   label=cluster_label, s=15)
    ax.set_xlabel('PCo1')
    ax.set_ylabel('PCo2')
    ax.set_zlabel('PCo3')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title('SARS-CoV-2-Variants')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.show()

def get_ids_and_labels(infile):
    labels = []
    ids = []
    with open(infile, 'r') as f:
        for line in f:
            labels.append(line.split(',')[0].strip())
            ids.append(line.split(',')[1].strip())
    return ids, labels