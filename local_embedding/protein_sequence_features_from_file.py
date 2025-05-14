# Define dictionaries for amino acid classification
import csv

non_polar = set(['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'P', 'V', 'W'])
polar_neutral = set(['N', 'Q', 'S', 'T', 'Y'])
acidic = set(['D', 'E'])
basic = set(['K', 'R'])

# Function to convert amino acid sequence to feature vectors
def convert_to_feature_vector(sequence):
    feature_vector = [0] * 64
    for i in range(len(sequence) - 2):
        three_mer = sequence[i:i+3]
        # Assign index based on amino acid category
        if three_mer[0] in non_polar:
            index = 0
        elif three_mer[0] in polar_neutral:
            index = 1
        elif three_mer[0] in acidic:
            index = 2
        elif three_mer[0] in basic:
            index = 3
        else:
            continue

        # Update the corresponding position in the feature vector
        index = index * 16 + (ord(three_mer[1]) - 65) * 4 + (ord(three_mer[2]) - 65)
        if index < 64:
            feature_vector[index] += 1

    # Normalize feature vector to be between 0 and 1
    max_count = max(feature_vector)
    if max_count != 0:
        feature_vector = [count / max_count for count in feature_vector]

    return feature_vector
# Function to read protein sequences from a FASTA file
def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ''
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

# Read protein sequences from a FASTA file
fasta_file_path = 'ic_idmapping_2023_10_18.fasta'
protein_sequences = read_fasta_file(fasta_file_path)

# Convert and store feature vectors in a CSV file
output_file_path = 'ic_output_feature_vectors.csv'
with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Feature' + str(i) for i in range(1, 65)])  # Write header row
    for sequence in protein_sequences:
        feature_vector = convert_to_feature_vector(sequence)
        csv_writer.writerow(feature_vector)
