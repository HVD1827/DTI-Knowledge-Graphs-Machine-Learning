import numpy as np

# lire le fichier contenant le graphe
with open('KGE/data/ic/target_target_sim_results_ic.txt') as f:
    data = f.readlines()

# extraire les noms des drugs et targets sans répétition
drugs = sorted(list(set([line.split()[0] for line in data])))
targets = sorted(list(set([line.split()[1] for line in data])))

# créer une matrice 2D remplie de zéros
matrix = np.zeros((len(drugs), len(targets)))

# remplir la matrice avec les valeurs d'interaction
for line in data:
    drug, target, interaction = line.split()
    matrix[drugs.index(drug), targets.index(target)] = float(interaction)

# afficher la matrice avec les noms des drugs et targets
print('\t' + '\t'.join(targets))
for i, drug in enumerate(drugs):
    row = '\t'.join([f'{value:.2f}' for value in matrix[i]])
    print(f'{drug}\t{row}')

# Ouvrir le fichier contenant le graphe
with open("KGE/data/ic/target_target_sim_results_ic.txt", "r") as f:
    lines = f.readlines()

# Extraire les noms de médicaments et de cibles uniques
drugs = list(set([line.split()[0] for line in lines]))
targets = list(set([line.split()[1] for line in lines]))

# Initialiser la matrice d'adjacence avec des zéros
matrix = [[0.0] * len(targets) for i in range(len(drugs))]

# Remplir la matrice d'adjacence avec les valeurs de similarité
for line in lines:
    drug, target, sim = line.split()
    i = drugs.index(drug)
    j = targets.index(target)
    matrix[i][j] = float(sim)

# Afficher la matrice d'adjacence avec les noms de médicaments et de cibles
with open("../ic_dataset/target_target_sim.txt", "w") as f:
    # Écrire les noms de cibles dans la première ligne
    f.write("\t" + "\t".join(targets) + "\n")

    # Écrire les noms de médicaments et les valeurs de similarité
    for i in range(len(drugs)):
        f.write(drugs[i] + "\t")
        for j in range(len(targets)):
            f.write(str(matrix[i][j]) + "\t")
        f.write("\n")