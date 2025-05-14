import csv
import numpy
import pandas as pd
import numpy as np
import sys
import os
import tarfile
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


entity_emb = np.load('../data/ckpts/DTIOG_DistMult_entity.npy')
rel_emb = np.load('../data/ckpts/DTIOG_DistMult_relation.npy')

Entities_Embedding_file = '../data/ckpts/entities.tsv'
Entities_Embedding_List = {}
IdEntiy__Embedding_List = {}
with open(Entities_Embedding_file) as f:
    for line in f:
        it = line.strip().split("\t")
        id = str(it[0])
        entity = str(it[1])
        if entity not in Entities_Embedding_List:
            Entities_Embedding_List[entity] = set()
        Entities_Embedding_List[entity].add(int(id))
        if id not in IdEntiy__Embedding_List:
            IdEntiy__Embedding_List[id] = set()
        IdEntiy__Embedding_List[id].add(str(entity))

############################################

embedlist = []

rel_id_DGLKE_List = {}
rel_idmap_file = 'gpcr_dataset/drug_list.txt'
with open("../data/source_drug_gpcr.list", 'w+') as feRel:
    with open(rel_idmap_file) as f:
        for line in f:
            it = line.strip().split(" ")
            relid = it[0]
            if str(relid) not in rel_id_DGLKE_List:
                rel_id_DGLKE_List[str(relid)] = set()
                rel_id_DGLKE_List[str(relid)].add(str("rel"))
                try:
                    entity = str(relid)
                    strentity = ''
                    strentity = 'Compound::' + entity
                    ids = Entities_Embedding_List[strentity]
                    for id in ids:
                        feRel.writelines("{}\n".format(id))
                        embedlist.append(id)
                except:
                    print("drug not found")
                    pass

rel_id_DGLKE_List = {}
rel_idmap_file = 'gpcr_dataset/target_list.txt'
with open("../data/train/target_gpcr.list", 'w+') as feRel:
    with open(rel_idmap_file) as f:
        for line in f:
            it = line.strip().split(" ")
            relid = it[0]
            if str(relid) not in rel_id_DGLKE_List:
                rel_id_DGLKE_List[str(relid)] = set()
                rel_id_DGLKE_List[str(relid)].add(str("rel"))
                try:
                    entity = str(relid)
                    strentity = ''
                    strentity = 'Human:' + entity
                    ids = Entities_Embedding_List[strentity]
                    for id in ids:
                        feRel.writelines("{}\n".format(id))
                        embedlist.append(id)
                except:
                    print(entity)
                    print("target not found")
                    pass
Entities_Embedding_List = {}
with open("../data/train/dict_gpcr_target.list", 'w+') as feRel:
    for i, x in enumerate(embedlist):
        coll = IdEntiy__Embedding_List[str(x)]
        for z in coll:
            if z not in Entities_Embedding_List:
                Entities_Embedding_List[z] = set()
            Entities_Embedding_List[z].add(int(i))
            if str(z).startswith("Human:"):
                trt = str(z).split("Human:")
                feRel.writelines("{}\t{}\n".format(i, trt[1]))
            if str(z).startswith("Compound::"):
                trt = str(z).split("Compound::")
                feRel.writelines("{}\t{}\n".format(i, trt[1]))

embnympy = numpy.array(embedlist)
axis = 0
# filter_indices=[1]
# print(np.take(entity_emb, embedlist, axis))
newvectoremb = np.take(entity_emb, embedlist, axis)
# print(newvectoremb[:2])
# print("-------")
# print(entity_emb[:2])
# newvectoremb2=entity_emb[np.in1d(entity_emb[:, 0], embnympy)]
# print(newvectoremb2)
##--------------------------------------------------------
similarity = cosine_similarity(newvectoremb)
similarity = similarity.flatten()
print(similarity.shape)

# cleanup self-compare and dup-compare
s = similarity < 0.99
s = np.unique(similarity[s])


plt.xlabel('Cosine similarity')
plt.ylabel('Number of entity pairs')
plt.hist(s)
plt.savefig("plot1_e.eps")
