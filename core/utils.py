import pickle
import os
import numpy as np
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sph
from fcmeans import FCM
from sklearn.cluster import MiniBatchKMeans

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')

def get_clustered_elements(c_labels, embeddings, current_rel):
    labels = list(embeddings.keys())
    idx = labels.index(current_rel)
    current_cluster_idx = c_labels[idx]
    curr_cluster = []
    for c, i in enumerate(c_labels):
        if i == current_cluster_idx: curr_cluster.append(labels[c])
    curr_cluster.remove(current_rel)
    return curr_cluster

def get_hierarchical_clusters(embeddings,type):
    rels = np.array(list(embeddings.values()))
    if type == 'semantic':
        pca_rels = []
        pca = PCA(n_components=1)
        for r in rels:
            pca_rels.append(np.squeeze(pca.fit_transform(r.reshape(-1, 1))))
        rels = np.array(pca_rels)
    pdist = sph.distance.pdist(rels, metric='euclidean')
    linkage = sph.linkage(pdist, method='complete')
    c_labels = sph.fcluster(linkage, 0.2 * rels.shape[0], 'maxclust')
    return c_labels

def get_entity_clusters(ent_embeddings):
    corr= np.array(list(ent_embeddings.values()))
    pdist = sph.distance.pdist(corr, metric='euclidean')
    linkage = sph.linkage(pdist,method='complete')
    c_labels=sph.fcluster(linkage,0.5*corr.shape[0],'maxclust')
    del corr, pdist,linkage
    return c_labels

def get_optimal_clusters(max_k,embeddings,type):
    rels = np.array(list(embeddings.values()))
    if type == 'semantic':
        pca_rels = []
        pca = PCA(n_components=1)
        for r in rels:
            pca_rels.append(np.squeeze(pca.fit_transform(r.reshape(-1,1))))
        rels = np.array(pca_rels)
    fcm=FCM(max_k)
    fcm.fit(rels)
    labels=fcm.predict(rels)
    return labels


def get_clusters(k,embeddings,current_rel,type):
    labels=list(embeddings.keys())
    rels=np.array(list(embeddings.values()))
    if type=='semantic':
        pca_rels=[]
        pca=PCA(n_components=1)
        for r in rels:
            pca_rels.append(np.squeeze(pca.fit_transform(r)))
        rels=np.array(pca_rels)
    n_clusters=int(len(labels)/k)
    kmeans_model=MiniBatchKMeans(init='k-means++',n_clusters=n_clusters,n_init=3)
    kmeans_model.fit(rels)
    c_labels=kmeans_model.labels_
    idx=labels.index(current_rel)
    current_cluster_idx=c_labels[idx]
    curr_cluster=[]
    for c,i in enumerate(c_labels):
        if i==current_cluster_idx:curr_cluster.append(labels[c])
    curr_cluster.remove(current_rel)
    return curr_cluster

def load_dataset(dataset_name,model_name,tmp=False):
    MODEL_DICT = {'semantic': ['ComplEx', 'DistMult', 'HolE', 'ANALOGY', 'SimplE', 'TuckER'],
                  'translation': ['TransE', 'TransH', 'TransR', 'TransD']}
    if tmp:
        prefix='tmp_'
    else:
        prefix=''
    rels = load_obj(os.path.join('datasets', prefix+dataset_name, prefix+dataset_name + '_' + model_name + '_relations'))
    ents = load_obj(os.path.join('datasets', prefix+dataset_name, prefix+dataset_name + '_' + model_name + '_entities'))
    preds = load_obj(os.path.join('datasets', prefix+dataset_name, prefix+dataset_name + '_' + model_name + '_predictions'))
    try:
        if model_name in MODEL_DICT['semantic']:
            type = 'semantic'
        elif model_name in MODEL_DICT['translation']:
            type = 'translation'
    except:
        print('Your model is not supported by the framework')
        quit(1)
    if type == 'semantic':
        rels = {k: np.diag(v) for k, v in rels.items()}
    known_facts = load_obj(os.path.join('../datasets', prefix+dataset_name, prefix+dataset_name+ '_' + model_name + + '_known_facts'))
    return rels, ents, known_facts, preds, type

