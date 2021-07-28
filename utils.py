import pickle

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