import numpy as np
import operator

def sig(x, y, type):
    if type == 'semantic':
        result = -np.dot(x, np.transpose(y))
    elif type == 'translation':
        result = -(x + y)
    return 1 / (1 + np.exp(result))


def point_hess(e_o, nei, embd_e, embd_rel, type):
    dim = np.shape(e_o)[0]
    H = np.zeros((dim, dim))
    for i in nei:
        if type == 'semantic':
            X = np.multiply(np.reshape(embd_e[i[0]], (1, -1)), embd_rel[i[1]])
        elif type == 'translation':
            X = np.reshape(embd_e[i[0]] + embd_rel[i[1]], (-1, 1))
        sig_tri = sig(e_o, X, type)
        Sig = (sig_tri) * (1 - sig_tri)
        if type == 'semantic':
            H += Sig * np.dot(np.transpose(X), X)
        elif type == 'translation':
            H += Sig * (X + X)
    return H


def point_score(Y, X, e_o, H, type):
    sig_tri = sig(e_o, X, type)
    try:
        M = np.linalg.inv(H + (sig_tri) * (1 - sig_tri) * np.dot(np.transpose(X), X))
    except:
        return np.inf, []
    if type == 'semantic':
        Score = - np.dot(Y, np.transpose((1 - sig_tri) * np.dot(X, M)))
    elif type == 'translation':
        Score = - np.linalg.norm(Y + (np.transpose((1 - sig_tri) * np.dot(np.reshape(X, (-1, 1)).T, M))))
    return Score, M


def find_best_attack(h, t, cur_rel, ent_dict, rel_dict, facts, type):
    dict_s = {}
    triples = []
    nei1 = [(i[0], k, i[1]) for k, v in facts.items() for i in v if i[1] == h]
    nei2 = [(i[0], k, i[1]) for k, v in facts.items() for i in v if i[1] == t]
    e_o = ent_dict[t]
    e_s = ent_dict[h]
    if type == 'semantic':
        Y1 = np.dot(rel_dict[cur_rel], e_o)
        Y2 = np.dot(rel_dict[cur_rel], e_s)
    elif type == 'translation':
        Y1 = rel_dict[cur_rel] + e_o
        Y2 = rel_dict[cur_rel] + e_s
    if len(nei1) > 0:
        H1 = point_hess(e_o, nei1, ent_dict, rel_dict, type)
        if len(nei1) > 50:
            nei1 = nei1[:50]
        for i in nei1:
            e1 = i[0]
            rel = i[1]
            if type == 'semantic':
                pred = np.matmul(rel_dict[rel], ent_dict[e1])
            elif type == 'translation':
                pred = rel_dict[rel] + ent_dict[e1]
            score_t, M = point_score(Y1, pred, e_o, H1, type)
            dict_s[i] = score_t
    if len(nei2) > 0:
        H2 = point_hess(e_s, nei2, ent_dict, rel_dict, type)
        if len(nei2) > 50:
            nei2 = nei2[:50]
        for i in nei2:
            e1 = i[0]
            rel = i[1]
            if type == 'semantic':
                pred = np.matmul(rel_dict[rel], ent_dict[e1])
            elif type == 'translation':
                pred = rel_dict[rel] + ent_dict[e1]
            score_t, M = point_score(Y2, pred, e_s, H2, type)
            dict_s[i] = score_t
    if nei1 and nei2:
        sorted_score = sorted(dict_s.items(), key=operator.itemgetter(1))
        if sorted_score and sorted_score[0][1] < 0:
            triples = []
            i = 0
            while (i <= len(sorted_score) - 2) and len(triples) < 3:
                if abs(sorted_score[i][1] - sorted_score[i + 1][1]) >= 1000:
                    triples.append(sorted_score[i][0])
                    i += 1
                else:
                    break
    return triples
