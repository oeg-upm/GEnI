from core import utils, axiom_extraction, influence_detection, correlation_detection
import operator
import os
import numpy as np

def prepare_geni_data(model_dataset,model_name,prefix,user_th_value,tmp):
    print('[PREPROCESSING: CURRENT STEP] Loading data model')
    rel_dict, ent_dict, known_facts, validation_facts, type = utils.load_dataset(model_dataset, model_name, tmp)
    print('[PREPROCESSING: CURRENT STEP] Generating relation clusters')
    if os.path.exists(os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_r_labels.pkl')):
        r_labels = utils.load_obj(os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_r_labels'))
    else:
        r_labels = utils.get_optimal_clusters(int(len(rel_dict.keys()) / 2), rel_dict, type)
        utils.save_obj(r_labels, os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_r_labels'))
    print('[PREPROCESSING: CURRENT STEP] Generating entity clusters')
    if os.path.exists(os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_e_labels.pkl')):
        e_clusters = utils.load_obj(os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_e_labels'))
    else:
        e_clusters = utils.get_entity_clusters(ent_dict)
        utils.save_obj(e_clusters, os.path.join('datasets', prefix+model_dataset, prefix+model_dataset + '_' + model_name + '_e_labels'))
    TH_MAX = abs(np.max(np.array(list(ent_dict.values()))) - np.min(np.array(list(ent_dict.values()))))
    th_value = TH_MAX + (1 - user_th_value) * TH_MAX
    return rel_dict,ent_dict,known_facts,validation_facts,type,th_value,e_clusters,r_labels

def _phase_1(h,cur_rel,t,r_labels,rel_dict,known_facts,type,th_value):
    search_space = utils.get_clustered_elements(r_labels, rel_dict, cur_rel)
    cur_axioms = axiom_extraction.get_axioms(th_value, rel_dict, cur_rel, type, search_space)
    rules=None
    if cur_axioms['symmetric'] or cur_axioms['transitive'] or cur_axioms['equivalent'] or cur_axioms['inverse'] or \
            cur_axioms['chain']:
        trig_rules, resps = axiom_extraction.evaluate_fact_axiom((h, t, cur_rel), cur_axioms, known_facts)
        if resps:
            if trig_rules['symmetric']:
                rules = [('rules', 'symmetric')]
                print(
                    '[SUCCESS!] The relation %s is symmetric, so your fact can be inferred based on the known fact (%s,%s,%s)' % (
                    cur_rel, t, cur_rel, h))

            if trig_rules['transitive']:
                rules = [('rules', 'transitive', trig_rules['transitive'])]
                print('[SUCCESS!] The relation %s is transitive, so your fact can be inferred based on the known'
                      'facts (%s,%s,%s) ^ (%s,%s,%s) -> (%s,%s,%s)' % (
                      cur_rel, h, cur_rel, trig_rules['transitive'][0], trig_rules['transitive'][0]
                      , cur_rel, t, h, cur_rel, t))

            if trig_rules['equivalent']:
                rules = [('rules', 'equivalent', trig_rules['equivalent'][0])]
                print(
                    '[SUCCESS!] The relation %s has an equivalent relation %s, so your fact can be inferred based on the known fact (%s,%s,%s)' % (
                        cur_rel, trig_rules['equivalent'][0], h, trig_rules['equivalent'], t))
            if trig_rules['inverse']:
                rules = [('rules', 'inverse', trig_rules['inverse'][0])]
                print(
                    '[SUCCESS!] The relation %s has an inverse relation %s, so your fact can be inferred based on the known fact (%s,%s,%s)' % (
                        cur_rel, trig_rules['inverse'][0], t, trig_rules['inverse'], h))
            if trig_rules['chain']:
                rules = [('rules', 'chain', trig_rules['chain'])]
                print(
                    '[SUCCESS!] Your fact can be inferred using the rule chain (%s, %s, %s) ^ (%s, %s, %s) -> (%s,%s,%s)'
                    % (h, trig_rules['chain'][0][0][0], trig_rules['chain'][0][1][0], trig_rules['chain'][0][1][0],
                       trig_rules['chain'][0][0][1], t, h, cur_rel, t))
    return rules

def _phase_2(h,cur_rel,t,goal,ent_dict,rel_dict,e_clusters,known_facts,type,th_value):
    ent_matrixes=np.array(ent_dict.values())
    correlations=None
    head_corrs = correlation_detection.find_entity_correlations(h, ent_dict, ent_matrixes, e_clusters, th_value)
    tail_corrs = correlation_detection.find_entity_correlations(t, ent_dict, ent_matrixes, e_clusters, th_value)
    if goal == 'o':
        direct_correlations = correlation_detection.find_direct_correlations(t, cur_rel, tail_corrs, rel_dict,
                                                                             ent_dict, type, th_value)
    else:
        direct_correlations = correlation_detection.find_direct_correlations(h, cur_rel, head_corrs,
                                                                             rel_dict, ent_dict, type, th_value)
    if direct_correlations:
        eval_corrs = correlation_detection.evaluate_direct_correlations(goal, direct_correlations, known_facts)
        if 1 in list(eval_corrs.values()):
            correlations = [('correlation', 'direct', eval_corrs)]
            corrs = [k for k, v in eval_corrs.items() if v == 1]
            corrs = sorted(corrs, key=operator.itemgetter(2), reverse=True)
            print('[SUCCESS!] The predicate or your fact (%s,%s) is highly correlated with the existing '
                  'predicate (%s,%s)' % (cur_rel, t, corrs[0][1], corrs[0][0]))

    if goal == 'o':
        triangular_correlations = correlation_detection.find_triangular_correlations(h, cur_rel, rel_dict,
                                                                                     ent_dict, type, th_value)
    else:
        triangular_correlations = correlation_detection.find_triangular_correlations(t, cur_rel,
                                                                                     rel_dict, ent_dict,
                                                                                     type, th_value)
    if triangular_correlations:
        if goal == 'o':
            eval_triangular = correlation_detection.evaluate_triangular_correlations(h, triangular_correlations,
                                                                                     known_facts)
        else:
            eval_triangular = correlation_detection.evaluate_triangular_correlations(t,
                                                                                     triangular_correlations,
                                                                                     known_facts)
        if 1 in list(eval_triangular.values()):
            correlations = [('correlation', 'triangular', eval_triangular)]
            corrs = [k for k, v in eval_triangular.items() if v == 1]
            corrs = sorted(corrs, key=operator.itemgetter(2), reverse=True)
            print('[SUCCESS!] The predicate or your fact (%s,%s) is highly correlated with the existing '
                  'entity %s' % (cur_rel, t, corrs[0][0]))
    return correlations

def _phase_3(h,cur_rel,t,goal,ent_dict,rel_dict,known_facts,type):
    influence=None
    if goal == 'o':
        top_triple = influence_detection.find_best_attack(h, t, cur_rel, ent_dict, rel_dict, known_facts, type)
    else:
        top_triple = influence_detection.find_best_attack(t, h, cur_rel, ent_dict, rel_dict, known_facts, type)
    if top_triple:
        influence = [('influence', top_triple)]
        if len(top_triple) == 1:
            print('[SUCCESS!] The model is inferring the current fact mainly based on the fact (%s,%s,%s), '
                  'which has the greatest influence on its prediction' % (
                  top_triple[0][0], top_triple[0][1], top_triple[0][2]))
        else:
            str = ''
            for t in top_triple:
                str += ' (' + t[0] + ',' + t[1] + ',' + t[2] + ')'
            str += '.'
            print(
                '[SUCCESS!] The model is inferring the current fact mainly based on the following most influential facts ' + str)
    return influence

def _generate_insight(h,cur_rel,t,goal,ent_dict,e_clusters,rel_dict,r_labels,known_facts,type,th_value):
    rules = _phase_1(h, cur_rel, t, r_labels, rel_dict, known_facts, type, th_value)
    if not rules:
        correlations = _phase_2(h, cur_rel, t, goal, ent_dict, rel_dict, e_clusters, known_facts, type, th_value)
        if not correlations:
            influence = _phase_3(h, cur_rel, t, goal, ent_dict, rel_dict, known_facts, type)
            if influence:
                return influence
            else:
                print('[FAILED!] No support for your fact could be found')
                return [0]
        else:
            return correlations
    else:
        return rules

def predict_all(model_dataset,model_name,user_th,prefix,tmp):
    rel_dict, ent_dict, known_facts, predictions, type, th_value, e_clusters, r_labels = prepare_geni_data(
        model_dataset, model_name, prefix, user_th,tmp)
    fact_explaination={}
    for cur_rel, p in predictions.items():
        for f in p:
            goal = f[0]
            h = f[1]
            t = f[2]
            fact_explaination[(goal, h, cur_rel, t)]=_generate_insight(h,cur_rel,t,goal,ent_dict,e_clusters,
                                                                       rel_dict,r_labels,known_facts,type,th_value)
    return fact_explaination

def predict_single(fact, model_dataset, model_name, user_th, goal, prefix,tmp):
    rel_dict, ent_dict, known_facts, _, type, th_value, e_clusters, r_labels = prepare_geni_data(
        model_dataset, model_name, prefix, user_th,tmp)
    h = fact[0]
    cur_rel = fact[1]
    t = fact[2]
    if h not in list(ent_dict.keys()) or cur_rel not in list(rel_dict.keys()) or t not in list(ent_dict.keys()):
        print(
            '[ERROR!] Incorrect fact input. Check that both of the entities and the relation are spelled correctly and belong to the chosen dataset')
        quit(1)
    print('-->CURRENT FACT: (%s,%s,%s)' % (h, cur_rel, t))
    return _generate_insight(h,cur_rel,t,goal,ent_dict,e_clusters,rel_dict,r_labels,known_facts,type,th_value)
