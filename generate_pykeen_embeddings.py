import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df
import os
from core.utils import save_obj
import argparse
from pykeen.evaluation import RankBasedEvaluator

def _split_complex(x):
    dim = x.shape[-1] // 2
    return x[..., :dim], x[..., dim:]

def generate_embeddings(model_dataset, model_name, num_epochs, dim, ratios, tmp, goal,evaluate=False):
    if tmp:
        prefix = 'tmp_'
    else:
        prefix = ''
    triples_factory = get_dataset(dataset=model_dataset).training
    training_factory, testing_factory, validation_factory = triples_factory.split(ratios, random_state=1995)
    if not os.path.exists(os.path.join('datasets', prefix + model_dataset)):
        os.mkdir(os.path.join('datasets', prefix + model_dataset))

    if not os.path.exists(
            os.path.join('datasets', prefix + model_dataset, prefix + model_dataset + '_'+model_name+ '_known_facts.pkl')):
        facts = {r: [] for r in list(triples_factory.relation_to_id)}
        for f in training_factory.label_triples(triples=training_factory.mapped_triples):
            facts[f[1]].append([f[0], f[2]])
        save_obj(facts, os.path.join('datasets', prefix + model_dataset, prefix + model_dataset + '_'+model_name+ '_known_facts'))
    if model_name!='TransR':
        result = pipeline(
            model=model_name,
            training=training_factory,
            testing=testing_factory,
            validation=validation_factory,
            model_kwargs=dict(embedding_dim=dim),
            training_kwargs=dict(num_epochs=num_epochs),
        )
    else:
        result = pipeline(
            model=model_name,
            training=training_factory,
            testing=testing_factory,
            validation=validation_factory,
            model_kwargs=dict(embedding_dim=dim,relation_dim=dim),
            training_kwargs=dict(num_epochs=num_epochs),
        )

    model = result.model
    if evaluate:
        model = result.model
        evaluator = RankBasedEvaluator()
        test_results = evaluator.evaluate(
            model=model,
            mapped_triples=testing_factory.mapped_triples,
        )
        mrr=test_results.get_metric('mrr')
        hat1=test_results.get_metric('hits_at_1')
        print('[TRAINING RESULTS] MRR: %f' %mrr)
        print('[TRAINING RESULTS] H@1: %f' %hat1)
    entity_embeddings: torch.FloatTensor = model.entity_representations[0]
    entity_indices = triples_factory.entity_to_id
    entity_dict = {str(k): np.squeeze(entity_embeddings(torch.tensor([v],dtype=torch.long)).cpu().detach().numpy()) for k, v in entity_indices.items()}
    save_obj(entity_dict,
             os.path.join('datasets', prefix + model_dataset, prefix + model_dataset + '_' + model_name + '_entities'))

    relation_embeddings: torch.FloatTensor = model.relation_representations[0]
    relation_indices = triples_factory.relation_to_id
    if model_name=='ComplEx':
        relation_dict = {
            str(k): np.squeeze(relation_embeddings(torch.tensor([v], dtype=torch.long)).cpu().detach().numpy()) for k, v in relation_indices.items()}
    else:
        relation_dict = {str(k): _split_complex(np.squeeze(relation_embeddings(torch.tensor([v],dtype=torch.long)).cpu().detach().numpy())) for k, v in relation_indices.items()}
    save_obj(relation_dict,
             os.path.join('datasets', prefix + model_dataset, prefix + model_dataset + '_' + model_name + '_relations'))

    labelled_triples = testing_factory.label_triples(triples=testing_factory.mapped_triples)
    results = {r: [] for r in testing_factory.relation_id_to_label.values()}
    for t in labelled_triples:
        if goal in ['b','o']:
            if t[0].isnumeric():
                df = get_tail_prediction_df(model, int(t[0]), t[1], triples_factory=training_factory)
            else:
                df = get_tail_prediction_df(model, t[0], t[1], triples_factory=training_factory)
            df = df.sort_values(by=['score'], ascending=False)
            tail = df.iloc[0]['tail_label']
            del df
            results[t[1]].append(('o', t[0], tail))
        if goal in ['b','s']:
            if t[2].isnumeric():
                df = get_head_prediction_df(model, t[1], int(t[2]), triples_factory=training_factory)
            else:
                df = get_head_prediction_df(model, t[1], t[2], triples_factory=training_factory)
            df = df.sort_values(by=['score'], ascending=False)
            head = df.iloc[0]['head_label']
            del df
            results[t[1]].append(('s', head, t[2]))
    save_obj(results, os.path.join('datasets', prefix + model_dataset,
                                   prefix + model_dataset + '_' + model_name + '_predictions'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', help="Indicate a dataset to work with")
    parser.add_argument('--model', '-m', help="Indicate a valid KGE model")
    parser.add_argument('--epochs', '-e', help="Number of training epochs. If unspecified, it uses 100 by default", type=int)
    parser.add_argument('--dim', help="Embedding dimension. If unspecified, it uses 100 by default", type=int)
    parser.add_argument('--split', help="Training/Test/Validation split ratios. If unspecified, it uses 0.8 0.1 0.1",
                        nargs='+', type=float)
    parser.add_argument('--goal','-g', help="Specify the  type of predictions to generate \n "
                                            "-> \'o\' for object (tail) predictions\n"
                                            "-> \'s\' for subject (head) predictions\n"
                                            "-> \'b\' for both.\n"
                                            "If no value is specified, both predictions are computed by default")
    parser.add_argument('--tmp',
                        help="Whether the generated data is permanently stored or deleted once processed. It unspecified, data is stored permantently",
                        action="store_true")
    parser.add_argument('--eval',
                        help="Provide evaluation metrics",
                        action="store_true")
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    epochs = args.epochs
    if dataset is None or model is None:
        parser.error("It is required to specify a dataset, a model and a number of epochs for training")
        exit(1)
    if not epochs:
        epochs=100
    dim = args.dim
    if dim is None:
        dim = 100
    ratio = args.split
    if ratio:
        if sum(ratio)>1:
            parser.error("Ratios must sum a total of 1")
        if len(ratio)!=3:
            parser.error("Exactly 3 values are required")
    else:
        ratio = [0.8, 0.1, 0.1]
    tmp = args.tmp
    goal=args.goal
    if not goal:
        goal='b'
    elif goal not in ['s','o','b']:
        parser.error("There are only 3 possible values: o (object), s (subject) and b (both).")

    generate_embeddings(dataset, model, epochs, dim, ratio, tmp, goal)
