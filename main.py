import time
from core import geni_workflow
from core.utils import *
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', help="Indicate a dataset to work with")
    parser.add_argument('--model', '-m', help="Indicate a valid KGE model")
    parser.add_argument('--threshold','-th', help="User threshold value. Default value is 0.6 ",type=float)
    parser.add_argument('--fact','-f', help="Explain a single prediction in the format h r t ",nargs='+')
    parser.add_argument('--goal','-g',help="s if the head entity is predicted, o if the tail entity is predicted. Default value is o")
    parser.add_argument('--all', help="Explain all stored predictions")
    parser.add_argument('--save','-s',help="Save final results", action="store_true")
    parser.add_argument('--tmp',
                        help="Whether the generated data is permanently stored or deleted once processed. It unspecified, data is stored permantently",
                        action="store_true")
    args = parser.parse_args()
    model_dataset=args.dataset
    model_name=args.model
    user_th=args.args.threshold
    fact=args.model
    all=args.model
    goal=args.goal
    tmp=args.tmp
    save=args.save
    if tmp:
        prefix='tmp_'
    else:
        prefix=''

    if not model_dataset or not model_name:
        parser.error("[ERROR!] A dataset and a model must be specified!")
        quit(1)

    if not fact and not all:
        parser.error("[ERROR!] No input facts have been specified")
        quit(1)

    if not goal:
        goal='o'

    if not user_th:
        user_th=0.6

    start_time=time.time()
    fact_explaination={}
    if fact:
        fact_explaination[(goal, fact[0], fact[1], fact[2])]=geni_workflow.predict_single(fact,model_dataset,model_name,user_th,goal,prefix,tmp)

    elif all:
        fact_explaination=geni_workflow.predict_all(model_dataset,model_name,user_th,prefix,tmp)

    if save:
        save_obj(fact_explaination,prefix+model_dataset+'_'+model_name+'_GEnI_output')

    if tmp:
        for f in os.listdir(os.path.join('datasets', prefix+model_dataset)):
            if f.startswith('tmp_'): os.remove(f)
    print('EXECUTION TIME: %s' %(time.time()-start_time))

