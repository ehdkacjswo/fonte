import argparse, os, copy, itertools, pickle, sys, json
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result/'

def ccc():
    a = 1

def aaa(setting_dict={'adddel': 'all-uni', 'use_br': False, 'beta': 1.0, 'diff_type': 'gumtree_class', 'use_stopword': True}):
    fonte_iter_list, fonte_rank_list = [], []
    ensemble_iter_list, ensemble_rank_list = [], []
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    
    for project in os.listdir(RESULT_DATA_DIR):
        if project == 'Closure-131b':
            continue

        iter_dir = os.path.join(RESULT_DATA_DIR, project, 'iteration')

        # Fonte iteration
        with open(os.path.join(iter_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
        
        fonte_iter_list.append(fonte_dict['precise'])

        # Ensemble iteration
        with open(os.path.join(iter_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)
        
        ensemble_iter_list.append(ensemble_dict['precise'][frozenset(setting_dict.items())])

        

        
        # Rank
        [pid, vid] = project[:-1].split('-')
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        vote_dir = os.path.join(RESULT_DATA_DIR, project, 'vote')

        # Fonte voting
        with open(os.path.join(vote_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
        
        rank = fonte_dict['precise']['rank'].get(BIC, None)
        fonte_rank_list.append(rank)
        if rank is None:
            print('No BIC (Fonte) ', project)
        
        # Ensemble voting
        with open(os.path.join(vote_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)

        rank = ensemble_dict['precise'][frozenset(setting_dict.items())]['rank'].get(BIC, None)
        ensemble_rank_list.append(rank)
        if rank is None:
            print('No BIC (Ensemble) ', project)
        
        if fonte_rank_list[-1] > ensemble_rank_list[-1]:
            print('Better', project, fonte_rank_list[-1], ensemble_rank_list[-1])

        elif fonte_rank_list[-1] < ensemble_rank_list[-1]:
            print('Worse', project, fonte_rank_list[-1], ensemble_rank_list[-1])

    better, same, worse = 0, 0, 0
    reduced, increased, reduction = 0, 0, 0
    cost_saving = []
    
    for fonte_iter, ensemble_iter in zip(fonte_iter_list, ensemble_iter_list):
        if fonte_iter > ensemble_iter:
            better += 1
            reduced += fonte_iter - ensemble_iter
        elif fonte_iter < ensemble_iter:
            worse += 1
            increased += ensemble_iter - fonte_iter
        else:
            same += 1

        cost_saving.append(fonte_iter - ensemble_iter)
    
    print(better, same, worse)
    print(reduced, increased)
    
    """plt.figure(figsize=(9, 2))
    plt.title("# saved search iterations by changing the search algorithm to the weighted bisection")

    cost_saving = list(reversed(sorted(cost_saving)))

    w, p = wilcoxon(cost_saving)
    #To confirm that the median of the differences can be assumed to be positive, we use:
    w, p = wilcoxon(cost_saving, alternative='greater')
    print("Wilcoxon signed rank test", w, p)
    N = len(cost_saving)

    plt.bar(range(0, N), cost_saving,
        color=["red" if d < 0 else "green" for d in cost_saving])
    plt.axhline(0, color="black")

    plt.yticks(range(min(cost_saving), max(cost_saving)+1))

    plt.axvspan(-0.5, N * reduced - 0.5, facecolor='green', alpha=0.1)
    plt.axvspan(N * (reduced + same)-0.5, N-0.5, facecolor='red', alpha=0.1)

    if reduced > 0.05:
        plt.text(N * reduced/2 - 0.5, max(cost_saving)-1, f"{reduced*100:.1f}%", horizontalalignment="center")
    if same > 0.05:
        plt.text(N * (reduced + same/2) - 0.5, max(cost_saving)-1, f"{same*100:.1f}%", horizontalalignment="center")
    if increased > 0.05:
        plt.text(N * (reduced + same + increased/2) - 0.5, max(cost_saving)-1, f"{increased*100:.1f}%", horizontalalignment="center")

    plt.xlim((0-0.5, N-0.5))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    if b_col == "standard_bisection_on_C":
        plt.axhline(np.mean(cost_saving), color="black", linestyle="--", label=f"Average Saved Iterations: {np.mean(cost_saving).round(1)}")
        print("Average # Saved Iterations", np.mean(cost_saving))
        plt.legend(loc="upper right")

    savepath = os.path.join(RESULTS_DIR,
        f"RQ2_cost_saving_by_{w_col}_compared_to_{b_col}.pdf")
    plt.savefig(savepath, bbox_inches="tight")
    print(f"Saved to {savepath}")
    plt.show()"""

def bbb():
    fonte_iter = dict()
    ensemble_iter = dict()

    fonte_rank = dict()
    bug_rank = dict()
    ensemble_rank = dict()

    bug_no = 0
    bug_git = 0
    bug_base = 0
    bug_class = 0

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    for project in os.listdir(RESULT_DATA_DIR):
        if project == 'Closure-131b':
            continue

        iter_dir = os.path.join(RESULT_DATA_DIR, project, 'iteration')

        # Fonte iteration
        with open(os.path.join(iter_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)

        for stage2, num_iter in fonte_dict.items():
            fonte_iter[stage2] = fonte_iter.get(stage2, 0) + num_iter / 129
        
        # Ensemble iteration
        with open(os.path.join(iter_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)
        
        for stage2, sub_dict in ensemble_dict.items():
            if stage2 not in ensemble_iter:
                ensemble_iter[stage2] = dict()

            for setting, num_iter in sub_dict.items():
                setting_dict = dict(setting)
                if not setting_dict['use_br'] and setting_dict['adddel'] == 'all-uni' and setting_dict['diff_type'] == 'gumtree_class':
                    ensemble_iter[stage2][str(setting_dict)] = ensemble_iter[stage2].get(str(setting_dict), 0) + num_iter / 129
        
        # Vote
        [pid, vid] = project[:-1].split('-')
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        vote_dir = os.path.join(RESULT_DATA_DIR, project, 'vote')

        # Fonte voting
        with open(os.path.join(vote_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
        
        for stage2, fonte_df in fonte_dict.items():
            rank = fonte_df['rank'].get(BIC, None)
            if rank is not None:
                fonte_rank[stage2] = fonte_rank.get(stage2, 0) + 1 / (rank * 129)
            else:
                print('No BIC (Fonte) ', project)

        # Bug2Commit voting
        with open(os.path.join(vote_dir, 'bug2commit.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
        
        for stage2, sub_dict in bug_dict.items():
            if stage2 not in bug_rank:
                bug_rank[stage2] = dict()

            for setting, bug_df in sub_dict.items():
                setting_dict = dict(setting)
                setting_key = str(setting)

                if setting_dict['use_br']:
                    if setting_key not in bug_rank[stage2]:
                        bug_rank[stage2][setting_key] = \
                            {'MRR' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0}

                    rank = bug_df['rank'].get(BIC, None)
                    if rank is not None:
                        bug_rank[stage2][setting_key]['MRR'] +=  1 / (rank * 129)
                        bug_rank[stage2][setting_key]['acc@1'] += 1 if rank <= 1 else 0
                        bug_rank[stage2][setting_key]['acc@2'] += 1 if rank <= 2 else 0
                        bug_rank[stage2][setting_key]['acc@3'] += 1 if rank <= 3 else 0
                        bug_rank[stage2][setting_key]['acc@5'] += 1 if rank <= 5 else 0
                        bug_rank[stage2][setting_key]['acc@10'] += 1 if rank <= 10 else 0
                    else:
                        print('No BIC (Bug2Commit) ', project)
        
        # Ensemble voting
        with open(os.path.join(vote_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)

        for stage2, sub_dict in ensemble_dict.items():
            if stage2 not in ensemble_rank:
                ensemble_rank[stage2] = dict()

            for setting, ensemble_df in sub_dict.items():
                setting_dict = dict(setting)
                setting_key = str(setting)

                if not setting_dict['use_br'] and setting_dict['adddel'] == 'all-uni' and setting_dict['diff_type'] == 'gumtree_class':
                    if setting_key not in ensemble_rank[stage2]:
                        ensemble_rank[stage2][setting_key] = \
                            {'MRR' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0}

                    rank = ensemble_df['rank'].get(BIC, None)
                    if rank is not None:
                        ensemble_rank[stage2][setting_key]['MRR'] += 1 / (rank * 129)
                        ensemble_rank[stage2][setting_key]['acc@1'] += 1 if rank <= 1 else 0
                        ensemble_rank[stage2][setting_key]['acc@2'] += 1 if rank <= 2 else 0
                        ensemble_rank[stage2][setting_key]['acc@3'] += 1 if rank <= 3 else 0
                        ensemble_rank[stage2][setting_key]['acc@5'] += 1 if rank <= 5 else 0
                        ensemble_rank[stage2][setting_key]['acc@10'] += 1 if rank <= 10 else 0
                    else:
                        print('No BIC (Ensemble) ', project)
    
    #print(json.dumps(fonte_iter, sort_keys=True, indent=4))
    print(json.dumps(ensemble_iter, sort_keys=True, indent=4))

    #print(json.dumps(fonte_rank, sort_keys=True, indent=4))
    #print(json.dumps(bug_rank['skip'], indent=4))
    print(json.dumps(ensemble_rank, sort_keys=True, indent=4))

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    
    bbb()