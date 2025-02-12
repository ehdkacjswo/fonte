import argparse, os, copy, itertools, pickle, sys, json
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analyze/lib')
from result_gen import get_metric_dict

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result/'

def metric_converter(metric_list):
    res_dict = {metric_key : [] for metric_key in metric_list[0].keys()}

    for sub_dict in metric_list:
        for metric_key, metric_value in sub_dict.items():
            res_dict[metric_key].append(metric_value)
    
    return res_dict

def compare_settings(org_method, new_method, org_setting, new_setting):
    org_metric_dict = get_metric_dict(method=org_method, mode='project')
    org_metric = metric_converter(list(org_metric_dict[org_setting].values()))

    new_metric_dict = get_metric_dict(method=new_method, mode='project')
    new_metric = metric_converter(list(new_metric_dict[new_setting].values()))

    for metric_key in org_metric.keys():
        print(f'Metric) {metric_key}')

        org_list = org_metric[metric_key]
        new_list = new_metric[metric_key]

        cost_saving = [a - b for a, b in zip(org_list, new_list)]

        reduced = sum(x for x in cost_saving if x > 0)
        increased = sum(-x for x in cost_saving if x < 0)
        same = 0


        plt.figure(figsize=(9, 2))
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

        #plt.yticks(range(min(cost_saving), max(cost_saving)+1))

        #plt.axvspan(-0.5, N * reduced - 0.5, facecolor='green', alpha=0.1)
        #plt.axvspan(N * (reduced + same)-0.5, N-0.5, facecolor='red', alpha=0.1)

        #if reduced > 0.05:
        #    plt.text(N * reduced/2 - 0.5, max(cost_saving)-1, f"{reduced*100:.1f}%", horizontalalignment="center")
        #if same > 0.05:
        #    plt.text(N * (reduced + same/2) - 0.5, max(cost_saving)-1, f"{same*100:.1f}%", horizontalalignment="center")
        #if increased > 0.05:
        #    plt.text(N * (reduced + same + increased/2) - 0.5, max(cost_saving)-1, f"{increased*100:.1f}%", horizontalalignment="center")

        plt.xlim((0-0.5, N-0.5))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)

        plt.axhline(np.mean(cost_saving), color="black", linestyle="--", label=f"Average Saved Iterations: {np.mean(cost_saving).round(1)}")
        print("Average # Saved Iterations", np.mean(cost_saving))
        plt.legend(loc="upper right")

        savepath = os.path.join('/root/workspace', f"{metric_key}.pdf")
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Saved to {savepath}")
        plt.show()

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    #org_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'no_diff', 'adddel':'del'}.items())
    #new_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'gumtree_class', 'adddel':'all-uni'}.items())

    #compare_settings(org_method='bug2commit', new_method='bug2commit', org_setting=org_setting, new_setting=new_setting)

    org_setting = frozenset({'stage2':'precise'}.items())
    new_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'gumtree_class', 'adddel':'all-uni', 'beta':1.0}.items())

    compare_settings(org_method='fonte', new_method='ensemble', org_setting=org_setting, new_setting=new_setting)