import sys
import pandas as pd
from scipy.stats import wilcoxon
from collections import Counter

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analysis/lib/')
from result_gen import *
from analysis import *
from compare import *

# Check the token share ratio of files
def use_file():
    share_ratio_dict = token_share_ratio(stage2='precise', \
        feature_setting={'tracker': 'git', 'diff_tool' : 'file', 'diff_type' : 'base'})
    res_dict = dict()

    # Aggregate token share ratio data
    for project, bug_type_dict in share_ratio_dict.items():
        for bug_type, commit_type_dict in bug_type_dict.items():
            res_dict.setdefault(bug_type, {'MRR' : 0, 'BIC_ratio' : list(), 'avg_ratio' : list()})
                
            res_dict[bug_type]['MRR'] += 1 / (commit_type_dict['diff']['non_id']['rank'] * len(share_ratio_dict))
            res_dict[bug_type]['BIC_ratio'].append(commit_type_dict['diff']['non_id']['BIC_ratio'])
            res_dict[bug_type]['avg_ratio'].append(commit_type_dict['diff']['non_id']['avg_ratio'])
    
    # Print metric for each bug types
    for bug_type, metric_dict in res_dict.items():
        print(f'Bug) {bug_type}')
        print(f"MRR : {metric_dict['MRR']}")

        rel_BIC_ratio = sum(BIC_ratio / avg_ratio if avg_ratio > 0 else 1 \
            for BIC_ratio, avg_ratio in zip(metric_dict['BIC_ratio'], metric_dict['avg_ratio'])) \
            / len(share_ratio_dict)

        print(f"BIC_ratio : {sum(metric_dict['BIC_ratio']) / len(metric_dict['BIC_ratio'])}")
        print(f"Avg_ratio : {sum(metric_dict['avg_ratio']) / len(metric_dict['avg_ratio'])}")
        print(f'Relative BIC Ratio : {rel_BIC_ratio}')

        _, p = wilcoxon(metric_dict['BIC_ratio'], metric_dict['avg_ratio'], alternative='greater')
        print(f'WSR : {p}')

# Compare token share ratio of diff tools
def precise_diff():
    res_dict = dict()

    # Aggregate token share ratio data
    for diff_tool in ['file', 'base', 'gumtree']:
        res_dict[diff_tool] = dict()
        share_ratio_dict = token_share_ratio(stage2='precise', \
            feature_setting=feature_setting = {'tracker': 'git', 'diff_tool' : diff_tool, 'diff_type' : 'base'})

        for project, bug_type_dict in share_ratio_dict.items():
            for bug_type, commit_type_dict in bug_type_dict.items():
                res_dict[diff_tool].setdefault(bug_type, {'rank' : list(), 'rel_ratio' : list()})
                    
                res_dict[diff_tool][bug_type]['rank'].append(1 / commit_type_dict['diff']['non_id']['rank'])
                res_dict[diff_tool][bug_type]['rel_ratio'].append(\
                    commit_type_dict['diff']['non_id']['BIC_ratio'] / commit_type_dict['diff']['non_id']['avg_ratio'] \
                    if commit_type_dict['diff']['non_id']['avg_ratio'] > 0 else 1)
    
    # Compare diffencing tools
    for (org, new) in [('file', 'base'), ('base', 'gumtree')]:
        print(f'Comparing {org} and {new}')
        org_dict, new_dict = res_dict[org], res_dict[new]
        
        for bug_type in org_dict.keys():
            print(f'Bug) {bug_type}')

            org_metric_dict, new_metric_dict = org_dict[bug_type], new_dict[bug_type]

            print(f"Org_MRR : {sum(org_metric_dict['rank']) / len(org_metric_dict['rank'])}, New_MRR : {sum(new_metric_dict['rank']) / len(new_metric_dict['rank'])}")
            print(f"Org_rel_ratio : {sum(org_metric_dict['rel_ratio']) / len(org_metric_dict['rel_ratio'])}, New_rel_ratio : {sum(new_metric_dict['rel_ratio']) / len(new_metric_dict['rel_ratio'])}")

            _, ratio_p = wilcoxon(new_metric_dict['rel_ratio'], org_metric_dict['rel_ratio'], alternative='greater')
            _, rank_p = wilcoxon(new_metric_dict['rank'], org_metric_dict['rank'], alternative='greater')
            print(f'Ratio WSR: {ratio_p}, Rank WSR : {rank_p}')

# Compare token share ratio change when using full identifiers
def use_id():
    share_ratio_dict = token_share_ratio(stage2='precise', \
        feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'classify_id' : False})
    res_dict = dict()

    # Aggregate token share ratio data
    for project, bug_type_dict in share_ratio_dict.items():
        for bug_type, commit_type_dict in bug_type_dict.items():
            res_dict.setdefault(bug_type, dict())

            for commit_type, metric_dict in commit_type_dict.items():
                res_dict[bug_type].setdefault(commit_type, dict())

                for id_type in ['non_id', 'all']:
                    res_dict[bug_type][commit_type].setdefault(id_type, {'rank' : list(), 'rel_ratio' : list()})
                
                    res_dict[bug_type][commit_type][id_type]['rank'].append(1 / metric_dict[id_type]['rank'])
                    res_dict[bug_type][commit_type][id_type]['rel_ratio'].append(\
                        metric_dict[id_type]['BIC_ratio'] / metric_dict[id_type]['avg_ratio'] \
                        if metric_dict[id_type]['avg_ratio'] > 0 else 1)
    
    # Print metric for each bug types
    for bug_type, commit_type_dict in res_dict.items():
        for commit_type, id_type_dict in commit_type_dict.items():
            print(f'Bug) {bug_type}, Commit) {commit_type}')
            print(f"Org_MRR : {sum(id_type_dict['non_id']['rank']) / len(id_type_dict['non_id']['rank'])}, New_MRR : {sum(id_type_dict['all']['rank']) / len(id_type_dict['all']['rank'])}")
            print(f"Org_rel_ratio : {sum(id_type_dict['non_id']['rel_ratio']) / len(id_type_dict['non_id']['rel_ratio'])}, New_rel_ratio : {sum(id_type_dict['all']['rel_ratio']) / len(id_type_dict['all']['rel_ratio'])}")

            _, ratio_p = wilcoxon(id_type_dict['all']['rel_ratio'], id_type_dict['non_id']['rel_ratio'], alternative='greater')
            _, rank_p = wilcoxon(id_type_dict['all']['rank'], id_type_dict['non_id']['rank'], alternative='greater')
            print(f'Better ratio WSR: {ratio_p}, Better rank WSR : {rank_p}')

            _, ratio_p = wilcoxon(id_type_dict['all']['rel_ratio'], id_type_dict['non_id']['rel_ratio'], alternative='less')
            _, rank_p = wilcoxon(id_type_dict['all']['rank'], id_type_dict['non_id']['rank'], alternative='less')
            print(f'Worse ratio WSR: {ratio_p}, Worse rank WSR : {rank_p}')

# 
def use_id_proj(pid='Jsoup', vid='41'):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data (Feature, encoder, bug_feature)
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'encoder.pkl'), 'rb') as file:
        encoder_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'bug_feature.pkl'), 'rb') as file:
        bug_feature_dict = pickle.load(file)

    setting = frozenset({'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'classify_id' : False}.items())

    feature_dict = feature_dict['precise'][setting]
    encoder = encoder_dict['precise'][setting]
    bug_feature_dict = bug_feature_dict['precise'][setting]

    # Get vocabulary from encoder
    vocab = encoder.id_vocab.copy()
    vocab.update(encoder.non_id_vocab)
    vocab = {ind : word for word, ind in vocab.items()}

    # Get the set of commit types
    commit_type_set = next(iter(feature_dict.values())).keys()
    print(f'Num commits) {len(feature_dict)}')

    for bug_type, bug_feature in bug_feature_dict.items():
        #if bug_type.startswith('br'):
        #    continue

        bug_vec = bug_feature['id']

        for commit_type in commit_type_set:
            print(f'\nBug) {bug_type}, Commit) {commit_type}')

            for ind, freq in bug_vec.most_common():
                num_commit, sum_freq = 0, 0
                
                for commit_feature in feature_dict.values():
                    if commit_feature[commit_type]['id'][ind] > 0:
                        num_commit += 1
                        sum_freq += commit_feature[commit_type]['id'][ind]
            
                if num_commit > 0:
                    BIC_freq, avg_freq = feature_dict[BIC][commit_type]['id'][ind], sum_freq / num_commit

                    if BIC_freq > avg_freq:
                        #print(vocab)
                        #print(f'ID) {vocab[ind]}')

                        _, non_id_vec = encoder.encode([vocab[ind]], update_vocab=False, mode='text')
                        print(non_id_vec)
                        print(''.join([f'{vocab[non_id_ind]} : {non_id_vec[non_id_ind]}' for non_id_ind in non_id_vec]))
                        print(f'Bug freq) {freq}, Num_commit) {num_commit}, BIC_freq) {BIC_freq}, Mean freq) {avg_freq}')

# Print distribution of identifiers & token share ratio
def classify_id():
    feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'classify_id' : True}
    
    # Count distribution of identifiers
    id_dist_dict = {'class' : 0, 'method' : 0, 'variable' : 0}

    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit
        with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)

        # Count tokens of identifiers
        id_cnt_dict = {'class' : 0, 'method' : 0, 'variable' : 0}

        for commit, feature_type_dict in feature_dict['precise'][frozenset(feature_setting.items())].items():
            for id_type in id_cnt_dict.keys():
                feature = feature_type_dict[id_type]
                id_cnt_dict[id_type] += sum((feature['id'] + feature['non_id']).values())

        total_id_token = sum(id_cnt_dict.values())
        for id_type in id_dist_dict.keys():
            id_dist_dict[id_type] += id_cnt_dict[id_type] / total_id_token if total_id_token > 0 else 1 / 3
    
    print('Identifer token proportion')
    for id_type, id_dist in id_dist_dict.items():
        print(f'{id_type}) {id_dist / len(GT)}')
    
    # Aggregate token share ratio data
    res_dict = dict()
    share_ratio_dict = token_share_ratio(classify_id=False, stage2='precise', feature_setting=feature_setting)
    
    for project, bug_type_dict in share_ratio_dict.items():
        for bug_type, commit_type_dict in bug_type_dict.items():
            res_dict.setdefault(bug_type, dict(){'MRR' : 0, 'BIC_ratio' : list(), 'avg_ratio' : list()})

            for commit_type in ['class', 'method', 'variable']:
                metric_dict = commit_type_dict[commit_type]
                    
                res_dict[bug_type]['MRR'] += 1 / (commit_type_dict['diff']['all']['rank'] * len(share_ratio_dict))
                res_dict[bug_type]['BIC_ratio'].append(commit_type_dict['diff']['all']['BIC_ratio'])
                res_dict[bug_type]['avg_ratio'].append(commit_type_dict['diff']['all']['avg_ratio'])
    
    # Print metric for each bug types
    for bug_type, metric_dict in res_dict.items():
        print(f'Bug) {bug_type}')
        print(f"MRR : {metric_dict['MRR']}")

        rel_BIC_ratio = sum(BIC_ratio / avg_ratio if avg_ratio > 0 else 1 \
            for BIC_ratio, avg_ratio in zip(metric_dict['BIC_ratio'], metric_dict['avg_ratio'])) \
            / len(share_ratio_dict)

        print(f"BIC_ratio : {sum(metric_dict['BIC_ratio']) / len(metric_dict['BIC_ratio'])}")
        print(f"Avg_ratio : {sum(metric_dict['avg_ratio']) / len(metric_dict['avg_ratio'])}")
        print(f'Relative BIC Ratio : {rel_BIC_ratio}')

        _, p = wilcoxon(metric_dict['BIC_ratio'], metric_dict['avg_ratio'], alternative='greater')
        print(f'WSR : {p}')

if __name__ == "__main__":
    #use_file()
    #precise_diff()
    use_id_proj(pid='Jsoup', vid='41')