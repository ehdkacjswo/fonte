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
    share_ratio_dict = token_share_ratio(stage2='skip', \
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
        share_ratio_dict = token_share_ratio(stage2='skip', \
            feature_setting={'tracker': 'git', 'diff_tool' : diff_tool, 'diff_type' : 'base'})

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

# Check token share ratio change by ignoring style change
def style_change():
    res_dict = dict()

    # Aggregate token share ratio data
    for stage2 in ['skip', 'precise']:
        res_dict[stage2] = dict()
        share_ratio_dict = token_share_ratio(stage2=stage2, \
            feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'base'})

        for project, bug_type_dict in share_ratio_dict.items():
            for bug_type, commit_type_dict in bug_type_dict.items():
                res_dict[stage2].setdefault(bug_type, {'rank' : list(), 'rel_ratio' : list()})
                    
                res_dict[stage2][bug_type]['rank'].append(1 / commit_type_dict['diff']['non_id']['rank'])
                res_dict[stage2][bug_type]['rel_ratio'].append(\
                    commit_type_dict['diff']['non_id']['BIC_ratio'] / commit_type_dict['diff']['non_id']['avg_ratio'] \
                    if commit_type_dict['diff']['non_id']['avg_ratio'] > 0 else 1)
    
    # Compare stage2
    org_dict, new_dict = res_dict['skip'], res_dict['precise']
    
    for bug_type in org_dict.keys():
        print(f'Bug) {bug_type}')

        org_metric_dict, new_metric_dict = org_dict[bug_type], new_dict[bug_type]

        print(f"Org_MRR : {sum(org_metric_dict['rank']) / len(org_metric_dict['rank'])}, New_MRR : {sum(new_metric_dict['rank']) / len(new_metric_dict['rank'])}")
        print(f"Org_rel_ratio : {sum(org_metric_dict['rel_ratio']) / len(org_metric_dict['rel_ratio'])}, New_rel_ratio : {sum(new_metric_dict['rel_ratio']) / len(new_metric_dict['rel_ratio'])}")

        #_, ratio_p = wilcoxon(new_metric_dict['rel_ratio'], org_metric_dict['rel_ratio'], alternative='greater')
        #_, rank_p = wilcoxon(new_metric_dict['rank'], org_metric_dict['rank'], alternative='greater')
        #print(f'Ratio WSR: {ratio_p}, Rank WSR : {rank_p}')

        _, ratio_p = wilcoxon(new_metric_dict['rel_ratio'], org_metric_dict['rel_ratio'], alternative='greater')
        _, rank_p = wilcoxon(new_metric_dict['rank'], org_metric_dict['rank'], alternative='greater')
        print(f'Better ratio WSR: {ratio_p}, Better rank WSR : {rank_p}')

        _, ratio_p = wilcoxon(new_metric_dict['rel_ratio'], org_metric_dict['rel_ratio'], alternative='less')
        _, rank_p = wilcoxon(new_metric_dict['rank'], org_metric_dict['rank'], alternative='less')
        print(f'Worse ratio WSR: {ratio_p}, Worse rank WSR : {rank_p}')

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
def use_id_proj(pid='Jsoup', vid='15'):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data (Feature, encoder, bug_feature)
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'encoder.pkl'), 'rb') as file:
        encoder_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'bug_feature.pkl'), 'rb') as file:
        bug_feature_dict = pickle.load(file)
    
    share_ratio_dict = token_share_ratio_proj(pid=pid, vid=vid, stage2='precise', feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'classify_id' : False}, BIC=BIC)

    setting = frozenset({'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'classify_id' : False}.items())

    feature_dict = feature_dict['precise'][setting]
    encoder = encoder_dict['precise'][setting]
    bug_feature_dict = bug_feature_dict['precise'][setting]

    # Get vocabulary from encoder
    vocab = encoder.id_vocab | encoder.non_id_vocab
    vocab = {ind : word for word, ind in vocab.items()}

    # Get the set of commit types
    print(f'Num commits) {len(feature_dict)}')

    # Get data for every token (Number of commits containing token, Frequency on BIC, Average frequency)
    commit_token_data = dict()

    for commit, commit_type_dict in feature_dict.items():
        for commit_type, commit_feature in commit_type_dict.items():
            commit_token_data.setdefault(commit_type, dict())

            for token_id, freq in (commit_feature['id'] + commit_feature['non_id']).items():
                commit_token_data[commit_type].setdefault(token_id, {'num_commits' : 0, 'BIC_freq' : 0, 'avg_freq' : 0})
                
                commit_token_data[commit_type][token_id]['num_commits'] += 1
                commit_token_data[commit_type][token_id]['avg_freq'] += freq

                if commit == BIC:
                    commit_token_data[commit_type][token_id]['BIC_freq'] = freq
        
    for commit_type_data in commit_token_data.values():
        for data_dict in commit_type_data.values():
            data_dict['avg_freq'] /= data_dict['num_commits']

    # Check shared identifier tokens with bug features
    for commit_type, commit_type_data in commit_token_data.items():
        for bug_type, bug_feature in bug_feature_dict.items():
            print(f'\nCommit) {commit_type}, Bug) {bug_type} ({sum(bug_feature["id"].values()) + sum(bug_feature["non_id"].values())} tokens)')

            # Share ratio data
            share_ratio_data = share_ratio_dict[bug_type][commit_type]
            print(f'BIC Rank) {share_ratio_data["non_id"]["rank"]} -> {share_ratio_data["all"]["rank"]}')
            print(f'Relative ratio) {share_ratio_data["non_id"]["BIC_ratio"] / share_ratio_data["non_id"]["avg_ratio"] if share_ratio_data["non_id"]["avg_ratio"] > 0 else 1:.3f} -> {share_ratio_data["all"]["BIC_ratio"]/ share_ratio_data["all"]["avg_ratio"] if share_ratio_data["all"]["avg_ratio"] > 0 else 1:.3f}')

            # 
            for ind, freq in bug_feature['id'].most_common():
                if ind in commit_type_data: #and commit_type_data[ind]['BIC_freq'] > commit_type_data[ind]['avg_freq']: # BIC contains more token than average
                    
                    # Ignore tokens that are tokenized as the same...?
                    _, non_id_vec = encoder.encode([vocab[ind]], update_vocab=False, mode='text')
                    if non_id_vec[ind] > 0:
                        continue
                        
                    print(f'\nToken) {vocab[ind]}')
                    for sub_token_id, freq in non_id_vec.items():
                        print(f'[{vocab[sub_token_id]}] Bug freq) {(bug_feature["id"][sub_token_id] + bug_feature["non_id"][sub_token_id])}, Num_commits) {commit_type_data[sub_token_id]["num_commits"]}, BIC_freq) {commit_type_data[sub_token_id]["BIC_freq"]}, Mean_freq) {commit_type_data[sub_token_id]["avg_freq"]:.3f}')
                    print(f'Bug freq) {freq + bug_feature["non_id"][ind]}, Num_commit) {commit_type_data[ind]["num_commits"]}, BIC_freq) {commit_type_data[ind]["BIC_freq"]}, Mean freq) {commit_type_data[ind]["avg_freq"]:.3f}')

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
        print(f'{id_type}) {id_dist / len(GT):.3f}')
    
    # Aggregate token share ratio data
    res_dict = dict()
    share_ratio_dict = token_share_ratio(stage2='precise', feature_setting=feature_setting)
    
    for project, bug_type_dict in share_ratio_dict.items():
        for bug_type, commit_type_dict in bug_type_dict.items():
            res_dict.setdefault(bug_type, dict())

            for commit_type in ['class', 'method', 'variable']:
                res_dict[bug_type].setdefault(commit_type, {'rank' : list(), 'rel_ratio' : list()})

                res_dict[bug_type][commit_type]['rank'].append(1 / commit_type_dict[commit_type]['all']['rank'])
                res_dict[bug_type][commit_type]['rel_ratio'].append(\
                    commit_type_dict[commit_type]['all']['BIC_ratio'] / commit_type_dict[commit_type]['all']['avg_ratio'] \
                    if commit_type_dict[commit_type]['all']['avg_ratio'] > 0 else 1)
    
    # Print metric for each bug types
    for bug_type, commit_type_dict in res_dict.items():
        print(f'\nBug) {bug_type}')

        for commit_type, metric_dict in commit_type_dict.items():
            print(f'\nCommit) {commit_type}')
            print(f"MRR : {sum(metric_dict['rank']) / len(metric_dict['rank']):.3f}")
            print(f"Rel_ratio : {sum(metric_dict['rel_ratio']) / len(metric_dict['rel_ratio']):.3f}")
        
        print()
        for metric_type in ['rank', 'rel_ratio']:
            try:
                _, kruskal_p = kruskal(commit_type_dict['class'][metric_type], \
                    commit_type_dict['method'][metric_type], commit_type_dict['variable'][metric_type])
                print(f'[{metric_type}] Kruskal P-value) {kruskal_p:.3f}')
            except:
                print(f'Identical {metric}?')
        


if __name__ == "__main__":
    #use_file()
    #precise_diff()
    #style_change()
    #use_id()
    use_id_proj(pid='Csv', vid='12')
    #classify_id()