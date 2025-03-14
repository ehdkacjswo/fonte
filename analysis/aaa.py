# Compare the extra scores based on metrics
# Metrics : acc(1,2,3,5,10), MRR, Mean number of iterations (score rank)
def compare_bug2commit_simple(fixed=False, use_br=True):
    tot_metric_dict = get_metric_dict(bug2commit=True)
    
    # Possible options for each parameters
    option_dict = {
        'score_mode' : ['score', 'rank', 'both'],
        'use_br' : ['True'] if use_br else ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['True'] if fixed else ['skip', 'True', 'False'],
        'use_stopword' : ['True'] if fixed else ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
    }

    setting_list = list(itertools.product(option_dict['score_mode'], option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        if param == 'use_br':
            continue
        
        if fixed and (param == 'stage2' or param == 'use_stopword'):
            continue

        print(f'Working on {param}')
        
        metric_dict = {'rank' : dict(), 'num_iters' : dict()}
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    for metric, val in tot_metric_dict[setting].items(): # Gather the metrics
                        metric_dict[metric][option] = metric_dict[metric].get(option, []) + [val]
                    break

        for metric, val_dict in metric_dict.items():
            data = [val_dict[option] for option in param_option_list]

            plt.figure()
            plt.boxplot(data, tick_labels=param_option_list)
            plt.title(f"{param}-{metric}")
            plt.xlabel("Options")
            plt.grid(True)
            plt.savefig(f"/root/workspace/analyze/plot/bug2commit/{'fixed_' if fixed else ''}{'' if use_br else 'no_br_'}{param}_{metric}.png")
            plt.close()  # Close the figure to free memory

def compare_bug2commit_complex(use_br=False):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    tot_metric_dict = dict()
    
    # Iterate through every projects
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        [pid, vid] = folder[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        
        # Bug2Commit score and number of iterations
        scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/scores.hdf'))
        with open(os.path.join(DIFF_DATA_DIR, f'{folder}/num_iters.pkl'), 'rb') as file:
            num_iters_dict = pickle.load(file)

        # Iterate through extra scores of every settings (use_br, use_diff, stage2, use_stopword, adddel)
        for setting, row in scores_df.iterrows():
            if not use_br and setting[0] == 'True':
                continue

            commit_df = row['commit_hash'].dropna()
            rank_df = row['rank'].dropna()

            # Index of the BIC
            BIC_ind = commit_df.loc[commit_df == BIC].index[0]
            setting_tup = tuple(setting)

            if setting_tup not in tot_metric_dict:
                tot_metric_dict[setting_tup] = dict()
            
            setting_metric_dict = tot_metric_dict[setting_tup]

            # Acc data
            BIC_rank = rank_df.loc[BIC_ind]
            for n in [1, 2, 3, 5, 10]:
                acc_str = f'acc@{n}'
                setting_metric_dict[acc_str] = setting_metric_dict.get(acc_str, 0) + (1 if BIC_rank <= n else 0)

            # MRR data
            # 130 is number of projects
            setting_metric_dict['MRR'] = setting_metric_dict.get('MRR', 0) + 1 / (BIC_rank * 130)

            # Mean of number of iterations
            # Using Bug2Commit alone could possibly cause BIC score to be 0, so use only score mode 'rank'
            setting_metric_dict['num_iters'] = setting_metric_dict.get('num_iters', 0) + \
                num_iters_dict[('None', 'rank', '(\'add\', 0.0)') + setting_tup][0] / 130

    # Possible options for each parameters
    option_dict = {
        'use_br' : ['True', 'False'] if use_br else ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
        #'adddel' : ['all-sep']
    }

    setting_list = list(itertools.product(option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        if not use_br and param == 'use_br':
            continue
        print(f'Working on {param}')
        
        option_metric_dict = dict()
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    if option not in option_metric_dict:
                        option_metric_dict[option] = dict()

                    for metric, val in tot_metric_dict[setting].items(): # Gather the metrics
                        option_metric_dict[option][metric] = \
                            option_metric_dict[option].get(metric, []) + [val]
                    break
        
        # Compare two different
        for ind1 in range(len(param_option_list)):
            for ind2 in range(ind1 + 1, len(param_option_list)):
                option1 = param_option_list[ind1]
                option2 = param_option_list[ind2]

                print(f'Comparing {option1} and {option2}')

                for metric in option_metric_dict[option1].keys():
                    val1 = mean(option_metric_dict[option1][metric])
                    val2 = mean(option_metric_dict[option2][metric])

                    try:
                        pvalue = wilcoxon(option_metric_dict[option1][metric], option_metric_dict[option2][metric]).pvalue
                    except:
                        pvalue = 'Identical'

                    print(f'{metric} for {option1} : {val1}')
                    print(f'{metric} for {option2} : {val2}')
                    print(f'P-value : {pvalue}')

        # Dictionary for drawing box plot
        metric_plot_dict = dict()

        for option in param_option_list:
            metric_dict = option_metric_dict[option]

            for metric, val in metric_dict.items():
                if metric not in metric_plot_dict:
                    metric_plot_dict[metric] = []
                
                metric_plot_dict[metric].append(val)

        for metric, val in metric_plot_dict.items():
            pvalue = friedmanchisquare(*val).pvalue if len(param_option_list) > 2 else wilcoxon(*val).pvalue
            plt.figure()
            plt.boxplot(val, tick_labels=param_option_list)
            plt.title(f"{param}-{metric}-{pvalue}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.grid(True)
            plt.show()
            plt.close()  # Close the figure to free memory

def compare_all_simple():
    if os.path.isfile('/root/workspace/analyze/data/fonte_metrics.pkl'):
        with open('/root/workspace/analyze/data/fonte_metrics.pkl', 'rb') as file:
            metric_dict = pickle.load(file)

    else:
        GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
        metric_dict = dict()

        for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
            [pid, vid] = folder[:-1].split("-")
            BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
            
            fonte_scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/fonte_scores.hdf'))

            # Iterate through extra scores of every settings
            for setting, row in fonte_scores_df.iterrows():
                commit_df = row['commit'].dropna()
                score_df = row['vote'].dropna()
                rank_df = score_df.rank(method='max', ascending=False)

                # Index of the BIC
                BIC_ind = commit_df.loc[commit_df == BIC].index[0]
                setting_tup = tuple(setting)

                if setting_tup not in metric_dict:
                    metric_dict[setting_tup] = dict()
                
                setting_metric_dict = metric_dict[setting_tup]

                # Acc data
                BIC_rank = rank_df.loc[BIC_ind]
                for n in [1, 2, 3, 5, 10]:
                    acc_str = f'acc@{n}'
                    setting_metric_dict[acc_str] = setting_metric_dict.get(acc_str, 0) + (1 if BIC_rank <= n else 0)

                # MRR data
                # 130 is number of projects
                BIC_rev_rank = 1 / BIC_rank
                setting_metric_dict['MRR'] = setting_metric_dict.get('MRR', 0) + BIC_rev_rank / 130

                # Score data
                BIC_score = score_df.loc[BIC_ind]
                score_mean = score_df.mean()
                
                setting_metric_dict['score_diff'] = \
                    setting_metric_dict.get('score_diff', []) + [BIC_score - score_mean]
                setting_metric_dict['score_ratio'] = \
                    setting_metric_dict.get('score_ratio', []) + [0 if score_mean == 0 else BIC_score / score_mean]
                
                # Rank data
                rev_rank_df = 1 / rank_df
                rev_rank_mean = rev_rank_df.mean()

                setting_metric_dict['rev_rank_diff'] = \
                    setting_metric_dict.get('rev_rank_diff', []) + [BIC_rev_rank - rev_rank_mean]
                setting_metric_dict['rev_rank_ratio'] = \
                    setting_metric_dict.get('rev_rank_ratio', []) + [BIC_rev_rank / rev_rank_mean]

                # Both data
                both_df = score_df / rank_df
                BIC_both = both_df.loc[BIC_ind]
                both_mean = both_df.mean()

                setting_metric_dict['both_diff'] = \
                    setting_metric_dict.get('both_diff', []) + [BIC_both - both_mean]
                setting_metric_dict['both_ratio'] = \
                    setting_metric_dict.get('both_ratio', []) + [0 if both_mean == 0 else BIC_both / both_mean]
        
        with open('/root/workspace/analyze/data/fonte_metrics.pkl', 'wb') as file:
            pickle.dump(metric_dict, file)

    # Possible options for each parameters
    """option_dict = {
        #'use_br' : ['True', 'False'],
        'use_br' : ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
        #'adddel' : ['all-sep']
    }

    setting_list = list(itertools.product(option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(param_list = ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        print(f'Working on {param}')
        
        option_metric_dict = dict()
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    if setting not in option_metric_dict:
                        option_metric_dict[option] = dict()

                    for metric, val in metric_dict[setting].items(): # Gather the metrics
                        option_metric_dict[option][metric] = \
                            option_metric_dict[option].get(metric, []) + (val if type(val) is list else [val])
                    break
        
        for ind1 in range(len(param_option_list)):
            for ind2 in range(ind1 + 1, len(param_option_list)):
                option1 = param_option_list[ind1]
                option2 = param_option_list[ind2]

                print(f'Comparing {option1} and {option2}')

                for metric in option_metric_dict[option1].keys():
                    val1 = mean(option_metric_dict[option1][metric])
                    val2 = mean(option_metric_dict[option2][metric])

                    try:
                        pvalue = kruskal(option_metric_dict[option1][metric], option_metric_dict[option2][metric]).pvalue
                    except:
                        pvalue = 'Identical'

                    print(f'{metric} for {option1} : {val1}')
                    print(f'{metric} for {option2} : {val2}')
                    print(f'P-value : {pvalue}')"""