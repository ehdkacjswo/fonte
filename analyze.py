from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy

# Data of Fonte
data = pd.read_csv('./total_data.csv')

# Number of iterations
standard = data.loc[:, 'standard_iter'].values
weight = data.loc[:, 'weight_iter'].values

iter_diff = standard - weight # Bigger is better

# Number of commits and rank of BIC
num_commits = data.loc[:, 'num_commits'].values
rank_BIC = data.loc[:, 'rank_BIC'].values

rank_log = (numpy.log2(num_commits / rank_BIC)).reshape(-1, 1) # Bigger is better

num_commits = data.loc[:, 'top_score_sum'].values
rank_BIC = data.loc[:, 'BIC_score_sum'].values

rank_log = (rank_BIC / num_commits).reshape(-1, 1) # Bigger is better

model = LinearRegression()
model.fit(X=rank_log, y=iter_diff)

pred = model.predict(rank_log)

plt.scatter(rank_log, iter_diff)
plt.plot(rank_log, pred, color='green')
plt.savefig('foo1.png')