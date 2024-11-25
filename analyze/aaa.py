import pandas as pd

# Load significant post-hoc results
data = pd.read_csv("/root/workspace/analyze/data/bug2commit/posthoc_results.csv")

# Extract all unique settings
all_settings = set(data["setting_1"]).union(data["setting_2"])

# Initialize the candidate set with all settings
candidate_set = set(all_settings)

# Iterate through all settings
for setting in all_settings:
    
    # Setting must not have large value for any other settings
    larger_cases = data[
        (data["Winner"] != setting) & (data["Loser"] == setting) & (data["estimate"] > 0) & (data["p.value"] < 0.05)
    ]
    if not larger_cases.empty:
        candidate_set.discard(setting)
        continue

    # Condition 2: Exclude if it doesn't show smaller values than all outside settings
    smaller_cases = data[
        (data["Winner"] == setting) & (data["Loser"] != setting) & (data["estimate"] < 0) & (data["p.value"] < 0.05)
    ]
    outside_settings = all_settings - candidate_set
    if not smaller_cases["Loser"].isin(outside_settings).all():
        candidate_set.discard(setting)

for setting in candidate_set:
    smaller_cases = data[
        (data["Winner"] == setting) & (data["Loser"] != setting) & (data["estimate"] < 0) & (data["p.value"] < 0.05)
    ]
    outside_settings = all_settings - candidate_set
    if not smaller_cases["Loser"].isin(outside_settings).all():
        print(setting)

# Print the final set
print("Settings in the statistically meaningful set:", candidate_set)

# Save filtered results
relevant_pairs = data[
    (data["Winner"].isin(candidate_set) & ~data["Loser"].isin(candidate_set)) |
    (~data["Winner"].isin(candidate_set) & data["Loser"].isin(candidate_set))
]
relevant_pairs.to_csv("filtered_all_pairs.csv", index=False)

# Define a function to find the optimal set of settings
def find_optimal_set(posthoc_results):
    # Filter for statistically significant comparisons (p-value < 0.05)
    significant_results = posthoc_results[posthoc_results['p.value'] < 0.05]
    
    # Parse the winners and losers from the significant results
    winners = significant_results['Winner'].tolist()
    losers = significant_results['Loser'].tolist()
    
    # Create a dictionary of comparisons
    comparisons = {}
    for _, row in significant_results.iterrows():
        winner, loser = row['Winner'], row['Loser']
        comparisons.setdefault(winner, {'wins_against': set(), 'loses_against': set()})
        comparisons.setdefault(loser, {'wins_against': set(), 'loses_against': set()})
        comparisons[winner]['wins_against'].add(loser)
        comparisons[loser]['loses_against'].add(winner)
    
    # Start with an empty set and progressively build it
    all_settings = set(winners + losers)
    current_set = set()
    
    for candidate in all_settings:
        # Check if adding the candidate satisfies both conditions
        can_add = True
        for setting in all_settings - {candidate}:
            if setting in current_set:
                # Condition: Candidate must not lose to any setting in the current set
                if candidate in comparisons.get(setting, {}).get('wins_against', set()):
                    can_add = False
                    break
            else:
                # Condition: Candidate must beat all settings outside the current set
                if setting not in comparisons.get(candidate, {}).get('wins_against', set()):
                    can_add = False
                    break
        
        # Add to the set if conditions are met
        if can_add:
            current_set.add(candidate)
    
    return current_set

# Apply the function
optimal_set = find_optimal_set(data)
print("Optimal set of settings:", optimal_set)