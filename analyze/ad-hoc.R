library(emmeans)
library(ARTool)
library(purrr)
library(dplyr)
library(tidyr)

# Load your data
data <- read.csv("/root/workspace/analyze/data/all_metrics.csv", stringsAsFactors = FALSE)

data <- data %>%
  mutate(
    project = as.factor(project),
    score_mode = as.factor(score_mode),
    use_diff = as.factor(use_diff),
    stage2 = as.factor(stage2),
    use_stopword = as.factor(use_stopword),
    adddel = as.factor(adddel),
    DependentName = as.factor(DependentName),
    DependentValue = as.numeric(DependentValue)
  )

# List of dependent variables to analyze
dependent_vars <- c("rank", "num_iters")

results <- list()

for (dep_var in dependent_vars) {
  subset_data <- data %>% filter(DependentName == dep_var)
  
  # Fit ART model
  art_model <- art(DependentValue ~ stage2 + (1|project), data = subset_data)
  
  # Perform post-hoc test for interaction of score_mode and use_diff
  posthoc_interaction <- art.con(art_model, "stage2")
  
  # Save results
  results[[dep_var]] <- summary(posthoc_interaction)
}

# Print results for each dependent variable
results