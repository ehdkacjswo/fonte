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
independent_vars <- c("score_mode", "use_diff", "stage2", "use_stopword", "adddel")
output_dir <- "/root/workspace/analyze/data/"
results <- list()

for (dep_var in dependent_vars) {
  subset_data <- data %>% filter(DependentName == dep_var)
    
    # Fit ART model
    art_model <- art(DependentValue ~ score_mode * use_diff * stage2 * use_stopword * adddel + (1|project), data = subset_data)

  for (ind_var in independent_vars) {
    
    #art_model <- art(DependentValue ~ score_mode * use_diff + (1|project), data = subset_data)
    
    # Perform post-hoc test for interaction of score_mode and use_diff
    #posthoc_interaction <- art.con(art_model, "score_mode:use_diff:stage2:use_stopword:adddel")
    posthoc_interaction <- art.con(art_model, ind_var)
    
    # Convert results to a data frame
    contrast_df <- as.data.frame(posthoc_interaction)

    # Filter for significant cases (p-value < 0.05)
    # Filter significant results
    significant_contrasts <- contrast_df %>%
      #filter(p.value < 0.05) %>%
      mutate(
        p.value.adjusted = p.adjust(p.value, method = "bonferroni"),
        Winner = sub(" - .*", "", contrast),  # Extract the first setting (winner)
        Loser = sub(".* - ", "", contrast)   # Extract the second setting (loser)
      )

    # Print significant cases
    output_file <- file.path(output_dir, paste0("post_hoc_", dep_var, "_", ind_var,".csv"))
    write.csv(significant_contrasts, file = output_file, row.names = FALSE)

      # Save results
    #results[[dep_var]] <- summary(posthoc_interaction)
  }
}

# Print results for each dependent variable
#results