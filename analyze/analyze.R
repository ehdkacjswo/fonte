#install.packages(c("ARTool", "dplyr", "purrr"), dependencies = TRUE)
#install.packages("pak")
library(pak)
#update.packages(checkBuilt = TRUE, ask = FALSE)
#pak::pkg_install(c("ARTool", "dplyr", "purrr"))

library(ARTool)
library(dplyr)
library(tidyr)
library(purrr)
library(progress)

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

# Generate all combinations of independent variables
independent_vars <- c("score_mode", "use_diff", "stage2", "use_stopword", "adddel")
iv_combinations <- map(1:length(independent_vars), ~combn(independent_vars, ., simplify = FALSE)) %>%
  unlist(recursive = FALSE)

# Define function to perform ART ANOVA
perform_art_anova <- function(dv_name, iv_combination) {
  # Filter data for the dependent variable
  dv_data <- data %>% filter(DependentName == dv_name)
  
  # Create formula dynamically
  formula <- as.formula(
    paste("DependentValue ~", paste(iv_combination, collapse = " * "), "+ (1 | project)")
  )
  
  # Perform ART
  art_model <- art(formula, data = dv_data)
  
  # Get ANOVA results
  anova_results <- anova(art_model)
  
  return(list(dv_name = dv_name, iv_combination = iv_combination, model = art_model, anova = anova_results))
}

# Define function to track progress
run_analysis_with_progress <- function(dv_name, iv_combinations) {
  # Create a progress bar
  pb <- progress_bar$new(
    format = "  Running ART ANOVA [:bar] :percent ETA: :eta",
    total = length(iv_combinations), 
    clear = FALSE, 
    width = 60
  )
  
  # Run ART ANOVA for all combinations with progress tracking
  results <- map(iv_combinations, function(iv_combination) {
    pb$tick()  # Update progress bar
    perform_art_anova(dv_name, iv_combination)
  })
  
  return(results)
}

# Perform ART ANOVA for all dependent variables and combinations of IVs
dependent_vars <- unique(data$DependentName)

# Use walk to run the analysis and track progress for each dependent variable
final_results <- map(dependent_vars, ~run_analysis_with_progress(.x, iv_combinations))

# Combine all results
final_results_combined <- bind_rows(final_results)

# Save the results
write.csv(final_results_combined, "/root/workspace/analyze/data/rm_anova.csv", row.names = FALSE)