# Load necessary libraries
library(ARTool)
library(dplyr)
library(tidyr)
library(purrr)
library(progress)

# Define independent variable combinations
independent_vars <- c("stage2", "score_mode", "use_diff", "use_stopword", "adddel")
selected_iv_combinations <- unlist(
  lapply(5:5, function(k) combn(independent_vars, k, simplify = FALSE)),
  recursive = FALSE
)

# Define dependent variables to analyze
selected_dependent_vars <- c("rank", "num_iters")

# Load and preprocess data
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

# Filter the data
#data <- data %>% filter(stage2 == "True")

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

# Modify run_subset_analysis to use write.table() for appending
run_subset_analysis <- function(dv_name, iv_combinations, output_file) {
  pb <- progress_bar$new(
    format = "Processing [:bar] :percent (:current/:total) Dependent: :dv IVs: :ivs ETA: :eta",
    total = length(iv_combinations),
    clear = FALSE,
    width = 80
  )
  
  results <- list()
  first_write <- TRUE  # Track whether to write headers or append

  for (i in seq_along(iv_combinations)) {
    iv_combination <- iv_combinations[[i]]
    pb$tick(tokens = list(dv = dv_name, ivs = paste(iv_combination, collapse = ", ")))
    
    tryCatch({
      # Perform ART ANOVA
      result <- perform_art_anova(dv_name, iv_combination)
      
      # Extract and save relevant data
      anova_table <- result$anova %>%
        mutate(
          DependentName = dv_name,
          IVs = paste(iv_combination, collapse = " + ")
        )
      
      # Use write.table() to handle appending
      write.table(
        anova_table,
        file = output_file,
        sep = ",",
        row.names = FALSE,
        col.names = first_write,  # Include column headers only for the first write
        append = !first_write,
        quote = FALSE
      )
      
      first_write <- FALSE  # After the first write, append subsequent results
      results[[i]] <- anova_table
    }, error = function(e) {
      message(paste("Error in combination:", paste(iv_combination, collapse = ", ")))
      message("Error message:", e$message)
    })
  }
  
  return(results)
}

# Perform analysis for selected dependent variables and IV combinations
output_dir <- "art_anova_results_subset"
dir.create(output_dir, showWarnings = FALSE)

final_results <- map(selected_dependent_vars, function(dv) {
  output_file <- file.path(output_dir, paste0("all_results_", dv, ".csv"))
  run_subset_analysis(dv, selected_iv_combinations, output_file)
})

# Combine all results into a single data frame (bind_rows will now work)
final_results_combined <- bind_rows(final_results)