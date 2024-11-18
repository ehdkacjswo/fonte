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
library(car)

# Define the specific independent variable combinations you want to analyze
selected_iv_combinations <- list(
  c("use_stopword", "stage2"),
  c("use_stopword"),
  c("stage2")
)

# Define the specific dependent variables to analyze
selected_dependent_vars <- c("rank", "num_iters")  # Replace with your choices

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

subset_data <- data %>% filter(stage2 == "True")

library(car)

perform_art_anova <- function(dependent_var, iv_combination) {
  # Filter data for the current dependent variable
  data_subset <- subset_data %>% filter(DependentName == dependent_var)
  
  # Generate formula with all interactions
  formula <- as.formula(
    paste("DependentValue ~ (", paste(iv_combination, collapse = " + "), ")^2 + (1 | project)")
  )
  
  # Perform ART
  art_model <- tryCatch({
    art(formula, data = data_subset)
  }, error = function(e) {
    stop(paste("Error in combination:", paste(iv_combination, collapse = ", "), "\nError message:", e$message))
  })
  
  # Conduct ANOVA with type III sums of squares to include p-values
  anova_results <- tryCatch({
    Anova(art_model, type = 3)
  }, error = function(e) {
    stop(paste("Error in combination:", paste(iv_combination, collapse = ", "), "\nError message:", e$message))
  })
  
  # Convert to a data frame and clean up column names
  anova_df <- as.data.frame(anova_results)
  anova_df <- tibble::rownames_to_column(anova_df, "Effect") %>%
    rename(Df = `Df`, F = `F`, p = `Pr(>F)`)
  
  list(
    model = art_model,
    anova = anova_df
  )
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
output_dir <- "/root/workspace/aaa"
dir.create(output_dir, showWarnings = FALSE)

final_results <- map(selected_dependent_vars, function(dv) {
  output_file <- file.path(output_dir, paste0("results_", dv, ".csv"))
  run_subset_analysis(dv, selected_iv_combinations, output_file)
})

# Combine all results into a single data frame (bind_rows will now work)
final_results_combined <- bind_rows(final_results)