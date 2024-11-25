# Load necessary libraries
library(emmeans)
library(ARTool)
library(purrr)
library(dplyr)
library(tidyr)
library(optparse)
library(progress)

# Command line parser
option_list <- list(
  make_option(c("-e", "--exclude"), type = "character", default = "use_diff", 
              help = "Comma-separated list of independent variables to exclude"),
  make_option(c("-b", "--bug2commit"), type = "logical", default = TRUE, 
              help = "Use bug2commit only[default: TRUE]"),
  make_option(c("-f", "--fix"), type = "logical", default = TRUE, 
              help = "Fix stage2, use_stopword as True [default: TRUE]"),
  make_option(c("-h", "--hsfl"), type = "logical", default = FALSE, 
              help = "Fix use_stopword as True [default: FALSE]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Parse excluded variables
excluded_vars <- unlist(strsplit(opt$exclude, ","))
if (opt$fix) excluded_vars <- setdiff(excluded_vars, c("use_stopword", "stage2"))
output_file <- "/root/workspace/analyze/data/all/post_hoc/use_br_fix_posthoc_results.csv"

# Load data
if (opt$bug2commit) {
  data <- read.csv("/root/workspace/analyze/data/bug2commit/metrics.csv", stringsAsFactors = FALSE)

  data <- data %>%
    mutate(
      project = as.factor(project),
      score_mode = as.factor(score_mode),
      use_br = as.factor(use_br),
      use_diff = as.factor(use_diff),
      stage2 = as.factor(stage2),
      use_stopword = as.factor(use_stopword),
      adddel = as.factor(adddel),
      DependentName = as.factor(DependentName),
      DependentValue = as.numeric(DependentValue)
    )
} else {
  data <- read.csv("/root/workspace/analyze/data/all/use_br_metrics.csv", stringsAsFactors = FALSE)

  data <- data %>%
  mutate(
    project = as.factor(project),
    HSFL = as.factor(HSFL),
    score_mode = as.factor(score_mode),
    ensemble = as.factor(ensemble),
    use_br = as.factor(use_br),
    use_diff = as.factor(use_diff),
    stage2 = as.factor(stage2),
    use_stopword = as.factor(use_stopword),
    adddel = as.factor(adddel),
    DependentName = as.factor(DependentName),
    DependentValue = as.numeric(DependentValue)
  )

  if (!opt$hsfl) {
    data <- data %>% filter(HSFL == "False")
    data <- data[,!names(data) %in% c("HSFL")]
    excluded_vars <- setdiff(excluded_vars, c("HSFL"))
  }
}

if (opt$fix) {
  data <- data %>% filter(use_stopword == "True" & stage2 == 'True')
  data <- data[,!names(data) %in% c("stage2", "use_stopword")]
  excluded_vars <- setdiff(excluded_vars, c("stage2", "use_stopword"))
}

# Function to get unique combinations of excluded variable values
get_excluded_levels <- function(data, excluded_vars) {
  if (length(excluded_vars) == 0) return(NULL)
  data %>%
    select(all_of(excluded_vars)) %>%
    distinct() %>%
    drop_na()
}

# Perform ART ANOVA and post-hoc contrast test
perform_art_posthoc <- function(dv_name, excluded_values, excluded_vars) {
  # Filter data for the dependent variable and excluded values
  filtered_data <- data %>% filter(DependentName == dv_name)
  for (var in excluded_vars) {
    filtered_data <- filtered_data %>% filter(!!sym(var) == excluded_values[[var]])
  }
  
  # Remaining independent variables
  remaining_vars <- setdiff(names(filtered_data), c(excluded_vars, "DependentValue", "project", "DependentName"))
  formula_str <- paste("DependentValue ~", paste(remaining_vars, collapse = " * "), "+ (1|project)")
  art_model <- art(as.formula(formula_str), data = filtered_data)
  
  # Perform post-hoc contrast test
  interaction_term <- paste(remaining_vars, collapse = ":")
  posthoc_results <- art.con(art_model, interaction_term, adjust = "bonferroni")
  contrast_df <- as.data.frame(posthoc_results)
  
  # Add excluded variable levels
  #for (var in excluded_vars) {
  #  contrast_df[[var]] <- excluded_values[[var]]
  #}
  
  return(contrast_df)
}

# Main analysis loop
run_analysis <- function(dependent_vars, excluded_vars, output_file) {
  excluded_levels <- get_excluded_levels(data, excluded_vars)
  print(excluded_levels)
  all_results <- list()

  pb <- progress_bar$new(
    format = "Processing [:bar] :percent (:current/:total) Dependent: :dv Levels: :levels ETA: :eta",
    total = length(dependent_vars) * (if (is.null(excluded_levels)) 1 else nrow(excluded_levels)),
    clear = FALSE,
    width = 80
  )
  
  for (dv_name in dependent_vars) {
    if (is.null(excluded_levels)) {
      # If no excluded variables, run directly
      result <- perform_art_posthoc(dv_name, list(), excluded_vars)
      result_df <- as.data.frame(result)

      # Filter significant cases (p-value < 0.05)
      significant_contrasts <- result_df %>%
        mutate(
          setting_1 = sub(" - .*", "", contrast),  # Extract first option
          setting_2 = sub(".* - ", "", contrast),  # Extract second option
        ) %>%
        filter(p.value < 0.05)
      all_results[[dv_name]] <- significant_contrasts
      pb$tick(tokens = list(dv = dv_name, levels = "none"))
    } else {
      for (i in seq_len(nrow(excluded_levels))) {
        excluded_values <- excluded_levels[i, ] %>% 
          as.list() %>%
          setNames(names(excluded_levels)) %>% # Ensure the list retains variable names
          lapply(as.character)
        print(excluded_values)
        result <- perform_art_posthoc(dv_name, excluded_values, excluded_vars)
        result_df <- as.data.frame(result)

        # Filter significant cases (p-value < 0.05)
        significant_contrasts <- result_df %>%
          mutate(
            setting_1 = sub(" - .*", "", contrast),  # Extract first option
            setting_2 = sub(".* - ", "", contrast),  # Extract second option
          ) #%>%
          #filter(p.value < 0.05)
        all_results[[paste(dv_name, paste(excluded_values, collapse = "_"), sep = "_")]] <- significant_contrasts
        pb$tick(tokens = list(dv = dv_name, levels = paste(unlist(excluded_values), collapse = ", ")))
      }
    }
  }

  # Combine results and save to a single file
  combined_results <- bind_rows(all_results, .id = "Test")
  #write.csv(combined_results, file = output_file, row.names = FALSE)
}

# List of dependent variables
dependent_vars <- c("rank", "num_iters")
run_analysis(dependent_vars, excluded_vars, output_file)
