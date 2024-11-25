# Load necessary libraries
library(ARTool)
library(dplyr)
library(tidyr)
library(purrr)
library(progress)
library(optparse)

# Define command-line arguments
option_list <- list(
  make_option(c("-e", "--exclude"), type = "character", default = "score_mode,use_diff", 
              help = "Comma-separated list of independent variables to exclude"),
  make_option(c("-b", "--bug2commit"), type = "logical", default = TRUE, 
              help = "Use bug2commit only[default: TRUE]"),
  make_option(c("-f", "--fix"), type = "character", default = "use_stopword:True,stage2:True,use_br:False", 
              help = "Comma-separated list of parameters and values to fix (e.g., use_stopword:False,stage2:True)")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Parse excluded variables
excluded_vars <- unlist(strsplit(opt$exclude, split = ","))

# Parse fix parameters and values
fix_parameters <- if (opt$fix != "") unlist(strsplit(opt$fix, split = ",")) else NULL

# Convert the fix parameters to a named list
fix_conditions <- list()
if (!is.null(fix_parameters)) {
  fix_conditions <- lapply(fix_parameters, function(param) {
    parts <- unlist(strsplit(param, ":"))
    setNames(list(parts[2]), parts[1])
  })
  fix_conditions <- do.call(c, fix_conditions)
}

# Define dependent variables to analyze
selected_dependent_vars <- c("rank", "num_iters")

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
  data <- read.csv("/root/workspace/analyze/data/all/metrics.csv", stringsAsFactors = FALSE)

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
}

# Apply fix conditions dynamically
if (!is.null(fix_conditions)) {
  for (param in names(fix_conditions)) {
    value <- fix_conditions[[param]]
    data <- data %>% filter(!!sym(param) == value)
    data <- data[, !names(data) %in% param] # Remove fixed column
    excluded_vars <- setdiff(excluded_vars, param)
  }
}

# Ensure excluded variables exist in the dataset
if (!all(excluded_vars %in% colnames(data))) {
  stop("One or more excluded variables do not exist in the dataset.")
}

# Function to get unique combinations of excluded variable values
get_excluded_levels <- function(data, excluded_vars) {
  if (length(excluded_vars) == 0) return(NULL)
  data %>%
    select(all_of(excluded_vars)) %>%
    distinct() %>%
    drop_na()
}

# Define function to perform ART ANOVA
perform_art_anova <- function(dv_name, excluded_vars, excluded_values, file_prefix) {
  # Filter data based on excluded variable values
  filtered_data <- data %>% filter(DependentName == dv_name)
  for (var in excluded_vars) {
    filtered_data <- filtered_data %>% filter(!!sym(var) == excluded_values[[var]])
  }
  #print(filtered_data)
  #print(excluded_vars)
  #print(excluded_values)

  # Check if filtered data is empty
  if (nrow(filtered_data) == 0) {
    warning(paste("Filtered data is empty for Dependent:", dv_name, "and Fixed Levels:", paste(excluded_values, collapse = ", ")))
    return(NULL)
  }
  
  # Get remaining independent 
  remaining_vars <- setdiff(names(filtered_data), c(excluded_vars, "DependentValue", "project", "DependentName"))
  
  # Create formula dynamically
  formula <- as.formula(
    paste("DependentValue ~", paste(remaining_vars, collapse = " * "), "+ (1 | project)")
  )
  
  # Perform ART
  art_model <- art(formula, data = filtered_data)
  
  # Get ANOVA results
  anova_results <- anova(art_model)
  
  # Add excluded variables and their levels to results
  anova_results <- anova_results %>%
    mutate(
      DependentName = dv_name,
      FixedVars = paste(paste(excluded_vars, excluded_values, sep = "="), collapse = ", "),
      InteractionTerm = rownames(anova_results),
      PValue = `Pr(>F)`
    ) %>%
    select(FixedVars, InteractionTerm, PValue, DependentName)
  
  return(anova_results)
}

# Run ART ANOVA for all dependent variables and excluded combinations
run_analysis <- function(dependent_vars, excluded_vars, file_prefix) {
  excluded_levels <- get_excluded_levels(data, excluded_vars)

  pb <- progress_bar$new(
    format = "Processing [:bar] :percent (:current/:total) Dependent: :dv ETA: :eta",
    total = length(dependent_vars),
    clear = FALSE,
    width = 80
  )
  
  results <- list()
  for (dv_name in dependent_vars) {
    if (is.null(excluded_levels)) {
      # If no excluded variables, run directly
      results[[dv_name]] <- perform_art_anova(dv_name, excluded_vars, list(), file_prefix)
    } else {
      for (i in seq_len(nrow(excluded_levels))) {
        excluded_values <- excluded_levels[i, ] %>% 
          as.list() %>%
          setNames(names(excluded_levels)) %>% # Ensure the list retains variable names
          lapply(as.character)
        
        tryCatch({
          # Perform ART ANOVA
          result <- perform_art_anova(dv_name, excluded_vars, excluded_values, file_prefix)
          if (!is.null(result)) {
            results <- append(results, list(result))
          }
        }, error = function(e) {
          message(paste("Error for Dependent:", dv_name, "Fixed Levels:", paste(excluded_values, collapse = ", ")))
          message("Error message:", e$message)
        })
      }
    }

    pb$tick(tokens = list(dv = dv_name, levels = "none"))
  }
  
  # Combine results and write to file
  final_results <- bind_rows(results)
  write.csv(final_results, file = paste0(file_prefix, ".csv"), row.names = FALSE)
}

# Output file path
file_prefix <- paste0("/root/workspace/analyze/data/",
  if (opt$bug2commit) "bug2commit" else "all",
  "/art_anova/",
  if (opt$fix != "") paste0(opt$fix, ",") else "",
  paste0(excluded_vars, collapse = ","))

# Run the analysis
run_analysis(selected_dependent_vars, excluded_vars, file_prefix)
