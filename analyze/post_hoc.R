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
  make_option(c("-e", "--exclude"), type = "character", default = "", 
              help = "Comma-separated list of independent variables to exclude"),
  make_option(c("-b", "--bug2commit"), type = "logical", default = FALSE, 
              help = "Use bug2commit only[default: TRUE]"),
  make_option(c("-f", "--fix"), type = "character", default = "use_br:False,use_stopword:True,HSFL:False", 
              help = "Comma-separated list of parameters and values to fix (e.g., use_stopword:False,stage2:True)")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Load data
if (opt$bug2commit) {
  data <- read.csv("/root/workspace/analyze/data/bug2commit/metrics.csv", stringsAsFactors = FALSE)
} else {
  data <- read.csv("/root/workspace/analyze/data/all/metrics.csv", stringsAsFactors = FALSE)
}

data <- data %>%
  mutate(
    across(!DependentValue, as.factor)
    DependentValue = as.numeric(DependentValue)
  )

# Parse excluded variables and reorder them in same order of columns in the data
excluded_vars <- unlist(strsplit(opt$exclude, ","))
excluded_vars <- intersect(colnames(data), excluded_vars)

# Parse fix parameters and values
fix_parameters <- if (opt$fix != "") unlist(strsplit(opt$fix, split = ",")) else NULL

# Convert the fix parameters to a named list
if (!is.null(fix_parameters)) {
  fix_conditions <- lapply(fix_parameters, function(param) {
    parts <- unlist(strsplit(param, ":"))
    setNames(list(parts[2]), parts[1])
  })
  fix_conditions <- do.call(c, fix_conditions)

  # Filter parameters in data and reorder them in same order of columns in the data
  fix_conditions <- fix_conditions[names(fix_conditions) %in% colnames(data)]
  fix_conditions <- fix_conditions[match(colnames(data), names(fix_conditions), nomatch = 0)]
} else {
  fix_conditions <- list()
}

# Ensure excluded variables exist in the dataset
if (!all(names(fix_conditions) %in% colnames(data))) {
  stop("One or more excluded variables do not exist in the dataset.")
}

# Fix the paremeters and remove from excluded variables
for (param in names(fix_conditions)) {
  value <- fix_conditions[[param]]
  data <- data %>% filter(!!sym(param) == value)
  data <- data[, !names(data) %in% param] # Remove fixed column
  excluded_vars <- setdiff(excluded_vars, param) # Remove from excluding list
}

# Ensure excluded variables exist in the dataset
if (!all(excluded_vars %in% colnames(data))) {
  stop("One or more excluded variables do not exist in the dataset.")
}

# Get unique combinations of excluded variable values
get_excluded_levels <- function(data, excluded_vars) {
  if (length(excluded_vars) == 0) return(NULL)
  data %>%
    select(all_of(excluded_vars)) %>%
    distinct() %>%
    drop_na()
}

# Perform ART post-hoc contrast test
perform_art_posthoc <- function(dv_name, excluded_vars, excluded_values) {
  # Filter data for the dependent variable and excluded values
  filtered_data <- data %>% filter(DependentName == dv_name)
  for (var in excluded_vars) {
    filtered_data <- filtered_data %>% filter(!!sym(var) == excluded_values[[var]])
    filtered_data <- filtered_data[, !names(filtered_data) %in% var] # Remove fixed column
  }
  
  # Remaining independent variables
  remaining_vars <- setdiff(names(filtered_data), c(excluded_vars, "DependentValue", "project", "DependentName"))
  formula_str <- paste("DependentValue ~", paste(remaining_vars, collapse = " * "), "+ (1|project)")
  art_model <- art(as.formula(formula_str), data = filtered_data)
  
  # Perform post-hoc contrast test
  interaction_term <- paste(remaining_vars, collapse = ":")
  posthoc_results <- art.con(art_model, interaction_term)
  contrast_df <- as.data.frame(posthoc_results)
  
  # Add excluded variable levels
  for (var in excluded_vars) {
    contrast_df[[var]] <- excluded_values[[var]]
  }
  
  return(contrast_df)
}

# Main analysis loop
run_analysis <- function(dependent_vars, excluded_vars, output_file) {
  excluded_levels <- get_excluded_levels(data, excluded_vars)
  all_results <- list()

  pb <- progress_bar$new(
    format = "Processing [:bar] :percent (:current/:total) Dependent: :dv Levels: :levels ETA: :eta",
    total = length(dependent_vars) * (if (is.null(excluded_levels)) 1 else nrow(excluded_levels)),
    clear = FALSE,
    width = 80
  )
  
  # No data excluded
  if (is.null(excluded_levels)) {
    for (dv_name in dependent_vars) {
      result <- perform_art_posthoc(dv_name, excluded_vars, list())
      result_df <- as.data.frame(result)

      # Filter significant cases (p-value < 0.05)
      significant_contrasts <- result_df %>%
        mutate(
          setting_1 = sub(" - .*", "", contrast),  # Extract first option
          setting_2 = sub(".* - ", "", contrast),  # Extract second option
          metric = dv_name
        ) %>%
        select(p.value, setting_1, setting_2, metric) %>%
        filter(p.value < 0.05)

      all_results <- append(all_results, list(significant_contrasts))
      pb$tick(tokens = list(dv = dv_name, levels = "none"))
    }
  } else {
    for (dv_name in dependent_vars) {
      for (i in seq_len(nrow(excluded_levels))) {
        # Get the value of 
        excluded_values <- excluded_levels[i, ] %>% 
          as.list() %>%
          setNames(names(excluded_levels)) %>% # Ensure the list retains variable names
          lapply(as.character)
        
        result <- perform_art_posthoc(dv_name, excluded_vars, excluded_values)
        result_df <- as.data.frame(result)

        exclude_string <- paste(excluded_values, collapse = ",")

        # Filter significant cases (p-value < 0.05)
        significant_contrasts <- result_df %>%
          mutate(
            setting_1 = sub(" - .*", "", contrast),  # Extract first option
            setting_2 = sub(".* - ", "", contrast),  # Extract second option
            metric = dv_name,
            exclude_string = exclude_string
          ) %>%
          select(union(excluded_vars, c('metric', 'p.value', 'setting_1', 'setting_2'))) #%>%
          #filter(p.value < 0.05)

        all_results <- append(all_results, list(significant_contrasts))
        pb$tick(tokens = list(dv = dv_name, levels = paste(unlist(excluded_values), collapse = ", ")))
      }
    }
  }

  # Combine results and save to a single file
  combined_results <- bind_rows(all_results)
  write.csv(combined_results, file = output_file, row.names = FALSE)
}

# List of dependent variables
dependent_vars <- c("rank", "num_iters")

fix_string <- paste(
  paste(names(fix_conditions), fix_conditions, sep = ":"),
  collapse = ","
)

exclude_string <- paste(excluded_vars, collapse = ',')

setting_string <- paste0(
  fix_string,
  if (fix_string != '' & exclude_string != '') ',' else '',
  exclude_string 
)

# Output file path
output_file <- paste0("/root/workspace/analyze/data/",
  if (opt$bug2commit) "bug2commit" else "all",
  "/post_hoc/",
  setting_string,
  '.csv')

run_analysis(dependent_vars, excluded_vars, output_file)
