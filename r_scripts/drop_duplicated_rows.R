source("r_scripts/requirements.R")
library(tidyverse)
library(arrow)

df <- read_parquet("data/dat_train.parquet")
df %>%
  distinct(SEA5, klartext, .keep_all = TRUE) %>% 
  write_parquet("data/training_data/raw_no_dupl2.parquet")
