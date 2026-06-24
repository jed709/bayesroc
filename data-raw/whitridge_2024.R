# Preparation of the shipped `whitridge_2024` dataset.
# Raw source: data-raw/whitridge_2024.rds. Re-run this script to regenerate
# data/whitridge_2024.rda after changing the raw file.

whitridge_2024 <- readRDS("data-raw/whitridge_2024.rds")

# Ship only experiment 1B; drop the now-constant `experiment` column.
whitridge_2024 <- whitridge_2024[whitridge_2024$experiment == "1B", ]
whitridge_2024$experiment <- NULL
rownames(whitridge_2024) <- NULL

# Tidy storage types; keep `condition` factor ("new" first = the lure level).
whitridge_2024$participant <- as.integer(whitridge_2024$participant)
whitridge_2024$old         <- as.integer(whitridge_2024$old)
whitridge_2024$conf        <- as.integer(whitridge_2024$conf)

usethis::use_data(whitridge_2024, overwrite = TRUE, compress = "xz")
