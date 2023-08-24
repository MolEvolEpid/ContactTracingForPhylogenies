library(SEEPS)
require(future)
source("./SerializeData.R")
# Generate data
# -------------
n_examples_per_time <- 100
sample_sizes <- c(15, 20, 30, 40, 50)
status <- "Test"
save_letters <- "5d"
discovery_p <- 0.1
# Time points
RNGkind("L'Ecuyer-CMRG")  # Thread-safe RNG
time_points <- c(0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120)
future::plan(future::multisession, workers = 10)  # scale me up in production

simulate_serialize(sample_sizes = sample_sizes, time_points = time_points,
                   n_examples_per_time = n_examples_per_time,
                   save_letters = save_letters, status = status,
                   discovery_p = discovery_p)
