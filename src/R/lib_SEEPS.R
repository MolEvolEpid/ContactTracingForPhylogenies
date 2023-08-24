require(SEEPS)
require(future)
source("./sim_Env_Clinical_SEEPS.R")
# Provide functions to generate data with parallelism

SEEPS_driver <- function(time_point, sample_size, discovery_p) {  # nolint
    pop_size <- 10 ** runif(n = 1, min = 3, max = 4)
    R0 <- runif(n = 1, min = 1.5, max = 5)    # nolint

    params <- list("rate_function_parameters" = list("R0" = R0),
                   "minimum_population" = sample_size,
                   "maximum_population_target" = pop_size,
                   "contact_tracing_discovery_probability" = discovery_p,
                   "total_steps_after_exp_phase" = time_point,
                   "mutation_rate" = 0.0067,
                   nonzero_I = FALSE,
                   # Use mutation rate for V3 from Leitner & Albert, 1999.
                   "a" = 5, "b" = 5)
    # call simulate_all_paradigms_HIV_V3 with the above parameters
    # and return the result
    # The simulation is modified, see the local file
    res <- simulate_all_paradigms_HIV_V3Clinical(params)
    return(res)
}

SEEPS_parallel <- function(n_examples, time_point, sample_size, discovery_p) {  # nolint
    results <- future.apply::future_lapply(1:n_examples, function(x)  # nolint
        SEEPS_driver(time_point, sample_size, discovery_p),
        future.seed = TRUE)
    return(results)
}
