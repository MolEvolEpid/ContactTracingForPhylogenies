library(SEEPS)
require(future)

source("./lib_SEEPS.R")

simulate_serialize <- function(sample_sizes, time_points, n_examples_per_time,
                               save_letters, status, discovery_p) {

    results <- list()
    index <- 1
    set.seed(1947)  # for reproducibility
    for (sample_size in sample_sizes) {
        # if discovery_p is a float, form it into a vector
        if (length(discovery_p) == 1) {
            discovery_p <- c(discovery_p)
        }
        for (discovery_p_value in discovery_p) {
            for (tp in time_points) {
                print(paste("Sample size:", sample_size, "Time point:", tp,
                            "Discovery p:", discovery_p_value))
                # Print out all the parameters for the next function call
                print(paste("n_examples_per_time:", n_examples_per_time,
                            "time_point:", tp, "sample_size:", sample_size,
                            "discovery_p:", discovery_p_value))

                # Call SEEPS_parallel with the time point and other parameters
                result <- SEEPS_parallel(n_examples = n_examples_per_time,
                                        time_point = tp, sample_size = sample_size,
                                        discovery_p = discovery_p_value)
                # Unpack the results into a data frame
                results[[index]] <- result
                index <- index + 1
            }
        }
    }
    flat_results <- unlist(results, recursive = FALSE)
    df <- data.frame(# Unpack the list of parameter values first
                    a = numeric(),
                    b = numeric(),
                    minimum_population = numeric(),
                    total_steps_after_exp_phase = numeric(),
                    contact_tracing_discovery_probability = numeric(),
                    mutation_rate = numeric(),
                    minimum_population = numeric(),
                    maximum_population_target = numeric(),
                    R0 = numeric(),
                    nonzero_I = numeric(),

                    # sequences = I(list()),
                    matrix_seqs = I(list()),
                    matrix_seqs_raw = I(list()),
                    matrix_trans = I(list()),
                    matrix_phylo = I(list()),
                    matrix_seqs_names = I(list()),
                    matrix_seqs_raw_names = I(list()),
                    matrix_phylo_names = I(list()),
                    matrix_trans_names = I(list()),
                    sequence_length = numeric(),
                    # ei ratios
                    ei_ratio_phylo = numeric(),
                    ei_ratio_trans = numeric(),
                    ei_ratio_phylo_mu = numeric(),
                    ei_ratio_trans_mu = numeric(),
                    # sackin indices
                    sackin_trans = numeric(),
                    sackin_trans_mu = numeric(),
                    sackin_phylo = numeric(),
                    sackin_phylo_mu = numeric(),
                    # Cherries
                    cherries_trans =  numeric(),
                    cherries_trans_mu =  numeric(),
                    cherries_phylo =  numeric(),
                    cherries_phylo_mu =  numeric(),
                    # Contact tracing samples
                    ct_group_ids = I(list()),
                    ct_sample_ids = I(list()),
                    string_trans = character(),
                    string_phylo = character()
                    )
    index <- 1
    print("Y")
    for (record in flat_results) {
        # We need to extract the $rate_function_parameters
        # Store the list of unpacked parameter values
        params <- record$params
        df[[index, "a"]] <- params$a
        df[[index, "b"]] <- params$b
        df[[index, "minimum_population"]] <- params$minimum_population
        df[[index, "total_steps_after_exp_phase"]] <- params$total_steps_after_exp_phase
        df[[index, "contact_tracing_discovery_probability"]] <- params$contact_tracing_discovery_probability
        df[[index, "mutation_rate"]] <- params$mutation_rate
        df[[index, "minimum_population"]] <- params$minimum_population
        df[[index, "maximum_population_target"]] <- params$maximum_population_target
        df[[index, "R0"]] <- params$rate_function_parameters$R0
        df[[index, "nonzero_I"]] <- params$nonzero_I

        df[[index, "matrix_trans_names"]] <- rownames(record$matrix_trans)
        df[[index, "matrix_seqs_names"]] <- rownames(record$matrix_seqs)
        df[[index, "matrix_seqs_raw_names"]] <- rownames(record$matrix_seqs_raw)
        df[[index, "matrix_phylo_names"]] <- rownames(record$matrix_phylo)

        for (mat in list(record$matrix_seqs, record$matrix_phylo, record$matrix_trans, record$matrix_seqs_raw)) {
            rownames(mat) <- NULL
            colnames(mat) <- NULL
        }
        # matrix statistics
        df[[index, "matrix_seqs"]] <- as.vector(t(record$matrix_seqs))
        df[[index, "matrix_seqs_raw"]] <- as.vector(t(record$matrix_seqs_raw))
        df[[index, "matrix_trans"]] <- as.vector(t(record$matrix_trans))
        df[[index, "matrix_phylo"]] <- as.vector(t(record$matrix_phylo))
        # Length of reference sequence. We probably only need this once, but it's easier than storing it seperately
        df[[index, "sequence_length"]] <- record$sequence_length
        # ei ratios
        df[[index, "ei_ratio_trans"]] <- record$ei_ratio_trans
        df[[index, "ei_ratio_phylo"]] <- record$ei_ratio_phylo
        df[[index, "ei_ratio_trans_mu"]] <- record$ei_ratio_trans_mu
        df[[index, "ei_ratio_phylo_mu"]] <- record$ei_ratio_phylo_mu
        # cherries
        df[[index, "cherries_trans"]] <- record$cherries_trans
        df[[index, "cherries_trans_mu"]] <- record$cherries_trans_mu
        df[[index, "cherries_phylo"]] <- record$cherries_phylo
        df[[index, "cherries_phylo_mu"]] <- record$cherries_phylo_mu
        # sackin index
        df[[index, "sackin_trans"]] <- record$sackin_trans
        df[[index, "sackin_trans_mu"]] <- record$sackin_trans_mu
        df[[index, "sackin_phylo"]] <- record$sackin_phylo
        df[[index, "sackin_phylo_mu"]] <- record$sackin_phylo_mu
        # Contact tracing
        df[[index, "ct_group_ids"]] <- record$ct_group_ids
        df[[index, "ct_sample_ids"]] <- record$ct_sample_ids
        df[[index, "string_trans"]] <- record$string_trans
        df[[index, "string_phylo"]] <- record$string_phylo
        # update counter
        index <- index + 1
    }
    print("Finished building data frame! Beginning serialization.")
    # Store the result in a file
    # data/large_data is sym-linked onto scratch (rotating media)
    # fname is prefix_{batch_ID}_{sample_size}.parquet
    fname <- paste0("../../synth_data/SEEPS_", save_letters, "/",
                    sample_size, "/SEEPS_", status, ".parquet")
    if (!file.exists(fname)) {
        # Create the directory path if it doesn't exist
        dir.create(dirname(fname), showWarnings = FALSE, recursive = TRUE)
    }
    arrow::write_parquet(x = df, sink = fname)
    # return(df)
}
