
source("./lib/TreeStatFrontend.R")

get_biphasic_HIV_rate_function <- function(front_density_factor, front_cutoff,
                                           target_length, params) {


    # Parameterize the distribution
    total_density <- target_length + (front_density_factor - 1) * front_cutoff
    front_mass <- (front_density_factor * front_cutoff)  #  / total_density
    # Density for each time step
    fdf <- ((front_mass / front_cutoff) * params[["R0"]]) / total_density
    tdf <- (params[["R0"]] / total_density)

    # This function must conform to the API requirements
    rate_fn <- function(current_step, birth_step, ...) {
        rates <- ((current_step - birth_step) < front_cutoff) * fdf
        rates <- rates + ((current_step - birth_step) >= front_cutoff) * tdf
        return(rates)
    }

    return(rate_fn)
}

simulate_SE_EU_HIV <- function(sample_size, ct_p, sample_times_data, sample_times_times, pop_size = 200,
                                 l1 = 1, l2 = 2, R0_init = 5, stretch = 0, init_pop_size = 20,
                                 front_ratio = 20, mu_factor = 1, R0 = 3) {

    # Parameters
    INITIAL_GROWTH_RATE <- R0_init  # FAST growth rate
    SLOW_GROWTH_RATE <- R0    # SLOW growth rate
    initial_growth_pop_size <- init_pop_size
    slow_growth_pop_size <- pop_size  # used to

    # The number of steps in the constant phase before the first spike.
    first_constant_phase_endpoint <- l1  # * 12 previously
    second_constant_phase_endpoint <- l2

    # Now draw some sampling times
    draw <- sample.int(sum(sample_times_data), sample_size, replace = FALSE)
    print(draw)
    print(length(draw))
    items <- findInterval(draw, cumsum(sample_times_data), rightmost.closed = TRUE)
    # print(items)
    # print(length(items))
    sample_times <- sample_times_times[items]
    # print(sample_times)
    # print(length(sample_times))
    # Agglomerate the sample times together.
    # biphasic_rate_function_fast <- SEEPS::get_biphasic_HIV_rate(
    #     params = list("R0" = INITIAL_GROWTH_RATE))
    biphasic_rate_function_fast <- get_biphasic_HIV_rate_function(
        front_ratio, 3, 24, params = list("R0" = INITIAL_GROWTH_RATE))
    # biphasic_rate_function_slow <- SEEPS::get_biphasic_HIV_rate(
    #     params = list("R0" = SLOW_GROWTH_RATE))
    biphasic_rate_function_slow <- get_biphasic_HIV_rate_function(
        front_ratio, 3, 24, params = list("R0" = SLOW_GROWTH_RATE))

    sim_parameters <- SEEPS::wrap_parameters(
        minimum_population = 5,
        offspring_rate_fn = biphasic_rate_function_fast,
        maximum_population_target = initial_growth_pop_size,
        total_steps = 0,
        spike_root = FALSE)

    state <- SEEPS::initialize(sim_parameters)
    # Simulate the first phase
    state <- SEEPS::gen_exp_phase(state, sim_parameters)
    # Determine the number of steps taken in the first phase
    first_phase_steps <- state$curr_step
    # Simulate until first_constant_phase steps are taken
    state <- SEEPS::gen_const_phase(state, sim_parameters,
                                    num_steps = first_constant_phase_endpoint - first_phase_steps)
    # Now the slower growth phase
    sim_parameters$offspring_rate_fn <- biphasic_rate_function_slow
    sim_parameters$maximum_population_target <- slow_growth_pop_size
    # Second exp phase
    state <- SEEPS::gen_exp_phase(state, sim_parameters)
    current_time <- state$curr_step

    # Simulate until the second constant phase
    state <- SEEPS::gen_const_phase(state, sim_parameters,
                                    num_steps = second_constant_phase_endpoint - current_time)

    # Now we mask down the population to 2k individuals randomly.
    # This accounts for the fact that the Thai population must export sequences to the EU.
    # This is a simplification, but it's very nice.
    # prop_forward <- sample(state$active, 400, replace = FALSE)
    # sample <- SEEPS::keep_samples(state, prop_forward)
    # Now we want to start sampling, forward in time.


    states_all <- list()

    #
    samples_all <- list()
    num_steps_total <- length(sample_times_data)
    ptr <- 1
    for (annual_steps in 1:num_steps_total) {
        # Simulate until the next sample time
        state <- SEEPS::gen_const_phase(state, sim_parameters,
                                        num_steps = 12)  # Go forward one year
        states_all[[ptr]] <- state
        # Now we want to sample the population
        if (annual_steps %in% sample_times) {
            # Determine the number of samples to take
            # Count the number of times annual_steps is in sample_times
            num_samples <- sum(sample_times == annual_steps)
            # Now sample the population
            sample_taken <- SEEPS::contact_traced_uniform_restarts_ids(
                state$active, state$parents, num_samples, ct_p)
            # Now we remove the samples from the population
            state <- SEEPS::remove_samples(state, sample_taken$samples)
            samples_all[[ptr]] <- list("curr_step" = state$curr_step, "samples" = sample_taken$samples)
        }
        ptr <- ptr + 1
    }
    # print("Finished sampling.")
    # Now we want to build out the sample tree
    # First, loop over the samples_all list and write the curr_step into a list
    # and the samples into their own list
    sample_times_ <- list()
    sample_ids <- list()
    ptr <- 1
    for (i in 1:length(samples_all)) {
        if (!is.null(samples_all[[i]])) {  # Look before you leap
            sample_times_[[ptr]] <- samples_all[[i]]$curr_step + stretch
            sample_ids[[ptr]] <- samples_all[[i]]$samples
            ptr <- ptr + 1
        }
    }
    # print("The samples are")
    # print("Reducing genealogy...")
    # flush.console()
    genealogy <- SEEPS::reduce_transmission_history_bpb2(samples = sample_ids,
                                                         parents = state$parents,
                                                         current_step = sample_times_)
    # print("Reduced genealogy. Converting to phylogeny...")
    # flush.console()
    phylogeny <- SEEPS::geneology_to_phylogeny_bpb(
        transmission_history = genealogy$parents,
        infection_times = genealogy$transmission_times,
        sample_times = genealogy$sample_times,
        a = 5, b = 5, leaf_sample_ids = genealogy$transformed_sample_indices)
    # We're going to sample the edge lengths.
    V3_sequence <- SEEPS::lookup_sequence_by_name(organism_name = "HIV1",  # nolint: object_name_linter
                                                        region_name = "Env-Clinical")
    seq_len <- length(V3_sequence)
    phylogeny <- SEEPS::stochastify_transmission_history(
                            transmission_history = phylogeny$phylogeny,
                            rate = 0.0067 / (12 * mu_factor) * seq_len)
    # print("Converted to phylogeny. Converting to newick...")
    # print(phylogeny$geneology)
    flush.console()
    newick_tree_string <- SEEPS::phylogeny_to_newick(phylogeny$geneology, mode = "mu", label_mode = "abs")
    # print("Converted to newick. Computing statistics...")
    # flush.console()
    EIr = ExtIntRatio(newick_tree_string)
    sackin_index = sackinIndex(newick_tree_string)
    # print("Computed statistics. Returning.")
    # flush.console()
    return(list("phylogeny"=phylogeny, "newick_tree_string" = newick_tree_string,
                "EIr" = EIr, "sackin_index" = sackin_index))
}
