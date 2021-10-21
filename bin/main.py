def main() -> None:
    run_config = get_run_config()
    optimalization_config = get_optimizers_config(run_config)
    simulation_config = get_simulation_config(run_config)
    environment_config = get_environment_setup(run_config)
    # simulation setup has number of initial solutions field
    environment = Environment(environment_config)
    simulator = Simulator(simulation_config)
    initial_solutions = generate_initial_solutions(simulator, environment)
    initial_results = simulator.simulate(initial_solutions)
    # simulation setup has algorithms field (i.e. tabu search)
    optimizers = generate_optimizers(optimalization_config)
    optimized_solutions, optimized_results = generate_optimizations(
        optimizers, simulator, environment
    )
    compared_data = compare_simulation_data(initial_results, optimized_results)
    visualized_data = generate_visualized_data()
    save_data(initial_solutions, initial_results, optimized_solutions, optimized_results)
