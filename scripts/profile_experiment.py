import sys

sys.path.extend([".", ".."])

from pyinstrument import Profiler

from bin.experiment0 import experiment_setup, genetic_algorithm_tsp_simple_test


def main():
    profiler = Profiler()

    print("experiment setup")

    exp_args, _ = experiment_setup()

    print("starting profiling")

    with profiler:
        _ = genetic_algorithm_tsp_simple_test(**exp_args)

    print("profiling done")

    profiler.open_in_browser()
