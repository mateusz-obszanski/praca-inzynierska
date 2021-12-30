from argparse import ArgumentParser

import numpy as np


def main(n):
    rng = np.random.default_rng()
    execute_rand_experiments_tsp(n=n, rng=rng)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run multiple experiments")
    parser.add_argument("--n", help="no. of experiments to run")
    args = parser.parse_args()

    main(args.n)
