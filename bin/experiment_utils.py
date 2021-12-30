from pathlib import Path
import json
from typing import Any, TypeVar, Optional, Union
from traceback import format_exc
from math import isfinite
from statistics import stdev

import numpy as np
from rich.progress import track
from rich import print as rprint
import matplotlib.pyplot as plt
import seaborn as sns
from bin.experiment_vrpp import experiment_vrpp

from bin.rand_exp_config import generate_rand_conf
from bin.experiment_tsp import experiment_tsp
from bin.experiment_vrp import experiment_vrp
from bin.utils import get_rand_exp_map, get_datetime_str
from libs.data_loading.utils import ExperimentType
from libs.data_loading.loaders import get_env_data


Rng = TypeVar("Rng", bound=np.random.Generator)


exp_param_pools = {
    "population_size": list(range(10, 101, 10)),
}


def get_rand_exp_params(rng: Rng) -> tuple[dict[str, Any], Rng]:
    return {k: rng.choice(ps) for k, ps in exp_param_pools.items()}, rng


def _get_early_stop_n(early_stop: Union[str, int], generation_n: int) -> int:
    if isinstance(early_stop, str):
        if early_stop.endswith("%"):
            try:
                ratio = float(early_stop[:-1])
            except ValueError as e:
                raise ValueError(
                    f"Could not convert `early_stop` ({early_stop}) to ratio (`float`)"
                ) from e
            early_stop_n = round(ratio * generation_n)
        else:
            try:
                early_stop_n = int(early_stop)
            except ValueError as e:
                raise Exception(
                    f"Could not convert `early_stop` ({early_stop}) to integer"
                ) from e
    else:
        early_stop_n = early_stop
    return early_stop_n


def execute_rand_experiments_tsp(
    n: int,
    rng: Optional[np.random.Generator] = None,
    generation_n: int = 10000,
    exp_timeout: int = 5 * 60,
    early_stop: Union[int, str] = "10%",
):
    """
    `exp_timeout` - timeout of a single experiment,
    `early_stop` - if int - max. no. of iterations without finding
    a better solution, if str and end with % - ratio `<early stop iterations> / generation_n`
    """

    early_stop_n = _get_early_stop_n(early_stop, generation_n)
    if rng is None:
        rng = np.random.default_rng()
    exp_t = ExperimentType.TSP
    envs_dir = Path("data/environments/")
    results_dir = Path("data/experiments/runs/tsp/")
    confs_dir = Path("data/experiments/configs/tsp/")
    # t - start time, n - experiment number
    conf_fmt, results_fmt = (f"{pref}_{{t}}_no_{{n}}.json" for pref in ("conf", "exp"))
    for i in track(range(1, n + 1), description="Running experiments..."):
        datetime_str = get_datetime_str()
        conf_path, results_path = (
            str(d / fmt_str.format(t=datetime_str, n=i))
            for d, fmt_str in zip((confs_dir, results_dir), (conf_fmt, results_fmt))
        )
        env_p, rng = get_rand_exp_map(exp_t, envs_dir, rng)
        env_data = get_env_data(env_p)
        rand_params, rng = get_rand_exp_params(rng)
        conf = generate_rand_conf(
            exp_t,
            env_data,
            population_size=rand_params["population_size"],
            generation_n=generation_n,
            timeout=exp_timeout,
            early_stop_iters=early_stop_n,
            map_path=env_p,
        )
        conf.save_to_json(conf_path)
        try:
            experiment_tsp(
                exp_conf_path=conf_path,
                results_path=results_path,
                population_size=rand_params["population_size"],
                generation_n=generation_n,
                exp_timeout=exp_timeout,
                early_stop_n=early_stop_n,
                silent=True,
            )
        except Exception as e:
            rprint(
                f"[red]Exception during experiment #{i}:\n"
                f"{format_exc()}\n"
                f"""{type(e).__name__}: {", ".join(e.args)}"""
            )


def execute_rand_experiments_vrp(
    n: int,
    salesmen_n: int,
    rng: Optional[np.random.Generator] = None,
    generation_n: int = 10000,
    exp_timeout: int = 5 * 60,
    early_stop: Union[int, str] = "10%",
):
    """
    `exp_timeout` - timeout of a single experiment,
    `early_stop` - if int - max. no. of iterations without finding
    a better solution, if str and end with % - ratio `<early stop iterations> / generation_n`
    """

    early_stop_n = _get_early_stop_n(early_stop, generation_n)
    if rng is None:
        rng = np.random.default_rng()
    exp_t = ExperimentType.VRP
    envs_dir = Path("data/environments/")
    results_dir = Path("data/experiments/runs/vrp/")
    confs_dir = Path("data/experiments/configs/vrp/")
    # t - start time, n - experiment number
    conf_fmt, results_fmt = (f"{pref}_{{t}}_no_{{n}}.json" for pref in ("conf", "exp"))
    for i in track(range(1, n + 1), description="Running experiments..."):
        datetime_str = get_datetime_str()
        conf_path, results_path = (
            str(d / fmt_str.format(t=datetime_str, n=i))
            for d, fmt_str in zip((confs_dir, results_dir), (conf_fmt, results_fmt))
        )
        env_p, rng = get_rand_exp_map(exp_t, envs_dir, rng)
        env_data = get_env_data(env_p)
        rand_params, rng = get_rand_exp_params(rng)
        conf = generate_rand_conf(
            exp_t,
            env_data,
            population_size=rand_params["population_size"],
            generation_n=generation_n,
            timeout=exp_timeout,
            early_stop_iters=early_stop_n,
            map_path=env_p,
            salesmen_n=salesmen_n,
        )
        conf.save_to_json(conf_path)
        try:
            experiment_vrp(
                exp_conf_path=conf_path,
                results_path=results_path,
                population_size=rand_params["population_size"],
                generation_n=generation_n,
                exp_timeout=exp_timeout,
                early_stop_n=early_stop_n,
                salesmen_n=salesmen_n,
                silent=True,
            )
        except Exception as e:
            rprint(
                f"[red]Exception during experiment #{i}:\n"
                f"{format_exc()}\n"
                f"""{type(e).__name__}: {", ".join(e.args)}"""
            )


def execute_rand_experiments_vrpp(
    n: int,
    salesmen_n: int,
    fillval: int,
    rng: Optional[np.random.Generator] = None,
    generation_n: int = 10000,
    exp_timeout: int = 5 * 60,
    early_stop: Union[int, str] = "10%",
):
    """
    `exp_timeout` - timeout of a single experiment,
    `early_stop` - if int - max. no. of iterations without finding
    a better solution, if str and end with % - ratio `<early stop iterations> / generation_n`
    """

    early_stop_n = _get_early_stop_n(early_stop, generation_n)
    if rng is None:
        rng = np.random.default_rng()
    exp_t = ExperimentType.VRPP
    envs_dir = Path("data/environments/")
    results_dir = Path("data/experiments/runs/vrpp/")
    confs_dir = Path("data/experiments/configs/vrpp/")
    # t - start time, n - experiment number
    conf_fmt, results_fmt = (f"{pref}_{{t}}_no_{{n}}.json" for pref in ("conf", "exp"))
    for i in track(range(1, n + 1), description="Running experiments..."):
        datetime_str = get_datetime_str()
        conf_path, results_path = (
            str(d / fmt_str.format(t=datetime_str, n=i))
            for d, fmt_str in zip((confs_dir, results_dir), (conf_fmt, results_fmt))
        )
        env_p, rng = get_rand_exp_map(exp_t, envs_dir, rng)
        env_data = get_env_data(env_p)
        rand_params, rng = get_rand_exp_params(rng)
        conf = generate_rand_conf(
            exp_t,
            env_data,
            population_size=rand_params["population_size"],
            generation_n=generation_n,
            timeout=exp_timeout,
            early_stop_iters=early_stop_n,
            map_path=env_p,
            salesmen_n=salesmen_n,
            fillval=fillval,
        )
        conf.save_to_json(conf_path)
        try:
            experiment_vrpp(
                exp_conf_path=conf_path,
                results_path=results_path,
                population_size=rand_params["population_size"],
                generation_n=generation_n,
                exp_timeout=exp_timeout,
                early_stop_n=early_stop_n,
                salesmen_n=salesmen_n,
                fillval=fillval,
                silent=True,
            )
        except Exception as e:
            rprint(
                f"[red]Exception during experiment #{i}:\n"
                f"{format_exc()}\n"
                f"""{type(e).__name__}: {", ".join(e.args)}"""
            )


def get_best_curr_cost(exp_data: dict[str, Any]) -> Optional[float]:
    all_curr_best = exp_data["costs"]["current_best"]
    curr_best = tuple(x for x in all_curr_best if isfinite(x))
    if curr_best:
        return min(curr_best)
    else:
        return None


def get_cost_improvement(exp_data: dict[str, Any]) -> float:
    best = get_best_curr_cost(exp_data)
    if best is not None:
        first = exp_data["costs"]["current_best"][0]
        cost_improvement = (first - best) / first
    else:
        cost_improvement = 0
    return cost_improvement


def analyze_by_criterium(
    criterium: str,
    exp_data: dict[str, Any],
    conf_data: dict[str, Any],
    analyzed: dict[str, Any],
    cnt_map: dict[int, int],
):
    if criterium == "by_population_size":
        crit_case = len(conf_data["population"])
    elif criterium == "by_destination_n":
        crit_case = len(conf_data["dist_mx"])
    else:
        raise Exception(f"Unknown criterium: {criterium}")
    analyzed_by_crit = analyzed[criterium]
    if crit_case not in analyzed_by_crit:
        analyzed_by_crit[crit_case] = {
            "mean_exec_t": 0,
            "mean_cost_improvement": 0,
            "mean_iters": 0,
        }
    dest_n = len(conf_data["dist_mx"])
    dest_n_cnt = cnt_map.get(dest_n, 0)
    cost_improvement = get_cost_improvement(exp_data)
    iter_n = len(exp_data["costs"]["current_best"])
    for k, new_v in zip(
        ("mean_exec_t", "mean_cost_improvement", "mean_iters"),
        (exp_data["exec_time"], cost_improvement, iter_n),
    ):
        analyzed_by_crit[crit_case][k] = (
            dest_n_cnt * analyzed_by_crit[crit_case][k] + new_v
        ) / (dest_n_cnt + 1)

    cnt_map[dest_n] = dest_n_cnt + 1


def analyze_by_map(
    map_path: Union[str, Path],
    exp_data: dict[str, Any],
    analyzed: dict[str, Any],
    cnt_map: dict[str, int],
    min_cost_accumulator: dict[str, list[float]],
):
    map_path = str(map_path)
    by_map = analyzed["by_map"]
    if map_path not in by_map:
        by_map[map_path] = {
            "mean_exec_t": 0,
            "mean_cost_improvement": 0,
            "mean_iters": 0,
            "best_solution": [],
            "min_cost": float("inf"),
            "min_cost_mean": 0,
            "min_cost_stddev": float("nan"),
        }
    if map_path not in min_cost_accumulator:
        min_cost_accumulator[map_path] = []
    by_curr_map = by_map[map_path]
    curr_best = get_best_curr_cost(exp_data) or float("inf")
    min_cost_accumulator[map_path].append(curr_best)
    if curr_best < by_curr_map["min_cost"]:
        by_curr_map["min_cost"] = curr_best
        by_curr_map["best_solution"] = exp_data["best_sol"]
    cost_improvement = get_cost_improvement(exp_data)
    iter_n = len(exp_data["costs"]["current_best"])
    if map_path not in cnt_map:
        cnt_map[map_path] = 0
    map_cnt = cnt_map[map_path]
    for k, new_v in zip(
        ("mean_exec_t", "mean_cost_improvement", "mean_iters", "min_cost_mean"),
        (exp_data["exec_time"], cost_improvement, iter_n, by_curr_map["min_cost"]),
    ):
        by_curr_map[k] = (map_cnt * by_curr_map[k] + new_v) / (map_cnt + 1)
    min_costs = min_cost_accumulator[map_path]
    if len(min_costs) == 1:
        by_curr_map["min_cost_stddev"] = float("nan")
    else:
        by_curr_map["min_cost_stddev"] = stdev(min_costs)
    cnt_map[map_path] = map_cnt + 1


def analyze_data(dir_path: Union[str, Path]) -> dict[str, Any]:
    # by_map - by environment data file
    # by_destination_n - by map size (no. of destinations)
    # by_population_size - by GA's population size
    # each dict in "by_population_size" and "by_destination_n" has fields:
    # - "mean_exec_t",
    # - "mean_cost_improvement",
    # - "mean_iters",
    analyzed = {
        "by_destination_n": {},
        "by_population_size": {},
        "by_map": {},
    }
    by_dest_n_cnt = {}
    by_pop_size_cnt = {}
    by_map_cnt = {}
    by_map_cost_accumulator = {}
    for results_f in track(
        tuple(f for f in Path(dir_path).iterdir() if f.is_file()),
        description="Processing files...",
    ):
        with results_f.open("r") as f:
            exp_data = json.load(f)
            if "exception" in exp_data and exp_data["exception"]:
                continue
            with open(exp_data["experiment_config_path"], "r") as conf_f:
                conf_data = json.load(conf_f)
            for criterium, cnt_map in zip(
                ("by_destination_n", "by_population_size"),
                (by_dest_n_cnt, by_pop_size_cnt),
            ):
                analyze_by_criterium(criterium, exp_data, conf_data, analyzed, cnt_map)
                analyze_by_map(
                    conf_data["map_path"],
                    exp_data,
                    analyzed,
                    by_map_cnt,
                    by_map_cost_accumulator,
                )

    return analyzed


def generate_result_plots():
    # TODO
    ...
