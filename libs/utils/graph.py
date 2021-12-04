import numpy as np


class NoGreedySolutionError(Exception):
    ...


def graph_cycle_greedy_nn(
    cost_mx: np.ndarray, initial_vx: int, forbidden_val: float = -1
) -> list[int]:
    vx_num = cost_mx.shape[0]
    vxs = list(range(vx_num))

    Vx = int
    visited_neighbours: dict[Vx, set[Vx]] = {vx: set() for vx in vxs}

    neighbours_by_distance: dict[Vx, list[Vx]] = {}

    def register_neighbours_by_distance(vx: int):
        """
        Registers `vx`'s neighbours sorted by distance at neighbours_by_distance
        """
        neighbour_distances = cost_mx[vx, :]
        reachable_sorted_neighbours = [
            n
            for _, n in sorted(
                (d, n)
                for d, n in zip(neighbour_distances, vxs)
                if d > 0 and d != forbidden_val and np.isfinite(d)
            )
        ]
        neighbours_by_distance[vx] = reachable_sorted_neighbours

    current_vx = initial_vx
    solution = [-1 for _ in range(vx_num + 1)]
    i = 0
    solution_len = len(solution)
    last_i = solution_len - 1

    while i < solution_len and i > -1:
        solution[i] = current_vx
        if current_vx not in neighbours_by_distance.keys():
            register_neighbours_by_distance(current_vx)

        reachable_neighs_sorted = [
            n for n in neighbours_by_distance[current_vx] if n not in solution
        ]

        if i != last_i:
            next_to_visit = next(
                (
                    n
                    for n in reachable_neighs_sorted
                    if n not in visited_neighbours[current_vx]
                ),
                None,
            )
        else:
            next_to_visit = (
                initial_vx if initial_vx in reachable_neighs_sorted else None
            )

        if next_to_visit is None:
            # go back
            i -= 1
            current_vx = solution[i]
            continue

        # advance
        visited_neighbours[current_vx].add(next_to_visit)
        current_vx = next_to_visit
        i += 1

    solution[-1] = initial_vx

    if any(v == -1 for v in solution):
        raise NoGreedySolutionError(str(solution))

    return solution
