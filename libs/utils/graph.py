import numpy as np


def graph_cycle_greedy_nn(cost_mx: np.ndarray, initial_vx: int) -> list[int]:
    vx_num = cost_mx.shape[0]
    vxs = list(range(vx_num))

    Vx = int
    visited_neighbours: dict[Vx, set[Vx]] = {vx: set() for vx in vxs}

    # ommit the first nonpositive elements
    neighbours_by_distance: dict[Vx, list[Vx]] = {}

    def register_neighbours_by_distance(vx: int):
        """
        Registers `vx`'s neighbours sorted by distance at neighbours_by_distance
        """
        neighbour_distances = cost_mx[vx, :]
        reachable_sorted_neighbours = sorted(
            n for d, n in zip(neighbour_distances, vxs) if d > 0
        )
        reachable_sorted_neighbours = [
            n
            for d, n in sorted(
                (d, n) for d, n in zip(neighbour_distances, vxs) if d > 0
            )
        ]
        neighbours_by_distance[vx] = reachable_sorted_neighbours

    current_vx = initial_vx
    solution = [-1 for _ in range(vx_num)]
    i = 0

    while i < vx_num and i > -1:
        solution[i] = current_vx
        if current_vx not in neighbours_by_distance.keys():
            register_neighbours_by_distance(current_vx)

        neighbours_to_visit = [
            n for n in neighbours_by_distance[current_vx] if n not in solution
        ]

        try:
            next_to_visit = next(
                n
                for n in neighbours_to_visit
                if n not in visited_neighbours[current_vx]
            )
        except StopIteration:
            # go back
            i -= 1
            current_vx = solution[i]
            continue

        # advance
        visited_neighbours[current_vx].add(next_to_visit)
        current_vx = next_to_visit
        i += 1

    return solution
