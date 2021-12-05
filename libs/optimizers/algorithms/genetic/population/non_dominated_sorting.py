# TODO remove module

def fast_nds():
    """
    Based on implementation at
    https://discourse.julialang.org/t/fast-non-dominated-sorting/24164/4
    by Igor_Douven
    """


def dominates(x, y):
    # dominates(x, y)
    # = all(i -> x[i] <= y[i], eachindex(x))
    # && any(i -> x[i] < y[i], eachindex(x))

    return all(x[i] <= y[i] for i in range(len(x)))
