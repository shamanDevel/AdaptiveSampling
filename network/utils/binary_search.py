from typing import Callable

def binarySearch(
    f : Callable[[float], float],
    start : float,
    target : float,
    num_steps : int,
    min_value : float = None,
    max_value : float = None):
    """
    Binary search implementation.
    Returns the value so that f(value) is close to target.
    It assumes that f is monotonically increasing
    """

    # search min/max
    v = f(start)
    if v == target: return start
    if v < target:
        # search upper bound
        lower_bound = start
        for i in range(num_steps):
            if max_value is None:
                start *= 2
            else:
                start = 0.5 * (max_value + start)
            v = f(start)
            if v > target:
                upper_bound = start
                num_steps -= i + 1
                break
            else:
                lower_bound = start
        else:
            return start # unable to find upper bound
    else:
        # search lower bound
        upper_bound = start
        for i in range(num_steps):
            if max_value is None:
                start /= 2
            else:
                start = 0.5 * (min_value + start)
            v = f(start)
            if v < target:
                lower_bound = start
                num_steps -= i + 1
                break
            else:
                upper_bound = start
        else:
            return start # unable to find lower bound

    # run loop
    mid = 0.5 * (lower_bound + upper_bound)
    for i in range(num_steps):
        v = f(mid)
        if v > target:
            upper_bound = mid
        else:
            lower_bound = mid
        mid = 0.5 * (lower_bound + upper_bound)
    
    return mid

if __name__ == "__main__":
    import math
    def f(x):
        v = 1.2*x + math.sin(x)
        print("f(%f)=%f"%(x,v))
        return v

    def run(start, target, min, max):
        print()
        print("Run, start={}, target={}, min={}, max={}".format(start, target, min, max))
        result = binarySearch(f, start, target, 10, min, max)
        print("Result: {} with value {}".format(result, f(result)))

    run(0.5, 4, 0, None)
    run(0.5, 0.1, 0, None)
    run(10, 4, None, None)
    run(0.5, 0.9, 0, 1)
