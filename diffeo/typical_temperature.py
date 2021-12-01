import math


def typical_temperature(delta, cut, n):
    """
    Taken from:
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L25
    """
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return 4 * delta ** 2 / (math.pi * n ** 2 * log)
