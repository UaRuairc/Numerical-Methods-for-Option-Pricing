from time import perf_counter
from contextlib import contextmanager

@contextmanager
def timed():
    t_ms = []
    start = perf_counter()
    try:
        yield t_ms
    finally:
        time_taken = perf_counter() - start
        t_ms.append(time_taken*1000)