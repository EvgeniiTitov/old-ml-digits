import functools
import time
import typing as t


def timer(func: t.Callable) -> t.Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> t.Any:
        s = time.perf_counter()
        res = func(*args, **kwargs)
        print(
            f"Function {func.__name__} took: "
            f"{time.perf_counter() - s: .4f} seconds"
        )
        return res

    return wrapper
