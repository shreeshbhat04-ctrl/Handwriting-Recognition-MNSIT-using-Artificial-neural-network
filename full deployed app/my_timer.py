import functools
import time

def my_timer(orig_func):
    """
    Decorator to print the execution time of a function.
    """
    @functools.wraps(orig_func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        # Execute the actual function
        value = orig_func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time to run {orig_func.__name__}: {elapsed_time:0.4f} seconds")
        # CRITICAL: We must return the value so Streamlit can receive the text
        return value
    return wrapper_timer