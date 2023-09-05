from functools import wraps

def kwargs_decorator(dict_kwargs):
    def wrapper(f):
        @wraps(f)
        def inner_wrapper(*args, **kwargs):
            new_kwargs = {**dict_kwargs, **kwargs}
            return f(*args, **new_kwargs)

        return inner_wrapper

    return wrapper