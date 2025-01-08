try:
    # for Python before 3.11
    # tomllib is expected to not exists as tomllib points to this file
    from pip._vendor.tomli import *
except ModuleNotFoundError:
    from tomli import *
