from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sonora")
except PackageNotFoundError:
    __version__ = "3.0.0"

__author__ = "Daniel Radu"
