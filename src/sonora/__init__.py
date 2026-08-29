from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sonora")
except PackageNotFoundError:
    __version__ = "2.7.2"

__author__ = "Daniel Radu"
