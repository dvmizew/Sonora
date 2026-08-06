from importlib.metadata import version

try:
    __version__ = version("sonora")
except Exception:
    __version__ = "0.1.0"

__author__ = "Daniel Radu"
