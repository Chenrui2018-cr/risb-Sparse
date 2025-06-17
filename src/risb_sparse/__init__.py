"""Library to use rotationally invariant slave-boson mean-field theory for lattice strongly correlated electrons."""

from importlib.metadata import PackageNotFoundError, version

from .solve_lattice_ED import LatticeSolver  # noqa: F401

try:
    __version__ = version("risb")
except PackageNotFoundError:
    __version__ = "unknown version"
