"""
Cost functions for different algorithms and scenarios.
"""


from .base import *
from .. import Environment, EnvironmentStatic, EnvironmentDynamic, EnvironmentTSPSimple
from ...solution.representation import (
    SolutionRepresentation as SolutionRepresentation,
    SolutionRepresentationTSP as SolutionRepresentationTSP,
)
