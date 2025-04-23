# src/priming_evaluation/__init__.py

"""
Package for native priming evaluation logic.
Replaces functionality previously dependent on the legacy `diagnnose` library.
"""

# Import key functions to make them available when importing the package
from .data_loader import create_priming_dataloader
from .evaluator import run_native_priming_eval

# Optionally define what gets imported with 'from . import *'
__all__ = [
    "create_priming_dataloader",
    "run_native_priming_eval",
]