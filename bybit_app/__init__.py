"""Bybit Spot Guardian package initialisation."""

from .utils.error_handling import install_global_exception_handlers

# Ensure global exception hooks are active as soon as the package is imported.
install_global_exception_handlers()

