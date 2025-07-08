"""Import helper for optional dependencies like PyMC."""

import warnings
from contextlib import contextmanager
from typing import Generator


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is not available."""
    pass


@contextmanager
def try_import() -> Generator[None, None, None]:
    """
    Context manager for importing optional dependencies.
    
    Usage:
        with try_import() as _import:
            import pymc as pm  # This will be handled gracefully if not available
    """
    try:
        yield
    except ImportError as e:
        # For now, we'll just issue a warning and continue
        # In production, you might want to fall back to simpler algorithms
        warnings.warn(
            f"Optional dependency not available: {e}. "
            "AB-MCTS-M will use simplified logic instead of full PyMC implementation.",
            UserWarning
        )
