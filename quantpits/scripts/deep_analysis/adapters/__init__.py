"""
Adapter registry for the MAS Deep Analysis Feedback Loop.

Maps action_type strings to their corresponding adapter classes.
Phase 4 only includes TrainingAdapter (for adjust_hyperparam).
"""

# Adapter type → adapter class mapping (populated at import time)
ADAPTER_REGISTRY = {}


def register_adapter(action_type: str):
    """Decorator to register an adapter class for a given action_type."""
    def decorator(cls):
        ADAPTER_REGISTRY[action_type] = cls
        return cls
    return decorator


def get_adapter(action_type: str):
    """Look up the adapter class for a given action_type.

    Returns:
        The adapter class, or None if no adapter is registered.
    """
    return ADAPTER_REGISTRY.get(action_type)


# Import adapters so they self-register via @register_adapter
from quantpits.scripts.deep_analysis.adapters.training_adapter import TrainingAdapter  # noqa: F401, E402
from quantpits.scripts.deep_analysis.adapters.model_selection_adapter import ModelSelectionAdapter  # noqa: F401, E402
from quantpits.scripts.deep_analysis.adapters.data_split_adapter import DataSplitAdapter  # noqa: F401, E402
