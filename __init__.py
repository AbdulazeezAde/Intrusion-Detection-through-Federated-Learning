"""
Intrusion Detection through Federated Learning
Package initialization with import shims for robust module loading.
"""

import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import shims: Expose key classes/functions at package level
try:
    from .data_pipeline import load_and_preprocess_nslkdd, partition_data_for_fl, create_dataloader
    from .model import IDS_NeuralNet
    from .client import FederatedClient
    from .server import FederatedServer
except ImportError:
    # Fallback if relative imports fail (e.g., running as script)
    try:
        from data_pipeline import load_and_preprocess_nslkdd, partition_data_for_fl, create_dataloader
        from model import IDS_NeuralNet
        from client import FederatedClient
        from server import FederatedServer
    except ImportError:
        pass # Allow loading even if dependencies aren't ready yet

__version__ = "1.0.0"
__all__ = [
    "load_and_preprocess_nslkdd",
    "partition_data_for_fl",
    "create_dataloader",
    "IDS_NeuralNet",
    "FederatedClient",
    "FederatedServer"
]
