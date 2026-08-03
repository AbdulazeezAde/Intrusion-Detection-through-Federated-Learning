"""
Graph Utilities for Federated Graph Neural Networks (FedGNN)
Converts NSL-KDD tabular data into graph structures for GNN training.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_graph_from_flow_data(
    X: np.ndarray, 
    y: np.ndarray, 
    k_neighbors: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a graph from NSL-KDD flow data using k-Nearest Neighbors.
    
    Each sample becomes a node, and edges connect each node to its k nearest neighbors.
    
    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels vector (n_samples,).
        k_neighbors: Number of neighbors for edge construction.
        
    Returns:
        edge_index: Tensor of shape (2, num_edges) containing edge connections.
        node_features: Tensor of shape (n_samples, n_features) containing node features.
        labels: Tensor of shape (n_samples,) containing node labels.
        mask: Boolean tensor indicating valid nodes.
    """
    n_samples = X.shape[0]
    
    # Ensure features are float32 for GNN compatibility
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    
    # Build k-NN graph
    # Use fewer neighbors if dataset is small
    actual_k = min(k_neighbors, n_samples - 1)
    
    if actual_k < 1:
        # Handle edge case: single sample
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        nbrs = NearestNeighbors(n_neighbors=actual_k + 1, metric='euclidean', n_jobs=-1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Create edge list (bidirectional)
        src_nodes = []
        dst_nodes = []
        
        for i in range(n_samples):
            for j_idx in range(1, actual_k + 1):  # Skip self (index 0)
                neighbor_idx = indices[i, j_idx]
                src_nodes.append(i)
                dst_nodes.append(neighbor_idx)
                # Add reverse edge for undirected graph
                src_nodes.append(neighbor_idx)
                dst_nodes.append(i)
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    # Convert to tensors
    node_features = torch.FloatTensor(X)
    labels = torch.LongTensor(y) if y.dtype == np.int64 or y.dtype == np.int32 else torch.LongTensor(y.astype(np.int64))
    mask = torch.ones(n_samples, dtype=torch.bool)
    
    return edge_index, node_features, labels, mask


def create_client_graph_data(
    X_client: np.ndarray, 
    y_client: np.ndarray, 
    k_neighbors: int = 10
) -> dict:
    """
    Create graph data structure for a single client.
    
    Args:
        X_client: Client's feature matrix.
        y_client: Client's labels.
        k_neighbors: Number of neighbors for graph construction.
        
    Returns:
        Dictionary containing graph data compatible with PyTorch Geometric.
    """
    edge_index, node_features, labels, mask = build_graph_from_flow_data(
        X_client, y_client, k_neighbors
    )
    
    graph_data = {
        'edge_index': edge_index,
        'x': node_features,
        'y': labels,
        'mask': mask,
        'num_nodes': len(node_features)
    }
    
    return graph_data


def partition_data_for_fedgnn(
    X: np.ndarray, 
    y: np.ndarray, 
    num_clients: int, 
    iid: bool = True,
    k_neighbors: int = 10
) -> List[dict]:
    """
    Partition data for Federated GNN simulation.
    
    Args:
        X: Global feature matrix.
        y: Global labels.
        num_clients: Number of clients.
        iid: Whether to use IID partitioning.
        k_neighbors: Number of neighbors for graph construction.
        
    Returns:
        List of graph data dictionaries, one per client.
    """
    from data_pipeline import partition_data_for_fl
    
    # First partition the tabular data
    partitions = partition_data_for_fl(X, y, num_clients, iid=iid)
    
    # Convert each partition to graph format
    client_graphs = []
    for X_client, y_client in partitions:
        graph_data = create_client_graph_data(X_client, y_client, k_neighbors)
        client_graphs.append(graph_data)
    
    return client_graphs
