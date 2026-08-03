"""
Federated GraphSAGE Model for Intrusion Detection
Implements a Graph Neural Network compatible with Federated Averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict, Dict


class GraphSAGEConv(nn.Module):
    """
    Simplified GraphSAGE convolution layer without external dependencies.
    Implements mean aggregation for neighbor features.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super(GraphSAGEConv, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)  # Concat [self, neighbor_mean]
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GraphSAGE convolution.
        
        Args:
            x: Node features (num_nodes, in_features).
            edge_index: Edge connections (2, num_edges).
            
        Returns:
            Updated node features (num_nodes, out_features).
        """
        num_nodes = x.shape[0]
        device = x.device
        
        # Compute neighbor mean aggregation
        # edge_index[0] -> source, edge_index[1] -> destination
        row, col = edge_index[0], edge_index[1]
        
        # Aggregate messages from neighbors
        neighbor_agg = torch.zeros((num_nodes, x.shape[1]), dtype=x.dtype, device=device)
        neighbor_count = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # Scatter add for neighbor aggregation
        neighbor_agg.index_add_(0, col, x[row])
        neighbor_count.index_add_(0, col, torch.ones_like(row))
        
        # Avoid division by zero
        neighbor_count = torch.clamp(neighbor_count, min=1)
        neighbor_mean = neighbor_agg / neighbor_count.unsqueeze(1).float()
        
        # Concatenate self features with neighbor mean
        combined = torch.cat([x, neighbor_mean], dim=1)
        
        # Apply linear transformation
        out = self.linear(combined)
        return out


class FedGraphSAGE(nn.Module):
    """
    Federated GraphSAGE model for Intrusion Detection.
    
    Architecture:
        - Input: Node features from network flows
        - Layer 1: GraphSAGE Conv + ReLU + BatchNorm + Dropout
        - Layer 2: GraphSAGE Conv + ReLU + BatchNorm + Dropout
        - Global Mean Pooling
        - Linear Classifier
    
    Compatible with Federated Averaging via standard state_dict methods.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        output_dim: int = 1,
        dropout_rate: float = 0.4
    ):
        """
        Initialize the Federated GraphSAGE model.
        
        Args:
            input_dim: Number of input features per node.
            hidden_dim: Dimension of hidden layer embeddings.
            output_dim: Number of output classes (1 for binary with BCE).
            dropout_rate: Dropout probability for regularization.
        """
        super(FedGraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # GraphSAGE Layer 1
        self.conv1 = GraphSAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # GraphSAGE Layer 2
        self.conv2 = GraphSAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GraphSAGE network.
        
        Args:
            x: Node feature matrix (num_nodes, input_dim).
            edge_index: Edge index tensor (2, num_edges).
            
        Returns:
            Logits for each node (num_nodes, output_dim).
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global classification (apply to each node)
        # In graph classification, we would pool; here we classify each flow sample
        logits = self.classifier(x)
        
        return logits.squeeze(-1)  # Shape: (num_nodes,) for BCEWithLogitsLoss
    
    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        """
        Retrieve model weights for federated averaging.
        
        Returns:
            OrderedDict containing the model's state dictionary.
        """
        return self.state_dict()
    
    def set_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        """
        Load weights from federated averaging.
        
        Args:
            state_dict: OrderedDict containing aggregated weights.
        """
        self.load_state_dict(state_dict)


class FedGraphSAGEGraphClassifier(nn.Module):
    """
    Alternative: Graph-level classification version of GraphSAGE.
    Uses global mean pooling to produce a single prediction per graph.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        output_dim: int = 1,
        dropout_rate: float = 0.4
    ):
        super(FedGraphSAGEGraphClassifier, self).__init__()
        
        self.conv1 = GraphSAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GraphSAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional global pooling.
        
        Args:
            x: Node features.
            edge_index: Edge connections.
            batch: Batch vector for graph pooling (optional).
            
        Returns:
            Graph-level predictions.
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        if batch is not None:
            # Global mean pooling over graphs
            num_graphs = batch.max().item() + 1
            graph_features = torch.zeros((num_graphs, x.shape[1]), dtype=x.dtype, device=x.device)
            graph_features.index_add_(0, batch, x)
            count = torch.bincount(batch, minlength=num_graphs).unsqueeze(1).float()
            graph_features = graph_features / torch.clamp(count, min=1)
            x = graph_features
        
        return self.classifier(x).squeeze(-1)
    
    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.state_dict()
    
    def set_weights(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
