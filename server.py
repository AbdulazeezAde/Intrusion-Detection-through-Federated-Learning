import torch
import torch.nn as nn
from typing import List, OrderedDict, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
import numpy as np


class FederatedServer:
    """
    Centralized Server for Federated Learning Intrusion Detection System.
    
    Implements the Federated Averaging (FedAvg) algorithm with weighted aggregation
    based on client sample counts.
    """
    
    def __init__(self, global_model: torch.nn.Module, device: torch.device = None):
        """
        Initialize the Federated Server.
        
        Args:
            global_model: The global PyTorch model to be trained.
            device: The device (CPU/CUDA) to run evaluation on.
        """
        self.global_model = global_model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
    def aggregate(
        self, 
        client_weights: List[OrderedDict[str, torch.Tensor]], 
        client_sample_counts: List[int]
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Perform Federated Averaging (FedAvg) to aggregate client weights.
        
        Formula: Global_Weight = Sum( (n_k / N) * Local_Weight_k )
        
        Args:
            client_weights: List of state dictionaries from participating clients.
            client_sample_counts: List of sample counts corresponding to each client.
            
        Returns:
            OrderedDict: The aggregated global state dictionary.
        """
        if len(client_weights) != len(client_sample_counts):
            raise ValueError("Length mismatch between weights and sample counts.")
        if len(client_weights) == 0:
            raise ValueError("No client weights provided.")
            
        total_samples = sum(client_sample_counts)
        aggregated_state_dict = OrderedDict()
        param_names = list(client_weights[0].keys())
        
        for param_name in param_names:
            accumulated_weight = torch.zeros_like(client_weights[0][param_name], dtype=torch.float64)
            
            for i, client_state in enumerate(client_weights):
                n_k = client_sample_counts[i]
                weight_ratio = n_k / total_samples
                client_param = client_state[param_name].to(torch.float64)
                accumulated_weight += weight_ratio * client_param
            
            aggregated_state_dict[param_name] = accumulated_weight.to(client_weights[0][param_name].dtype)
            
        return aggregated_state_dict
    
    def evaluate(self, test_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the global model on the test dataset.
        
        Args:
            test_dataloader: PyTorch DataLoader containing the test data.
            
        Returns:
            Dict: Dictionary containing accuracy, precision, recall, f1_score, and confusion_matrix.
        """
        self.global_model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
