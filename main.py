import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, OrderedDict
import gc
from sklearn.model_selection import StratifiedKFold

# Import local modules
from data_pipeline import load_and_preprocess_nslkdd, partition_data_for_fl
from model import IDS_NeuralNet
from client import FederatedClient
from server import FederatedServer


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Helper function to create a PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def stratified_cross_validation_split(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Generator for stratified k-fold cross-validation that handles imbalanced classes.
    Maintains class distribution across all folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def run_federated_experiment(
    train_partitions: List[Tuple[np.ndarray, np.ndarray]],
    test_dataloader: DataLoader,
    input_dim: int,
    num_rounds: int = 15,
    local_epochs: int = 5,
    learning_rate: float = 0.0005,
    use_smote: bool = False,
    use_transfer_learning: bool = False,
    pre_trained_weights: OrderedDict = None,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Run a complete Federated Learning experiment with optional SMOTE and Transfer Learning.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Global Model
    global_model = IDS_NeuralNet(input_dim=input_dim)
    
    # Transfer Learning: Initialize with pre-trained weights if available
    if use_transfer_learning and pre_trained_weights is not None:
        global_model.load_state_dict(pre_trained_weights)
        print("✓ Initialized with pre-trained weights (Transfer Learning)")
    
    global_model.to(device)
    
    # Initialize Server
    server = FederatedServer(global_model=global_model, device=device)
    
    # Initialize Clients
    clients = []
    for i, (X_client, y_client) in enumerate(train_partitions):
        client_dataloader = create_dataloader(X_client, y_client, batch_size=64, shuffle=True)
        client = FederatedClient(
            client_id=i,
            dataloader=client_dataloader,
            device=device,
            model=global_model
        )
        clients.append(client)
    
    # Metrics storage
    history = {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}
    
    exp_name = "WITH SMOTE + Transfer Learning" if (use_smote and use_transfer_learning) else \
               "WITH SMOTE" if use_smote else \
               "WITH Transfer Learning" if use_transfer_learning else "WITHOUT SMOTE"
    
    print(f"--- Starting Experiment: {exp_name} ---")
    
    for round_idx in range(1, num_rounds + 1):
        print(f"Round {round_idx}/{num_rounds}...", end=" ")
        
        # 1. Local Training Phase
        client_weights = []
        client_sample_counts = []
        
        for client in clients:
            weights, num_samples = client.train_local(
                epochs=local_epochs, 
                lr=learning_rate, 
                apply_smote=use_smote, mu=0.01
            )
            client_weights.append(weights)
            client_sample_counts.append(num_samples)
        
        # 2. Aggregation Phase (FedAvg)
        aggregated_weights = server.aggregate(client_weights, client_sample_counts)
        
        # Update global model weights
        server.global_model.load_state_dict(aggregated_weights)
        
        # 3. Evaluation Phase
        metrics = server.evaluate(test_dataloader)
        
        history['accuracy'].append(metrics['accuracy'])
        history['f1_score'].append(metrics['f1_score'])
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        # Memory management
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Final Evaluation
    final_metrics = server.evaluate(test_dataloader)
    print(f"Final Confusion Matrix ({exp_name}):\n{final_metrics['confusion_matrix']}")
    
    return history, server.global_model.get_weights()


def plot_results(results_dict: Dict, save_path: str = "fl_results_comparison.png"):
    """Plot and compare the results of multiple experiments."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Baseline': 'gray', 'SMOTE': 'blue', 'Transfer Learning': 'green', 'SMOTE + TL': 'red'}
    markers = {'Baseline': 'o', 'SMOTE': 's', 'Transfer Learning': '^', 'SMOTE + TL': 'D'}
    
    for name, results in results_dict.items():
        rounds = list(range(1, len(results['accuracy']) + 1))
        ax[0].plot(rounds, results['accuracy'], label=name, marker=markers.get(name, 'o'), 
                   linestyle='-', color=colors.get(name, None), linewidth=2)
        ax[1].plot(rounds, results['f1_score'], label=name, marker=markers.get(name, 'o'), 
                   linestyle='-', color=colors.get(name, None), linewidth=2)
    
    ax[0].set_title('Global Accuracy over Communication Rounds')
    ax[0].set_xlabel('Communication Round')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].set_title('Global F1-Score over Communication Rounds')
    ax[1].set_xlabel('Communication Round')
    ax[1].set_ylabel('F1-Score')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Results plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 5
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========================================================================
    # 1. Data Loading and Preprocessing
    # ========================================================================
    print("Loading and preprocessing NSL-KDD dataset...")
    TRAIN_PATH = "data/KDDTrain+.csv"
    TEST_PATH = "data/KDDTest+.csv"
    
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_nslkdd(TRAIN_PATH, TEST_PATH)
        input_dim = X_train.shape[1]
        print(f"Data loaded. Input dimension: {input_dim}")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    except FileNotFoundError:
        print(f"Error: Dataset files not found. Run 'python data_setup.py' first.")
        exit(1)
    
    test_dataloader = create_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========================================================================
    # 2. Data Partitioning for FL Simulation
    # ========================================================================
    print(f"Partitioning data for {NUM_CLIENTS} clients (IID)...")
    np.random.seed(42)
    partitions_base = partition_data_for_fl(X_train, y_train, num_clients=NUM_CLIENTS, iid=True)
    np.random.seed(42)
    partitions_smote = partition_data_for_fl(X_train, y_train, num_clients=NUM_CLIENTS, iid=True)
    
    # ========================================================================
    # 3. Transfer Learning: Pre-train on full dataset
    # ========================================================================
    print("\n--- Pre-training Global Model (Transfer Learning Step) ---")
    pretrain_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    pretrain_model = IDS_NeuralNet(input_dim=input_dim).to(device)
    
    # Class-weighted loss for pre-training
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([len(y_train)/(2*class_counts[i]) if class_counts[i]>0 else 1.0 
                                       for i in range(2)]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(pretrain_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    for epoch in range(10):
        pretrain_model.train()
        total_loss = 0
        for inputs, labels in pretrain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(pretrain_model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(pretrain_loader))
    
    pre_trained_weights = pretrain_model.get_weights()
    print("✓ Pre-training complete. Weights ready for transfer learning.\n")
    
    # ========================================================================
    # 4. Run Experiments
    # ========================================================================
    results_dict = {}
    
    # Experiment 1: Baseline (No SMOTE, No Transfer Learning)
    results_dict['Baseline'], _ = run_federated_experiment(
        train_partitions=partitions_base,
        test_dataloader=test_dataloader,
        input_dim=input_dim,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        use_smote=False,
        use_transfer_learning=False,
        device=device
    )
    
    # Experiment 2: With SMOTE only
    results_dict['SMOTE'], _ = run_federated_experiment(
        train_partitions=partitions_smote,
        test_dataloader=test_dataloader,
        input_dim=input_dim,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        use_smote=True,
        use_transfer_learning=False,
        device=device
    )
    
    # Experiment 3: With Transfer Learning only
    results_dict['Transfer Learning'], _ = run_federated_experiment(
        train_partitions=partitions_base,
        test_dataloader=test_dataloader,
        input_dim=input_dim,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        use_smote=False,
        use_transfer_learning=True,
        pre_trained_weights=pre_trained_weights,
        device=device
    )
    
    # Experiment 4: With SMOTE + Transfer Learning (IDS-INT approach)
    results_dict['SMOTE + TL'], _ = run_federated_experiment(
        train_partitions=partitions_smote,
        test_dataloader=test_dataloader,
        input_dim=input_dim,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        use_smote=True,
        use_transfer_learning=True,
        pre_trained_weights=pre_trained_weights,
        device=device
    )
    
    # ========================================================================
    # 5. Visualization and Summary
    # ========================================================================
    plot_results(results_dict)
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*70)
    for name, results in results_dict.items():
        print(f"\n{name}:")
        print(f"  Final Accuracy: {results['accuracy'][-1]:.4f}")
        print(f"  Final F1-Score: {results['f1_score'][-1]:.4f}")
        print(f"  Final Precision: {results['precision'][-1]:.4f}")
        print(f"  Final Recall: {results['recall'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)
