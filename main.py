import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pickle
from STAG import STAG_Network
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np

"""
Graph Classifier - Offshore Pipeline Leakage Detection Based on STAG Network

End-to-end multi-valve leakage detection using STAG network:
1. Time-Frequency Domain Graph (TFDG) as input
2. STAG network for feature extraction
3. Graph-level classification for leakage state prediction
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(train_path, test_path):
    """
    Load training and test set graph data
    
    Args:
        train_path: Path to training set pkl file
        test_path: Path to test set pkl file
    
    Returns:
        train_data: List of training graphs
        test_data: List of test graphs
    """
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Move data to device
    for data in train_data + test_data:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
    
    return train_data, test_data


def train():
    """
    Train STAG network for offshore pipeline leakage detection
    
    Training configuration:
    - Batch size: 32
    - Learning rate: 0.001 (Adam optimizer)
    - Training epochs: 100
    - Number of STAG layers: 3
    """
    # Hyperparameter configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    HIDDEN_DIM = 128
    NUM_STAG_LAYERS = 3
    
    # Load data
    print("Loading data...")
    train_data, test_data = load_data('train_graphs.pkl', 'test_graphs.pkl')
    
    # Ensure labels start from 0
    train_labels = torch.cat([data.y for data in train_data])
    test_labels = torch.cat([data.y for data in test_data])
    min_label = min(train_labels.min(), test_labels.min())
    if min_label != 0:
        for data in train_data + test_data:
            data.y = data.y - min_label
    
    # Get number of classes
    num_classes = int(max(train_labels.max(), test_labels.max()) + 1)
    print(f"Dataset info: train_samples={len(train_data)}, test_samples={len(test_data)}, num_classes={num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Check if edge attributes exist (conditional entropy weights from TFDG)
    edge_attr_dim = 0
    if hasattr(train_data[0], 'edge_attr') and train_data[0].edge_attr is not None:
        edge_attr_dim = train_data[0].edge_attr.shape[1]
        print(f"Edge attributes detected: dimension = {edge_attr_dim}")
    
    # Initialize STAG network
    print(f"\nInitializing STAG network...")
    print(f"- Input feature dimension: {train_data[0].x.shape[1]}")
    print(f"- Hidden dimension: {HIDDEN_DIM}")
    print(f"- Number of STAG layers: {NUM_STAG_LAYERS}")
    print(f"- Edge attribute dimension: {edge_attr_dim}")
    print(f"- Number of output classes: {num_classes}")
    
    model = STAG_Network(
        input_dim=train_data[0].x.shape[1], 
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_STAG_LAYERS,
        K=3,
        pseudo_coord_dim=3,
        edge_attr_dim=edge_attr_dim,
        dropout=0.3
    ).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting training...")
    print(f"- Optimizer: Adam (lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"- Loss function: CrossEntropyLoss")
    print(f"- Training epochs: {EPOCHS}")
    print(f"- Batch size: {BATCH_SIZE}")
    print("-" * 80)

    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward propagation with edge attributes (conditional entropy weights from TFDG)
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            out = model(data.x, data.edge_index, data.batch, edge_attr)
            
            # Ensure correct label format
            target = data.y.squeeze().long()
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            # Compute loss and backpropagate
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        # Evaluation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                # Forward propagation with edge attributes
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                out = model(data.x, data.edge_index, data.batch, edge_attr)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        # Compute evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / num_batches
        
        # Print training information
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'Acc: {accuracy:.4f} | '
              f'F1: {f1:.4f} | '
              f'Recall: {recall:.4f}')
        
        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    print("-" * 80)
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    train()
