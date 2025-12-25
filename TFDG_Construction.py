import pandas as pd
import torch
import torch.fft as fft
import networkx as nx
from scipy.fftpack import next_fast_len
from sklearn.neighbors import NearestNeighbors
import pickle
from torch_geometric.data import Data
import numpy as np

"""
Time-Frequency Domain Graph (TFDG) Construction

Based on the methodology in the paper:
1. Fourier autocorrelation for node feature extraction
2. Information-theoretic conditional entropy for edge weight computation

Theoretical Foundation:
- Fourier autocorrelation maps signal energy distribution to delay domain via power spectrum inverse transform
- Peak positions encode dominant frequencies of periodic oscillations
- Decay rates reflect impulse transient durations
- Conditional entropy quantifies information dependency strength between sensors
"""


def fourier_autocorrelation(v):
    """
    Compute node features via Fourier autocorrelation
    
    Mathematical derivation:
    1. Zero-pad signal v[n] (n ∈ [0, N-1]) to FFT length N_opt
    2. Apply Fourier transform: V[k] = Σ v_padded[n] * e^(-j2πkn/N_opt)
    3. Compute power spectral density: S[k] = |V[k]|^2
    4. Inverse transform to obtain autocorrelation sequence s[n]
    5. Extract last N samples: V[m] = s[N_opt - N + m]
    
    Args:
        v: Sensor-acquired signal tensor, shape: (N,)
    
    Returns:
        Node feature vector, shape: (N,)
    """
    N = v.size(0)
    
    # Step 1: Compute optimal FFT length
    N_opt = next_fast_len(N)
    
    # Step 2: Fourier transform (use rfft for single signal)
    V = fft.rfft(v, n=N_opt)
    
    # Step 3: Compute power spectral density S[k] = |V[k]|^2
    S = torch.abs(V) ** 2
    
    # Step 4: Inverse Fourier transform to get autocorrelation sequence
    s = fft.irfft(S, n=N_opt)
    
    # Step 5: Extract last N samples to form node features
    # V[m] = s[N_opt - N + m], m = 0, 1, ..., N-1
    node_feature = s[N_opt - N:]
    
    return node_feature


def estimate_conditional_entropy_from_samples(features_i, features_j, bins=50):
    """
    Estimate conditional entropy H(v_i | v_j) from multiple samples
    
    Mathematical formulas:
    P(v_i, v_j) = (1/N_s) * Σ H(v_i^(n), v_j^(n))
    P(v_i | v_j) = P(v_i, v_j) / P(v_j)
    H(v_i | v_j) = -Σ P(v_i, v_j) log P(v_i | v_j)
    
    Args:
        features_i: Features of sensor i across all samples, shape: (N_s, feature_dim)
        features_j: Features of sensor j across all samples, shape: (N_s, feature_dim)
        bins: Number of bins for histogram
    
    Returns:
        Conditional entropy value
    """
    # Flatten features for histogram computation
    x = features_i.flatten()
    y = features_j.flatten()
    
    # Compute normalized 2D histogram to estimate joint distribution
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_2d = hist_2d / np.sum(hist_2d)  # Normalize
    
    # Compute marginal distribution P(v_j)
    p_y = np.sum(hist_2d, axis=0)
    
    # Compute conditional entropy H(v_i | v_j)
    conditional_entropy = 0.0
    for j in range(bins):
        if p_y[j] > 0:
            # Conditional distribution P(v_i | v_j)
            p_x_given_y = hist_2d[:, j] / p_y[j]
            # Remove zero probabilities to avoid log(0)
            p_x_given_y = p_x_given_y[p_x_given_y > 0]
            # Accumulate -Σ P(v_i | v_j) log P(v_i | v_j)
            conditional_entropy += p_y[j] * (-np.sum(p_x_given_y * np.log(p_x_given_y)))
    
    return conditional_entropy


def construct_tfdg(file_path, sheet_names, k_neighbors=4):
    """
    Construct Time-Frequency Domain Graph (TFDG)
    
    Pipeline:
    1. Load multi-sensor data
    2. For each sample:
       a. Extract node features using Fourier autocorrelation
       b. Establish candidate edges by selecting k nearest neighbors via Euclidean distance
       c. Estimate conditional entropy across all samples
       d. Compute symmetrized edge weights: w_ij = 1 / (1 + (H(v_i|v_j) + H(v_j|v_i))/2)
    3. Construct weighted adjacency matrix and return graph structure
    
    Args:
        file_path: Path to Excel file
        sheet_names: List of sheet names corresponding to sensors
        k_neighbors: Number of neighbors in KNN
    
    Returns:
        List of graphs, one graph per sample
    """
    # Read all sheet data (first row not used as column names)
    sheets_data = {sheet: pd.read_excel(file_path, sheet_name=sheet, header=None) 
                   for sheet in sheet_names}
    
    num_samples = len(sheets_data[sheet_names[0]])
    num_sensors = len(sheet_names)
    
    # Step 1: Extract node features for all samples
    all_node_features = []  # shape: (num_samples, num_sensors, feature_dim)
    
    for i in range(num_samples):
        sample_features = []
        for sheet_name in sheet_names:
            # Read sensor signal (last column is label, not included in signal)
            signal = sheets_data[sheet_name].iloc[i, :-1].values
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            
            # Extract node features using Fourier autocorrelation
            node_feature = fourier_autocorrelation(signal_tensor)
            sample_features.append(node_feature.numpy())
        
        all_node_features.append(sample_features)
    
    all_node_features = np.array(all_node_features)  # shape: (num_samples, num_sensors, feature_dim)
    
    # Step 2: Select k nearest neighbors for each node based on Euclidean distance
    # Use mean features across all samples to establish topology
    mean_features = np.mean(all_node_features, axis=0)  # shape: (num_sensors, feature_dim)
    
    knn = NearestNeighbors(n_neighbors=min(k_neighbors, num_sensors))
    knn.fit(mean_features)
    distances, indices = knn.kneighbors(mean_features)
    
    # Step 3: Estimate conditional entropy across all samples and compute edge weights
    edge_weights = {}
    
    for i in range(num_sensors):
        for neighbor_idx in indices[i]:
            if neighbor_idx != i:
                # Extract features of sensor i and j across all samples
                features_i = all_node_features[:, i, :]  # shape: (num_samples, feature_dim)
                features_j = all_node_features[:, neighbor_idx, :]
                
                # Compute bidirectional conditional entropy
                H_i_given_j = estimate_conditional_entropy_from_samples(features_i, features_j)
                H_j_given_i = estimate_conditional_entropy_from_samples(features_j, features_i)
                
                # Symmetrized edge weight: w_ij = 1 / (1 + (H(v_i|v_j) + H(v_j|v_i))/2)
                avg_conditional_entropy = (H_i_given_j + H_j_given_i) / 2.0
                weight = 1.0 / (1.0 + avg_conditional_entropy)
                
                edge_weights[(i, neighbor_idx)] = weight
    
    # Step 4: Construct graph structure for each sample
    all_graphs = []
    
    for sample_idx in range(num_samples):
        G = nx.Graph()
        
        # Add nodes with their features
        for sensor_idx, sheet_name in enumerate(sheet_names):
            node_name = f"{sheet_name}"
            node_feature = all_node_features[sample_idx, sensor_idx, :]
            G.add_node(node_name, feature=node_feature)
        
        # Add edges with their weights
        for (i, j), weight in edge_weights.items():
            node_i = sheet_names[i]
            node_j = sheet_names[j]
            G.add_edge(node_i, node_j, weight=weight)
        
        # Add graph label (from last column of first sheet)
        label = sheets_data[sheet_names[0]].iloc[sample_idx, -1]
        G.graph['label'] = label
        
        all_graphs.append(G)
    
    return all_graphs


def convert_to_pyg_data(nx_graph):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Args:
        nx_graph: NetworkX graph object
    
    Returns:
        PyG Data object
    """
    # Get node features
    node_features = [data['feature'] for _, data in nx_graph.nodes(data=True)]
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Create mapping from node names to IDs
    node_id_map = {node: i for i, node in enumerate(nx_graph.nodes)}
    
    # Get edge indices and edge weights
    edges = list(nx_graph.edges)
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_ids = [(node_id_map[u], node_id_map[v]) for u, v in edges]
        edge_index = torch.tensor(edge_ids, dtype=torch.long).t().contiguous()
        
        edge_weights = [nx_graph[u][v]['weight'] for u, v in edges]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Get graph label
    y = torch.tensor([nx_graph.graph['label']], dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


if __name__ == "__main__":
    # File paths and sensor sheet names
    train_file_path = 'train.xlsx'
    test_file_path = 'test.xlsx'
    sheet_names = ['Sensor3', 'Sensor4', 'Sensor5', 'Sensor6']
    
    # Construct TFDG for training set
    print("Constructing Time-Frequency Domain Graphs for training set...")
    train_graphs = construct_tfdg(train_file_path, sheet_names, k_neighbors=4)
    print(f"Generated {len(train_graphs)} graphs for training set")
    
    # Construct TFDG for test set
    print("\nConstructing Time-Frequency Domain Graphs for test set...")
    test_graphs = construct_tfdg(test_file_path, sheet_names, k_neighbors=4)
    print(f"Generated {len(test_graphs)} graphs for test set")
    
    # Print training set graph information
    print("\nTraining set graph information:")
    for i, graph in enumerate(train_graphs[:5]):
        num_nodes = len(graph.nodes)
        node_feature_length = len(next(iter(graph.nodes(data=True)))[1]['feature'])
        num_edges = len(graph.edges)
        print(f"Graph {i+1}: nodes={num_nodes}, feature_dim={node_feature_length}, edges={num_edges}")
    
    # Print test set graph information
    print("\nTest set graph information:")
    for i, graph in enumerate(test_graphs[:5]):
        num_nodes = len(graph.nodes)
        node_feature_length = len(next(iter(graph.nodes(data=True)))[1]['feature'])
        num_edges = len(graph.edges)
        print(f"Graph {i+1}: nodes={num_nodes}, feature_dim={node_feature_length}, edges={num_edges}")
    
    # Convert to PyG format and save
    print("\nConverting and saving graph structures...")
    train_pyg_data = [convert_to_pyg_data(g) for g in train_graphs]
    with open('train_graphs.pkl', 'wb') as f:
        pickle.dump(train_pyg_data, f)
    
    test_pyg_data = [convert_to_pyg_data(g) for g in test_graphs]
    with open('test_graphs.pkl', 'wb') as f:
        pickle.dump(test_pyg_data, f)
    
    print("Graph structures saved as train_graphs.pkl and test_graphs.pkl")
    print("\nTime-Frequency Domain Graph construction completed!")
