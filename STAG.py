import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import degree

"""
Spatial-Topological Adaptive Graph Neural Network (STAG)

Based on the paper methodology, STAG systematically enhances multi-valve leakage detection through three synergistic modules:
1. Pseudo-Coordinate Positional Encoding
2. Spatial-Aware Convolution Kernels
3. Degree-Adaptive Scaling
"""


class STAG_Layer(nn.Module):
    """
    STAG Graph Convolutional Layer
    
    Implements the complete STAG layer as defined in the paper, including:
    - Pseudo-coordinate positional encoding to capture global topological dependencies
    - Spatial-aware convolution kernels to adaptively adjust information propagation strength
    - Degree-adaptive scaling to balance gradient contributions across sensors with different connectivity
    - Residual connections and batch normalization
    """
    
    def __init__(self, input_dim, hidden_dim, K=3, pseudo_coord_dim=3, edge_attr_dim=0):
        """
        Initialize STAG layer
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            K: Maximum number of random walk steps
            pseudo_coord_dim: Pseudo-coordinate dimension
            edge_attr_dim: Edge attribute dimension (e.g., physical distance, conditional entropy weight)
        """
        super(STAG_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.pseudo_coord_dim = pseudo_coord_dim
        self.edge_attr_dim = edge_attr_dim
        
        # Kernel function ψ_θ: MLP maps relative positional encoding to channel-wise modulation weights
        # Input: [edge_attr, P^(0), P^(1), ..., P^(K-1)]
        # Total input dimension: edge_attr_dim + K
        kernel_input_dim = edge_attr_dim + K
        self.kernel_mlp = nn.Sequential(
            nn.Linear(kernel_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Depthwise separable convolution: W ∈ R^(d_h × d_h)
        # Reduces parameter complexity from O(d_h^2 K) to O(d_h K + d_h^2)
        self.depthwise_conv = nn.Linear(input_dim, hidden_dim)
        self.pointwise_conv = nn.Linear(hidden_dim, hidden_dim)
        
        # Degree-adaptive scaling parameters: θ_1, θ_2 ∈ R^(d_h)
        self.deg_coef_1 = nn.Parameter(torch.ones(hidden_dim))
        self.deg_coef_2 = nn.Parameter(torch.ones(hidden_dim))
        
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def generate_pseudo_coordinates(self, edge_index, num_nodes, edge_attr=None):
        """
        Generate pseudo-coordinate positional encoding
        
        Mathematical derivation:
        1. Construct random walk matrix: M = D^(-1)A
        2. Compute k-step transition probabilities: P^(k) = M^k, k = 0, 1, ..., K-1
        3. Symmetric degree normalization: P_ij^scaled = P_ij^(k) * d_i^(1/2) * d_j^(-1/2)
        4. SVD decomposition: P^scaled = U Σ V^T
        5. Extract pseudo-coordinates: C = U_{:,1:d} Σ_{1:d,1:d}
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Total number of nodes
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional, e.g., conditional entropy weights)
        
        Returns:
            pseudo_coords: Pseudo-coordinate matrix [num_nodes, pseudo_coord_dim]
            relative_encodings: Relative positional encoding dict {(i,j): [e_ij, P^(0)_ij, ..., P^(K-1)_ij]}
        """
        device = edge_index.device
        
        # Step 1: Construct sparse adjacency matrix
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]),
             (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
            shape=(num_nodes, num_nodes)
        )
        
        # Compute degree matrix D = diag(Σ_j A_ij)
        degrees = np.array(adj.sum(1)).flatten()
        degrees = degrees + 1e-6  # Prevent division by zero
        d_inv = np.power(degrees, -1.0)
        d_mat_inv = sp.diags(d_inv)
        
        # Step 2: Compute random walk matrix M = D^(-1)A
        M = torch.FloatTensor(d_mat_inv.dot(adj).todense()).to(device)
        
        # Compute k-step transition probabilities
        P_list = []
        P_list.append(torch.eye(num_nodes, device=device))  # P^(0) = I
        
        current_M = M
        for k in range(1, self.K):
            P_list.append(current_M.clone())  # P^(k) = M^k
            current_M = current_M @ M
        
        # Step 3: Symmetric degree normalization
        # P_ij^scaled = P_ij^(k) * d_i^(1/2) * d_j^(-1/2)
        degrees_tensor = torch.FloatTensor(degrees).to(device)
        d_sqrt = torch.sqrt(degrees_tensor)  # d_i^(1/2)
        d_inv_sqrt = 1.0 / torch.sqrt(degrees_tensor)  # d_j^(-1/2)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        
        # Apply degree normalization to each k-step transition probability
        P_scaled_list = []
        for P_k in P_list:
            scaling_matrix = d_sqrt.unsqueeze(1) * d_inv_sqrt.unsqueeze(0)
            P_scaled = P_k * scaling_matrix
            P_scaled_list.append(P_scaled)
        
        # Concatenate all transition probabilities across steps
        P_scaled = torch.stack(P_scaled_list, dim=2)  # [num_nodes, num_nodes, K]
        P_flat = P_scaled.reshape(num_nodes, -1)  # [num_nodes, num_nodes * K]
        
        # Steps 4-5: SVD decomposition and extract pseudo-coordinates
        # P^scaled = U Σ V^T
        U, S, V = torch.svd(P_flat)
        # C = U_{:,1:d} Σ_{1:d,1:d}
        pseudo_coords = U[:, :self.pseudo_coord_dim] * S[:self.pseudo_coord_dim].unsqueeze(0)
        
        # Construct relative positional encoding: P_{i,j} = [e_ij, P^(0)_ij, P^(1)_ij, ..., P^(K-1)_ij]
        # If edge attributes are provided, concatenate them with transition probabilities
        relative_encodings = {}
        for edge_idx in range(edge_index.shape[1]):
            i = edge_index[0, edge_idx].item()
            j = edge_index[1, edge_idx].item()
            
            # Extract transition probabilities for sensor pair (i,j) at each step
            trans_probs = torch.stack([P_list[k][i, j] for k in range(self.K)])
            
            # Concatenate edge attributes if provided: [e_ij, P^(0)_ij, ..., P^(K-1)_ij]
            if edge_attr is not None and self.edge_attr_dim > 0:
                edge_feature = edge_attr[edge_idx]
                if edge_feature.dim() == 0:
                    edge_feature = edge_feature.unsqueeze(0)
                rel_enc = torch.cat([edge_feature, trans_probs])
            else:
                rel_enc = trans_probs
            
            relative_encodings[(i, j)] = rel_enc
        
        return pseudo_coords, relative_encodings
    
    def spatial_aware_convolution(self, h, edge_index, relative_encodings):
        """
        Spatial-aware convolution
        
        Mathematical formula:
        h_i^conv = (1/|N(i)|) Σ_{j∈N(i)} h_j ⊙ ψ_θ(P_{i,j}) + b
        
        where P_{i,j} = [e_ij, P^(0)_ij, P^(1)_ij, ..., P^(K-1)_ij]
        - e_ij: edge attributes (e.g., conditional entropy weight from TFDG)
        - P^(k)_ij: k-step transition probability
        
        Args:
            h: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            relative_encodings: Relative positional encoding dict {(i,j): [e_ij, P^(0), ..., P^(K-1)]}
        
        Returns:
            h_conv: Convolved node features [num_nodes, hidden_dim]
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        
        # Compute modulation weights for each edge: ψ_θ(P_{i,j})
        modulation_weights = []
        for edge_idx in range(num_edges):
            i = edge_index[0, edge_idx].item()
            j = edge_index[1, edge_idx].item()
            rel_enc = relative_encodings[(i, j)]
            weight = self.kernel_mlp(rel_enc)  # [hidden_dim]
            modulation_weights.append(weight)
        
        modulation_weights = torch.stack(modulation_weights)  # [num_edges, hidden_dim]
        
        # Get neighbor node features and apply modulation weights
        # h_j ⊙ ψ_θ(P_{i,j})
        neighbor_features = h[edge_index[1]]  # [num_edges, hidden_dim]
        modulated_features = neighbor_features * modulation_weights  # Element-wise multiplication
        
        # Aggregate neighbor information: (1/|N(i)|) Σ_{j∈N(i)} ...
        h_conv = scatter(modulated_features, edge_index[0], dim=0, 
                        dim_size=num_nodes, reduce='mean')
        
        return h_conv
    
    def degree_adaptive_scaling(self, h, edge_index, num_nodes):
        """
        Degree-adaptive scaling
        
        Mathematical formula:
        h_i' = h_i^out ⊙ θ_1 + sqrt(deg(i)) · h_i^out ⊙ θ_2
        
        Args:
            h: Convolved node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Total number of nodes
        
        Returns:
            h_scaled: Scaled node features [num_nodes, hidden_dim]
        """
        # Compute node degrees
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        sqrt_deg = torch.sqrt(deg + 1e-6).unsqueeze(1)  # [num_nodes, 1]
        
        # Apply degree-adaptive scaling: h' = h ⊙ θ_1 + sqrt(deg) · h ⊙ θ_2
        h_scaled = h * self.deg_coef_1 + sqrt_deg * h * self.deg_coef_2
        
        return h_scaled
    
    def forward(self, x, edge_index, edge_attr=None, pseudo_coords=None, relative_encodings=None):
        """
        STAG layer forward propagation
        
        Complete pipeline:
        1. Generate pseudo-coordinate positional encoding (if not provided)
        2. Concatenate pseudo-coordinates to node features: h^(0) = [x, c]
        3. Depthwise separable convolution
        4. Spatial-aware convolution: h^agg = W(Σ h_j ⊙ ψ_θ(P_{i,j}))
           where P_{i,j} = [e_ij, P^(0)_ij, ..., P^(K-1)_ij]
        5. Degree-adaptive scaling: h^scaled = h^agg ⊙ θ_1 + sqrt(deg) · h^agg ⊙ θ_2
        6. Residual connections and batch normalization: h^(l+1) = BN(FFN(BN(h^scaled + h^(l))) + h^(l))
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional, from TFDG)
            pseudo_coords: Pseudo-coordinates [num_nodes, pseudo_coord_dim] (optional)
            relative_encodings: Relative positional encoding dict (optional)
        
        Returns:
            h_out: Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Step 1: Generate pseudo-coordinate positional encoding
        if pseudo_coords is None or relative_encodings is None:
            pseudo_coords, relative_encodings = self.generate_pseudo_coordinates(
                edge_index, num_nodes, edge_attr
            )
        
        # Step 2: Concatenate pseudo-coordinates to node features
        # h^(0) = [x_i, c_i]
        h = torch.cat([x, pseudo_coords], dim=1)  # [num_nodes, input_dim + pseudo_coord_dim]
        
        # Save input for residual connection
        h_residual = h
        
        # Step 3: Depthwise separable convolution
        h = self.depthwise_conv(h)
        h = F.relu(h)
        h = self.pointwise_conv(h)
        
        # Step 4: Spatial-aware convolution
        h_agg = self.spatial_aware_convolution(h, edge_index, relative_encodings)
        
        # Step 5: Degree-adaptive scaling
        h_scaled = self.degree_adaptive_scaling(h_agg, edge_index, num_nodes)
        
        # Step 6: Residual connections and batch normalization
        # First residual: BN(h^scaled + h^(l))
        if h_residual.size(1) != h_scaled.size(1):
            # Use linear transformation to align dimensions if mismatch
            if not hasattr(self, 'residual_proj'):
                self.residual_proj = nn.Linear(
                    h_residual.size(1), h_scaled.size(1)
                ).to(h_scaled.device)
            h_residual = self.residual_proj(h_residual)
        
        h = self.bn1(h_scaled + h_residual)
        
        # Feed-forward network
        h_ffn = self.ffn(h)
        
        # Second residual: BN(FFN(h) + h)
        h_out = self.bn2(h_ffn + h)
        
        return h_out


class STAG_Network(nn.Module):
    """
    Complete STAG Network
    
    Consists of L stacked STAG layers followed by global sum pooling and classifier for graph-level prediction
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, K=3, 
                 pseudo_coord_dim=3, edge_attr_dim=0, dropout=0.3):
        """
        Initialize STAG network
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classification classes
            num_layers: Number of STAG layers (L)
            K: Maximum number of random walk steps
            pseudo_coord_dim: Pseudo-coordinate dimension
            edge_attr_dim: Edge attribute dimension (from TFDG conditional entropy weights)
            dropout: Dropout ratio
        """
        super(STAG_Network, self).__init__()
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        
        # Stack L STAG layers
        self.stag_layers = nn.ModuleList()
        
        # First layer: input_dim + pseudo_coord_dim -> hidden_dim
        self.stag_layers.append(
            STAG_Layer(input_dim + pseudo_coord_dim, hidden_dim, K, pseudo_coord_dim, edge_attr_dim)
        )
        
        # Intermediate layers: hidden_dim + pseudo_coord_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.stag_layers.append(
                STAG_Layer(hidden_dim + pseudo_coord_dim, hidden_dim, K, pseudo_coord_dim, edge_attr_dim)
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        STAG network forward propagation
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional, from TFDG)
        
        Returns:
            out: Classification output [batch_size, num_classes]
        """
        # Generate pseudo-coordinates only in first layer, reuse in subsequent layers
        pseudo_coords = None
        relative_encodings = None
        
        # Pass through L STAG layers
        for layer_idx, stag_layer in enumerate(self.stag_layers):
            if layer_idx == 0:
                # First layer generates pseudo-coordinates with edge attributes
                h = stag_layer(x, edge_index, edge_attr, pseudo_coords, relative_encodings)
                # Save pseudo-coordinates for subsequent layers
                pseudo_coords, relative_encodings = stag_layer.generate_pseudo_coordinates(
                    edge_index, x.size(0), edge_attr
                )
            else:
                # Subsequent layers use pseudo-coordinates from first layer
                h = stag_layer(h, edge_index, edge_attr, pseudo_coords, relative_encodings)
        
        # Global sum pooling: Aggregate all node features in graph to graph-level representation
        # graph_representation = Σ_{i∈G} h_i
        batch_size = batch.max().item() + 1
        graph_repr = scatter(h, batch, dim=0, dim_size=batch_size, reduce='sum')
        
        # Classification
        out = self.classifier(graph_repr)
        
        return out


if __name__ == "__main__":
    print("Testing STAG network...")
    
    # Create test data
    num_nodes = 8
    input_dim = 256
    hidden_dim = 128
    num_classes = 11
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3],
        [1, 2, 2, 3, 3, 0, 0]
    ], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Test with edge attributes (e.g., conditional entropy weights from TFDG)
    edge_attr = torch.randn(edge_index.size(1), 1)  # [num_edges, 1]
    
    # Create STAG network with edge attributes
    model = STAG_Network(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=3,
        K=3,
        pseudo_coord_dim=3,
        edge_attr_dim=1  # Use conditional entropy weight from TFDG
    )
    
    # Forward propagation
    out = model(x, edge_index, batch, edge_attr)
    print(f"Input feature dimension: {x.shape}")
    print(f"Edge attribute dimension: {edge_attr.shape}")
    print(f"Output dimension: {out.shape}")
    print(f"Expected output dimension: [batch_size={batch.max().item()+1}, num_classes={num_classes}]")
    
    print("\nTest with edge attributes completed!")
    
    # Test without edge attributes
    print("\nTesting without edge attributes...")
    model_no_edge = STAG_Network(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=3,
        K=3,
        pseudo_coord_dim=3,
        edge_attr_dim=0
    )
    out_no_edge = model_no_edge(x, edge_index, batch)
    print(f"Output dimension (no edge attr): {out_no_edge.shape}")
    print("\nAll tests completed!")
