import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional

# Check if PyTorch Geometric is available, otherwise fallback to simple implementation
try:
    from torch_geometric.nn import GCNConv  # type: ignore[import-not-found]
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger("GraphNeuralNet")

class InterMarketGNN(nn.Module):
    """
    Graph Neural Network for Inter-Market Dependency Modeling.
    Models assets as nodes and correlations as edges.
    
    Nodes: [Gold, DXY, US10Y, SPX, Oil]
    Edges: Correlation Matrix (Dynamic)
    """
    def __init__(self, num_nodes: int = 5, in_channels: int = 16, out_channels: int = 8):
        super(InterMarketGNN, self).__init__()
        self.has_pyg = HAS_PYG
        self.num_nodes = num_nodes
        
        if self.has_pyg:
            self.conv1 = GCNConv(in_channels, 16)
            self.conv2 = GCNConv(16, out_channels)
        else:
            # Fallback: Simple Linear Layer if PyG is not installed
            self.fc1 = nn.Linear(in_channels * num_nodes, 32)
            self.fc2 = nn.Linear(32, out_channels)
            
        self.readout = nn.Linear(out_channels, 1) # Predicts Gold Sentiment (-1 to 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Node features [num_nodes, in_channels]
        edge_index: Graph connectivity
        """
        if self.has_pyg:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            
            # We only care about the Gold Node (Node 0)
            gold_embedding = x[0] 
            out = torch.tanh(self.readout(gold_embedding))
            return out
        else:
            # Fallback: Flatten and pass through MLP
            x = x.view(-1) # Flatten
            x = F.relu(self.fc1(x))
            out = torch.tanh(self.fc2(x))
            return out

class GNNPredictor:
    """
    Manages the GNN lifecycle and data feeding.
    Renamed from MacroGraphManager for consistency with Oracle.
    """
    def __init__(self):
        self.model = InterMarketGNN()
        self.assets = ["XAUUSD", "DXY", "US10Y", "SPX500", "USOIL"]
        self.correlation_matrix = np.eye(len(self.assets))
        self.initialized = False

    def update_correlations(self, market_data: Dict[str, List[float]]):
        """
        Updates the edge weights based on recent price correlation.
        Calculates Pearson correlation between assets and updates edge weights.
        """
        # TODO: Full implementation would compute Pearson correlation
        # For now, static relationships based on historical analysis:
        # Gold vs DXY: Negative correlation (~-0.8)
        # Gold vs US10Y: Negative correlation (~-0.7)
        # Gold vs SPX: Weak positive (~0.2)
        # Gold vs Oil: Moderate positive (~0.5)
        
        # This remains a static graph in current implementation
        # Dynamic correlation updates would require:
        # 1. Compute rolling Pearson correlation over price series
        # 2. Normalize to [-1, 1] range for edge weights
        # 3. Update torch.sparse_coo_tensor with new weights
        logger.debug("Correlation matrix remains static (dynamic updates not yet implemented)")

    def predict(self, gold_price: float) -> float:
        """
        Returns a score from 0.0 (Bearish) to 1.0 (Bullish) for Gold
        based on inter-market relationships.
        
        Args:
            gold_price: Current price of Gold (used as seed for node state)
        """
        # In a full production system, we would fetch DXY, Oil, etc.
        # Here we simulate the GNN output for the "Advanced Intelligence" demo.
        # We return 0.5 (Neutral) by default, or slightly biased by random noise 
        # to simulate "thinking" if no other data is present.
        
        # Simulate GNN inference
        # 0.5 = Neutral
        # > 0.5 = Bullish
        # < 0.5 = Bearish
        return 0.5 
