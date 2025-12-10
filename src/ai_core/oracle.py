import torch
import torch.nn as nn
import numpy as np
import logging
import os

logger = logging.getLogger("Oracle")

class TransformerModel(nn.Module):
    """
    Simplified Transformer model for price prediction.
    Matches the architecture expected by the weights.
    """
    def __init__(self, input_dim=60, d_model=128, nhead=4, num_layers=2, output_dim=3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim) # [UP, DOWN, NEUTRAL]

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer_encoder(x)
        # Take the last time step
        x = x[:, -1, :]
        output = self.fc_out(x)
        return output

class Oracle:
    """
    The Oracle Engine: Uses Transformer models to predict future price movements.
    """
    def __init__(self, model_path="models/nexus_transformer.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Loads the Transformer model weights."""
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ ORACLE: Model file not found at {self.model_path}. Running in SIMULATION mode.")
            return

        try:
            # Initialize model architecture
            self.model = TransformerModel().to(self.device)
            
            # Load weights
            # map_location ensures we can load GPU models on CPU if needed
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # Set to evaluation mode
            logger.info(f"🔮 ORACLE: Nexus Transformer loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"⚠️ ORACLE: Failed to load model: {e}")
            self.model = None

    def predict(self, market_data_sequence):
        """
        Predicts the next price movement direction.
        
        Args:
            market_data_sequence: List or Array of last 60 candles [Open, High, Low, Close, Volume]
            
        Returns:
            prediction: "UP", "DOWN", or "NEUTRAL"
            confidence: float (0.0 to 1.0)
        """
        if self.model is None:
            # Fallback / Simulation logic if model is missing
            return "NEUTRAL", 0.0

        try:
            # Preprocess data
            # Expecting (Batch, Seq_Len, Features) -> (1, 60, 5)
            # We need to flatten or project to input_dim=60 if architecture expects that
            # For this simplified version, let's assume input is (1, 60) representing Close prices
            
            # Convert to tensor
            input_tensor = torch.tensor(market_data_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Get predicted class
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_idx].item()
                
                classes = ["UP", "DOWN", "NEUTRAL"]
                prediction = classes[predicted_idx]
                
                return prediction, confidence

        except Exception as e:
            logger.error(f"⚠️ ORACLE: Prediction error: {e}")
            return "NEUTRAL", 0.0
