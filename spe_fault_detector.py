import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import time

class SPEFaultDetector:
    """
    SPE-based fault detector using transformer autoencoder architecture.
    This class encapsulates the optimized transformer model for fault detection
    using Squared Prediction Error (SPE) metrics.
    """
    
    def __init__(self, input_dim, hidden_dim=None, num_heads=4, num_layers=2, confidence=0.99):
        """
        Initialize the SPE Fault Detector
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int or None
            Dimension of hidden layers. If None, it will be automatically determined
        num_heads : int
            Number of attention heads in transformer
        num_layers : int
            Number of transformer layers
        confidence : float
            Confidence level for control limit (0-1)
        """
        self.input_dim = input_dim
        self.confidence = confidence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default hidden dim if not provided
        if hidden_dim is None:
            self.hidden_dim = min(32, input_dim)
        else:
            self.hidden_dim = hidden_dim
            
        # Initialize model
        self.model = self._create_model(num_heads, num_layers)
        self.model.to(self.device)
        
        # Control limits will be set during training
        self.spe_limit = None
        self.is_trained = False
        
        print(f"SPE Fault Detector initialized. Using device: {self.device}")
    
    def _create_model(self, num_heads, num_layers):
        """Create the transformer autoencoder model"""
        return TransformerAutoencoder(
            self.input_dim, 
            self.hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    def train(self, train_data, epochs=100, batch_size=32, lr=0.001, validation_split=0.1):
        """
        Train the model on normal operating data
        
        Parameters:
        -----------
        train_data : numpy array
            Training data (normal operating conditions only)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        validation_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        train_losses : list
            Training loss history
        val_losses : list
            Validation loss history
        """
        # Standardize data
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_data)
        
        # Split data into training and validation
        indices = np.random.permutation(len(X_train))
        val_size = int(len(X_train) * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train_split = X_train[train_indices]
        X_val = X_train[val_indices]
        
        # Create data loaders
        train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        
        train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=batch_size)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"Starting training for {epochs} epochs...")
        best_val_loss = float('inf')
        best_model = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            
            for batch_data in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                
                loss = nn.MSELoss()(reconstructed, batch_data)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(self.device)
                    reconstructed = self.model(batch_data)
                    loss = nn.MSELoss()(reconstructed, batch_data)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Calculate control limits
        self.calculate_control_limits(X_train)
        
        self.is_trained = True
        return train_losses, val_losses
    
    def save_model(self, filepath='spe_fault_detector.pth'):
        """Save the trained model to a file"""
        if not self.is_trained:
            print("Warning: Model not trained yet. No model saved.")
            return False
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'spe_limit': self.spe_limit,
            'scaler': self.scaler
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='spe_fault_detector.pth'):
        """Load a trained model from a file"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Recreate model with correct dimensions
            self.input_dim = checkpoint['input_dim']
            self.hidden_dim = checkpoint['hidden_dim']
            
            # Load model state
            self.model = self._create_model(num_heads=4, num_layers=2)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            
            # Load control limit and scaler
            self.spe_limit = checkpoint['spe_limit']
            self.scaler = checkpoint['scaler']
            
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def calculate_control_limits(self, train_data):
        """
        Calculate SPE control limit from training data
        
        Parameters:
        -----------
        train_data : numpy array
            Training data (normal operating conditions)
        """
        # Calculate SPE values for training data
        spe_train = self.calculate_spe(train_data)
        
        # Calculate control limit using kernel density estimation
        self.spe_limit = self._calculate_control_limit(spe_train, self.confidence)
        print(f"SPE control limit: {self.spe_limit:.2f} (confidence: {self.confidence*100:.1f}%)")
    
    def _calculate_control_limit(self, values, confidence=0.99):
        """Calculate control limit using kernel density estimation"""
        from scipy import stats
        
        # Ensure values are flattened to 1D
        values_flat = values.flatten() if hasattr(values, 'flatten') else np.array(values)
        
        # Handle extreme cases
        if np.std(values_flat) < 1e-6 or len(values_flat) < 5:
            return np.percentile(values_flat, 99) + 1e-3
        
        try:
            # KDE-based calculation
            values_reshaped = values_flat.reshape(-1, 1)
            kde = stats.gaussian_kde(values_reshaped.T, bw_method='scott')
            
            # Create evaluation grid
            x_min, x_max = np.min(values_flat), np.max(values_flat)
            range_width = x_max - x_min
            x = np.linspace(x_min - 0.1 * range_width, x_max + 0.5 * range_width, 1000)
            
            # Calculate PDF and CDF
            pdf = kde(x)
            cdf = np.zeros_like(x)
            dx = x[1] - x[0]
            for i in range(1, len(x)):
                cdf[i] = cdf[i-1] + pdf[i-1] * dx
            
            # Normalize CDF
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            
            # Find KDE-based control limit at desired confidence level
            limit_idx = np.searchsorted(cdf, confidence)
            if limit_idx < len(x):
                return x[limit_idx]
            else:
                return x[-1]
        except:
            # Fallback if KDE fails
            print("Warning: KDE calculation failed, using percentile method")
            return np.percentile(values_flat, confidence * 100)
    
    def calculate_spe(self, data):
        """
        Calculate SPE (Q statistic) for given data
        
        Parameters:
        -----------
        data : numpy array
            Input data
            
        Returns:
        --------
        spe_values : numpy array
            SPE values for each sample
        """
        # Standardize data if model is trained
        if hasattr(self, 'scaler'):
            data = self.scaler.transform(data)
        
        # Configure model for evaluation
        self.model.eval()
        batch_size = 32
        spe_values = []
        
        # Calculate SPE in batches
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(self.device)
                reconstructed = self.model(batch)
                # SPE is squared reconstruction error
                error = torch.sum((batch - reconstructed)**2, dim=1)
                spe_values.extend(error.cpu().numpy())
        
        return np.array(spe_values)
    
    def detect_faults(self, data):
        """
        Detect faults in new data using SPE metrics
        
        Parameters:
        -----------
        data : numpy array
            New data to analyze
            
        Returns:
        --------
        fault_flags : numpy array
            Boolean array indicating fault detection (True=fault)
        spe_values : numpy array
            SPE values for each sample
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        # Calculate SPE for test data
        spe_values = self.calculate_spe(data)
        
        # Apply control limit to detect faults
        fault_flags = (spe_values > self.spe_limit)
        
        return fault_flags, spe_values
    
    def evaluate_performance(self, test_data, fault_start_idx):
        """
        Evaluate fault detection performance metrics
        
        Parameters:
        -----------
        test_data : numpy array
            Test data containing both normal and fault conditions
        fault_start_idx : int
            Index at which faults begin in the test data
            
        Returns:
        --------
        metrics : dict
            Performance metrics including false alarm rate, miss rate, detection time
        """
        # Detect faults
        fault_flags, spe_values = self.detect_faults(test_data)
        
        # Calculate performance metrics
        # False alarm rate
        if fault_start_idx > 0:
            false_alarms = np.sum(fault_flags[:fault_start_idx])
            false_rate = 100 * false_alarms / fault_start_idx
        else:
            false_rate = 0
        
        # Miss rate
        if fault_start_idx < len(test_data):
            misses = np.sum(~fault_flags[fault_start_idx:])
            miss_rate = 100 * misses / (len(test_data) - fault_start_idx)
        else:
            miss_rate = 0
        
        # Detection time (consecutive samples required)
        consecutive = 3
        detection_time = None
        
        for i in range(fault_start_idx, len(fault_flags) - consecutive + 1):
            if all(fault_flags[i:i+consecutive]):
                detection_time = i - fault_start_idx
                break
        
        # Compile metrics
        metrics = {
            'false_alarm_rate': false_rate,
            'miss_rate': miss_rate,
            'detection_time': detection_time,
            'spe_values': spe_values,
            'fault_flags': fault_flags
        }
        
        return metrics
    
    def plot_results(self, test_data, fault_start_idx, save_path=None):
        """
        Plot SPE values and fault detection results
        
        Parameters:
        -----------
        test_data : numpy array
            Test data containing both normal and fault conditions
        fault_start_idx : int
            Index at which faults begin in the test data
        save_path : str or None
            Path to save the plot, if None, plot will be displayed
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        # Get metrics and values
        metrics = self.evaluate_performance(test_data, fault_start_idx)
        spe_values = metrics['spe_values']
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot normal data in green
        if fault_start_idx > 0:
            plt.plot(range(1, fault_start_idx+1), spe_values[:fault_start_idx], 'g', label='Normal')
        
        # Plot fault data in red
        if fault_start_idx < len(test_data):
            plt.plot(range(fault_start_idx+1, len(spe_values)+1), spe_values[fault_start_idx:], 'r', label='Fault')
        
        # Plot control limit
        plt.axhline(y=self.spe_limit, color='k', linestyle='--', label='Control Limit')
        
        # Plot fault start line
        plt.axvline(x=fault_start_idx, color='m', linestyle='-', label='Fault Time')
        
        # Add labels and title
        plt.title('SPE-Based Fault Detection')
        plt.xlabel('Sample')
        plt.ylabel('SPE (Q Statistics)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        det_time = metrics['detection_time'] if metrics['detection_time'] is not None else "Not detected"
        plt.text(0.02, 0.95, f"False Alarm Rate: {metrics['false_alarm_rate']:.2f}%", transform=plt.gca().transAxes)
        plt.text(0.02, 0.90, f"Miss Rate: {metrics['miss_rate']:.2f}%", transform=plt.gca().transAxes)
        plt.text(0.02, 0.85, f"Detection Time: {det_time}", transform=plt.gca().transAxes)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return metrics


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for process monitoring
    Optimized for SPE (Q statistic) fault detection
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        
        # Ensure hidden_dim is compatible with num_heads
        if hidden_dim % num_heads != 0:
            num_heads = hidden_dim // (hidden_dim // num_heads)
            print(f"Adjusted number of attention heads to {num_heads} to match hidden dimension {hidden_dim}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Transformer decoder
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        # Input embedding
        embedded = self.feature_embedding(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(embedded)
        
        # Bottleneck
        bottleneck_features = self.bottleneck(encoded)
        
        # Self-attention for feature emphasis
        attn_output, _ = self.attention(bottleneck_features, bottleneck_features, bottleneck_features)
        
        # Transformer decoder
        decoded = self.transformer_decoder(attn_output + bottleneck_features)
        
        # Output projection
        output = self.output_layer(decoded)
        
        return output


def demo():
    """Demonstrate usage of the SPE fault detector"""
    # Generate mock data
    np.random.seed(42)
    input_dim = 20
    
    # Normal training data (500 samples)
    train_data = np.random.randn(500, input_dim) * 0.1
    
    # Test data with fault after sample 200
    test_data = np.random.randn(500, input_dim) * 0.1
    fault_start = 200
    
    # Add fault signature to test data
    for i in range(fault_start, 500):
        test_data[i, :] += 0.2 * np.random.randn(input_dim) + 0.1
    
    # Create and train fault detector
    detector = SPEFaultDetector(input_dim=input_dim, hidden_dim=24)
    detector.train(train_data, epochs=50, batch_size=32)
    
    # Save model
    detector.save_model('spe_detector_demo.pth')
    
    # Evaluate performance
    metrics = detector.evaluate_performance(test_data, fault_start)
    
    # Print results
    print("\n===== Fault Detection Results =====")
    print(f"False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")
    print(f"Miss Rate: {metrics['miss_rate']:.2f}%")
    print(f"Detection Time: {metrics['detection_time'] if metrics['detection_time'] is not None else 'Not detected'}")
    
    # Plot results
    detector.plot_results(test_data, fault_start, save_path='spe_detector_demo.png')
    
    return detector, metrics


if __name__ == "__main__":
    demo()
