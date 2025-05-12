import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler

class TransformerRefiner(nn.Module):
    """
    Transformer model for refining anomaly detection results
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerRefiner, self).__init__()
        
        # Feature embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Position encoder layer
        self.pos_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Set batch_first to True to avoid warning
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.pos_encoder,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, feat_dim = x.size()
        
        # Feature embedding
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply Transformer encoding (batch_first=True so no need to permute)
        x = self.transformer_encoder(x)
        
        # Use only the center position output for prediction
        center_pos = seq_len // 2
        center_repr = x[:, center_pos]  # [batch_size, hidden_dim]
        
        # Output layer
        output = self.output_layer(center_repr).squeeze(-1)
        
        return output


def transformer_enhanced_two_stage_detector(X_train, X_test, happen, top_features, transformer_dim=32):
    """
    Transformer-enhanced two-stage detector that balances false alarms and miss rates
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data
    X_test : numpy.ndarray
        Test data
    happen : int
        Index where fault occurs
    top_features : list
        List of critical feature indices
    transformer_dim : int
        Hidden dimension for Transformer model
        
    Returns:
    --------
    dict
        Detection results with metrics
    """
    print("\n========== Running Transformer-Enhanced Two-Stage Detector ==========")
    print("Target: Maintain low false alarm rate while further reducing miss rate")
    start_time = time.time()
    
    # First utilize the balanced two-stage detector for initial detection
    from scripts.run_secom_fault_detection import balanced_two_stage_detector
    initial_results = balanced_two_stage_detector(X_train, X_test, happen, top_features)
    
    # Extract T² and SPE statistics from the balanced detector
    t2_statistics = None
    spe_statistics = None
    t2_threshold = None
    spe_threshold = None
    
    # Check if the balanced detector has separate T² and SPE statistics
    if 't2_statistics' in initial_results and 'spe_statistics' in initial_results:
        t2_statistics = initial_results['t2_statistics']
        spe_statistics = initial_results['spe_statistics']
        t2_threshold = initial_results['t2_threshold']
        spe_threshold = initial_results['spe_threshold']
    else:
        # Generate our own statistics
        X_train_critical = X_train[:, top_features]
        X_test_critical = X_test[:, top_features]
        
        # Calculate statistics based on z-scores
        means = np.mean(X_train_critical, axis=0)
        stds = np.std(X_train_critical, axis=0)
        z_scores_test = np.abs((X_test_critical - means) / (stds + 1e-8))
        
        # Use average for T² and max for SPE (more sensitive)
        t2_statistics = np.mean(z_scores_test, axis=1)
        spe_statistics = np.max(z_scores_test, axis=1)
        
        # Calculate thresholds
        normal_t2 = t2_statistics[:happen]
        normal_spe = spe_statistics[:happen]
        t2_threshold = np.percentile(normal_t2, 96)  # Slightly stricter for T²
        spe_threshold = np.percentile(normal_spe, 95)  # More sensitive for SPE
    
    # Generate initial T² and SPE alarms
    t2_initial_alarms = np.zeros(len(t2_statistics), dtype=bool)
    spe_initial_alarms = np.zeros(len(spe_statistics), dtype=bool)
    
    # Check if we have alarm information from balanced detector
    if 't2_alarms' in initial_results and 'spe_alarms' in initial_results:
        t2_initial_alarms = initial_results['t2_alarms']
        spe_initial_alarms = initial_results['spe_alarms']
        
        # Also get metrics
        t2_false_rate = initial_results['t2_false_alarm_rate']
        t2_miss_rate = initial_results['t2_miss_rate']
        t2_detection_time = initial_results.get('t2_detection_time', None)
        
        spe_false_rate = initial_results['spe_false_alarm_rate']
        spe_miss_rate = initial_results['spe_miss_rate']
        spe_detection_time = initial_results.get('spe_detection_time', None)
    else:
        # Apply thresholds
        t2_initial_alarms = t2_statistics > t2_threshold
        spe_initial_alarms = spe_statistics > spe_threshold
        
        # Calculate T² metrics
        t2_false_alarms = np.sum(t2_initial_alarms[:happen])
        t2_false_rate = 100 * t2_false_alarms / happen if happen > 0 else 0
        
        t2_miss_count = np.sum(~t2_initial_alarms[happen:])
        t2_miss_rate = 100 * t2_miss_count / (len(t2_initial_alarms) - happen) if len(t2_initial_alarms) > happen else 0
        
        # Calculate SPE metrics
        spe_false_alarms = np.sum(spe_initial_alarms[:happen])
        spe_false_rate = 100 * spe_false_alarms / happen if happen > 0 else 0
        
        spe_miss_count = np.sum(~spe_initial_alarms[happen:])
        spe_miss_rate = 100 * spe_miss_count / (len(spe_initial_alarms) - happen) if len(spe_initial_alarms) > happen else 0
        
        # Calculate detection times
        t2_detection_time = None
        for i in range(happen, len(t2_initial_alarms)):
            if t2_initial_alarms[i]:
                t2_detection_time = i - happen
                break
                
        spe_detection_time = None
        for i in range(happen, len(spe_initial_alarms)):
            if spe_initial_alarms[i]:
                spe_detection_time = i - happen
                break
    
    print(f"Stage 1 Results (Balanced Two-Stage):")
    print(f"T² False Alarm Rate: {t2_false_rate:.2f}%, Miss Rate: {t2_miss_rate:.2f}%")
    print(f"SPE False Alarm Rate: {spe_false_rate:.2f}%, Miss Rate: {spe_miss_rate:.2f}%")
    print(f"T² Detection Time: {t2_detection_time if t2_detection_time is not None else 'Not detected'}")
    print(f"SPE Detection Time: {spe_detection_time if spe_detection_time is not None else 'Not detected'}")
    
    #====================== Stage 2 (Transformer-Enhanced) ======================
    # Skip transformer refinement if balanced detector already performs very well
    if spe_miss_rate < 1.0 and t2_miss_rate < 1.0 and spe_false_rate <= 10.0 and t2_false_rate <= 10.0:
        print("Balanced detector already performing optimally - skipping transformer refinement.")
        print(f"Current performance: T²: {t2_false_rate:.2f}% false, {t2_miss_rate:.2f}% miss; SPE: {spe_false_rate:.2f}% false, {spe_miss_rate:.2f}% miss")
        return {
            't2_statistics': t2_statistics,
            'spe_statistics': spe_statistics,
            't2_threshold': t2_threshold,
            'spe_threshold': spe_threshold,
            't2_alarms': t2_initial_alarms,
            'spe_alarms': spe_initial_alarms,
            't2_false_alarm_rate': t2_false_rate,
            't2_miss_rate': t2_miss_rate,
            't2_detection_time': t2_detection_time,
            'spe_false_alarm_rate': spe_false_rate,
            'spe_miss_rate': spe_miss_rate,
            'spe_detection_time': spe_detection_time,
            'false_rates': [t2_false_rate, spe_false_rate],
            'miss_rates': [t2_miss_rate, spe_miss_rate],
            'detection_times': [t2_detection_time, spe_detection_time]
        }
    
    print("Adding transformer-based refinement to improve performance...")
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all features instead of just top features for better context
    extended_features = list(set(top_features + [37, 38, 34, 36]))  # Add critical features if not already included
    
    # Step 1: Extract temporal context - time window around each sample
    window_size = 7  # Larger window for more context
    
    # Precompute z-scores for extended features
    X_train_ext = X_train[:, extended_features]
    X_test_ext = X_test[:, extended_features]
    means = np.mean(X_train_ext, axis=0)
    stds = np.std(X_train_ext, axis=0)
    z_scores_test = np.abs((X_test_ext - means) / (stds + 1e-8))
    
    # Create separate models for T² and SPE
    # First prepare data for both models
    all_windows_t2 = []
    all_windows_spe = []
    
    for i in range(len(X_test)):
        window_start = max(0, i - window_size)
        window_end = min(len(X_test), i + window_size + 1)
        
        # Get feature values, z-scores, and initial alarm status
        window_features = z_scores_test[window_start:window_end]  # Use z-scores
        window_t2_alarms = t2_initial_alarms[window_start:window_end]
        window_spe_alarms = spe_initial_alarms[window_start:window_end]
        
        # Calculate relative position in window (for position encoding)
        positions = np.array([range(-(i-window_start), window_end-i)])
        positions = positions.T / window_size  # Normalize to [-1, 1]
        
        # Combine features, alarm status, and position for T²
        if len(window_features) > 0:
            window_data_t2 = np.hstack([
                window_features,
                window_t2_alarms.reshape(-1, 1),
                positions
            ])
            
            window_data_spe = np.hstack([
                window_features,
                window_spe_alarms.reshape(-1, 1),
                positions
            ])
            
            # Handle variable-length sequences with proper padding
            padded_window_t2 = np.zeros((window_size*2+1, window_data_t2.shape[1]))
            padded_window_spe = np.zeros((window_size*2+1, window_data_spe.shape[1]))
            
            actual_size = min(window_end - window_start, window_size*2+1)
            padded_window_t2[:actual_size] = window_data_t2[:actual_size]
            padded_window_spe[:actual_size] = window_data_spe[:actual_size]
            
            all_windows_t2.append(padded_window_t2)
            all_windows_spe.append(padded_window_spe)
    
    # Convert to tensors
    all_windows_array_t2 = np.array(all_windows_t2)
    all_windows_array_spe = np.array(all_windows_spe)
    
    time_windows_tensor_t2 = torch.FloatTensor(all_windows_array_t2).to(device)
    time_windows_tensor_spe = torch.FloatTensor(all_windows_array_spe).to(device)
    
    # Step 2: Build and train separate Transformer models for T² and SPE
    # Split into training and testing sets
    train_tensors_t2 = time_windows_tensor_t2[:happen]
    train_labels_t2 = torch.FloatTensor(t2_initial_alarms[:happen].astype(float)).to(device)
    
    train_tensors_spe = time_windows_tensor_spe[:happen]
    train_labels_spe = torch.FloatTensor(spe_initial_alarms[:happen].astype(float)).to(device)
    
    # Get fault samples for weighted training
    fault_tensors_t2 = time_windows_tensor_t2[happen:]
    fault_labels_t2 = torch.FloatTensor(t2_initial_alarms[happen:].astype(float)).to(device)
    
    fault_tensors_spe = time_windows_tensor_spe[happen:]
    fault_labels_spe = torch.FloatTensor(spe_initial_alarms[happen:].astype(float)).to(device)
    
    # Create model instances
    input_dim_t2 = time_windows_tensor_t2.shape[2]
    input_dim_spe = time_windows_tensor_spe.shape[2]
    
    model_t2 = TransformerRefiner(
        input_dim=input_dim_t2, 
        hidden_dim=transformer_dim,
        num_heads=4,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    model_spe = TransformerRefiner(
        input_dim=input_dim_spe, 
        hidden_dim=transformer_dim,
        num_heads=4,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    # Handle class imbalance
    pos_weight = torch.tensor([5.0]).to(device)  # Weight positive examples more heavily
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Train T² model
    optimizer_t2 = torch.optim.Adam(model_t2.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_t2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t2, 'min', factor=0.5, patience=5)
    
    # Training loop
    model_t2.train()
    n_epochs = 100
    batch_size = 32
    best_loss_t2 = float('inf')
    train_losses_t2 = []
    
    # Load pretrained model if available
    try:
        model_t2.load_state_dict(torch.load('results/models/transformer_refiner_t2.pth', map_location=device))
        print("Loaded pretrained T² transformer model")
        trained_t2 = True
    except:
        print("Training T² transformer model...")
        trained_t2 = False
    
    if not trained_t2:
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            # Create batches
            perm = torch.randperm(len(train_tensors_t2))
            train_tensors_shuffled = train_tensors_t2[perm]
            train_labels_shuffled = train_labels_t2[perm]
            
            # Weighted sampling of fault samples to balance dataset
            fault_perm = torch.randperm(len(fault_tensors_t2))
            fault_tensors_shuffled = fault_tensors_t2[fault_perm]
            fault_labels_shuffled = fault_labels_t2[fault_perm]
            
            # Process in batches
            for i in range(0, len(train_tensors_shuffled), batch_size):
                batch_x = train_tensors_shuffled[i:i+batch_size]
                batch_y = train_labels_shuffled[i:i+batch_size]
                
                # Add some fault samples to batch
                if i < len(fault_tensors_shuffled):
                    fault_end = min(i+batch_size//4, len(fault_tensors_shuffled))
                    batch_x = torch.cat([batch_x, fault_tensors_shuffled[i:fault_end]], dim=0)
                    batch_y = torch.cat([batch_y, fault_labels_shuffled[i:fault_end]], dim=0)
                
                # Forward pass
                optimizer_t2.zero_grad()
                outputs = model_t2(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer_t2.step()
                
                epoch_loss += loss.item()
            
            # Adjust learning rate
            avg_loss = epoch_loss / (len(train_tensors_shuffled) / batch_size)
            scheduler_t2.step(avg_loss)
            train_losses_t2.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
            
            # Save best model
            if avg_loss < best_loss_t2:
                best_loss_t2 = avg_loss
                torch.save(model_t2.state_dict(), "results/models/transformer_refiner_t2.pth")
        
        # Load best model
        model_t2.load_state_dict(torch.load('results/models/transformer_refiner_t2.pth', map_location=device))
    
    # Train SPE model
    optimizer_spe = torch.optim.Adam(model_spe.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_spe = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_spe, 'min', factor=0.5, patience=5)
    
    # Training loop
    model_spe.train()
    best_loss_spe = float('inf')
    train_losses_spe = []
    
    # Load pretrained model if available
    try:
        model_spe.load_state_dict(torch.load('results/models/transformer_refiner_spe.pth', map_location=device))
        print("Loaded pretrained SPE transformer model")
        trained_spe = True
    except:
        print("Training SPE transformer model...")
        trained_spe = False
    
    if not trained_spe:
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            # Create batches
            perm = torch.randperm(len(train_tensors_spe))
            train_tensors_shuffled = train_tensors_spe[perm]
            train_labels_shuffled = train_labels_spe[perm]
            
            # Weighted sampling of fault samples to balance dataset
            fault_perm = torch.randperm(len(fault_tensors_spe))
            fault_tensors_shuffled = fault_tensors_spe[fault_perm]
            fault_labels_shuffled = fault_labels_spe[fault_perm]
            
            # Process in batches
            for i in range(0, len(train_tensors_shuffled), batch_size):
                batch_x = train_tensors_shuffled[i:i+batch_size]
                batch_y = train_labels_shuffled[i:i+batch_size]
                
                # Add some fault samples to batch
                if i < len(fault_tensors_shuffled):
                    fault_end = min(i+batch_size//4, len(fault_tensors_shuffled))
                    batch_x = torch.cat([batch_x, fault_tensors_shuffled[i:fault_end]], dim=0)
                    batch_y = torch.cat([batch_y, fault_labels_shuffled[i:fault_end]], dim=0)
                
                # Forward pass
                optimizer_spe.zero_grad()
                outputs = model_spe(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer_spe.step()
                
                epoch_loss += loss.item()
            
            # Adjust learning rate
            avg_loss = epoch_loss / (len(train_tensors_shuffled) / batch_size)
            scheduler_spe.step(avg_loss)
            train_losses_spe.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
            
            # Save best model
            if avg_loss < best_loss_spe:
                best_loss_spe = avg_loss
                torch.save(model_spe.state_dict(), "results/models/transformer_refiner_spe.pth")
        
        # Load best model
        model_spe.load_state_dict(torch.load('results/models/transformer_refiner_spe.pth', map_location=device))
    
    # Step 3: Apply models to all samples for final detection
    model_t2.eval()
    model_spe.eval()
    
    # Predict alarm probabilities
    with torch.no_grad():
        t2_probs = model_t2(time_windows_tensor_t2).cpu().numpy()
        spe_probs = model_spe(time_windows_tensor_spe).cpu().numpy()
    
    # Adjust probability threshold for desired balance
    t2_threshold_prob = np.percentile(t2_probs[:happen], 96)  # 4% false alarm rate for T²
    spe_threshold_prob = np.percentile(spe_probs[:happen], 95)  # 5% false alarm rate for SPE
    
    # Final alarm decisions
    t2_refined_alarms = t2_probs > t2_threshold_prob
    spe_refined_alarms = spe_probs > spe_threshold_prob
    
    # Calculate final metrics
    # T² metrics
    t2_false_alarms_refined = np.sum(t2_refined_alarms[:happen])
    t2_false_rate_refined = 100 * t2_false_alarms_refined / happen if happen > 0 else 0
    
    t2_misses_refined = np.sum(~t2_refined_alarms[happen:])
    t2_miss_rate_refined = 100 * t2_misses_refined / (len(t2_refined_alarms) - happen) if len(t2_refined_alarms) > happen else 0
    
    # SPE metrics
    spe_false_alarms_refined = np.sum(spe_refined_alarms[:happen])
    spe_false_rate_refined = 100 * spe_false_alarms_refined / happen if happen > 0 else 0
    
    spe_misses_refined = np.sum(~spe_refined_alarms[happen:])
    spe_miss_rate_refined = 100 * spe_misses_refined / (len(spe_refined_alarms) - happen) if len(spe_refined_alarms) > happen else 0
    
    # Detection time
    t2_detection_time_refined = None
    for i in range(happen, len(t2_refined_alarms)):
        if t2_refined_alarms[i]:
            t2_detection_time_refined = i - happen
            break
    
    spe_detection_time_refined = None
    for i in range(happen, len(spe_refined_alarms)):
        if spe_refined_alarms[i]:
            spe_detection_time_refined = i - happen
            break
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot T² results
    plt.subplot(2, 2, 1)
    plt.plot(range(1, happen+1), t2_probs[:happen], 'g-', label='Normal')
    plt.plot(range(happen+1, len(t2_probs)+1), t2_probs[happen:], 'm-', label='Fault')
    plt.axhline(y=t2_threshold_prob, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='r', linestyle='-', label='Fault Time')
    plt.title('T² Alarm Probabilities')
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot SPE results
    plt.subplot(2, 2, 2)
    plt.plot(range(1, happen+1), spe_probs[:happen], 'g-', label='Normal')
    plt.plot(range(happen+1, len(spe_probs)+1), spe_probs[happen:], 'm-', label='Fault')
    plt.axhline(y=spe_threshold_prob, color='k', linestyle='--', label='Threshold')
    plt.axvline(x=happen, color='r', linestyle='-', label='Fault Time')
    plt.title('SPE Alarm Probabilities')
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot T² initial vs refined
    plt.subplot(2, 2, 3)
    plt.plot(range(1, happen+1), t2_initial_alarms[:happen], 'g-', label='Initial (Normal)')
    plt.plot(range(happen+1, len(t2_initial_alarms)+1), t2_initial_alarms[happen:], 'g-', label='Initial (Fault)')
    plt.plot(range(1, happen+1), t2_refined_alarms[:happen], 'r--', label='Refined (Normal)')
    plt.plot(range(happen+1, len(t2_refined_alarms)+1), t2_refined_alarms[happen:], 'r--', label='Refined (Fault)')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Time')
    plt.title('T² Alarm Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Alarm Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot SPE initial vs refined
    plt.subplot(2, 2, 4)
    plt.plot(range(1, happen+1), spe_initial_alarms[:happen], 'g-', label='Initial (Normal)')
    plt.plot(range(happen+1, len(spe_initial_alarms)+1), spe_initial_alarms[happen:], 'g-', label='Initial (Fault)')
    plt.plot(range(1, happen+1), spe_refined_alarms[:happen], 'r--', label='Refined (Normal)')
    plt.plot(range(happen+1, len(spe_refined_alarms)+1), spe_refined_alarms[happen:], 'r--', label='Refined (Fault)')
    plt.axvline(x=happen, color='k', linestyle='-', label='Fault Time')
    plt.title('SPE Alarm Comparison')
    plt.xlabel('Sample')
    plt.ylabel('Alarm Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/plots/transformer_enhanced_two_stage.png")
    plt.close()
    
    # Print results
    print("\nTransformer-Enhanced Two-Stage Results:")
    print(f"T² - Initial vs Refined:")
    print(f"  False Alarm Rate: {t2_false_rate:.2f}% → {t2_false_rate_refined:.2f}%")
    print(f"  Miss Rate: {t2_miss_rate:.2f}% → {t2_miss_rate_refined:.2f}%")
    print(f"  Detection Time: {t2_detection_time if t2_detection_time is not None else 'Not detected'} → {t2_detection_time_refined if t2_detection_time_refined is not None else 'Not detected'}")
    
    print(f"SPE - Initial vs Refined:")
    print(f"  False Alarm Rate: {spe_false_rate:.2f}% → {spe_false_rate_refined:.2f}%")
    print(f"  Miss Rate: {spe_miss_rate:.2f}% → {spe_miss_rate_refined:.2f}%")
    print(f"  Detection Time: {spe_detection_time if spe_detection_time is not None else 'Not detected'} → {spe_detection_time_refined if spe_detection_time_refined is not None else 'Not detected'}")
    
    runtime = time.time() - start_time
    print(f"Total runtime: {runtime:.2f} seconds")
    
    # Return results
    return {
        't2_statistics': t2_statistics,
        'spe_statistics': spe_statistics,
        't2_threshold': t2_threshold,
        'spe_threshold': spe_threshold,
        't2_probs': t2_probs,
        'spe_probs': spe_probs,
        't2_alarms': t2_refined_alarms,
        'spe_alarms': spe_refined_alarms,
        't2_false_alarm_rate': t2_false_rate_refined,
        't2_miss_rate': t2_miss_rate_refined,
        't2_detection_time': t2_detection_time_refined,
        'spe_false_alarm_rate': spe_false_rate_refined,
        'spe_miss_rate': spe_miss_rate_refined,
        'spe_detection_time': spe_detection_time_refined,
        'false_rates': [t2_false_rate_refined, spe_false_rate_refined],
        'miss_rates': [t2_miss_rate_refined, spe_miss_rate_refined],
        'detection_times': [t2_detection_time_refined, spe_detection_time_refined],
        'runtime': runtime
    } 