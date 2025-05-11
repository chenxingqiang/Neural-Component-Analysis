"""
Fault Detector Factory Module

This module provides a factory pattern implementation for creating different types of
fault detection methods, allowing easy switching between detection approaches.
"""

import numpy as np
from spe_fault_detector import SPEFaultDetector
from enhanced_transformer_detection import EnhancedTransformerModel

class FaultDetector:
    """Base interface for all fault detection methods"""
    
    def train(self, train_data):
        """Train the detector on normal operating data"""
        raise NotImplementedError("Subclasses must implement train()")
    
    def detect(self, test_data):
        """Detect faults in test data"""
        raise NotImplementedError("Subclasses must implement detect()")
    
    def evaluate(self, test_data, fault_start_idx):
        """Evaluate detector performance"""
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save(self, filepath):
        """Save the model"""
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, filepath):
        """Load the model"""
        raise NotImplementedError("Subclasses must implement load()")


class SPEFaultDetectorAdapter(FaultDetector):
    """Adapter for the SPE-based fault detector"""
    
    def __init__(self, input_dim, hidden_dim=None, confidence=0.99):
        self.detector = SPEFaultDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            confidence=confidence
        )
    
    def train(self, train_data, **kwargs):
        """Train the SPE detector"""
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 0.001)
        validation_split = kwargs.get('validation_split', 0.1)
        
        return self.detector.train(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            validation_split=validation_split
        )
    
    def detect(self, test_data):
        """Detect faults using SPE metrics"""
        fault_flags, spe_values = self.detector.detect_faults(test_data)
        return {
            'fault_flags': fault_flags,
            'metric_values': spe_values,
            'detection_type': 'SPE'
        }
    
    def evaluate(self, test_data, fault_start_idx):
        """Evaluate detector performance"""
        return self.detector.evaluate_performance(test_data, fault_start_idx)
    
    def save(self, filepath):
        """Save the model"""
        return self.detector.save_model(filepath)
    
    def load(self, filepath):
        """Load the model"""
        return self.detector.load_model(filepath)


class EnhancedTransformerAdapter(FaultDetector):
    """Adapter for the Enhanced Transformer model"""
    
    def __init__(self, input_dim, hidden_dim=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else min(32, input_dim)
        self.model = EnhancedTransformerModel(input_dim, self.hidden_dim)
        self.is_trained = False
        self.t2_limit = None
        self.spe_limit = None
    
    def train(self, train_data, **kwargs):
        """Train the enhanced transformer model"""
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        
        result = self.model.train_model(train_data, epochs, batch_size)
        self.is_trained = True
        
        # Calculate control limits
        self.t2_limit, self.spe_limit = self.model.calculate_control_limits(train_data)
        
        return result
    
    def detect(self, test_data):
        """Detect faults using T² and SPE metrics"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        t2_values, spe_values = self.model.calculate_statistics(test_data)
        
        t2_flags = t2_values > self.t2_limit
        spe_flags = spe_values > self.spe_limit
        
        # Combined detection (either T² or SPE exceeds limit)
        combined_flags = np.logical_or(t2_flags, spe_flags)
        
        return {
            'fault_flags': combined_flags,
            't2_values': t2_values,
            'spe_values': spe_values,
            'detection_type': 'Combined'
        }
    
    def evaluate(self, test_data, fault_start_idx):
        """Evaluate detector performance"""
        result = self.detect(test_data)
        
        fault_flags = result['fault_flags']
        t2_values = result['t2_values']
        spe_values = result['spe_values']
        
        # Calculate false alarm rate
        false_alarms = np.sum(fault_flags[:fault_start_idx])
        false_rate = 100 * false_alarms / fault_start_idx if fault_start_idx > 0 else 0
        
        # Calculate miss rate
        misses = np.sum(~fault_flags[fault_start_idx:])
        miss_rate = 100 * misses / (len(fault_flags) - fault_start_idx) if fault_start_idx < len(fault_flags) else 0
        
        # Detection time
        consecutive = 3
        detection_time = None
        
        for i in range(fault_start_idx, len(fault_flags) - consecutive + 1):
            if all(fault_flags[i:i+consecutive]):
                detection_time = i - fault_start_idx
                break
        
        return {
            'false_alarm_rate': false_rate,
            'miss_rate': miss_rate,
            'detection_time': detection_time,
            't2_values': t2_values,
            'spe_values': spe_values,
            'fault_flags': fault_flags
        }
    
    def save(self, filepath):
        """Save the model"""
        return self.model.save_model(filepath)
    
    def load(self, filepath):
        """Load the model"""
        success = self.model.load_model(filepath)
        if success:
            self.is_trained = True
        return success


class FaultDetectorFactory:
    """Factory for creating different fault detectors"""
    
    @staticmethod
    def create_detector(detector_type, input_dim, **kwargs):
        """
        Create a fault detector of the specified type
        
        Parameters:
        -----------
        detector_type : str
            Type of detector ('spe', 'transformer', 'hybrid')
        input_dim : int
            Input dimension for the model
        **kwargs : dict
            Additional parameters for the specific detector
            
        Returns:
        --------
        detector : FaultDetector
            A detector instance of the requested type
        """
        if detector_type.lower() == 'spe':
            hidden_dim = kwargs.get('hidden_dim', None)
            confidence = kwargs.get('confidence', 0.99)
            return SPEFaultDetectorAdapter(input_dim, hidden_dim, confidence)
        
        elif detector_type.lower() == 'transformer':
            hidden_dim = kwargs.get('hidden_dim', None)
            return EnhancedTransformerAdapter(input_dim, hidden_dim)
        
        elif detector_type.lower() == 'hybrid':
            # Hybrid detector that combines SPE and T² methods
            # Implementation left for future work
            raise NotImplementedError("Hybrid detector not yet implemented")
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")


# Example usage
def detector_factory_demo():
    """Demonstrate use of the FaultDetectorFactory"""
    from enhanced_transformer_detection import load_data
    
    # Load data
    X_train, X_test, happen = load_data(is_mock=True)
    input_dim = X_train.shape[1]
    
    print(f"Creating SPE-based fault detector for input dimension {input_dim}...")
    
    # Create detector using factory
    detector = FaultDetectorFactory.create_detector(
        detector_type='spe',
        input_dim=input_dim,
        hidden_dim=32,
        confidence=0.99
    )
    
    # Train or load the detector
    try:
        print("Trying to load pre-trained model...")
        success = detector.load('factory_spe_detector.pth')
        if not success:
            raise FileNotFoundError("Model not found")
    except:
        print("Training new model...")
        detector.train(X_train, epochs=50)
        detector.save('factory_spe_detector.pth')
    
    # Evaluate performance
    metrics = detector.evaluate(X_test, happen)
    
    # Print results
    print("\n===== Detector Performance =====")
    print(f"False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")
    print(f"Miss Rate: {metrics['miss_rate']:.2f}%")
    print(f"Detection Time: {metrics['detection_time'] if metrics['detection_time'] is not None else 'Not detected'}")
    
    return detector, metrics


if __name__ == "__main__":
    detector_factory_demo()

